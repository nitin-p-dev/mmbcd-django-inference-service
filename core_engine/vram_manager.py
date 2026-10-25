"""
core_engine/vram_manager.py
===========================
Thread-safe, context-managed sequential Model LifeCycle and VRAM Manager.

Enforces Architectural Mandate #1: STRICT SEQUENTIAL VRAM EXECUTION (Zero Overlap).
Ensures that two heavy deep learning models (FocalNet-DINO and MMBCD) NEVER reside
in GPU memory simultaneously.

Features:
- Global reentrant mutex locking for thread-safe sequential execution across workers
- Strict lifecycle guarantees: load -> infer -> explicit unload -> garbage collection -> CUDA cache purge
- Complete telemetry tracking: memory allocated, reserved, peak VRAM before/after inference
- Graceful CPU fallback when CUDA device is absent or disabled
- Emergency OOM recovery handling with custom VRAMOutOfMemoryException

Author: Principal AI Infrastructure & Software Architect
"""

from contextlib import contextmanager
import gc
import logging
import threading
from typing import Any, Dict, Generator, Optional
import psutil
import torch

from .base import BaseInferenceModel

logger = logging.getLogger("mmbcd_service.vram_manager")


class VRAMOutOfMemoryException(Exception):
    """Raised when GPU VRAM runs out during model loading or inference execution."""
    pass


class VRAMManager:
    """
    Singleton / Centralized Lifecycle and Memory Management Authority.
    Controls physical allocation, context scopes, and deterministic eviction.
    """

    _instance: Optional["VRAMManager"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "VRAMManager":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, default_device: Optional[str] = None) -> None:
        if getattr(self, "_initialized", False):
            return

        self._mutex = threading.RLock()
        self._active_model_name: Optional[str] = None
        self._active_model_ref: Optional[BaseInferenceModel] = None

        if default_device is not None:
            self.device = torch.device(default_device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._initialized = True
        logger.info(f"VRAMManager initialized. Primary Target Device: {self.device}")

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    @property
    def active_model(self) -> Optional[str]:
        return self._active_model_name

    def get_memory_snapshot(self) -> Dict[str, Any]:
        """
        Gathers accurate memory metrics for the active device.
        """
        snapshot: Dict[str, Any] = {
            "device": str(self.device),
            "is_cuda": self.is_cuda,
            "active_model": self._active_model_name,
        }

        if self.is_cuda:
            alloc_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            res_mb = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            max_alloc_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            total_mb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)

            snapshot.update({
                "allocated_mb": round(alloc_mb, 2),
                "reserved_mb": round(res_mb, 2),
                "max_allocated_mb": round(max_alloc_mb, 2),
                "total_device_mb": round(total_mb, 2),
                "free_device_mb": round(total_mb - res_mb, 2),
                "utilization_pct": round((res_mb / total_mb) * 100, 2) if total_mb > 0 else 0.0,
            })
        else:
            process = psutil.Process()
            mem_info = process.memory_info()
            snapshot.update({
                "process_rss_mb": round(mem_info.rss / (1024 ** 2), 2),
                "process_vms_mb": round(mem_info.vms / (1024 ** 2), 2),
                "system_ram_pct": psutil.virtual_memory().percent,
            })

        return snapshot

    def cleanup_memory(self, reset_peak_stats: bool = True) -> Dict[str, Any]:
        """
        Executes aggressive, multi-stage garbage collection and GPU cache purging.
        Guarantees that unreferenced computational graphs and weights are purged.
        """
        # 1. Python GC pass
        collected = gc.collect()

        # 2. PyTorch CUDA cache clearing
        if self.is_cuda:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            if reset_peak_stats:
                torch.cuda.reset_peak_memory_stats(self.device)

        snapshot = self.get_memory_snapshot()
        logger.debug(
            f"VRAM cleanup executed (collected {collected} objects). "
            f"Current state: {snapshot}"
        )
        return snapshot

    @contextmanager
    def execution_lock(self) -> Generator[None, None, None]:
        """
        Ensures atomic pipeline execution so concurrent requests do not interleave
        their model loading stages and cause VRAM collisions.
        """
        with self._mutex:
            yield

    @contextmanager
    def scoped_model(
        self,
        model: BaseInferenceModel,
    ) -> Generator[BaseInferenceModel, None, None]:
        """
        Context Manager enforcing ZERO-OVERLAP sequential execution.

        Usage:
            with vram_manager.scoped_model(focalnet_detector) as detector:
                rois = detector.predict(tensor)
            # Here, focalnet_detector is completely purged from VRAM.

            with vram_manager.scoped_model(mmbcd_classifier) as classifier:
                diagnosis = classifier.predict(rois, text)
            # Here, mmbcd_classifier is completely purged from VRAM.

        Guarantees:
        - Only ONE model is mounted in VRAM at any time.
        - Previous model, if any, is forcibly evicted.
        - Memory before and after is tracked and logged.
        - Out-of-Memory exceptions trigger emergency memory reclaim.
        """
        with self._mutex:
            # 1. Enforce zero overlap: If a model is somehow lingering, evict it
            if self._active_model_ref is not None and self._active_model_ref is not model:
                logger.warning(
                    f"Found lingering active model '{self._active_model_name}'. Forcing eviction."
                )
                try:
                    self._active_model_ref.unload()
                except Exception as e:
                    logger.error(f"Error during force-unload of '{self._active_model_name}': {e}")
                self._active_model_name = None
                self._active_model_ref = None
                self.cleanup_memory()

            # 2. Pre-load memory baseline
            mem_pre_load = self.get_memory_snapshot()
            logger.info(
                f"[VRAM Lifecycle] Loading model '{model.model_name}' on {self.device}. "
                f"Memory pre-load: {mem_pre_load}"
            )

            # 3. Load model with OOM trap
            try:
                model.load(self.device)
                self._active_model_name = model.model_name
                self._active_model_ref = model
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                    logger.critical(f"FATAL: Out-Of-Memory while loading '{model.model_name}': {e}")
                    self.emergency_cleanup()
                    raise VRAMOutOfMemoryException(
                        f"VRAM Out-Of-Memory while loading '{model.model_name}': {e}"
                    ) from e
                raise

            mem_post_load = self.get_memory_snapshot()
            logger.info(
                f"[VRAM Lifecycle] Model '{model.model_name}' successfully mounted. "
                f"Memory post-load: {mem_post_load}"
            )

            try:
                # 4. Yield execution context to caller
                yield model

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                    logger.critical(
                        f"FATAL: Out-Of-Memory during forward inference on '{model.model_name}': {e}"
                    )
                    self.emergency_cleanup()
                    raise VRAMOutOfMemoryException(
                        f"VRAM Out-Of-Memory during inference on '{model.model_name}': {e}"
                    ) from e
                raise

            finally:
                # 5. Deterministic Unload & VRAM purge
                logger.info(f"[VRAM Lifecycle] Unloading model '{model.model_name}' from VRAM...")
                try:
                    model.unload()
                except Exception as e:
                    logger.error(f"Error while unloading '{model.model_name}': {e}")

                self._active_model_name = None
                self._active_model_ref = None

                # 6. Aggressive cache flush
                mem_after_cleanup = self.cleanup_memory()
                logger.info(
                    f"[VRAM Lifecycle] Eviction & VRAM purge complete for '{model.model_name}'. "
                    f"Memory post-cleanup: {mem_after_cleanup}"
                )

    def emergency_cleanup(self) -> None:
        """
        Emergency evacuation called when an unhandled OOM error occurs.
        Drops all active references and reclaims maximum possible VRAM.
        """
        logger.warning("[VRAM Lifecycle] EMERGENCY CLEANUP INITIATED.")
        if self._active_model_ref is not None:
            try:
                self._active_model_ref.unload()
            except Exception as e:
                logger.error(f"Error in emergency unload: {e}")
            self._active_model_ref = None
            self._active_model_name = None

        self.cleanup_memory(reset_peak_stats=True)
        logger.warning("[VRAM Lifecycle] Emergency cleanup finished.")


# Global instance provider
_global_vram_manager: Optional[VRAMManager] = None


def get_vram_manager() -> VRAMManager:
    """Returns the singleton instance of VRAMManager."""
    global _global_vram_manager
    if _global_vram_manager is None:
        _global_vram_manager = VRAMManager()
    return _global_vram_manager
