"""
core_engine/engine.py
=====================
Production Medical Inference Orchestrator.

Enforces Architectural Mandates:
1. Strict Sequential VRAM Execution (Zero Overlap):
   [DICOM Preprocessing] -> [Load Model 1] -> [Detect ROIs] -> [Unload & Purge GPU]
   -> [Crop ROIs] -> [Load Model 2] -> [Predict Cancer Probability] -> [Unload & Purge GPU]
2. Medical Preprocessing: VOI LUT, dynamic aspect-ratio resizing, ImageNet normalization
3. Decoupling: Fully abstracted from Django REST framework request/response cycle

Author: Principal AI Infrastructure & Software Architect
"""

import logging
from pathlib import Path
import time
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch

from .base import (
    BBoxProposal,
    ClassificationResult,
    DetectionResult,
    InferencePipelineResponse,
    PreprocessedDICOMResult,
    RuntimeBackend,
)
from .dicom_processor import DICOMProcessor
from .model_wrappers import FocalNetDINOWrapper, MMBCDWrapper
from .vram_manager import VRAMManager, VRAMOutOfMemoryException, get_vram_manager

logger = logging.getLogger("mmbcd_service.orchestrator")


class MedicalInferenceEngine:
    """
    Production-grade Healthcare Inference Orchestrator.
    Executes the sequential multimodal pipeline with zero GPU memory overlap.
    """

    def __init__(
        self,
        focalnet_weights_path: Optional[str] = None,
        focalnet_config_path: Optional[str] = None,
        mmbcd_weights_path: Optional[str] = None,
        roberta_checkpoint_path: Optional[str] = None,
        backend: RuntimeBackend = RuntimeBackend.PYTORCH,
        device_override: Optional[str] = None,
        top_k: int = 5,
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.1,
    ) -> None:
        """
        Initializes the service engine and subordinate processors.
        Does NOT load models into GPU memory at initialization time,
        preserving idle resources.
        """
        self.vram_manager: VRAMManager = get_vram_manager()
        if device_override is not None:
            self.vram_manager.device = torch.device(device_override)

        self.backend = backend
        self.top_k = top_k
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Autodetect weights paths if relative paths provided
        base_dir = Path(__file__).resolve().parent.parent
        self.focalnet_weights = (
            focalnet_weights_path
            or str(base_dir / "weights" / "focalnet" / "focalnet_dino_best.pth")
        )
        self.focalnet_config = focalnet_config_path
        self.mmbcd_weights = (
            mmbcd_weights_path
            or str(base_dir / "weights" / "mmbcd" / "mmbcd_best.pth")
        )
        self.roberta_checkpoint = roberta_checkpoint_path

        # Instantiate DICOM processor
        self.dicom_processor = DICOMProcessor(
            target_size=(1024, 1024),
            crop_padding=15,
            apply_tissue_crop=True,
            preserve_aspect_ratio=True,
        )

        logger.info(
            f"MedicalInferenceEngine initialized. Target Device: {self.vram_manager.device}, "
            f"FocalNet weights: {self.focalnet_weights}, MMBCD weights: {self.mmbcd_weights}"
        )

    def predict(
        self,
        dicom_source: Union[str, Path, bytes, BinaryIO],
        clinical_history: str = "Routine screening mammogram, no focal symptoms reported.",
        uhid: Optional[str] = None,
        top_k: Optional[int] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> InferencePipelineResponse:
        """
        Executes end-to-end sequential inference on a mammogram DICOM.

        Execution Flow:
        1. Preprocess DICOM (CPU) -> Calibrated 1024x1024 tensor + metadata
        2. Acquire VRAM Mutex (thread-safe sequential barrier)
        3. Phase 1 (Model 1): Load FocalNet-DINO -> Predict ROIs -> Evict & Flush VRAM
        4. CPU ROI Cropping: Extract top-K bounding patches from preprocessed image
        5. Phase 2 (Model 2): Load MMBCD -> Predict Malignancy -> Evict & Flush VRAM
        6. Package structured JSON-ready diagnostic response

        Args:
            dicom_source: DICOM file path, raw bytes, or stream.
            clinical_history: Narrative clinical notes or clinical indication.
            uhid: Unique Hospital / Patient Identifier (optional).
            top_k: Number of highest-confidence ROIs to analyze (default: 5).
            conf_threshold: Confidence threshold for candidate detection.
            iou_threshold: IoU threshold for candidate NMS.

        Returns:
            InferencePipelineResponse containing complete clinical & operational data.

        Raises:
            DICOMProcessingError: If DICOM parsing/windowing fails.
            VRAMOutOfMemoryException: If GPU runs out of memory during inference.
            Exception: For unexpected system faults.
        """
        pipeline_start = time.perf_counter()
        timing: Dict[str, float] = {}
        memory_telemetry: Dict[str, Any] = {}
        warnings: List[str] = []

        k = top_k or self.top_k
        c_thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        i_thresh = iou_threshold if iou_threshold is not None else self.iou_threshold

        logger.info("Starting Medical Inference Pipeline...")

        # --------------------------------------------------------------------
        # STAGE 1: MEDICAL DICOM PREPROCESSING
        # --------------------------------------------------------------------
        t0 = time.perf_counter()
        preprocessed: PreprocessedDICOMResult = self.dicom_processor.process(dicom_source)
        timing["dicom_preprocessing_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Merge / override UHID if provided by clinician or API
        patient_meta = dict(preprocessed.metadata)
        if uhid:
            patient_meta["uhid"] = uhid
        elif "patient_id" in patient_meta:
            patient_meta["uhid"] = patient_meta["patient_id"]

        patient_meta["clinical_indication"] = clinical_history.strip()

        # --------------------------------------------------------------------
        # STAGE 2: SEQUENTIAL VRAM EXECUTION BARRIER
        # --------------------------------------------------------------------
        with self.vram_manager.execution_lock():
            memory_telemetry["baseline_pre_inference"] = self.vram_manager.get_memory_snapshot()

            # ----------------------------------------------------------------
            # PHASE 1: FOCALNET-DINO (LESION / ROI DETECTION)
            # ----------------------------------------------------------------
            t_det = time.perf_counter()
            focalnet_detector = FocalNetDINOWrapper(
                checkpoint_path=self.focalnet_weights,
                config_path=self.focalnet_config,
                backend=self.backend,
                confidence_threshold=c_thresh,
                iou_threshold=i_thresh,
                top_k=k,
            )

            # Strict context: FocalNet-DINO is loaded -> inferred -> completely unloaded
            with self.vram_manager.scoped_model(focalnet_detector) as detector:
                detection_result: DetectionResult = detector.predict(
                    image_tensor=preprocessed.tensor,
                    original_shape=preprocessed.cropped_dims,
                    top_k=k,
                    conf_threshold=c_thresh,
                    iou_threshold=i_thresh,
                )

            timing["focalnet_detection_ms"] = round((time.perf_counter() - t_det) * 1000.0, 2)
            memory_telemetry["post_phase1_cleanup"] = self.vram_manager.get_memory_snapshot()

            # VERIFY ZERO OVERLAP
            assert self.vram_manager.active_model is None, "Invariance violation: Model 1 still in memory!"

            # ----------------------------------------------------------------
            # INTERMEDIATE STAGE: ROI CROPPING & PATCH EXTRACTION
            # ----------------------------------------------------------------
            t_crop = time.perf_counter()
            roi_crops = self._extract_roi_patches(
                source_image=preprocessed.pil_image,
                proposals=detection_result.proposals,
                target_dims=preprocessed.target_dims,
            )
            timing["roi_cropping_ms"] = round((time.perf_counter() - t_crop) * 1000.0, 2)

            # ----------------------------------------------------------------
            # PHASE 2: MMBCD (MULTIMODAL MALIGNANCY DIAGNOSIS)
            # ----------------------------------------------------------------
            t_cls = time.perf_counter()
            mmbcd_classifier = MMBCDWrapper(
                checkpoint_path=self.mmbcd_weights,
                roberta_checkpoint=self.roberta_checkpoint,
                backend=self.backend,
                roi_img_size=224,
            )

            # Strict context: MMBCD is loaded -> inferred -> completely unloaded
            with self.vram_manager.scoped_model(mmbcd_classifier) as classifier:
                classification_result: ClassificationResult = classifier.predict(
                    roi_crops=roi_crops,
                    clinical_text=clinical_history,
                )

            timing["mmbcd_classification_ms"] = round((time.perf_counter() - t_cls) * 1000.0, 2)
            memory_telemetry["post_phase2_cleanup"] = self.vram_manager.get_memory_snapshot()

            # VERIFY ZERO OVERLAP
            assert self.vram_manager.active_model is None, "Invariance violation: Model 2 still in memory!"

        timing["total_pipeline_ms"] = round((time.perf_counter() - pipeline_start) * 1000.0, 2)

        logger.info(
            f"End-to-end inference complete in {timing['total_pipeline_ms']:.2f}ms. "
            f"Diagnosis: {classification_result.prediction_label} "
            f"(P_Malignant: {classification_result.malignancy_score:.4f})"
        )

        # --------------------------------------------------------------------
        # STAGE 3: ASSEMBLE ENTERPRISE RESPONSE
        # --------------------------------------------------------------------
        detection_payload = {
            "rois_detected": [p.to_dict() for p in detection_result.proposals],
            "total_candidates_found": detection_result.total_candidates,
            "rois_evaluated": len(detection_result.proposals),
            "inference_time_ms": round(detection_result.inference_time_ms, 2),
        }

        classification_payload = classification_result.to_dict()

        return InferencePipelineResponse(
            status="SUCCESS",
            patient_metadata=patient_meta,
            detection=detection_payload,
            classification=classification_payload,
            timing_metrics_ms=timing,
            memory_metrics=memory_telemetry,
            errors_or_warnings=warnings,
        )

    def _extract_roi_patches(
        self,
        source_image: Image.Image,
        proposals: List[BBoxProposal],
        target_dims: Tuple[int, int],
    ) -> List[Image.Image]:
        """
        Crops lesion candidate regions from the standardized PIL image canvas.
        Replicates MMBCD data.py `convert_yolo_pascal` and `create_crops`.
        """
        w_img, h_img = source_image.size
        crops: List[Image.Image] = []

        for p in proposals:
            # Reconstruct Pascal coordinates [x1, y1, x2, y2]
            cx, cy, w, h = p.cx, p.cy, p.w, p.h

            x1 = max(0, int((cx - w / 2.0) * w_img))
            y1 = max(0, int((cy - h / 2.0) * h_img))
            x2 = min(w_img, int((cx + w / 2.0) * w_img))
            y2 = min(h_img, int((cy + h / 2.0) * h_img))

            # Ensure non-zero crop area
            if x2 <= x1:
                x2 = min(w_img, x1 + 32)
            if y2 <= y1:
                y2 = min(h_img, y1 + 32)

            crop = source_image.crop((x1, y1, x2, y2))
            crops.append(crop)

        # Fallback if no crops generated
        if not crops:
            logger.warning("No ROIs available for cropping; defaulting to center patch.")
            cx, cy = w_img // 2, h_img // 2
            crops.append(source_image.crop((cx - 112, cy - 112, cx + 112, cy + 112)))

        return crops


# Singleton instance accessor
_global_engine: Optional[MedicalInferenceEngine] = None


def get_inference_engine() -> MedicalInferenceEngine:
    """Returns the globally configured singleton MedicalInferenceEngine."""
    global _global_engine
    if _global_engine is None:
        _global_engine = MedicalInferenceEngine()
    return _global_engine
