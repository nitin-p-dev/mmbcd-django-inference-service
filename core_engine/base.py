"""
core_engine/base.py
===================
Enterprise architectural abstraction layer defining contracts, data schemas,
and interfaces for deep learning inference backends (PyTorch, TensorRT, Mock).

Author: Principal AI Infrastructure & Software Architect
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import torch


class ModelType(str, Enum):
    FOCALNET_DINO = "focalnet_dino"
    MMBCD = "mmbcd"


class RuntimeBackend(str, Enum):
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    ONNX_RUNTIME = "onnx_runtime"
    SIMULATION = "simulation"


@dataclass(frozen=True)
class BBoxProposal:
    """
    Standardized bounding box proposal structure.
    Coordinates are normalized [0.0, 1.0] in YOLO format [cx, cy, w, h]
    and also converted to absolute pixel [x1, y1, x2, y2].
    """
    cx: float
    cy: float
    w: float
    h: float
    confidence: float
    class_id: int = 0
    label: str = "lesion"
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox_normalized": {
                "cx": round(float(self.cx), 5),
                "cy": round(float(self.cy), 5),
                "w": round(float(self.w), 5),
                "h": round(float(self.h), 5),
            },
            "bbox_pixel": {
                "x1": int(self.x1),
                "y1": int(self.y1),
                "x2": int(self.x2),
                "y2": int(self.y2),
            },
            "confidence": round(float(self.confidence), 4),
            "class_id": self.class_id,
            "label": self.label,
        }


@dataclass
class DetectionResult:
    """Detection stage output containing proposed regions of interest."""
    proposals: List[BBoxProposal]
    total_candidates: int
    inference_time_ms: float
    model_name: str
    backend: RuntimeBackend


@dataclass
class ClassificationResult:
    """Multimodal classification stage output."""
    malignancy_score: float
    benign_score: float
    predicted_class: int  # 0: Benign, 1: Malignant
    prediction_label: str  # 'Malignant' or 'Benign'
    confidence: float
    cross_attention_weights: Optional[List[float]] = None
    inference_time_ms: float = 0.0
    model_name: str = ""
    backend: RuntimeBackend = RuntimeBackend.PYTORCH

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction_label,
            "predicted_class": self.predicted_class,
            "confidence": round(float(self.confidence), 4),
            "malignancy_probability": round(float(self.malignancy_score), 4),
            "benign_probability": round(float(self.benign_score), 4),
            "cross_attention_weights": [round(float(w), 4) for w in self.cross_attention_weights]
            if self.cross_attention_weights is not None
            else None,
            "inference_time_ms": round(float(self.inference_time_ms), 2),
        }


@dataclass
class PreprocessedDICOMResult:
    """Structured container for preprocessed DICOM data and clinical tags."""
    tensor: torch.Tensor                     # Shape: (1, 3, H, W) normalized ImageNet
    original_image: Any                      # np.ndarray uint8 2D grayscale
    pil_image: Any                           # PIL.Image.Image in RGB
    metadata: Dict[str, Any]                 # Extracted DICOM headers
    original_dims: Tuple[int, int]           # (Height, Width) of raw DICOM
    cropped_dims: Tuple[int, int]            # (Height, Width) post black-border crop
    target_dims: Tuple[int, int]             # (Height, Width) fed to detector


@dataclass
class InferencePipelineResponse:
    """Complete orchestrated response delivered to caller/API."""
    status: str
    patient_metadata: Dict[str, Any]
    detection: Dict[str, Any]
    classification: Dict[str, Any]
    timing_metrics_ms: Dict[str, float]
    memory_metrics: Dict[str, Any]
    errors_or_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "patient_metadata": self.patient_metadata,
            "detection": self.detection,
            "classification": self.classification,
            "timing_metrics_ms": self.timing_metrics_ms,
            "memory_metrics": self.memory_metrics,
            "errors_or_warnings": self.errors_or_warnings,
        }


class BaseInferenceModel(ABC):
    """
    Abstract Model Wrapper Interface.
    Enables plug-and-play interchangeability between native PyTorch modules,
    TorchScript, and high-performance TensorRT runtime execution engines
    without breaking API contracts or the sequential lifecycle orchestrator.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        backend: RuntimeBackend = RuntimeBackend.PYTORCH,
    ) -> None:
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.backend = backend
        self._is_loaded: bool = False
        self._device: torch.device = torch.device("cpu")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> torch.device:
        return self._device

    @abstractmethod
    def load(self, device: torch.device) -> None:
        """
        Loads weights into memory and mounts the network onto target device.
        Must handle CUDA/CPU context placement gracefully.
        """
        pass

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Executes forward inference under torch.no_grad() or TensorRT context.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Releases references and unbinds computational graphs to ensure
        deterministic garbage collection and complete VRAM reclamation.
        """
        pass
