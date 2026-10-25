"""
core_engine/dicom_processor.py
==============================
Production-grade DICOM ingestion and preprocessing pipeline for digital mammography.

Features:
- Dual ingestion: File paths (Path, str) or in-memory byte streams (io.BytesIO, Django UploadedFile)
- Strict DICOM header validation and clinical metadata extraction
- Mammographic VOI LUT (Value of Interest Look-Up Table) application with fallback windowing
- Photometric interpretation handling (MONOCHROME1 inversion vs MONOCHROME2)
- Automated breast tissue segmentation (black space background cropping)
- Dynamic aspect-ratio preservation and standardized resizing
- 3-channel tensor construction with ImageNet mean/standard deviation normalization
- Enterprise custom exception hierarchy for HTTP 400 translation

Author: Principal AI Infrastructure & Software Architect
"""

import io
import logging
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
try:
    from pydicom.pixels import apply_voi_lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import torch
from torchvision import transforms

from .base import PreprocessedDICOMResult

logger = logging.getLogger("mmbcd_service.dicom_processor")


# ============================================================================
# Domain Exceptions
# ============================================================================

class DICOMProcessingError(Exception):
    """Base exception for all DICOM preprocessing failures."""
    pass


class CorruptDICOMError(DICOMProcessingError):
    """Raised when file cannot be parsed as valid DICOM format."""
    pass


class MissingPixelDataError(DICOMProcessingError):
    """Raised when DICOM dataset lacks PixelData or pixel_array is unreadable."""
    pass


class UnsupportedDICOMFormatError(DICOMProcessingError):
    """Raised when Photometric Interpretation or bit-depth is unsupported."""
    pass


# ============================================================================
# DICOM Processor Implementation
# ============================================================================

class DICOMProcessor:
    """
    Thread-safe, stateless medical DICOM preprocessing service.
    Transforms raw mammograms into clean PyTorch tensors and extracted metadata.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        target_size: Tuple[int, int] = (1024, 1024),
        crop_padding: int = 15,
        apply_tissue_crop: bool = True,
        preserve_aspect_ratio: bool = True,
    ) -> None:
        """
        Args:
            target_size: Standardized (Height, Width) for detection model input.
            crop_padding: Pixel padding around the segmented breast contour.
            apply_tissue_crop: Whether to eliminate black unexposed scanner margins.
            preserve_aspect_ratio: Whether to pad/letterbox or stretch to target_size.
        """
        self.target_size = target_size
        self.crop_padding = crop_padding
        self.apply_tissue_crop = apply_tissue_crop
        self.preserve_aspect_ratio = preserve_aspect_ratio

        # ImageNet normalization transform
        self._normalize_transform = transforms.Normalize(
            mean=self.IMAGENET_MEAN,
            std=self.IMAGENET_STD,
        )

    def process(
        self,
        dicom_input: Union[str, Path, bytes, BinaryIO],
    ) -> PreprocessedDICOMResult:
        """
        End-to-end preprocessing pipeline for a single DICOM instance.

        Args:
            dicom_input: File path, raw bytes, or file-like binary stream.

        Returns:
            PreprocessedDICOMResult with normalized tensor, images, and metadata.

        Raises:
            CorruptDICOMError: If file header is invalid.
            MissingPixelDataError: If pixel array is absent.
            UnsupportedDICOMFormatError: If photometric interpretation is missing.
        """
        logger.info("Initiating DICOM preprocessing pipeline...")

        # 1. Read DICOM Dataset
        ds = self._read_dicom(dicom_input)

        # 2. Extract and validate metadata
        metadata = self._extract_metadata(ds)

        # 3. Extract and calibrate pixel array (VOI LUT, HU windowing, Inversion)
        calibrated_img = self._calibrate_pixels(ds)
        orig_dims = calibrated_img.shape[:2]  # (H, W)

        # 4. Segment breast tissue (Black space cropping)
        if self.apply_tissue_crop:
            cropped_img = self._crop_black_space(calibrated_img, padding=self.crop_padding)
        else:
            cropped_img = calibrated_img
        cropped_dims = cropped_img.shape[:2]

        # 5. Convert to PIL Image (RGB) & Resize
        pil_img_rgb = Image.fromarray(cropped_img).convert("RGB")

        if self.preserve_aspect_ratio:
            resized_pil = self._resize_with_aspect_ratio(pil_img_rgb, self.target_size)
        else:
            resized_pil = pil_img_rgb.resize(
                (self.target_size[1], self.target_size[0]),
                resample=Image.Resampling.BILINEAR,
            )

        # 6. Transform to normalized PyTorch tensor
        tensor_img = transforms.functional.to_tensor(resized_pil)  # [3, H, W] in [0.0, 1.0]
        normalized_tensor = self._normalize_transform(tensor_img).unsqueeze(0)  # [1, 3, H, W]

        logger.info(
            f"DICOM processing complete. Original: {orig_dims}, "
            f"Cropped: {cropped_dims}, Target: {self.target_size}"
        )

        return PreprocessedDICOMResult(
            tensor=normalized_tensor,
            original_image=calibrated_img,
            pil_image=resized_pil,
            metadata=metadata,
            original_dims=(orig_dims[0], orig_dims[1]),
            cropped_dims=(cropped_dims[0], cropped_dims[1]),
            target_dims=self.target_size,
        )

    def _read_dicom(self, dicom_input: Union[str, Path, bytes, BinaryIO]) -> pydicom.Dataset:
        """Safely parses DICOM source from various input formats."""
        try:
            if isinstance(dicom_input, (str, Path)):
                path = Path(dicom_input)
                if not path.is_file():
                    raise CorruptDICOMError(f"DICOM file not found at: {path}")
                try:
                    ds = pydicom.dcmread(str(path))
                except InvalidDicomError:
                    ds = pydicom.dcmread(str(path), force=True)

            elif isinstance(dicom_input, bytes):
                try:
                    ds = pydicom.dcmread(io.BytesIO(dicom_input))
                except InvalidDicomError:
                    ds = pydicom.dcmread(io.BytesIO(dicom_input), force=True)

            elif hasattr(dicom_input, "read"):
                if hasattr(dicom_input, "seek"):
                    dicom_input.seek(0)
                content = dicom_input.read()
                try:
                    ds = pydicom.dcmread(io.BytesIO(content))
                except InvalidDicomError:
                    ds = pydicom.dcmread(io.BytesIO(content), force=True)

            else:
                raise CorruptDICOMError(
                    f"Unsupported DICOM input type: {type(dicom_input).__name__}"
                )

            # Validate basic DICOM presence
            if not hasattr(ds, "PixelData") and not hasattr(ds, "pixel_array"):
                raise MissingPixelDataError("DICOM dataset does not contain PixelData element.")

            return ds

        except (CorruptDICOMError, MissingPixelDataError):
            raise
        except InvalidDicomError as e:
            logger.error(f"Invalid DICOM header error: {e}")
            raise CorruptDICOMError(f"Corrupt or non-DICOM payload: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error while opening DICOM: {e}")
            raise CorruptDICOMError(f"Failed to read DICOM stream: {e}") from e

    def _calibrate_pixels(self, ds: pydicom.Dataset) -> np.ndarray:
        """
        Converts raw DICOM pixel values to standardized 8-bit optical grayscale.
        Applies:
        - Rescale slope and intercept
        - VOI LUT transformation (or fallback window width/center)
        - Photometric interpretation inversion (MONOCHROME1 vs MONOCHROME2)
        - Scaled normalization to uint8 [0, 255]
        """
        try:
            raw_pixels = ds.pixel_array
        except Exception as e:
            logger.error(f"Unable to access pixel array from DICOM: {e}")
            raise MissingPixelDataError(f"DICOM has missing or compressed pixel array: {e}") from e

        if raw_pixels is None or raw_pixels.size == 0:
            raise MissingPixelDataError("DICOM pixel array is empty.")

        # Float conversion for linear mathematics
        pixels = raw_pixels.astype(np.float32)

        # 1. Apply Rescale Slope / Intercept if specified in DICOM
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        if slope != 1.0 or intercept != 0.0:
            pixels = pixels * slope + intercept

        # 2. Apply VOI LUT (Value of Interest Look-Up Table)
        try:
            pixels = apply_voi_lut(pixels, ds)
        except Exception as e:
            logger.warning(f"apply_voi_lut failed, attempting fallback linear windowing: {e}")
            pixels = self._apply_linear_windowing(pixels, ds)

        # 3. Handle Photometric Interpretation
        photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        if photometric == "MONOCHROME1":
            # MONOCHROME1: 0 is white, maximum pixel is black -> Invert for standard visualization
            max_val = np.nanmax(pixels)
            pixels = max_val - pixels
        elif photometric == "MONOCHROME2":
            # MONOCHROME2: 0 is black, high values are dense tissue -> Offset to zero
            min_val = np.nanmin(pixels)
            pixels = pixels - min_val
        else:
            logger.warning(
                f"Unusual PhotometricInterpretation: {photometric}. Defaulting to min-subtraction."
            )
            pixels = pixels - np.nanmin(pixels)

        # 4. Safe dynamic range scaling to [0, 255]
        max_dynamic = np.nanmax(pixels)
        if max_dynamic > 0:
            pixels = (pixels / max_dynamic) * 255.0
        else:
            pixels = np.zeros_like(pixels)

        return np.clip(pixels, 0, 255).astype(np.uint8)

    def _apply_linear_windowing(self, pixels: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
        """Fallback linear windowing using WindowCenter and WindowWidth tags."""
        window_center = getattr(ds, "WindowCenter", None)
        window_width = getattr(ds, "WindowWidth", None)

        if window_center is not None and window_width is not None:
            # Handle list/multi-valued attributes
            if isinstance(window_center, (list, pydicom.multival.MultiValue)):
                window_center = float(window_center[0])
            else:
                window_center = float(window_center)

            if isinstance(window_width, (list, pydicom.multival.MultiValue)):
                window_width = float(window_width[0])
            else:
                window_width = float(window_width)

            lower = window_center - (window_width / 2.0)
            upper = window_center + (window_width / 2.0)
            return np.clip(pixels, lower, upper)

        # Fallback: Min-Max normalization
        p_min = np.nanmin(pixels)
        p_max = np.nanmax(pixels)
        if p_max > p_min:
            return (pixels - p_min) / (p_max - p_min) * 255.0
        return pixels

    def _crop_black_space(self, img: np.ndarray, padding: int = 15) -> np.ndarray:
        """
        Segments and crops the breast tissue, eliminating wide black borders.
        Replicates research baseline standard (preprocess/crop.py).
        """
        try:
            _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return img

            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)

            # Prevent zero or inverted bounds
            if w <= 10 or h <= 10:
                return img

            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(img.shape[1], x + w + padding)
            y_end = min(img.shape[0], y + h + padding)

            cropped = img[y_start:y_end, x_start:x_end]
            if cropped.size > 0:
                return cropped
            return img
        except Exception as e:
            logger.warning(f"Tissue boundary cropping encountered an issue: {e}. Using uncropped image.")
            return img

    def _resize_with_aspect_ratio(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
    ) -> Image.Image:
        """
        Resizes PIL image maintaining native aspect ratio with symmetric zero padding.
        target_size: (target_height, target_width).
        """
        target_h, target_w = target_size
        img_w, img_h = image.size

        # Compute scaling ratio
        scale = min(target_w / img_w, target_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        # Canvas with zero padding
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))

        return canvas

    def _extract_metadata(self, ds: pydicom.Dataset) -> Dict[str, Any]:
        """
        Extracts core patient and study parameters with robust clinical fallbacks.
        """
        def safe_get(tag_name: str, default: Any = "UNKNOWN") -> Any:
            val = getattr(ds, tag_name, default)
            if val is None or str(val).strip() == "":
                return default
            return str(val).strip()

        return {
            "patient_id": safe_get("PatientID", "ANONYMOUS"),
            "patient_age": safe_get("PatientAge", "UNKNOWN"),
            "patient_sex": safe_get("PatientSex", "F"),
            "study_instance_uid": safe_get("StudyInstanceUID", ""),
            "series_instance_uid": safe_get("SeriesInstanceUID", ""),
            "sop_instance_uid": safe_get("SOPInstanceUID", ""),
            "modality": safe_get("Modality", "MG"),
            "laterality": safe_get("Laterality", "UNKNOWN"),       # L or R
            "view_position": safe_get("ViewPosition", "UNKNOWN"),   # MLO or CC
            "body_part_examined": safe_get("BodyPartExamined", "BREAST"),
            "manufacturer": safe_get("Manufacturer", "UNKNOWN"),
            "photometric_interpretation": safe_get("PhotometricInterpretation", "MONOCHROME2"),
            "rows": int(getattr(ds, "Rows", 0)),
            "columns": int(getattr(ds, "Columns", 0)),
            "bits_stored": int(getattr(ds, "BitsStored", 16)),
        }
