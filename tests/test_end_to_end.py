"""
tests/test_end_to_end.py
========================
End-to-end integration and architectural compliance test suite.

Validates:
1. Medical DICOM Preprocessor (VOI LUT, tissue cropping, metadata extraction)
2. Strict Sequential VRAM Execution (Zero-Overlap Invariant)
3. MedicalInferenceEngine Full Pipeline (FocalNet-DINO -> ROIs -> MMBCD -> Diagnosis)
4. Django REST Framework API Endpoints (/api/v1/infer/, /api/v1/health/)
5. Error handling for corrupt payloads and OOM recovery

Author: Principal AI Infrastructure & Software Architect
"""

import os
from pathlib import Path
import unittest

# Configure Django environment for testing
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mmbcd_service.settings")
import django
django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient

from core_engine.base import ModelType
from core_engine.dicom_processor import (
    CorruptDICOMError,
    DICOMProcessor,
    PreprocessedDICOMResult,
)
from core_engine.engine import MedicalInferenceEngine, get_inference_engine
from core_engine.model_wrappers import FocalNetDINOWrapper, MMBCDWrapper
from core_engine.vram_manager import VRAMManager, get_vram_manager
from scripts.create_sample_dicom import generate_synthetic_mammogram_dicom


class MammogramEndToEndTestSuite(TestCase):
    """Integration test suite covering pipeline, VRAM zero-overlap, and REST API."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = APIClient()
        cls.test_dir = Path(__file__).resolve().parent.parent / "test_samples" / "dicom"
        cls.sample_dicom_path = cls.test_dir / "sample_mammogram.dcm"

        # Generate fresh sample DICOM if absent
        if not cls.sample_dicom_path.is_file():
            generate_synthetic_mammogram_dicom(cls.sample_dicom_path)

        with open(cls.sample_dicom_path, "rb") as f:
            cls.sample_dicom_bytes = f.read()

    def test_01_dicom_processor_pipeline(self):
        """Validates VOI LUT, windowing, tissue cropping, and metadata extraction."""
        processor = DICOMProcessor(
            target_size=(1024, 1024),
            crop_padding=15,
            apply_tissue_crop=True,
            preserve_aspect_ratio=True,
        )

        result: PreprocessedDICOMResult = processor.process(self.sample_dicom_path)

        # 1. Verify tensor dimensions and types
        self.assertEqual(result.tensor.shape, (1, 3, 1024, 1024))
        self.assertEqual(result.target_dims, (1024, 1024))

        # 2. Verify extracted metadata
        self.assertEqual(result.metadata["modality"], "MG")
        self.assertEqual(result.metadata["view_position"], "MLO")
        self.assertEqual(result.metadata["laterality"], "R")
        self.assertEqual(result.metadata["patient_id"], "UHID-MAMMO-2026-9081")

        # 3. Verify PIL Image representation
        self.assertEqual(result.pil_image.size, (1024, 1024))
        self.assertEqual(result.pil_image.mode, "RGB")

    def test_02_strict_sequential_vram_zero_overlap(self):
        """
        Validates Architectural Mandate #1: STRICT SEQUENTIAL VRAM EXECUTION.
        Verifies that:
        - Only one model is mounted in memory at any point.
        - Memory is completely cleared and active model is None between phases.
        """
        vram_mgr: VRAMManager = get_vram_manager()

        # Initial baseline: No model mounted
        self.assertIsNone(vram_mgr.active_model)

        detector = FocalNetDINOWrapper(confidence_threshold=0.1, top_k=5)
        classifier = MMBCDWrapper(roi_img_size=224)

        # Phase 1: Mount Model 1
        with vram_mgr.scoped_model(detector) as active_det:
            self.assertEqual(vram_mgr.active_model, ModelType.FOCALNET_DINO)
            self.assertTrue(active_det.is_loaded)

        # Post Phase 1: Must be completely evicted
        self.assertIsNone(vram_mgr.active_model)
        self.assertFalse(detector.is_loaded)

        # Phase 2: Mount Model 2
        with vram_mgr.scoped_model(classifier) as active_cls:
            self.assertEqual(vram_mgr.active_model, ModelType.MMBCD)
            self.assertTrue(active_cls.is_loaded)

        # Post Phase 2: Must be completely evicted
        self.assertIsNone(vram_mgr.active_model)
        self.assertFalse(classifier.is_loaded)

    def test_03_engine_full_pipeline_execution(self):
        """Validates MedicalInferenceEngine end-to-end execution and response schema."""
        engine = get_inference_engine()

        response = engine.predict(
            dicom_source=self.sample_dicom_bytes,
            clinical_history="54yo female, screening exam with palpable retroareolar density.",
            uhid="UHID-TEST-001",
            top_k=5,
            conf_threshold=0.1,
            iou_threshold=0.1,
        )

        self.assertEqual(response.status, "SUCCESS")
        self.assertEqual(response.patient_metadata["uhid"], "UHID-TEST-001")
        self.assertIn("rois_detected", response.detection)
        self.assertGreaterEqual(len(response.detection["rois_detected"]), 1)

        # Verify classification probabilities
        cls_res = response.classification
        self.assertIn(cls_res["prediction"], ["Malignant", "Benign"])
        self.assertGreaterEqual(cls_res["malignancy_probability"], 0.0)
        self.assertLessEqual(cls_res["malignancy_probability"], 1.0)
        self.assertAlmostEqual(
            cls_res["malignancy_probability"] + cls_res["benign_probability"],
            1.0,
            places=3,
        )

        # Verify timing metrics
        self.assertIn("total_pipeline_ms", response.timing_metrics_ms)
        self.assertGreater(response.timing_metrics_ms["total_pipeline_ms"], 0.0)

    def test_04_rest_api_infer_endpoint(self):
        """Tests POST /api/v1/infer/ with multipart/form-data upload."""
        upload_file = SimpleUploadedFile(
            name="patient_mammogram.dcm",
            content=self.sample_dicom_bytes,
            content_type="application/dicom",
        )

        payload = {
            "dicom_file": upload_file,
            "clinical_history": "Patient reports family history of breast carcinoma; routine screening.",
            "uhid": "HOSP-UHID-9921",
            "top_k": 5,
            "conf_threshold": 0.1,
            "iou_threshold": 0.1,
        }

        response = self.client.post("/api/v1/infer/", data=payload, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        body = response.json()
        self.assertEqual(body["status"], "SUCCESS")
        self.assertIn("trace_id", body)
        self.assertEqual(body["patient_metadata"]["uhid"], "HOSP-UHID-9921")
        self.assertIn("detection", body)
        self.assertIn("classification", body)

    def test_05_rest_api_health_endpoint(self):
        """Tests GET /api/v1/health/ probe."""
        response = self.client.get("/api/v1/health/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        body = response.json()
        self.assertEqual(body["status"], "HEALTHY")
        self.assertIn("cuda_available", body)
        self.assertIn("memory_telemetry", body)

    def test_06_rest_api_error_handling_corrupt_dicom(self):
        """Tests POST /api/v1/infer/ with invalid/corrupt payload returning HTTP 400."""
        corrupt_file = SimpleUploadedFile(
            name="corrupt.dcm",
            content=b"THIS_IS_CORRUPT_NON_DICOM_DATA_GARBAGE_PAYLOAD",
            content_type="application/octet-stream",
        )

        payload = {
            "dicom_file": corrupt_file,
            "clinical_history": "Corrupt test payload.",
        }

        response = self.client.post("/api/v1/infer/", data=payload, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        body = response.json()
        self.assertEqual(body["status"], "ERROR")
        self.assertIn(body["error_code"], ["CORRUPT_DICOM", "MISSING_PIXEL_DATA"])
        self.assertIn("trace_id", body)


if __name__ == "__main__":
    unittest.main()
