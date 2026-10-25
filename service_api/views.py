"""
service_api/views.py
====================
Django REST Framework API Views for Mammogram Cancer Detection Inference.

Endpoints:
- POST /api/v1/infer/   : Primary multipart DICOM inference endpoint
- GET  /api/v1/health/  : System telemetry, GPU/VRAM health check endpoint

Author: Principal AI Infrastructure & Software Architect
"""

import logging
import uuid

from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from core_engine.dicom_processor import (
    CorruptDICOMError,
    DICOMProcessingError,
    MissingPixelDataError,
    UnsupportedDICOMFormatError,
)
from core_engine.engine import get_inference_engine
from core_engine.vram_manager import VRAMOutOfMemoryException, get_vram_manager
from .serializers import InferenceRequestSerializer

logger = logging.getLogger("mmbcd_service.api")


class InferAPIView(APIView):
    """
    Primary Inference Endpoint for Multimodal Mammogram Cancer Detection.
    Accepts multipart/form-data with a raw DICOM file and optional clinical narrative.
    """
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs) -> Response:
        trace_id = str(uuid.uuid4())
        logger.info(f"[{trace_id}] Received inference request from {request.META.get('REMOTE_ADDR')}")

        # 1. Validate Form Input
        serializer = InferenceRequestSerializer(data=request.data)
        if not serializer.is_valid():
            logger.warning(f"[{trace_id}] Request validation failed: {serializer.errors}")
            return Response(
                {
                    "status": "ERROR",
                    "error_code": "INVALID_INPUT",
                    "trace_id": trace_id,
                    "details": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        data = serializer.validated_data
        dicom_file = data["dicom_file"]
        clinical_history = data.get("clinical_history", "")
        uhid = data.get("uhid", "")
        top_k = data.get("top_k", 5)
        conf_threshold = data.get("conf_threshold", 0.1)
        iou_threshold = data.get("iou_threshold", 0.1)

        # 2. Execute Orchestrated Engine
        try:
            engine = get_inference_engine()
            pipeline_result = engine.predict(
                dicom_source=dicom_file,
                clinical_history=clinical_history,
                uhid=uhid,
                top_k=top_k,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

            response_data = pipeline_result.to_dict()
            response_data["trace_id"] = trace_id
            return Response(response_data, status=status.HTTP_200_OK)

        # 3. Medical DICOM Processing Exceptions -> HTTP 400
        except CorruptDICOMError as e:
            logger.error(f"[{trace_id}] Corrupt DICOM uploaded: {e}")
            return Response(
                {
                    "status": "ERROR",
                    "error_code": "CORRUPT_DICOM",
                    "trace_id": trace_id,
                    "message": str(e),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        except MissingPixelDataError as e:
            logger.error(f"[{trace_id}] Missing DICOM pixel data: {e}")
            return Response(
                {
                    "status": "ERROR",
                    "error_code": "MISSING_PIXEL_DATA",
                    "trace_id": trace_id,
                    "message": str(e),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        except (UnsupportedDICOMFormatError, DICOMProcessingError) as e:
            logger.error(f"[{trace_id}] DICOM processing failure: {e}")
            return Response(
                {
                    "status": "ERROR",
                    "error_code": "UNSUPPORTED_DICOM_FORMAT",
                    "trace_id": trace_id,
                    "message": str(e),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 4. VRAM Out-of-Memory Exception -> HTTP 503 Service Unavailable
        except VRAMOutOfMemoryException as e:
            logger.critical(f"[{trace_id}] GPU VRAM Out-Of-Memory failure: {e}")
            return Response(
                {
                    "status": "ERROR",
                    "error_code": "GPU_VRAM_EXHAUSTED",
                    "trace_id": trace_id,
                    "message": "The GPU inference server temporarily exhausted VRAM resources. Please retry shortly.",
                    "details": str(e),
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={"Retry-After": "5"},
            )

        # 5. Unexpected System Faults -> HTTP 500
        except Exception as e:
            logger.exception(f"[{trace_id}] Unhandled internal exception: {e}")
            return Response(
                {
                    "status": "ERROR",
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "trace_id": trace_id,
                    "message": "An unexpected error occurred during deep learning inference.",
                    "details": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class HealthCheckAPIView(APIView):
    """
    Readiness & Liveness probe returning GPU/VRAM state and system telemetry.
    """
    def get(self, request, *args, **kwargs) -> Response:
        vram_mgr = get_vram_manager()
        snapshot = vram_mgr.get_memory_snapshot()

        return Response(
            {
                "status": "HEALTHY",
                "service": "MMBCD Mammography Inference Service",
                "version": "1.0.0",
                "device": str(vram_mgr.device),
                "cuda_available": vram_mgr.is_cuda,
                "active_model_in_vram": vram_mgr.active_model,
                "memory_telemetry": snapshot,
            },
            status=status.HTTP_200_OK,
        )
