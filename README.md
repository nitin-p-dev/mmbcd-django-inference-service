# MMBCD Enterprise Mammography Inference Service

A production-grade, modular, and highly optimized Django-based deep learning inference service for mammogram cancer detection. The architecture orchestrates two state-of-the-art research models under a **Strict Sequential VRAM Execution Protocol**:

1. **FocalNet-DINO**: Region-of-Interest (lesion / mass / microcalcification) detection.
2. **MMBCD**: Multimodal Breast Cancer Detection fusing RoBERTa textual clinical history with Vision Transformer (ViT) ROI crop representations via Cross-Attention.

---

## Architectural Overview & Execution Pipeline

```
                                      [ CLINICAL DICOM REQUEST ]
                                    (Multipart POST: .dcm + Text)
                                                  │
                                                  ▼
                                     ┌─────────────────────────┐
                                     │   1. DICOM Processor    │
                                     │ ─────────────────────── │
                                     │ • Read Pydicom Stream   │
                                     │ • VOI LUT / Windowing   │
                                     │ • Monochr. Inversion    │
                                     │ • Breast Tissue Crop    │
                                     │ • Aspect-Ratio Resize   │
                                     │ • ImageNet Normalization│
                                     └────────────┬────────────┘
                                                  │
                                                  ▼
                                     ┌─────────────────────────┐
                              ┌─────►│  VRAM Mutex Lock (Hold) │◄────┐
                              │      └────────────┬────────────┘     │
                              │                   │                  │
    STAGE 1: LESION DETECTION │                   ▼                  │ STAGE 2: MULTIMODAL DIAGNOSIS
┌─────────────────────────────┴┐     ┌─────────────────────────┐    ┌┴─────────────────────────────┐
│ 2. Scoped Load: FocalNet-DINO│     │ 4. Intermediate Extract │    │ 5. Scoped Load: MMBCD        │
│ ──────────────────────────── │     │ ─────────────────────── │    │ ──────────────────────────── │
│ • Mount Model 1 on VRAM      │     │ • Extract top-K ROIs    │    │ • Model 1 is 100% PURGED     │
│ • Predict Bounding Boxes     │     │ • Crop patches from PIL │    │ • Mount Model 2 on VRAM      │
│ • Apply NMS (IoU <= 0.1)     │     │ • Standardize (224x224) │    │ • RoBERTa Tokenize History   │
│ • Explicit Model Unload      │     └────────────┬────────────┘    │ • Cross-Attention Fusion     │
│ • gc.collect() & empty_cache │                  │                 │ • Softmax Malignancy Logits  │
└──────────────────────────────┘                  │                 │ • Explicit Model Unload      │
                                                  │                 │ • gc.collect() & empty_cache │
                                                  │                 └──────────────┬───────────────┘
                                                  │                                │
                                                  ▼                                ▼
                                     ┌─────────────────────────────────────────────┐
                                     │          VRAM Mutex Lock (Released)         │
                                     │ ─────────────────────────────────────────── │
                                     │ Active Model in VRAM: None (Zero-Overlap)   │
                                     └──────────────────────┬──────────────────────┘
                                                            │
                                                            ▼
                                             [ JSON CLINICAL DIAGNOSIS ]
                                           (ROIs, Malignancy %, Latencies)
```

---

## Key Architectural Highlights

### 1. Strict Sequential VRAM Execution (Zero Overlap)
- **Problem**: Mounting both FocalNet-DINO (transformer object detector) and MMBCD (RoBERTa + ViT + cross-attention) simultaneously easily exhausts GPU VRAM on commodity or shared enterprise medical inference GPUs (e.g., NVIDIA T4 16GB, RTX 3090/4090, A10/A30).
- **Solution**: The `core_engine.vram_manager.VRAMManager` enforces a strict context-managed lifecycle:
  - `with vram_manager.scoped_model(focalnet) as detector:` -> Model 1 loads, predicts ROIs, explicitly unloads, triggers `gc.collect()`, `torch.cuda.empty_cache()`, and resets peak memory stats.
  - VRAM is 100% purged before Model 2 is loaded.
  - Thread-safe reentrant mutex prevents concurrent web requests from colliding in VRAM.

### 2. Medical Preprocessing Pipeline
- Handled via `core_engine.dicom_processor.DICOMProcessor`:
  - Directly processes `.dcm` streams from HTTP uploads without writing temporary files to disk.
  - **VOI LUT**: Value of Interest Look-Up Table transformation (`apply_voi_lut`) with fallback linear windowing (`WindowCenter`, `WindowWidth`).
  - **Photometric Inversion**: Inverts `MONOCHROME1` to ensure dense fibroglandular tissue is bright and background is dark.
  - **Tissue Background Cropping**: Automated contour thresholding eliminates wide scanner borders.
  - **Standardized Resizing**: Standardized 1024x1024 letterboxing preserving diagnostic aspect ratios.

### 3. Decoupled & Modular Design
- Django REST Framework views (`service_api/views.py`) are pure HTTP adapters.
- Complete ML logic lives in `core_engine/engine.py` (`MedicalInferenceEngine`).
- `core_engine/base.py` provides an abstract base class `BaseInferenceModel`, enabling seamless transition between native PyTorch and NVIDIA TensorRT runtime engines without breaking API contracts.

---

## Directory Structure

```
mmbcd_enterprise_service/
├── core_engine/
│   ├── __init__.py
│   ├── base.py                   # Data schemas, dataclasses, and BaseInferenceModel
│   ├── dicom_processor.py        # DICOM reader, VOI LUT, tissue cropping, ImageNet norm
│   ├── vram_manager.py           # Thread-safe sequential VRAM lifecycle manager
│   ├── model_wrappers.py         # FocalNet-DINO and MMBCD wrappers with NMS
│   └── engine.py                 # MedicalInferenceEngine orchestrator
├── mmbcd_service/
│   ├── __init__.py
│   ├── settings.py               # Enterprise Django configuration
│   ├── urls.py                   # Root URL routing
│   ├── asgi.py
│   └── wsgi.py
├── service_api/
│   ├── __init__.py
│   ├── apps.py
│   ├── serializers.py            # DRF serializers with DICOM validation
│   ├── views.py                  # InferAPIView & HealthCheckAPIView
│   └── urls.py                   # /api/v1/infer/ & /api/v1/health/
├── tests/
│   ├── __init__.py
│   └── test_end_to_end.py        # Full integration test suite
├── scripts/
│   ├── create_sample_dicom.py    # Generates standard 16-bit synthetic mammograms
│   └── run_validation.py         # Standalone CLI validation runner
├── external/
│   ├── FocalNet-DINO/            # Cloned research repository
│   └── MMBCD/                    # Cloned research repository
├── weights/
│   ├── focalnet/                 # Put FocalNet-DINO .pth weights here
│   └── mmbcd/                    # Put MMBCD .pth weights here
├── test_samples/
│   └── dicom/                    # Sample DICOM files
├── logs/                         # Structured service runtime logs
├── Dockerfile                    # Production CUDA runtime container definition
├── docker-compose.yml            # NVIDIA GPU container compose specification
├── requirements.txt              # Production dependency lock
└── manage.py
```

---

## REST API Specifications

### 1. Multimodal Mammogram Inference: `POST /api/v1/infer/`
- **Content-Type**: `multipart/form-data`

#### Parameters:
| Field | Type | Required | Description |
|---|---|---|---|
| `dicom_file` | File (`.dcm`) | **Yes** | Raw binary mammogram DICOM |
| `clinical_history`| String | No | Patient clinical notes / indication narrative |
| `uhid` | String | No | Unique Hospital / Patient ID |
| `top_k` | Integer | No | Number of top lesion crops (default: `5`) |
| `conf_threshold` | Float | No | Candidate detection threshold (default: `0.1`) |
| `iou_threshold` | Float | No | Candidate NMS IoU threshold (default: `0.1`) |

#### Sample Request (`curl`):
```bash
curl -X POST http://localhost:8000/api/v1/infer/ \
  -F "dicom_file=@test_samples/dicom/sample_mammogram.dcm" \
  -F "clinical_history=54yo female, palpable mass in right upper outer quadrant, dense fibroglandular parenchyma." \
  -F "uhid=UHID-MAMMO-2026-9081" \
  -F "top_k=5"
```

#### Sample Response (`200 OK`):
```json
{
  "status": "SUCCESS",
  "trace_id": "388e7b27-4664-46aa-9555-9b4e2bfde140",
  "patient_metadata": {
    "uhid": "UHID-MAMMO-2026-9081",
    "patient_id": "UHID-MAMMO-2026-9081",
    "patient_age": "054Y",
    "patient_sex": "F",
    "modality": "MG",
    "laterality": "R",
    "view_position": "MLO",
    "rows": 1024,
    "columns": 800,
    "bits_stored": 12,
    "clinical_indication": "54yo female, palpable mass in right upper outer quadrant, dense fibroglandular parenchyma."
  },
  "detection": {
    "rois_detected": [
      {
        "bbox_normalized": {"cx": 0.492, "cy": 0.496, "w": 0.515, "h": 0.473},
        "bbox_pixel": {"x1": 162, "y1": 261, "x2": 520, "y2": 738},
        "confidence": 0.5293,
        "class_id": 1,
        "label": "suspect_lesion"
      }
    ],
    "total_candidates_found": 20,
    "rois_evaluated": 5,
    "inference_time_ms": 7.58
  },
  "classification": {
    "prediction": "Malignant",
    "predicted_class": 1,
    "confidence": 0.5080,
    "malignancy_probability": 0.5080,
    "benign_probability": 0.4920,
    "cross_attention_weights": [0.201, 0.198, 0.204, 0.199, 0.198],
    "inference_time_ms": 31.96
  },
  "timing_metrics_ms": {
    "dicom_preprocessing_ms": 42.45,
    "focalnet_detection_ms": 97.68,
    "roi_cropping_ms": 1.44,
    "mmbcd_classification_ms": 3571.83,
    "total_pipeline_ms": 3714.18
  },
  "memory_metrics": {
    "baseline_pre_inference": {"active_model": null},
    "post_phase1_cleanup": {"active_model": null},
    "post_phase2_cleanup": {"active_model": null}
  }
}
```

---

### 2. Service Health & VRAM Probe: `GET /api/v1/health/`
```bash
curl -X GET http://localhost:8000/api/v1/health/
```

```json
{
  "status": "HEALTHY",
  "service": "MMBCD Mammography Inference Service",
  "version": "1.0.0",
  "device": "cuda:0",
  "cuda_available": true,
  "active_model_in_vram": null,
  "memory_telemetry": {
    "device": "cuda:0",
    "is_cuda": true,
    "active_model": null,
    "allocated_mb": 0.0,
    "reserved_mb": 0.0,
    "total_device_mb": 16160.5,
    "free_device_mb": 16160.5,
    "utilization_pct": 0.0
  }
}
```

---

## Error Handling Specifications

| Scenario | HTTP Status | Error Code | Behavior |
|---|---|---|---|
| Corrupt DICOM / missing header | `400 Bad Request` | `CORRUPT_DICOM` | Rejects payload before GPU allocation |
| Non-mammogram / missing pixels | `400 Bad Request` | `MISSING_PIXEL_DATA` | Returns clear clinical validation error |
| Invalid parameters (e.g. top_k > 20) | `400 Bad Request` | `INVALID_INPUT` | DRF serializer validation |
| VRAM Out-of-Memory | `503 Service Unavailable` | `GPU_VRAM_EXHAUSTED` | Triggers emergency VRAM cleanup, returns `Retry-After: 5` |
| Unexpected System Error | `500 Internal Server Error`| `INTERNAL_SERVER_ERROR` | Logs stack trace with unique `trace_id` |

---

## Validation & Verification

### 1. Run Complete Test Suite
```bash
python manage.py test tests
```

### 2. Run Standalone CLI Diagnostic Tool
```bash
python scripts/run_validation.py
```

---

## Docker & Container Deployment

### Local Development / Evaluation:
```bash
python manage.py runserver 0.0.0.0:8000
```

### Docker Compose (GPU Enabled):
```bash
docker compose up --build
```
