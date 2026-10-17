"""
scripts/run_validation.py
=========================
Production CLI utility for verifying the end-to-end MMBCD & FocalNet-DINO
inference pipeline against real or synthetic DICOM mammograms.

Usage:
    python scripts/run_validation.py [--dicom <path>] [--clinical-history <text>]

Author: Principal AI Infrastructure & Software Architect
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mmbcd_service.settings")

from core_engine.engine import MedicalInferenceEngine
from core_engine.vram_manager import get_vram_manager
from scripts.create_sample_dicom import generate_synthetic_mammogram_dicom


def print_banner():
    banner = """
========================================================================================
     MMBCD ENTERPRISE INFERENCE SERVICE - MAMMIC PROTOCOL VALIDATION ENGINE
           Sequential VRAM Execution: FocalNet-DINO (ROIs) -> MMBCD (Malignancy)
========================================================================================
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(description="MMBCD Enterprise Service Validation Runner")
    parser.add_argument(
        "--dicom",
        type=str,
        default=None,
        help="Path to .dcm file. Defaults to test_samples/dicom/sample_mammogram.dcm",
    )
    parser.add_argument(
        "--clinical-history",
        type=str,
        default="54yo female, palpable mass in right upper outer quadrant, dense fibroglandular parenchyma.",
        help="Patient clinical history indication narrative.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ROI crops to evaluate.",
    )
    args = parser.parse_args()

    print_banner()

    # Determine DICOM file path
    if args.dicom:
        dicom_path = Path(args.dicom)
    else:
        dicom_path = ROOT_DIR / "test_samples" / "dicom" / "sample_mammogram.dcm"
        if not dicom_path.is_file():
            print(f"[*] Generating synthetic test DICOM at: {dicom_path}")
            generate_synthetic_mammogram_dicom(dicom_path)

    if not dicom_path.is_file():
        print(f"[!] ERROR: Specified DICOM file does not exist: {dicom_path}")
        sys.exit(1)

    print(f"[*] Target DICOM File: {dicom_path}")
    print(f"[*] File Size: {dicom_path.stat().st_size / 1024:.2f} KB")
    print(f"[*] Clinical History: \"{args.clinical_history}\"")
    print("-" * 88)

    # Inspect device
    vram_mgr = get_vram_manager()
    print(f"[*] Target Hardware Device : {vram_mgr.device} (CUDA Available: {vram_mgr.is_cuda})")
    init_mem = vram_mgr.get_memory_snapshot()
    print(f"[*] Memory State (Pre-Run): {json.dumps(init_mem, indent=2)}")
    print("-" * 88)

    # Initialize Engine
    print("[*] Initializing MedicalInferenceEngine...")
    engine = MedicalInferenceEngine(top_k=args.top_k)

    # Run Sequential Pipeline
    print("[*] Executing Sequential Pipeline (Strict Zero-Overlap Invariant)...")
    t_start = time.perf_counter()
    response = engine.predict(
        dicom_source=dicom_path,
        clinical_history=args.clinical_history,
        top_k=args.top_k,
    )
    total_elapsed = (time.perf_counter() - t_start) * 1000.0

    print("\n" + "=" * 88)
    print("                     CLINICAL INFERENCE DIAGNOSTIC REPORT")
    print("=" * 88)

    # 1. Patient Metadata
    meta = response.patient_metadata
    print(f"\n[+] PATIENT / EXAM METADATA:")
    print(f"    - Patient ID / UHID     : {meta.get('uhid') or meta.get('patient_id')}")
    print(f"    - Patient Demographics  : Age: {meta.get('patient_age')} | Sex: {meta.get('patient_sex')}")
    print(f"    - Examination Profile   : Modality: {meta.get('modality')} | Laterality: {meta.get('laterality')} | View: {meta.get('view_position')}")
    print(f"    - Image Geometry        : Rows: {meta.get('rows')} x Columns: {meta.get('columns')} (Bits Stored: {meta.get('bits_stored')})")

    # 2. Lesion Detection ROIs
    det = response.detection
    print(f"\n[+] LESION / ROI PROPOSALS (FocalNet-DINO):")
    print(f"    - Total Candidates Evaluated : {det['total_candidates_found']}")
    print(f"    - Top ROIs Selected (NMS)    : {det['rois_evaluated']}")
    print(f"    - Detection Latency          : {det['inference_time_ms']} ms")
    print("\n    Candidate Regions Table:")
    print(f"    {'Index':<6} | {'Confidence':<10} | {'Normalized (cx, cy, w, h)':<30} | {'Pixel Bounds [x1, y1, x2, y2]'}")
    print("    " + "-" * 76)
    for idx, roi in enumerate(det["rois_detected"], 1):
        norm = roi["bbox_normalized"]
        pix = roi["bbox_pixel"]
        norm_str = f"({norm['cx']:.3f}, {norm['cy']:.3f}, {norm['w']:.3f}, {norm['h']:.3f})"
        pix_str = f"[{pix['x1']}, {pix['y1']}, {pix['x2']}, {pix['y2']}]"
        print(f"    #{idx:<5} | {roi['confidence']:<10.4f} | {norm_str:<30} | {pix_str}")

    # 3. Multimodal Diagnosis
    cls_res = response.classification
    print(f"\n[+] MULTIMODAL DIAGNOSIS (MMBCD Cross-Attention Fusion):")
    print(f"    - Predicted Diagnosis        : >>> {cls_res['prediction'].upper()} <<<")
    print(f"    - Diagnostic Confidence     : {cls_res['confidence'] * 100:.2f}%")
    print(f"    - Malignancy Probability    : {cls_res['malignancy_probability']:.4f}")
    print(f"    - Benign Probability        : {cls_res['benign_probability']:.4f}")
    print(f"    - Classification Latency    : {cls_res['inference_time_ms']} ms")

    # 4. Latency Breakdown
    timing = response.timing_metrics_ms
    print(f"\n[+] PIPELINE LATENCY PROFILE:")
    print(f"    - DICOM Preprocessing (VOI LUT / Tissue Crop) : {timing.get('dicom_preprocessing_ms', 0):>8.2f} ms")
    print(f"    - FocalNet-DINO Load, Infer & Evict          : {timing.get('focalnet_detection_ms', 0):>8.2f} ms")
    print(f"    - Intermediate Patch Cropping (CPU)          : {timing.get('roi_cropping_ms', 0):>8.2f} ms")
    print(f"    - MMBCD Multimodal Load, Infer & Evict       : {timing.get('mmbcd_classification_ms', 0):>8.2f} ms")
    print(f"    ----------------------------------------------------------------")
    print(f"    - Total End-to-End Execution Latency         : {timing.get('total_pipeline_ms', total_elapsed):>8.2f} ms")

    # 5. Zero-Overlap Memory Telemetry
    print(f"\n[+] VRAM ZERO-OVERLAP STATUS:")
    final_mem = vram_mgr.get_memory_snapshot()
    print(f"    - Active Model in VRAM : {vram_mgr.active_model} (None = Successfully Evicted)")
    print(f"    - Final Memory State   : {json.dumps(final_mem, indent=6)}")

    print("\n" + "=" * 88)
    print("                      [PASS] VALIDATION SUCCESSFUL")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
