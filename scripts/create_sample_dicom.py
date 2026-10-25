"""
scripts/create_sample_dicom.py
==============================
Generates a realistic, standard-compliant synthetic digital mammography DICOM
file for validation, CI/CD, and integration testing.

Synthesizes:
- Medical DICOM preamble and file meta information
- Standard DICOM Part 3 clinical tags (MG modality, MLO view, Laterality, VOI LUT)
- Realistic breast phantom geometry: parabolic breast boundary, pectoral muscle,
  and a simulated suspect lesion in the upper quadrant.

Author: Principal AI Infrastructure & Software Architect
"""

from pathlib import Path
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


def generate_synthetic_mammogram_dicom(
    output_path: Path,
    height: int = 1024,
    width: int = 800,
) -> Path:
    """
    Creates a valid 16-bit mammography DICOM image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Create File Meta Information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    # 2. Create Dataset
    ds = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Patient & Clinical Metadata
    ds.PatientName = "MAMMOGRAM^TEST^PATIENT"
    ds.PatientID = "UHID-MAMMO-2026-9081"
    ds.PatientBirthDate = "19720514"
    ds.PatientSex = "F"
    ds.PatientAge = "054Y"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "MG"
    ds.Laterality = "R"
    ds.ViewPosition = "MLO"
    ds.BodyPartExamined = "BREAST"
    ds.Manufacturer = "EnterpriseMed Imaging Systems"
    ds.StudyDate = "20261018"
    ds.SeriesDescription = "Full Field Digital Mammography (Right MLO)"

    # Image Plane & Representation
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = height
    ds.Columns = width
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0  # Unsigned integer

    # Medical Windowing (VOI LUT parameters)
    ds.WindowCenter = 2048
    ds.WindowWidth = 4096
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1

    # 3. Synthesize Anatomical Tissue Gradients
    y_coords, x_coords = np.mgrid[:height, :width]
    image = np.zeros((height, width), dtype=np.float32)

    # Breast profile curve (right laterality facing left)
    # Parabolic breast contour
    center_y = height * 0.52
    breast_mask = ((y_coords - center_y) ** 2 / (0.45 * height) ** 2 +
                   (x_coords - width) ** 2 / (0.85 * width) ** 2) <= 1.0
    breast_mask &= (x_coords < width * 0.95)

    # Fibroglandular density gradient (dense core tapering toward skin margin)
    dist_from_margin = np.clip(1.0 - (x_coords / (width * 0.9)), 0.0, 1.0)
    density = np.sin(dist_from_margin * np.pi) * 2200.0 + 800.0
    image[breast_mask] = density[breast_mask]

    # Pectoral muscle shadow in upper right corner
    pectoral_mask = (y_coords < height * 0.35) & (x_coords > width * 0.65)
    pectoral_line = (y_coords / (height * 0.35)) + ((width - x_coords) / (width * 0.35))
    image[pectoral_mask & (pectoral_line < 1.0)] += 600.0

    # Suspect focal lesion / mass in upper-middle quadrant
    lesion_cy = int(height * 0.42)
    lesion_cx = int(width * 0.45)
    lesion_rad = 35.0
    lesion_dist = np.sqrt((x_coords - lesion_cx) ** 2 + (y_coords - lesion_cy) ** 2)
    lesion_mask = lesion_dist <= lesion_rad
    image[lesion_mask] += np.exp(-((lesion_dist[lesion_mask] / lesion_rad) ** 2)) * 950.0

    # Add subtle medical sensor noise
    noise = np.random.normal(0, 15, (height, width))
    image[breast_mask] += noise[breast_mask]

    # Clip to 12-bit dynamic range [0, 4095]
    final_pixels = np.clip(image, 0, 4095).astype(np.uint16)
    ds.PixelData = final_pixels.tobytes()

    ds.save_as(str(output_path), write_like_original=False)
    print(f"Synthetic mammogram DICOM successfully generated: {output_path}")
    return output_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "test_samples" / "dicom"
    sample_file = out_dir / "sample_mammogram.dcm"
    generate_synthetic_mammogram_dicom(sample_file)
