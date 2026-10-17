"""
service_api/serializers.py
==========================
Django REST Framework serializers for request validation and structured schemas.

Author: Principal AI Infrastructure & Software Architect
"""

from rest_framework import serializers


class InferenceRequestSerializer(serializers.Serializer):
    """
    Validates multipart/form-data payload for /api/v1/infer/
    """
    dicom_file = serializers.FileField(
        required=True,
        help_text="Raw medical DICOM (.dcm) binary file."
    )
    clinical_history = serializers.CharField(
        required=False,
        default="Routine screening mammogram, no focal symptoms reported.",
        allow_blank=True,
        max_length=2000,
        help_text="Patient clinical history or clinical indication narrative."
    )
    uhid = serializers.CharField(
        required=False,
        default="",
        allow_blank=True,
        max_length=128,
        help_text="Unique Hospital / Patient Identifier (optional)."
    )
    top_k = serializers.IntegerField(
        required=False,
        default=5,
        min_value=1,
        max_value=20,
        help_text="Number of top lesion candidate ROIs to evaluate (default: 5)."
    )
    conf_threshold = serializers.FloatField(
        required=False,
        default=0.1,
        min_value=0.0,
        max_value=1.0,
        help_text="Minimum confidence threshold for lesion candidates (default: 0.1)."
    )
    iou_threshold = serializers.FloatField(
        required=False,
        default=0.1,
        min_value=0.0,
        max_value=1.0,
        help_text="IoU threshold for Non-Maximum Suppression (default: 0.1)."
    )

    def validate_dicom_file(self, value):
        """
        Validates file size limit and filename extension.
        """
        # Maximum allowed size: 100 MB
        max_size = 100 * 1024 * 1024
        if value.size > max_size:
            raise serializers.ValidationError(
                f"DICOM file size ({value.size / (1024**2):.1f} MB) exceeds maximum allowed limit (100 MB)."
            )

        name = value.name.lower() if value.name else ""
        # Accept standard DICOM extensions or raw DICOM streams
        valid_exts = (".dcm", ".dicom", ".ima", "")
        if not any(name.endswith(ext) for ext in valid_exts):
            raise serializers.ValidationError(
                f"File '{value.name}' does not have a recognized DICOM extension (.dcm, .dicom)."
            )

        return value
