"""
service_api/apps.py
===================
Django Application Configuration for the Healthcare Inference Service.
"""

from django.apps import AppConfig


class ServiceApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "service_api"
    verbose_name = "Mammogram MMBCD Inference API"
