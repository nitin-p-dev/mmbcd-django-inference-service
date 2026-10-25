"""
service_api/urls.py
===================
API routing table for the mmbcd_enterprise_service API endpoints.
"""

from django.urls import path
from .views import InferAPIView, HealthCheckAPIView

app_name = "service_api"

urlpatterns = [
    path("infer/", InferAPIView.as_view(), name="infer"),
    path("health/", HealthCheckAPIView.as_view(), name="health"),
]
