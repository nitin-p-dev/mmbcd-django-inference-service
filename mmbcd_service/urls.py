"""
mmbcd_service/urls.py
=====================
Root URL Configuration for the MMBCD Inference Service.
"""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include("service_api.urls", namespace="service_api")),
]
