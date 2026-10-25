# syntax=docker/dockerfile:1
# ==============================================================================
# Production Dockerfile for MMBCD Healthcare Inference Service
# Base Image: Python 3.10 slim
# ==============================================================================

FROM python:3.10-slim

LABEL maintainer="Principal AI Infrastructure Architect"
LABEL description="Production Django service for Mammogram Cancer Detection using FocalNet-DINO & MMBCD"

# Optimize Python: prevent writing bytecode and force unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PORT=8000 \
    DJANGO_SETTINGS_MODULE=mmbcd_service.settings

# Install system dependencies (OpenCV / DICOM rendering & healthcheck probes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root application user for healthcare HIPAA / security compliance
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs /app/weights /app/test_samples && \
    chown -R appuser:appuser /app

# Copy dependency manifest and install dependencies with cache mount
COPY requirements.txt /app/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application codebase (weights/ and test_samples/ excluded via .dockerignore)
COPY --chown=appuser:appuser . /app/

# Switch to non-root user
USER appuser

# Expose HTTP port 8000
EXPOSE 8000

# Container Healthcheck verifying service liveness
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/ || exit 1

# Production WSGI server command using Gunicorn
CMD ["gunicorn", "mmbcd_service.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
