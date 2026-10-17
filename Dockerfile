# ==============================================================================
# Production Dockerfile for MMBCD Healthcare Inference Service
# Base Image: Official PyTorch 2.1.2 with CUDA 11.8 & cuDNN 8 runtime
# ==============================================================================

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

LABEL maintainer="Principal AI Infrastructure Architect"
LABEL description="Production Django service for Mammogram Cancer Detection using FocalNet-DINO & MMBCD"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PORT=8000 \
    DJANGO_SETTINGS_MODULE=mmbcd_service.settings

# Install system dependencies (OpenCV, DICOM rendering, healthcheck curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Create non-root application user for healthcare HIPAA / security compliance
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs /app/weights/focalnet /app/weights/mmbcd /app/test_samples && \
    chown -R appuser:appuser /app

# Copy dependency manifest and install wheels
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY --chown=appuser:appuser . /app/

# Switch to non-root user
USER appuser

# Expose HTTP port
EXPOSE 8000

# Container Healthcheck verifying service liveness and GPU readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/ || exit 1

# Production WSGI server command (1 worker, 4 threads to prevent GPU VRAM concurrency conflicts)
CMD ["gunicorn", "mmbcd_service.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "300", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
