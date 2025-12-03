# Atlas: Autonomously Teaching, Learning And Self-organizing System
# Multi-stage Dockerfile for cloud deployment

# ============================================
# Stage 1: Build Stage
# ============================================
FROM python:3.11-slim as builder

# Set build arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY self_organizing_av_system/requirements.txt /tmp/requirements.txt
COPY self_organizing_av_system/requirements_no_audio.txt /tmp/requirements_no_audio.txt

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU version (must use separate command with --index-url)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    opencv-python-headless \
    librosa \
    matplotlib \
    PyYAML \
    prometheus-client \
    redis \
    celery \
    boto3 \
    google-cloud-storage \
    azure-storage-blob \
    requests \
    pyyaml \
    watchdog \
    schedule

# ============================================
# Stage 2: Runtime Stage
# ============================================
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    ATLAS_HOME=/app \
    ATLAS_DATA_DIR=/data \
    ATLAS_CHECKPOINT_DIR=/data/checkpoints \
    ATLAS_LOG_DIR=/data/logs \
    ATLAS_INPUT_DIR=/data/input

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash atlas

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create directories
RUN mkdir -p /app /data/checkpoints /data/logs /data/input /data/output \
    && chown -R atlas:atlas /app /data

# Copy application code
WORKDIR /app
COPY --chown=atlas:atlas self_organizing_av_system/ /app/

# Copy cloud infrastructure code
COPY --chown=atlas:atlas cloud/ /app/cloud/

# Switch to non-root user
USER atlas

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command - can be overridden
ENTRYPOINT ["python"]
CMD ["-m", "cloud.atlas_service"]

# Expose ports
EXPOSE 8080 9090
