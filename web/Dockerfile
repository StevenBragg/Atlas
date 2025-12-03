# Atlas Web Stack Dockerfile
# Multi-stage build for React frontend + Python backend

# ============================================
# Stage 1: Build React Frontend
# ============================================
FROM node:20-alpine as frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm install

# Copy source code
COPY frontend/ ./

# Build production bundle
RUN npm run build

# ============================================
# Stage 2: Python Backend with Frontend
# ============================================
FROM python:3.11-slim as production

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy Atlas core (parent directory)
COPY ../self_organizing_av_system/ ./self_organizing_av_system/

# Copy built frontend to serve as static files
COPY --from=frontend-builder /app/frontend/dist ./static/

# Create directories
RUN mkdir -p checkpoints logs

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ATLAS_WEB_PORT=8000
ENV ATLAS_WEB_HOST=0.0.0.0
ENV ATLAS_CHECKPOINT_DIR=/app/checkpoints

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the backend with static file serving
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
