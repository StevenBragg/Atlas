# Atlas Cloud Deployment Guide

## Overview

This guide explains how to deploy Atlas to the cloud, enabling it to learn and grow autonomously. Atlas can be deployed using several methods:

1. **Docker Compose** - Local development and testing
2. **Kubernetes** - Production cloud deployment
3. **AWS with Terraform** - Full cloud infrastructure
4. **Salad Cloud** - Affordable distributed GPU network (recommended for cost-effective GPU learning)

## Quick Start

### Option 1: Docker Compose (Fastest)

```bash
# Build and start Atlas locally
docker-compose up -d

# View logs
docker-compose logs -f atlas

# Check health
curl http://localhost:8080/health

# View Grafana dashboards
open http://localhost:3000  # admin/atlas-admin
```

### Option 2: Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/rbac.yaml
kubectl apply -f kubernetes/storage.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

# Check status
kubectl -n atlas get pods

# View logs
kubectl -n atlas logs -f deployment/atlas-brain
```

### Option 3: Salad Cloud (Recommended for GPU Learning)

```bash
# Set up Salad Cloud credentials
export SALAD_API_KEY="your-api-key"
export SALAD_ORG_NAME="your-org"
export SALAD_PROJECT_NAME="atlas"
export CONTAINER_IMAGE="docker.io/YOUR_USERNAME/atlas:salad-latest"

# Deploy Atlas to Salad Cloud
cd salad-cloud
./deploy.sh deploy

# Check status
./deploy.sh status

# Scale replicas
./deploy.sh scale 5
```

See [salad-cloud/README.md](salad-cloud/README.md) for full documentation.

### Option 4: AWS with Terraform

```bash
cd terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var-file=terraform.tfvars

# Apply infrastructure
terraform apply -var-file=terraform.tfvars

# Get kubectl config
aws eks update-kubeconfig --region us-west-2 --name atlas-cluster
```

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              CLOUD INFRASTRUCTURE           │
                    │                                             │
┌───────────────┐   │  ┌─────────────────────────────────────┐   │
│  Data Sources │   │  │           Atlas Brain               │   │
│               │   │  │  ┌─────────────────────────────┐    │   │
│  - Video      │──▶│  │  │  Unified Super Intelligence │    │   │
│  - Audio      │   │  │  │  - Visual Processing        │    │   │
│  - Images     │   │  │  │  - Audio Processing         │    │   │
│  - Streams    │   │  │  │  - Multimodal Association   │    │   │
└───────────────┘   │  │  │  - Memory Systems           │    │   │
                    │  │  │  - Reasoning & Planning     │    │   │
                    │  │  │  - Self-Improvement         │    │   │
                    │  │  └─────────────────────────────┘    │   │
                    │  └─────────────────────────────────────┘   │
                    │                    │                       │
                    │                    ▼                       │
                    │  ┌─────────────────────────────────────┐   │
                    │  │         Persistent Storage          │   │
                    │  │  - Checkpoints (learned knowledge)  │   │
                    │  │  - Episodic Memory                  │   │
                    │  │  - Semantic Memory                  │   │
                    │  └─────────────────────────────────────┘   │
                    │                    │                       │
                    │                    ▼                       │
                    │  ┌─────────────────────────────────────┐   │
                    │  │           Monitoring                │   │
                    │  │  - Prometheus (metrics)             │   │
                    │  │  - Grafana (dashboards)             │   │
                    │  │  - Health checks                    │   │
                    │  └─────────────────────────────────────┘   │
                    └─────────────────────────────────────────────┘
```

## Components

### 1. Atlas Brain (`atlas-brain`)
The core learning system that:
- Processes audio-visual input continuously
- Learns through self-organization (no supervision needed)
- Builds episodic and semantic memory
- Develops abstract reasoning capabilities
- Self-improves over time

### 2. Data Ingestion (`atlas-ingestion`)
Continuously feeds data to Atlas:
- Processes video files
- Processes image files
- Generates synthetic training data
- Supports streaming sources

### 3. Persistent Storage
Stores learned knowledge:
- **Checkpoints**: Neural pathway states (.pkl files)
- **S3/MinIO**: Long-term storage for data and models

### 4. Monitoring
Observability stack:
- **Prometheus**: Collects metrics
- **Grafana**: Visualizes learning progress

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ATLAS_LOG_LEVEL` | Logging verbosity | `INFO` |
| `ATLAS_CHECKPOINT_DIR` | Checkpoint storage path | `/data/checkpoints` |
| `ATLAS_INPUT_DIR` | Input data directory | `/data/input` |
| `ATLAS_CHECKPOINT_INTERVAL` | Frames between checkpoints | `1000` |
| `ATLAS_MAX_CHECKPOINTS` | Max checkpoints to keep | `10` |
| `ATLAS_LEARNING_RATE` | Learning rate | `0.01` |
| `ATLAS_MULTIMODAL_SIZE` | Multimodal layer size | `128` |
| `ATLAS_ENABLE_UNIFIED_INTELLIGENCE` | Enable full cognitive systems | `true` |
| `ATLAS_ENABLE_EXPLORATION` | Enable synthetic data exploration | `true` |
| `ATLAS_HTTP_PORT` | HTTP server port | `8080` |

### Scaling

Atlas supports several scaling mechanisms:

1. **Vertical Scaling**: Increase CPU/memory for faster processing
2. **Horizontal Scaling**: Scale data ingestion pods
3. **GPU Acceleration**: Enable NVIDIA GPU nodes for faster learning

```yaml
# Enable GPU nodes in Terraform
enable_gpu_nodes = true
```

## Feeding Data to Atlas

### Option 1: Drop Files in Input Directory

```bash
# Copy videos/images to input directory
cp my_video.mp4 /data/input/
cp my_images/*.jpg /data/input/

# Atlas will automatically process and archive them
```

### Option 2: Upload to S3 (AWS)

```bash
# Upload training data
aws s3 cp training_videos/ s3://atlas-cluster-data-xxx/input/ --recursive
```

### Option 3: Stream Data (Advanced)

Implement custom data sources in `cloud/data_ingestion.py`.

## Monitoring Atlas's Growth

### Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "running",
  "uptime_seconds": 3600,
  "total_frames_processed": 50000,
  "total_learning_cycles": 50,
  "has_unified_intelligence": true
}
```

### Prometheus Metrics

```bash
curl http://localhost:8080/metrics
```

Key metrics:
- `atlas_frames_processed_total`: Total frames processed
- `atlas_learning_cycles_total`: Learning cycles completed
- `atlas_intelligence_score`: Current intelligence quotient
- `atlas_creativity_index`: Creativity measurement
- `atlas_processing_time_seconds`: Processing latency

### Grafana Dashboard

Access at `http://localhost:3000` (or your Grafana URL):
- **Atlas Overview**: Main dashboard showing learning progress
- **Processing Rate**: Frames per second
- **Intelligence Quotient**: Measured IQ over time
- **Memory Growth**: Episodic and semantic memory size

## How Atlas Learns Autonomously

1. **Self-Organizing Sensory Processing**
   - Visual features emerge from exposure to images
   - Audio features emerge from sound patterns
   - No labeled data required

2. **Multimodal Association**
   - Links between audio and visual patterns form automatically
   - Cross-modal prediction develops

3. **Memory Formation**
   - **Episodic**: Stores significant experiences
   - **Semantic**: Builds knowledge graph of concepts
   - **Working**: Maintains active context

4. **Self-Improvement**
   - Monitors own performance
   - Optimizes learning parameters
   - Restructures neural pathways

5. **Exploration**
   - Generates synthetic patterns for diverse learning
   - Curiosity-driven discovery

## Production Recommendations

### Security
- Use private subnets for Atlas brain
- Enable encryption for S3 buckets
- Use IAM roles (IRSA) for service accounts
- Enable network policies in Kubernetes

### Reliability
- Set up multiple availability zones
- Configure automatic checkpoint backups
- Enable persistent volume snapshots
- Set up alerting for anomalies

### Cost Optimization
- Use spot instances for data ingestion
- Scale down during low activity
- Use lifecycle policies for old checkpoints
- Consider reserved instances for steady workloads

## Troubleshooting

### Atlas Not Processing Data

```bash
# Check pod logs
kubectl -n atlas logs deployment/atlas-brain

# Check input directory
kubectl -n atlas exec deployment/atlas-brain -- ls -la /data/input

# Check health
kubectl -n atlas exec deployment/atlas-brain -- curl localhost:8080/health
```

### Checkpoints Not Saving

```bash
# Check storage
kubectl -n atlas exec deployment/atlas-brain -- df -h /data/checkpoints

# Check permissions
kubectl -n atlas exec deployment/atlas-brain -- ls -la /data/checkpoints
```

### High Memory Usage

Reduce memory footprint:
```yaml
ATLAS_MULTIMODAL_SIZE: "64"  # Reduce from 128
```

## Support

- GitHub Issues: https://github.com/StevenBragg/Atlas/issues
- Documentation: See `README.md` and `SUPERINTELLIGENCE_ANALYSIS.md`
