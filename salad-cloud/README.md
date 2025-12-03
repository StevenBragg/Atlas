# Atlas on Salad Cloud

Deploy Atlas to Salad Cloud's distributed GPU network for cost-effective autonomous learning.

## Why Salad Cloud?

- **Affordable GPUs**: RTX 3090/4090 from $0.02-0.25/hour
- **Distributed Network**: 60,000+ GPUs available
- **Perfect for AI**: Designed for ML/AI workloads
- **Auto-scaling**: Scale replicas based on demand

## Quick Start

### 1. Get Salad Cloud Account

1. Sign up at [salad.com](https://salad.com)
2. Create an organization and project
3. Generate an API key from the portal

### 2. Set Environment Variables

```bash
export SALAD_API_KEY="your-api-key"
export SALAD_ORG_NAME="your-organization"
export SALAD_PROJECT_NAME="atlas"
export CONTAINER_IMAGE="docker.io/YOUR_USERNAME/atlas:salad-latest"
```

### 3. Deploy Atlas

```bash
cd salad-cloud

# Full deployment (build, push, deploy)
./deploy.sh deploy

# Or step by step:
./deploy.sh build    # Build Docker image
./deploy.sh push     # Push to registry
./deploy.sh deploy   # Create container group
```

### 4. Monitor

```bash
# Check status
./deploy.sh status

# View instances
./deploy.sh instances

# Scale replicas
./deploy.sh scale 5
```

## GPU Configurations

| Config | GPUs | CPU | RAM | Cost/hr | Best For |
|--------|------|-----|-----|---------|----------|
| `high_performance` | RTX 4090 | 8 | 32GB | ~$0.25 | Maximum learning speed |
| `balanced` | 3090/4090 | 4 | 16GB | ~$0.10 | Good balance |
| `cost_optimized` | Any | 2 | 8GB | ~$0.05 | Budget-friendly |

Set configuration:
```bash
export GPU_CONFIG=balanced  # or high_performance, cost_optimized
./deploy.sh deploy
```

## Architecture on Salad Cloud

```
┌─────────────────────────────────────────────────────────────┐
│                    SALAD CLOUD NETWORK                      │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Node 1     │  │  Node 2     │  │  Node 3     │   ...   │
│  │  RTX 4090   │  │  RTX 3090   │  │  RTX 3090Ti │         │
│  │             │  │             │  │             │         │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │         │
│  │ │  Atlas  │ │  │ │  Atlas  │ │  │ │  Atlas  │ │         │
│  │ │ Brain   │ │  │ │ Brain   │ │  │ │ Brain   │ │         │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                  │
│              ┌───────────▼───────────┐                      │
│              │   Container Gateway   │                      │
│              │   (Load Balancer)     │                      │
│              └───────────────────────┘                      │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
                    Your Application
```

## How Atlas Learns on Salad Cloud

1. **Distributed Learning**: Multiple Atlas instances learn in parallel
2. **GPU Acceleration**: CUDA-accelerated processing for faster learning
3. **Checkpoint Persistence**: Learned knowledge saved frequently (every 500 frames)
4. **Node Migration**: Graceful shutdown saves state before node preemption
5. **Self-Improvement**: Continuous optimization cycles

## Feeding Data to Atlas

### Option 1: Pre-loaded Data Volume
Include training data in your Docker image:
```dockerfile
COPY training_data/ /data/input/
```

### Option 2: External Storage
Mount external storage (coming soon with Salad Cloud storage integration)

### Option 3: Synthetic Exploration
Atlas generates its own training data when `ATLAS_ENABLE_EXPLORATION=true`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_LOG_LEVEL` | `INFO` | Logging verbosity |
| `ATLAS_CHECKPOINT_INTERVAL` | `500` | Frames between checkpoints |
| `ATLAS_MAX_CHECKPOINTS` | `3` | Max checkpoints to keep |
| `ATLAS_MULTIMODAL_SIZE` | `256` | Neural network size (GPU) |
| `ATLAS_LEARNING_RATE` | `0.01` | Learning rate |
| `ATLAS_ENABLE_UNIFIED_INTELLIGENCE` | `true` | Enable full cognitive systems |
| `ATLAS_ENABLE_EXPLORATION` | `true` | Generate synthetic training data |

## Health Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Liveness probe |
| `/ready` | Readiness probe |
| `/status` | Detailed status JSON |
| `/metrics` | Prometheus metrics |

## Commands Reference

```bash
# Build
./deploy.sh build

# Push to registry
./deploy.sh push

# Deploy (build + push + create)
./deploy.sh deploy

# Status
./deploy.sh status

# Instance details
./deploy.sh instances

# Start/Stop
./deploy.sh start
./deploy.sh stop

# Scale
./deploy.sh scale 5

# Delete
./deploy.sh delete

# List available GPUs
./deploy.sh gpus
```

## Troubleshooting

### Container won't start
- Check image is pushed correctly
- Verify API key permissions
- Check GPU availability in your region

### Frequent node preemption
- Increase replicas for resilience
- Use `restart_policy: always`
- Checkpoints save state automatically

### Slow learning
- Use `high_performance` GPU config
- Increase `ATLAS_MULTIMODAL_SIZE`
- Ensure GPU is being utilized (check `/status` endpoint)

## Cost Estimation

| Replicas | GPU Config | Estimated Cost/Day |
|----------|------------|-------------------|
| 3 | balanced | ~$7.20 |
| 5 | balanced | ~$12.00 |
| 3 | high_performance | ~$18.00 |
| 10 | cost_optimized | ~$12.00 |

## Resources

- [Salad Cloud Documentation](https://docs.salad.com)
- [Container Engine Guide](https://docs.salad.com/container-engine/salad-container-engine)
- [GPU Classes API](https://docs.salad.com/api-reference/container-groups/list-gpu-classes)
