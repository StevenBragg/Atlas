#!/bin/bash
# ============================================
# Atlas Salad Cloud Deployment Script
# ============================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SALAD_API_URL="https://api.salad.com/api/public"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Required environment variables
: "${SALAD_API_KEY:?Please set SALAD_API_KEY environment variable}"
: "${SALAD_ORG_NAME:?Please set SALAD_ORG_NAME environment variable}"
: "${SALAD_PROJECT_NAME:?Please set SALAD_PROJECT_NAME environment variable}"

# Optional configuration
CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker.io/YOUR_USERNAME/atlas:salad-latest}"
GPU_CONFIG="${GPU_CONFIG:-balanced}"
REPLICAS="${REPLICAS:-3}"

echo -e "${BLUE}"
echo "============================================"
echo "  ATLAS - Salad Cloud Deployment"
echo "  Distributed GPU Learning Network"
echo "============================================"
echo -e "${NC}"

function print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

function print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

function print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function check_prerequisites() {
    print_step "Checking prerequisites..."

    # Check for required tools
    for cmd in curl jq docker; do
        if ! command -v $cmd &> /dev/null; then
            print_error "$cmd is required but not installed."
            exit 1
        fi
    done

    # Verify API key
    print_info "Verifying Salad Cloud API key..."
    response=$(curl -s -w "%{http_code}" -o /tmp/salad_verify.json \
        -H "Salad-Api-Key: ${SALAD_API_KEY}" \
        "${SALAD_API_URL}/organizations/${SALAD_ORG_NAME}/projects")

    if [ "$response" != "200" ]; then
        print_error "Invalid API key or organization. HTTP: $response"
        cat /tmp/salad_verify.json
        exit 1
    fi

    print_info "API key verified successfully"
}

function get_gpu_classes() {
    print_step "Fetching available GPU classes..."

    curl -s -H "Salad-Api-Key: ${SALAD_API_KEY}" \
        "${SALAD_API_URL}/organizations/${SALAD_ORG_NAME}/gpu-classes" \
        | jq -r '.items[] | "\(.id): \(.name) - \(.description)"'

    echo ""
}

function build_image() {
    print_step "Building Salad Cloud optimized Docker image..."

    cd "$PROJECT_ROOT"

    # Build the image
    docker build -f Dockerfile.salad -t atlas:salad-latest .

    print_info "Image built successfully: atlas:salad-latest"
}

function push_image() {
    print_step "Pushing image to container registry..."

    # Tag for registry
    docker tag atlas:salad-latest "$CONTAINER_IMAGE"

    # Push
    docker push "$CONTAINER_IMAGE"

    print_info "Image pushed: $CONTAINER_IMAGE"
}

function create_container_group() {
    print_step "Creating container group on Salad Cloud..."

    # Load GPU config
    local gpu_config_file="$SCRIPT_DIR/gpu-configs.json"
    local cpu=$(jq -r ".gpu_configurations.${GPU_CONFIG}.cpu" "$gpu_config_file")
    local memory=$(jq -r ".gpu_configurations.${GPU_CONFIG}.memory" "$gpu_config_file")
    local gpu_classes=$(jq -c ".gpu_configurations.${GPU_CONFIG}.gpu_classes" "$gpu_config_file")

    print_info "Using GPU config: $GPU_CONFIG"
    print_info "CPU: $cpu, Memory: ${memory}MB"
    print_info "Replicas: $REPLICAS"

    # Create deployment payload
    local payload=$(cat <<EOF
{
  "name": "atlas-brain",
  "display_name": "Atlas Autonomous Learning System",
  "container": {
    "image": "${CONTAINER_IMAGE}",
    "resources": {
      "cpu": ${cpu},
      "memory": ${memory},
      "gpu_classes": ${gpu_classes}
    },
    "environment_variables": {
      "ATLAS_LOG_LEVEL": "INFO",
      "ATLAS_CHECKPOINT_DIR": "/data/checkpoints",
      "ATLAS_INPUT_DIR": "/data/input",
      "ATLAS_CHECKPOINT_INTERVAL": "500",
      "ATLAS_MAX_CHECKPOINTS": "3",
      "ATLAS_ENABLE_UNIFIED_INTELLIGENCE": "true",
      "ATLAS_ENABLE_EXPLORATION": "true",
      "ATLAS_LEARNING_RATE": "0.01",
      "ATLAS_MULTIMODAL_SIZE": "256",
      "ATLAS_HTTP_PORT": "8080"
    }
  },
  "autostart_policy": true,
  "restart_policy": "always",
  "replicas": ${REPLICAS},
  "networking": {
    "protocol": "http",
    "port": 8080,
    "auth": false
  },
  "startup_probe": {
    "http": {
      "path": "/health",
      "port": 8080,
      "scheme": "http",
      "headers": []
    },
    "initial_delay_seconds": 60,
    "period_seconds": 10,
    "timeout_seconds": 5,
    "failure_threshold": 30
  },
  "liveness_probe": {
    "http": {
      "path": "/health",
      "port": 8080,
      "scheme": "http",
      "headers": []
    },
    "initial_delay_seconds": 60,
    "period_seconds": 30,
    "timeout_seconds": 10,
    "failure_threshold": 3
  },
  "readiness_probe": {
    "http": {
      "path": "/ready",
      "port": 8080,
      "scheme": "http",
      "headers": []
    },
    "initial_delay_seconds": 30,
    "period_seconds": 10,
    "timeout_seconds": 5,
    "failure_threshold": 3
  }
}
EOF
)

    # Deploy
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Salad-Api-Key: ${SALAD_API_KEY}" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        -d "$payload" \
        "${SALAD_API_URL}/organizations/${SALAD_ORG_NAME}/projects/${SALAD_PROJECT_NAME}/containers")

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" = "201" ] || [ "$http_code" = "200" ]; then
        print_info "Container group created successfully!"
        echo "$body" | jq .
    else
        print_error "Failed to create container group. HTTP: $http_code"
        echo "$body" | jq .
        exit 1
    fi
}

function get_status() {
    print_step "Getting container group status..."

    response=$(curl -s \
        -H "Salad-Api-Key: ${SALAD_API_KEY}" \
        "${SALAD_API_URL}/organizations/${SALAD_ORG_NAME}/projects/${SALAD_PROJECT_NAME}/containers/atlas-brain")

    echo "$response" | jq '{
        name: .name,
        status: .current_state.status,
        replicas: .replicas,
        running_replicas: .current_state.instance_status_counts.running,
        pending_replicas: .current_state.instance_status_counts.pending,
        access_url: .networking.dns
    }'
}

function get_instances() {
    print_step "Getting instance details..."

    curl -s \
        -H "Salad-Api-Key: ${SALAD_API_KEY}" \
        "${SALAD_API_URL}/organizations/${SALAD_ORG_NAME}/projects/${SALAD_PROJECT_NAME}/containers/atlas-brain/instances" \
        | jq '.instances[] | {machine_id, state, update_time}'
}

function start_container_group() {
    print_step "Starting container group..."

    curl -s -X POST \
        -H "Salad-Api-Key: ${SALAD_API_KEY}" \
        "${SALAD_API_URL}/organizations/${SALAD_ORG_NAME}/projects/${SALAD_PROJECT_NAME}/containers/atlas-brain/start"

    print_info "Start command sent"
}

function stop_container_group() {
    print_step "Stopping container group..."

    curl -s -X POST \
        -H "Salad-Api-Key: ${SALAD_API_KEY}" \
        "${SALAD_API_URL}/organizations/${SALAD_ORG_NAME}/projects/${SALAD_PROJECT_NAME}/containers/atlas-brain/stop"

    print_info "Stop command sent"
}

function delete_container_group() {
    print_warning "This will delete the container group!"
    read -p "Are you sure? (y/n) " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cancelled"
        exit 0
    fi

    print_step "Deleting container group..."

    curl -s -X DELETE \
        -H "Salad-Api-Key: ${SALAD_API_KEY}" \
        "${SALAD_API_URL}/organizations/${SALAD_ORG_NAME}/projects/${SALAD_PROJECT_NAME}/containers/atlas-brain"

    print_info "Container group deleted"
}

function scale_replicas() {
    local new_replicas=${1:-$REPLICAS}
    print_step "Scaling to $new_replicas replicas..."

    curl -s -X PATCH \
        -H "Salad-Api-Key: ${SALAD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"replicas\": $new_replicas}" \
        "${SALAD_API_URL}/organizations/${SALAD_ORG_NAME}/projects/${SALAD_PROJECT_NAME}/containers/atlas-brain"

    print_info "Scale command sent"
}

function show_help() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Salad Cloud Docker image"
    echo "  push        Push image to container registry"
    echo "  deploy      Build, push, and create container group"
    echo "  status      Get container group status"
    echo "  instances   Get instance details"
    echo "  start       Start the container group"
    echo "  stop        Stop the container group"
    echo "  delete      Delete the container group"
    echo "  scale N     Scale to N replicas"
    echo "  gpus        List available GPU classes"
    echo ""
    echo "Environment variables:"
    echo "  SALAD_API_KEY       Your Salad Cloud API key (required)"
    echo "  SALAD_ORG_NAME      Your organization name (required)"
    echo "  SALAD_PROJECT_NAME  Your project name (required)"
    echo "  CONTAINER_IMAGE     Container image URL (default: docker.io/YOUR_USERNAME/atlas:salad-latest)"
    echo "  GPU_CONFIG          GPU config: high_performance, balanced, cost_optimized (default: balanced)"
    echo "  REPLICAS            Number of replicas (default: 3)"
    echo ""
    echo "Example:"
    echo "  export SALAD_API_KEY=your-api-key"
    echo "  export SALAD_ORG_NAME=your-org"
    echo "  export SALAD_PROJECT_NAME=atlas"
    echo "  export CONTAINER_IMAGE=docker.io/myuser/atlas:salad-latest"
    echo "  $0 deploy"
}

# Main
case "${1:-help}" in
    build)
        check_prerequisites
        build_image
        ;;
    push)
        check_prerequisites
        push_image
        ;;
    deploy)
        check_prerequisites
        build_image
        push_image
        create_container_group
        echo ""
        print_info "Deployment initiated! Atlas will start learning on Salad Cloud."
        print_info "Check status with: $0 status"
        ;;
    status)
        check_prerequisites
        get_status
        ;;
    instances)
        check_prerequisites
        get_instances
        ;;
    start)
        check_prerequisites
        start_container_group
        ;;
    stop)
        check_prerequisites
        stop_container_group
        ;;
    delete)
        check_prerequisites
        delete_container_group
        ;;
    scale)
        check_prerequisites
        scale_replicas "${2:-3}"
        ;;
    gpus)
        check_prerequisites
        get_gpu_classes
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
