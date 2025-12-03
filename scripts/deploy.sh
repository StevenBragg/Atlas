#!/bin/bash
# ============================================
# Atlas Cloud Deployment Script
# ============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEPLOYMENT_TYPE=${1:-docker}
ENVIRONMENT=${ENVIRONMENT:-dev}
AWS_REGION=${AWS_REGION:-us-west-2}
CLUSTER_NAME=${CLUSTER_NAME:-atlas-cluster}

echo -e "${BLUE}"
echo "============================================"
echo "  ATLAS Cloud Deployment"
echo "  Autonomously Teaching, Learning And Self-organizing"
echo "============================================"
echo -e "${NC}"

function print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

function print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function deploy_docker() {
    print_step "Deploying Atlas with Docker Compose..."

    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    }

    # Build images
    print_step "Building Docker images..."
    docker-compose build

    # Start services
    print_step "Starting services..."
    docker-compose up -d

    # Wait for health check
    print_step "Waiting for Atlas to start..."
    sleep 30

    # Check health
    if curl -s http://localhost:8080/health | grep -q "running"; then
        echo -e "${GREEN}Atlas is running successfully!${NC}"
        echo ""
        echo "Access points:"
        echo "  - Atlas Health: http://localhost:8080/health"
        echo "  - Atlas Status: http://localhost:8080/status"
        echo "  - Grafana:      http://localhost:3000 (admin/atlas-admin)"
        echo "  - Prometheus:   http://localhost:9091"
        echo "  - MinIO:        http://localhost:9001 (atlas/atlas-secret-key)"
    else
        print_warning "Atlas may still be starting. Check logs with: docker-compose logs -f atlas"
    fi
}

function deploy_kubernetes() {
    print_step "Deploying Atlas to Kubernetes..."

    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl and try again."
        exit 1
    fi

    # Build and push Docker image (if using local registry)
    print_step "Building Docker image..."
    docker build -t atlas:latest .

    # Apply Kubernetes manifests in order
    print_step "Creating namespace..."
    kubectl apply -f kubernetes/namespace.yaml

    print_step "Creating config..."
    kubectl apply -f kubernetes/configmap.yaml

    print_step "Creating RBAC..."
    kubectl apply -f kubernetes/rbac.yaml

    print_step "Creating storage..."
    kubectl apply -f kubernetes/storage.yaml

    print_step "Creating deployments..."
    kubectl apply -f kubernetes/deployment.yaml

    print_step "Creating services..."
    kubectl apply -f kubernetes/service.yaml

    print_step "Creating autoscaling..."
    kubectl apply -f kubernetes/hpa.yaml

    # Wait for deployment
    print_step "Waiting for Atlas to start..."
    kubectl -n atlas rollout status deployment/atlas-brain --timeout=300s

    echo -e "${GREEN}Atlas deployed to Kubernetes successfully!${NC}"
    echo ""
    echo "Check status with:"
    echo "  kubectl -n atlas get pods"
    echo "  kubectl -n atlas logs -f deployment/atlas-brain"
}

function deploy_aws() {
    print_step "Deploying Atlas to AWS with Terraform..."

    # Check if Terraform is available
    if ! command -v terraform &> /dev/null; then
        print_error "terraform not found. Please install Terraform and try again."
        exit 1
    fi

    # Check if AWS CLI is configured
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS CLI not configured. Please run 'aws configure' and try again."
        exit 1
    fi

    cd terraform

    # Initialize Terraform
    print_step "Initializing Terraform..."
    terraform init

    # Create tfvars if not exists
    if [ ! -f terraform.tfvars ]; then
        print_step "Creating terraform.tfvars from example..."
        cp variables.tfvars.example terraform.tfvars
        print_warning "Please review terraform.tfvars before proceeding"
        echo "Edit terraform/terraform.tfvars with your settings, then run this script again."
        exit 0
    fi

    # Plan
    print_step "Planning infrastructure..."
    terraform plan -var-file=terraform.tfvars -out=tfplan

    # Confirm
    echo ""
    read -p "Do you want to apply this plan? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Deployment cancelled"
        exit 0
    fi

    # Apply
    print_step "Applying infrastructure..."
    terraform apply tfplan

    # Get kubectl config
    print_step "Configuring kubectl..."
    aws eks update-kubeconfig --region ${AWS_REGION} --name ${CLUSTER_NAME}

    cd ..

    # Deploy Kubernetes manifests
    deploy_kubernetes

    echo -e "${GREEN}Atlas deployed to AWS successfully!${NC}"
}

function show_status() {
    print_step "Atlas Status"
    echo ""

    case $DEPLOYMENT_TYPE in
        docker)
            docker-compose ps
            echo ""
            echo "Health:"
            curl -s http://localhost:8080/health | python3 -m json.tool 2>/dev/null || echo "Atlas not responding"
            ;;
        kubernetes|aws)
            kubectl -n atlas get pods
            echo ""
            kubectl -n atlas get svc
            ;;
    esac
}

function show_logs() {
    print_step "Atlas Logs"
    echo ""

    case $DEPLOYMENT_TYPE in
        docker)
            docker-compose logs -f atlas
            ;;
        kubernetes|aws)
            kubectl -n atlas logs -f deployment/atlas-brain
            ;;
    esac
}

function stop_atlas() {
    print_step "Stopping Atlas..."

    case $DEPLOYMENT_TYPE in
        docker)
            docker-compose down
            ;;
        kubernetes|aws)
            kubectl -n atlas scale deployment atlas-brain --replicas=0
            kubectl -n atlas scale deployment atlas-ingestion --replicas=0
            ;;
    esac

    echo -e "${GREEN}Atlas stopped${NC}"
}

function destroy_infrastructure() {
    print_warning "This will destroy all AWS infrastructure including data!"
    read -p "Are you sure? (type 'yes' to confirm) " -r
    echo ""
    if [[ ! $REPLY == "yes" ]]; then
        print_warning "Destruction cancelled"
        exit 0
    fi

    cd terraform
    terraform destroy -var-file=terraform.tfvars
    cd ..

    echo -e "${GREEN}Infrastructure destroyed${NC}"
}

# Main
case ${2:-deploy} in
    deploy)
        case $DEPLOYMENT_TYPE in
            docker)
                deploy_docker
                ;;
            kubernetes|k8s)
                deploy_kubernetes
                ;;
            aws|terraform)
                deploy_aws
                ;;
            *)
                echo "Usage: $0 [docker|kubernetes|aws] [deploy|status|logs|stop|destroy]"
                exit 1
                ;;
        esac
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    stop)
        stop_atlas
        ;;
    destroy)
        destroy_infrastructure
        ;;
    *)
        echo "Usage: $0 [docker|kubernetes|aws] [deploy|status|logs|stop|destroy]"
        exit 1
        ;;
esac
