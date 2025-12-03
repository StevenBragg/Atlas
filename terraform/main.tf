# ============================================
# Atlas Cloud Infrastructure - Terraform
# ============================================
# Supports: AWS, GCP, Azure (with provider modules)
# Default: AWS EKS
# ============================================

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  # Remote state storage (uncomment for production)
  # backend "s3" {
  #   bucket         = "atlas-terraform-state"
  #   key            = "atlas/terraform.tfstate"
  #   region         = "us-west-2"
  #   encrypt        = true
  #   dynamodb_table = "atlas-terraform-locks"
  # }
}

# ============================================
# Variables
# ============================================

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "atlas-cluster"
}

variable "node_instance_types" {
  description = "EC2 instance types for worker nodes"
  type        = list(string)
  default     = ["t3.xlarge", "t3.2xlarge"]
}

variable "gpu_instance_types" {
  description = "EC2 instance types for GPU nodes (for accelerated learning)"
  type        = list(string)
  default     = ["g4dn.xlarge", "g4dn.2xlarge"]
}

variable "min_nodes" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 2
}

variable "max_nodes" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 10
}

variable "enable_gpu_nodes" {
  description = "Enable GPU nodes for accelerated learning"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "Atlas"
    ManagedBy   = "Terraform"
    Purpose     = "Autonomous-Learning"
  }
}

# ============================================
# Provider Configuration
# ============================================

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = var.tags
  }
}

# ============================================
# Data Sources
# ============================================

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# ============================================
# VPC Configuration
# ============================================

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment == "dev" ? true : false
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Required tags for EKS
  public_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                    = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"           = "1"
  }

  tags = merge(var.tags, {
    Environment = var.environment
  })
}

# ============================================
# EKS Cluster
# ============================================

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  # Enable IRSA
  enable_irsa = true

  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa_role.iam_role_arn
    }
  }

  # Managed node groups
  eks_managed_node_groups = {
    # Main compute nodes
    atlas_compute = {
      name            = "atlas-compute"
      instance_types  = var.node_instance_types
      min_size        = var.min_nodes
      max_size        = var.max_nodes
      desired_size    = var.min_nodes

      capacity_type = "ON_DEMAND"  # Use SPOT for cost savings in dev

      labels = {
        role    = "compute"
        project = "atlas"
      }

      taints = []

      # Disk configuration
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 100
            volume_type           = "gp3"
            iops                  = 3000
            throughput            = 125
            encrypted             = true
            delete_on_termination = true
          }
        }
      }
    }
  }

  # Fargate profiles (optional, for serverless workloads)
  # fargate_profiles = {
  #   atlas_serverless = {
  #     name = "atlas-serverless"
  #     selectors = [
  #       {
  #         namespace = "atlas"
  #         labels = {
  #           serverless = "true"
  #         }
  #       }
  #     ]
  #   }
  # }

  tags = merge(var.tags, {
    Environment = var.environment
  })
}

# GPU Node Group (conditional)
resource "aws_eks_node_group" "gpu_nodes" {
  count = var.enable_gpu_nodes ? 1 : 0

  cluster_name    = module.eks.cluster_name
  node_group_name = "atlas-gpu"
  node_role_arn   = module.eks.eks_managed_node_groups["atlas_compute"].iam_role_arn
  subnet_ids      = module.vpc.private_subnets

  instance_types = var.gpu_instance_types
  ami_type       = "AL2_x86_64_GPU"
  capacity_type  = "ON_DEMAND"

  scaling_config {
    desired_size = 1
    max_size     = 3
    min_size     = 0
  }

  labels = {
    role         = "gpu"
    project      = "atlas"
    accelerator  = "nvidia"
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags = merge(var.tags, {
    Name        = "atlas-gpu-nodes"
    Environment = var.environment
  })

  depends_on = [module.eks]
}

# ============================================
# IAM Roles for Service Accounts (IRSA)
# ============================================

module "ebs_csi_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${var.cluster_name}-ebs-csi"
  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }

  tags = var.tags
}

# Atlas service account role
module "atlas_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${var.cluster_name}-atlas-sa"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["atlas:atlas-sa"]
    }
  }

  role_policy_arns = {
    s3_access = aws_iam_policy.atlas_s3_access.arn
  }

  tags = var.tags
}

# S3 Access Policy for Atlas
resource "aws_iam_policy" "atlas_s3_access" {
  name        = "${var.cluster_name}-atlas-s3-access"
  description = "Allow Atlas to access S3 for data and checkpoints"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.atlas_data.arn,
          "${aws_s3_bucket.atlas_data.arn}/*",
          aws_s3_bucket.atlas_checkpoints.arn,
          "${aws_s3_bucket.atlas_checkpoints.arn}/*"
        ]
      }
    ]
  })

  tags = var.tags
}

# ============================================
# S3 Buckets for Data Storage
# ============================================

resource "aws_s3_bucket" "atlas_data" {
  bucket = "${var.cluster_name}-data-${data.aws_caller_identity.current.account_id}"

  tags = merge(var.tags, {
    Name    = "atlas-data"
    Purpose = "Training data storage"
  })
}

resource "aws_s3_bucket_versioning" "atlas_data" {
  bucket = aws_s3_bucket.atlas_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "atlas_data" {
  bucket = aws_s3_bucket.atlas_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket" "atlas_checkpoints" {
  bucket = "${var.cluster_name}-checkpoints-${data.aws_caller_identity.current.account_id}"

  tags = merge(var.tags, {
    Name    = "atlas-checkpoints"
    Purpose = "Model checkpoint storage"
  })
}

resource "aws_s3_bucket_versioning" "atlas_checkpoints" {
  bucket = aws_s3_bucket.atlas_checkpoints.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "atlas_checkpoints" {
  bucket = aws_s3_bucket.atlas_checkpoints.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Lifecycle policy for checkpoints
resource "aws_s3_bucket_lifecycle_configuration" "atlas_checkpoints" {
  bucket = aws_s3_bucket.atlas_checkpoints.id

  rule {
    id     = "cleanup-old-checkpoints"
    status = "Enabled"

    filter {
      prefix = "checkpoints/"
    }

    # Move to cheaper storage after 30 days
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    # Move to glacier after 90 days
    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    # Delete after 365 days
    expiration {
      days = 365
    }
  }
}

# ============================================
# ECR Repository for Atlas Images
# ============================================

resource "aws_ecr_repository" "atlas" {
  name                 = "atlas"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = merge(var.tags, {
    Name = "atlas-ecr"
  })
}

resource "aws_ecr_lifecycle_policy" "atlas" {
  repository = aws_ecr_repository.atlas.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 20 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 20
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# ============================================
# Kubernetes Provider Configuration
# ============================================

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# ============================================
# Install Monitoring Stack (Prometheus + Grafana)
# ============================================

resource "helm_release" "prometheus_stack" {
  name             = "prometheus-stack"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  version          = "51.0.0"
  namespace        = "monitoring"
  create_namespace = true

  values = [
    <<-EOT
    grafana:
      enabled: true
      adminPassword: atlas-admin
      persistence:
        enabled: true
        size: 10Gi

    prometheus:
      prometheusSpec:
        retention: 30d
        storageSpec:
          volumeClaimTemplate:
            spec:
              accessModes: ["ReadWriteOnce"]
              resources:
                requests:
                  storage: 50Gi

    alertmanager:
      enabled: true
    EOT
  ]

  depends_on = [module.eks]
}

# ============================================
# Outputs
# ============================================

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "kubectl_config" {
  description = "kubectl config command"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

output "ecr_repository_url" {
  description = "ECR repository URL for Atlas images"
  value       = aws_ecr_repository.atlas.repository_url
}

output "s3_data_bucket" {
  description = "S3 bucket for training data"
  value       = aws_s3_bucket.atlas_data.bucket
}

output "s3_checkpoints_bucket" {
  description = "S3 bucket for model checkpoints"
  value       = aws_s3_bucket.atlas_checkpoints.bucket
}

output "atlas_irsa_role_arn" {
  description = "IAM role ARN for Atlas service account"
  value       = module.atlas_irsa_role.iam_role_arn
}
