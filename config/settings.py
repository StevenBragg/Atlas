"""
Secure configuration management for Atlas.
Loads sensitive values from environment variables only.
"""

import os
from pathlib import Path
from typing import Optional

# Load .env file if it exists (for local development)
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if key not in os.environ:
                    os.environ[key] = value


class SaladCloudConfig:
    """Salad Cloud configuration with guardrails."""
    
    # Spending limits
    MAX_MONTHLY_SPEND_USD = 100.0
    MAX_DAILY_SPEND_USD = 20.0
    
    # Instance limits
    MAX_GPU_INSTANCES = 2
    MAX_CPU_INSTANCES = 4
    MAX_INSTANCE_HOURS = 8
    
    # Allowed GPU types (cost-controlled, cheapest first)
    ALLOWED_GPU_TYPES = [
        "rtx3060",   # ~$0.20/hr - default for testing
        "rtx3070",   # ~$0.30/hr
        "rtx3080",   # ~$0.45/hr
        "rtx3090",   # ~$0.70/hr - training only
    ]
    
    # Approval thresholds
    APPROVAL_SPEND_THRESHOLD = 10.0
    APPROVAL_GPU_THRESHOLD = "rtx3080"
    APPROVAL_DURATION_THRESHOLD_HOURS = 4
    
    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """Get Salad Cloud API key from environment."""
        return os.environ.get('SALAD_CLOUD_API_KEY')
    
    @classmethod
    def validate_gpu(cls, gpu_type: str) -> tuple[bool, str]:
        """Validate GPU type against allowed list."""
        gpu = gpu_type.lower()
        if gpu not in cls.ALLOWED_GPU_TYPES:
            return False, f"GPU '{gpu}' not allowed. Use: {', '.join(cls.ALLOWED_GPU_TYPES)}"
        return True, "OK"
    
    @classmethod
    def validate_deployment(cls, config: dict) -> tuple[bool, str]:
        """Validate deployment config against guardrails."""
        gpu = config.get("gpu_class", "").lower()
        replicas = config.get("replicas", 1)
        
        # Check GPU type
        valid, msg = cls.validate_gpu(gpu)
        if not valid:
            return False, msg
        
        # Check instance count
        if replicas > cls.MAX_GPU_INSTANCES:
            return False, f"Max {cls.MAX_GPU_INSTANCES} GPU instances allowed"
        
        return True, "OK"


class GitHubConfig:
    """GitHub configuration."""
    
    @classmethod
    def get_token(cls) -> Optional[str]:
        """Get GitHub token from environment."""
        return os.environ.get('GITHUB_TOKEN')


# Validate on import
if __name__ == "__main__":
    # Test configuration
    salad_key = SaladCloudConfig.get_api_key()
    github_key = GitHubConfig.get_token()
    
    print("Configuration Status:")
    print(f"  Salad Cloud API Key: {'✓ Set' if salad_key else '✗ Not set'}")
    print(f"  GitHub Token: {'✓ Set' if github_key else '✗ Not set'}")
    print(f"  Max Monthly Spend: ${SaladCloudConfig.MAX_MONTHLY_SPEND_USD}")
    print(f"  Allowed GPUs: {', '.join(SaladCloudConfig.ALLOWED_GPU_TYPES)}")
