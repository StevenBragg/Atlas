#!/usr/bin/env python3
"""
Deploy Atlas to Salad Cloud
Simple deployment script using Salad Cloud API
"""

import os
import sys
import json
import urllib.request
import urllib.error

# Configuration
SALAD_API_KEY = os.environ.get('SALAD_API_KEY')
ORGANIZATION = 'stevenbragg'
PROJECT = 'atlas'
CONTAINER_GROUP = 'atlas-rtx3060'

def api_request(method, path, data=None):
    """Make Salad Cloud API request"""
    url = f"https://api.salad.com/api/public/organizations/{ORGANIZATION}{path}"
    
    headers = {
        'Salad-Api-Key': SALAD_API_KEY,
        'Content-Type': 'application/json'
    }
    
    if data:
        data = json.dumps(data).encode()
    
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"API Error: {e.code}")
        print(e.read().decode())
        return None

def create_project():
    """Create the atlas project if it doesn't exist"""
    print("Creating project 'atlas'...")
    result = api_request('POST', '/projects', {
        'name': PROJECT,
        'description': 'Atlas AI self-organizing learning system'
    })
    if result:
        print(f"✅ Project created: {result.get('id')}")
    return result

def create_container_group():
    """Create container group for Atlas"""
    print(f"Creating container group '{CONTAINER_GROUP}'...")
    
    config = {
        'name': CONTAINER_GROUP,
        'container': {
            'image': 'ghcr.io/stevenbragg/atlas/atlas-salad:latest',
            'resources': {
                'cpu': 4,
                'memory': 8192,
                'gpu_classes': ['f51baccc-dc95-40fb-a5d1-6d0ee0db31d2'],
                'storage_amount': 10737418240
            },
            'environment_variables': {
                'ATLAS_ENABLE_TEXT_LEARNING': 'true',
                'ATLAS_LOG_LEVEL': 'INFO'
            }
        },
        'replicas': 1,
        'country_codes': ['US', 'CA'],
        'restart_policy': 'always',
        'readiness_probe': {
            'http': {
                'port': 8080,
                'path': '/health'
            },
            'initial_delay_seconds': 60,
            'period_seconds': 30,
            'timeout_seconds': 10,
            'failure_threshold': 3
        }
    }
    
    result = api_request('POST', f'/projects/{PROJECT}/containers', config)
    if result:
        print(f"✅ Container group created!")
        print(f"   Status: {result.get('status')}")
        print(f"   Replicas: {result.get('replicas')}")
    return result

def get_container_group():
    """Get existing container group"""
    return api_request('GET', f'/projects/{PROJECT}/containers/{CONTAINER_GROUP}')

def main():
    if not SALAD_API_KEY:
        print("Error: SALAD_API_KEY environment variable required")
        sys.exit(1)
    
    print("=" * 60)
    print("Atlas Salad Cloud Deployment")
    print("=" * 60)
    print()
    
    # Try to create project (may already exist)
    create_project()
    
    # Check if container group exists
    existing = get_container_group()
    if existing:
        print(f"Container group '{CONTAINER_GROUP}' already exists")
        print(f"Status: {existing.get('status')}")
        print(f"Replicas: {existing.get('replicas')}")
    else:
        # Create container group
        result = create_container_group()
        if result:
            print()
            print("✅ Deployment initiated!")
            print(f"Monitor at: https://portal.salad.com/organizations/{ORGANIZATION}/projects/{PROJECT}")
        else:
            print("❌ Deployment failed")
            sys.exit(1)

if __name__ == "__main__":
    main()
