#!/usr/bin/env python3
"""Encrypt and create GitHub secret for Salad Cloud API key"""

import base64
import json
import urllib.request
import urllib.error

# GitHub API details
import os
REPO = "StevenBragg/Atlas"
TOKEN = os.environ.get("GITHUB_TOKEN")
SECRET_NAME = "SALAD_API_KEY"
SECRET_VALUE = os.environ.get("SALAD_API_KEY")

if not TOKEN or not SECRET_VALUE:
    print("Error: GITHUB_TOKEN and SALAD_API_KEY environment variables required")
    exit(1)

def get_public_key():
    """Get the public key for the repo"""
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/actions/secrets/public-key",
        headers={
            "Authorization": f"token {TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def encrypt_secret(public_key, secret_value):
    """Encrypt the secret using libsodium"""
    import nacl.public
    import nacl.encoding
    
    public_key_bytes = base64.b64decode(public_key)
    public_key_obj = nacl.public.PublicKey(public_key_bytes)
    
    sealed_box = nacl.public.SealedBox(public_key_obj)
    encrypted = sealed_box.encrypt(secret_value.encode())
    
    return base64.b64encode(encrypted).decode()

def create_secret(key_id, encrypted_value):
    """Create the secret on GitHub"""
    data = json.dumps({
        "encrypted_value": encrypted_value,
        "key_id": key_id
    }).encode()
    
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/actions/secrets/{SECRET_NAME}",
        data=data,
        headers={
            "Authorization": f"token {TOKEN}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        },
        method="PUT"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            print(f"✅ Secret '{SECRET_NAME}' created/updated successfully")
            return True
    except urllib.error.HTTPError as e:
        print(f"❌ Error: {e.code} - {e.read().decode()}")
        return False

if __name__ == "__main__":
    try:
        from nacl import public
    except ImportError:
        print("Installing PyNaCl...")
        import subprocess
        subprocess.run(["pip", "install", "pynacl", "-q"])
        from nacl import public
    
    print("Getting public key...")
    key_data = get_public_key()
    
    print("Encrypting secret...")
    encrypted = encrypt_secret(key_data["key"], SECRET_VALUE)
    
    print("Creating secret on GitHub...")
    create_secret(key_data["key_id"], encrypted)
