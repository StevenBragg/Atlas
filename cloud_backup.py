#!/usr/bin/env python3
"""
Atlas Cloud Backup

Automatically backs up Atlas checkpoints to Google Drive or S3.
Ensures learning is never lost.
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path

class CloudBackup:
    """Backup Atlas state to cloud storage"""
    
    def __init__(self, provider='gdrive'):
        self.provider = provider
        self.local_base = Path('/root/.openclaw/workspace/Atlas')
        
    def backup_to_gdrive(self, filepath):
        """Backup file to Google Drive"""
        # For now, we'll use a simple approach
        # In production, you'd use Google Drive API
        
        # Copy to a 'cloud_backup' directory that can be synced
        backup_dir = self.local_base / 'cloud_backup'
        backup_dir.mkdir(exist_ok=True)
        
        import shutil
        dest = backup_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filepath.name}"
        shutil.copy(filepath, dest)
        
        print(f"Backed up to: {dest}")
        return dest
        
    def backup_all(self):
        """Backup all Atlas state"""
        print("☁️  Starting cloud backup...")
        
        # Find all state files
        state_files = []
        
        # Teacher state
        teacher_state = self.local_base / 'teacher_state'
        if teacher_state.exists():
            state_files.extend(teacher_state.glob('*.pkl'))
            state_files.extend(teacher_state.glob('*.json'))
        
        # Improvements state
        improvements = self.local_base / 'improvements' / 'state'
        if improvements.exists():
            state_files.extend(improvements.glob('*.pkl'))
        
        # Checkpoints
        checkpoints = self.local_base / 'checkpoints'
        if checkpoints.exists():
            # Get latest checkpoint from each type
            for cp_type in ['autonomous_atlas', 'teacher', 'atlas']:
                latest = sorted(checkpoints.glob(f'{cp_type}_*'), 
                              key=lambda p: p.stat().st_mtime,
                              reverse=True)[:1]
                state_files.extend(latest)
        
        backed_up = []
        for f in state_files:
            try:
                dest = self.backup_to_gdrive(f)
                backed_up.append(dest)
            except Exception as e:
                print(f"Failed to backup {f}: {e}")
        
        print(f"✅ Backed up {len(backed_up)} files")
        return backed_up


def main():
    backup = CloudBackup()
    backup.backup_all()


if __name__ == "__main__":
    main()
