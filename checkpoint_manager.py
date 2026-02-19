"""
Atlas Checkpoint Manager

Manages saving and loading of Atlas's learning state.
Ensures no progress is lost and enables resume after restarts.
"""

import os
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path


class CheckpointManager:
    """Manages Atlas learning checkpoints"""
    
    def __init__(self, base_path='/root/.openclaw/workspace/Atlas/checkpoints'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Subdirectories
        self.weights_dir = self.base_path / 'weights'
        self.memory_dir = self.base_path / 'memory'
        self.config_dir = self.base_path / 'config'
        
        for d in [self.weights_dir, self.memory_dir, self.config_dir]:
            d.mkdir(exist_ok=True)
        
    def save_checkpoint(self, name, components):
        """
        Save a checkpoint of Atlas state.
        
        Args:
            name: Checkpoint name (e.g., 'autonomous_atlas', 'teacher')
            components: Dict of components to save
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = self.base_path / f'{name}_{timestamp}'
        checkpoint_dir.mkdir(exist_ok=True)
        
        saved_files = []
        
        for component_name, component in components.items():
            filepath = checkpoint_dir / f'{component_name}.pkl'
            
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(component, f)
                saved_files.append(filepath)
            except Exception as e:
                print(f"Warning: Could not save {component_name}: {e}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'name': name,
            'components': list(components.keys()),
            'files': [str(f.name) for f in saved_files]
        }
        
        with open(checkpoint_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Keep only last 5 checkpoints
        self._cleanup_old_checkpoints(name, keep=5)
        
        return checkpoint_dir
        
    def load_checkpoint(self, name, timestamp=None):
        """
        Load a checkpoint.
        
        Args:
            name: Checkpoint name
            timestamp: Specific timestamp, or None for latest
            
        Returns:
            Dict of loaded components
        """
        if timestamp:
            checkpoint_dir = self.base_path / f'{name}_{timestamp}'
        else:
            # Find latest
            checkpoints = sorted(
                self.base_path.glob(f'{name}_*'),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if not checkpoints:
                return None
            checkpoint_dir = checkpoints[0]
        
        if not checkpoint_dir.exists():
            return None
        
        # Load metadata
        with open(checkpoint_dir / 'metadata.json') as f:
            metadata = json.load(f)
        
        # Load components
        components = {}
        for filename in metadata['files']:
            component_name = filename.replace('.pkl', '')
            filepath = checkpoint_dir / filename
            
            try:
                with open(filepath, 'rb') as f:
                    components[component_name] = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load {component_name}: {e}")
        
        return components
        
    def _cleanup_old_checkpoints(self, name, keep=5):
        """Keep only the most recent checkpoints"""
        checkpoints = sorted(
            self.base_path.glob(f'{name}_*'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old in checkpoints[keep:]:
            shutil.rmtree(old)
            
    def list_checkpoints(self, name=None):
        """List available checkpoints"""
        if name:
            pattern = f'{name}_*'
        else:
            pattern = '*'
            
        checkpoints = []
        for cp_dir in self.base_path.glob(pattern):
            if cp_dir.is_dir() and (cp_dir / 'metadata.json').exists():
                with open(cp_dir / 'metadata.json') as f:
                    metadata = json.load(f)
                checkpoints.append(metadata)
                
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)


# Global checkpoint manager
checkpoint_manager = CheckpointManager()


def save_atlas_state(name, **components):
    """Convenience function to save Atlas state"""
    return checkpoint_manager.save_checkpoint(name, components)


def load_atlas_state(name, timestamp=None):
    """Convenience function to load Atlas state"""
    return checkpoint_manager.load_checkpoint(name, timestamp)


if __name__ == "__main__":
    # Test
    cm = CheckpointManager()
    print("Available checkpoints:")
    for cp in cm.list_checkpoints():
        print(f"  {cp['name']} - {cp['timestamp']}")
