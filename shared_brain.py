"""
Shared Atlas Brain

Singleton pattern to ensure all components use the same learning instance.
This allows the teacher, autonomous learner, and interactive mode to all
train the same brain.
"""

import os
import pickle
from pathlib import Path

# Global shared brain instance
_shared_brain = None

def get_shared_brain():
    """Get the singleton shared brain instance"""
    global _shared_brain
    
    if _shared_brain is None:
        from core.text_learning import TextLearningModule
        
        _shared_brain = TextLearningModule(embedding_dim=256)
        
        # Try to load existing state
        state_paths = [
            Path('/root/.openclaw/workspace/Atlas/improvements/state/text_learner.pkl'),
            Path('/root/.openclaw/workspace/Atlas/teacher_state/atlas_brain.pkl'),
            Path('/root/.openclaw/workspace/Atlas/shared_brain.pkl')
        ]
        
        for path in state_paths:
            if path.exists():
                try:
                    _shared_brain.load_state(path)
                    print(f"Loaded shared brain from {path}")
                    break
                except Exception as e:
                    print(f"Could not load from {path}: {e}")
    
    return _shared_brain

def save_shared_brain():
    """Save the shared brain state"""
    global _shared_brain
    
    if _shared_brain is None:
        return
    
    # Save to multiple locations for redundancy
    save_paths = [
        Path('/root/.openclaw/workspace/Atlas/shared_brain.pkl'),
        Path('/root/.openclaw/workspace/Atlas/teacher_state/atlas_brain.pkl'),
        Path('/root/.openclaw/workspace/Atlas/improvements/state/text_learner.pkl')
    ]
    
    for path in save_paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _shared_brain.save_state(path)
        except Exception as e:
            print(f"Could not save to {path}: {e}")
    
    stats = _shared_brain.get_stats()
    print(f"Shared brain saved: {stats['vocabulary_size']} words")
