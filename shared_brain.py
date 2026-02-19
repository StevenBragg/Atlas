"""
Atlas Shared Brain - Singleton with Memory Persistence

Ensures ALL Atlas components use the EXACT same brain instance.
Prevents vocabulary loss from multiple instances overwriting each other.
"""

import os
import pickle
import threading
from pathlib import Path
from typing import Optional

# Global singleton instance
_brain_instance = None
_brain_lock = threading.Lock()

def get_shared_brain():
    """
    Get the singleton shared brain instance.
    
    ALL Atlas components MUST use this function to access the brain.
    Never create TextLearningModule directly - always use this.
    """
    global _brain_instance
    
    with _brain_lock:
        if _brain_instance is None:
            from core.text_learning import TextLearningModule
            
            _brain_instance = TextLearningModule(embedding_dim=256)
            
            # Try to load existing large brain
            brain_path = Path('/root/.openclaw/workspace/Atlas/shared_brain.pkl')
            backup_path = Path('/root/.openclaw/workspace/Atlas/teacher_state/atlas_brain_merged.pkl')
            
            loaded = False
            for path in [brain_path, backup_path]:
                if path.exists():
                    try:
                        with open(path, 'rb') as f:
                            state = pickle.load(f)
                            vocab_size = len(state.get('token_to_idx', {}))
                            if vocab_size > 100:  # Only load if substantial
                                _brain_instance.token_to_idx = state['token_to_idx']
                                _brain_instance.idx_to_token = state['idx_to_token']
                                _brain_instance.embeddings = state['embeddings']
                                _brain_instance.context_weights = state['context_weights']
                                _brain_instance.token_counts = state['token_counts']
                                _brain_instance.total_tokens = state['total_tokens']
                                _brain_instance.next_idx = state['next_idx']
                                print(f"[SharedBrain] Loaded {vocab_size} words from {path}")
                                loaded = True
                                break
                    except Exception as e:
                        print(f"[SharedBrain] Could not load {path}: {e}")
            
            if not loaded:
                print("[SharedBrain] Created fresh brain instance")
    
    return _brain_instance

def save_shared_brain():
    """
    Save the shared brain to disk.
    
    ALL saves go through this function to ensure consistency.
    """
    global _brain_instance
    
    with _brain_lock:
        if _brain_instance is None:
            return
        
        stats = _brain_instance.get_stats()
        vocab_size = stats['vocabulary_size']
        
        # Only save if vocabulary is substantial (prevent small overwrites)
        if vocab_size < 100:
            print(f"[SharedBrain] Skipping save - vocabulary too small ({vocab_size})")
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
                _brain_instance.save_state(path)
            except Exception as e:
                print(f"[SharedBrain] Could not save to {path}: {e}")
        
        print(f"[SharedBrain] Saved {vocab_size} words")

def get_brain_stats():
    """Get current brain statistics"""
    brain = get_shared_brain()
    return brain.get_stats()

def reset_shared_brain():
    """Reset the shared brain (use with caution)"""
    global _brain_instance
    
    with _brain_lock:
        _brain_instance = None
        print("[SharedBrain] Reset complete")

# Prevent direct instantiation
# _prevent_direct_creation() temporarily disabled
def _prevent_direct_creation():
    """Monkey-patch to prevent direct TextLearningModule creation"""
    import sys
    sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')
    from core.text_learning import TextLearningModule
    
    original_init = TextLearningModule.__init__
    
    def patched_init(self, *args, **kwargs):
        import traceback
        stack = traceback.extract_stack()
        
        # Check if called from get_shared_brain
        allowed_callers = ['shared_brain.py', 'get_shared_brain']
        is_allowed = any(caller in str(frame) for frame in stack for caller in allowed_callers)
        
        if not is_allowed and not hasattr(self, '_shared_brain_bypass'):
            import warnings
            warnings.warn(
                "DIRECT TextLearningModule CREATION DETECTED! "
                "Use get_shared_brain() instead to ensure shared learning.",
                RuntimeWarning
            )
        
        return original_init(self, *args, **kwargs)
    
    text_learning.TextLearningModule.__init__ = patched_init

# Apply patch on import
# _prevent_direct_creation() temporarily disabled
