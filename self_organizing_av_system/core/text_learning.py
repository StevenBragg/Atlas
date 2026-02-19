"""
Text Learning Module for Atlas - Unified Brain with Biological Plasticity

Integrates with Atlas's unified brain architecture:
- Uses same Hebbian learning as visual/creative modules
- Applies biological pruning and forgetting
- Consolidates strongly-encoded memories
- Part of one shared neural system
"""

import numpy as np
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import logging
import time

logger = logging.getLogger(__name__)


class TextLearningModule:
    """
    Self-organizing text learning system with biological plasticity.
    
    Unified brain principles:
    - Predictive coding: predict next token from context
    - Hebbian learning with decay: co-occurring tokens strengthen, unused weaken
    - Memory consolidation: frequent tokens become permanent
    - Synaptic pruning: weak connections removed
    - Forgetting: unused memories fade over time
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        max_vocabulary: int = 100000,
        context_window: int = 5,
        learning_rate: float = 0.05,
        # Biological plasticity - unified with other brain regions
        hebbian_decay: float = 0.9995,      # Weight decay like creative canvas
        prune_threshold: float = 0.01,       # Prune weak associations
        forgetting_rate: float = 0.0001,     # Gradual forgetting
        consolidation_threshold: int = 10    # Exposures before permanent
    ):
        self.embedding_dim = embedding_dim
        self.max_vocabulary = max_vocabulary
        self.context_window = context_window
        self.learning_rate = learning_rate
        
        # Biological plasticity parameters (unified brain)
        self.hebbian_decay = hebbian_decay
        self.prune_threshold = prune_threshold
        self.forgetting_rate = forgetting_rate
        self.consolidation_threshold = consolidation_threshold
        
        # Vocabulary: token -> index
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self.next_idx = 0
        
        # Embeddings: distributed representations
        self.embeddings: Dict[int, np.ndarray] = {}
        
        # Context weights: Hebbian associations
        self.context_weights: Dict[Tuple[int, ...], np.ndarray] = {}
        self.context_weight_strength: Dict[Tuple[int, ...], float] = {}
        
        # Biological memory tracking
        self.token_counts = Counter()
        self.token_exposure_count: Dict[int, int] = {}  # For consolidation
        self.token_last_seen: Dict[int, float] = {}      # For forgetting
        self.consolidated_tokens: set = set()            # Permanent memories
        self.total_tokens = 0
        self.learning_step = 0
        
        # Special tokens
        self._add_special_tokens()
        
    def _add_special_tokens(self):
        """Add special tokens for text structure"""
        special = ['<PAD>', '<UNK>', '<START>', '<END>', '<SPACE>']
        for token in special:
            self._get_or_add_token(token)
    
    def _get_or_add_token(self, token: str) -> int:
        """Get token index, adding to vocabulary if new (with biological pruning)"""
        if token not in self.token_to_idx:
            # Check if we need to prune before adding
            if len(self.token_to_idx) >= self.max_vocabulary:
                self._prune_weakest_token()
            
            idx = self.next_idx
            self.next_idx += 1
            
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            # Initialize embedding with small random values
            self.embeddings[idx] = np.random.randn(self.embedding_dim) * 0.1
            self.token_exposure_count[idx] = 0
            self.token_last_seen[idx] = time.time()
            
        return self.token_to_idx[token]
    
    def _prune_weakest_token(self):
        """Biological pruning: remove least used, unconsolidated token"""
        # Find candidates for pruning (not consolidated, low exposure)
        candidates = [
            (idx, self.token_exposure_count.get(idx, 0), self.token_counts.get(idx, 0))
            for idx in self.embeddings.keys()
            if idx not in self.consolidated_tokens and idx < 5  # Not special tokens
        ]
        
        if candidates:
            # Prune token with lowest exposure count
            weakest = min(candidates, key=lambda x: (x[1], x[2]))
            idx_to_prune = weakest[0]
            
            # Find token name
            token_to_remove = None
            for token, idx in self.token_to_idx.items():
                if idx == idx_to_prune:
                    token_to_remove = token
                    break
            
            if token_to_remove:
                # Remove token and all its associations
                del self.token_to_idx[token_to_remove]
                del self.idx_to_token[idx_to_prune]
                del self.embeddings[idx_to_prune]
                if idx_to_prune in self.token_exposure_count:
                    del self.token_exposure_count[idx_to_prune]
                if idx_to_prune in self.token_last_seen:
                    del self.token_last_seen[idx_to_prune]
                
                # Remove context weights involving this token
                contexts_to_remove = [
                    ctx for ctx in self.context_weights.keys()
                    if idx_to_prune in ctx
                ]
                for ctx in contexts_to_remove:
                    del self.context_weights[ctx]
                    if ctx in self.context_weight_strength:
                        del self.context_weight_strength[ctx]
                
                logger.debug(f"Pruned token: {token_to_remove}")
    
    def _apply_forgetting(self):
        """Apply time-based forgetting to unused tokens"""
        current_time = time.time()
        tokens_to_forget = []
        
        for idx, last_seen in self.token_last_seen.items():
            if idx in self.consolidated_tokens:
                continue  # Don't forget consolidated memories
            
            time_since_seen = current_time - last_seen
            forget_probability = 1 - np.exp(-self.forgetting_rate * time_since_seen)
            
            if np.random.random() < forget_probability:
                tokens_to_forget.append(idx)
        
        # Gradually reduce embedding strength for forgotten tokens
        for idx in tokens_to_forget[:10]:  # Limit forgetting per step
            if idx in self.embeddings:
                self.embeddings[idx] *= 0.9  # Decay embedding
    
    def _consolidate_memories(self):
        """Consolidate frequently exposed tokens into permanent memory"""
        for idx, exposure in self.token_exposure_count.items():
            if exposure >= self.consolidation_threshold and idx not in self.consolidated_tokens:
                self.consolidated_tokens.add(idx)
                # Strengthen embedding
                self.embeddings[idx] *= 1.5
                self.embeddings[idx] = np.clip(self.embeddings[idx], -2, 2)
                token = self.idx_to_token.get(idx, '?')
                logger.debug(f"Consolidated token: {token}")
    
    def tokenize(self, text: str) -> List[str]:
        """Convert text to tokens"""
        text = text.lower().strip()
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens if tokens else ['<UNK>']
    
    def learn_from_text(self, text: str) -> Dict[str, any]:
        """
        Learn from text using unified brain biological mechanisms.
        """
        self.learning_step += 1
        
        # Apply biological processes periodically
        if self.learning_step % 100 == 0:
            self._apply_forgetting()
            self._consolidate_memories()
        
        tokens = self.tokenize(text)
        token_indices = [self._get_or_add_token(t) for t in tokens]
        
        predictions = []
        errors = []
        
        # Learn from each position
        for i in range(len(token_indices)):
            target = token_indices[i]
            
            # Update exposure tracking
            self.token_exposure_count[target] = self.token_exposure_count.get(target, 0) + 1
            self.token_last_seen[target] = time.time()
            
            # Get context window
            start = max(0, i - self.context_window)
            context = tuple(token_indices[start:i])
            
            # Predict target from context
            if context:
                pred_embedding = self._predict_from_context(context)
                target_embedding = self.embeddings[target]
                
                # Prediction error
                error = target_embedding - pred_embedding
                error_norm = np.linalg.norm(error)
                errors.append(error_norm)
                
                # Hebbian update with biological decay
                self._update_context_weights(context, target, error, error_norm)
                
                predictions.append({
                    'context': [self.idx_to_token.get(idx, '?') for idx in context],
                    'predicted': self.idx_to_token.get(target, '?'),
                    'error': float(error_norm)
                })
            
            # Update token statistics
            self.token_counts[target] += 1
            self.total_tokens += 1
        
        return {
            'tokens_processed': len(tokens),
            'unique_tokens': len(set(tokens)),
            'vocabulary_size': len(self.token_to_idx),
            'consolidated_tokens': len(self.consolidated_tokens),
            'avg_prediction_error': np.mean(errors) if errors else 0,
            'predictions': predictions[:5]
        }
    
    def _predict_from_context(self, context: Tuple[int, ...]) -> np.ndarray:
        """Predict token embedding from context"""
        if context not in self.context_weights:
            # Initialize with small random weights
            self.context_weights[context] = np.random.randn(
                len(context), self.embedding_dim
            ) * 0.1
            self.context_weight_strength[context] = 0.0
        
        # Apply Hebbian decay
        self.context_weights[context] *= self.hebbian_decay
        
        # Weighted combination
        weights = self.context_weights[context]
        context_embs = np.array([self.embeddings[idx] for idx in context])
        prediction = np.mean(context_embs * weights, axis=0)
        
        return prediction
    
    def _update_context_weights(self, context: Tuple[int, ...], 
                                target: int, 
                                error: np.ndarray,
                                error_norm: float):
        """Update weights with Hebbian learning and biological constraints"""
        context_embs = np.array([self.embeddings[idx] for idx in context])
        
        # Hebbian update: strengthen based on error reduction
        delta = self.learning_rate * np.outer(
            np.ones(len(context)),
            error
        )
        
        self.context_weights[context] += delta
        
        # Track connection strength
        self.context_weight_strength[context] = np.linalg.norm(self.context_weights[context])
        
        # Prune weak connections
        if self.context_weight_strength[context] < self.prune_threshold:
            del self.context_weights[context]
            if context in self.context_weight_strength:
                del self.context_weight_strength[context]
        else:
            # Normalize to prevent explosion
            self.context_weights[context] = np.clip(
                self.context_weights[context], -2, 2
            )
    
    def generate_text(self, prompt: str = "", max_length: int = 50) -> str:
        """Generate text from learned patterns"""
        tokens = self.tokenize(prompt) if prompt else ['<START>']
        token_indices = [self.token_to_idx.get(t, self.token_to_idx['<UNK>']) 
                        for t in tokens]
        
        for _ in range(max_length):
            start = max(0, len(token_indices) - self.context_window)
            context = tuple(token_indices[start:])
            
            if context in self.context_weights:
                pred_emb = self._predict_from_context(context)
                
                # Find closest token using cosine similarity
                similarities = []
                for idx, emb in self.embeddings.items():
                    norm_pred = np.linalg.norm(pred_emb)
                    norm_emb = np.linalg.norm(emb)
                    if norm_pred > 0 and norm_emb > 0:
                        sim = np.dot(pred_emb, emb) / (norm_pred * norm_emb + 1e-8)
                        # Boost consolidated tokens
                        if idx in self.consolidated_tokens:
                            sim *= 1.2
                        similarities.append((idx, sim))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                next_idx = similarities[0][0] if similarities else self.token_to_idx['<UNK>']
            else:
                # Prefer consolidated tokens for random selection
                consolidated = list(self.consolidated_tokens)
                if consolidated and np.random.random() < 0.7:
                    next_idx = np.random.choice(consolidated)
                else:
                    next_idx = np.random.choice(list(self.embeddings.keys()))
            
            if next_idx == self.token_to_idx.get('<END>'):
                break
            
            token_indices.append(next_idx)
        
        # Convert back to text
        result_tokens = [self.idx_to_token.get(idx, '<UNK>') for idx in token_indices]
        return ' '.join(result_tokens)
    
    def get_stats(self) -> Dict[str, any]:
        """Get learning statistics"""
        return {
            'vocabulary_size': len(self.token_to_idx),
            'total_tokens_seen': self.total_tokens,
            'unique_contexts': len(self.context_weights),
            'consolidated_tokens': len(self.consolidated_tokens),
            'learning_step': self.learning_step,
            'avg_token_exposure': np.mean(list(self.token_exposure_count.values())) if self.token_exposure_count else 0
        }
    
    def save_state(self, filepath: str):
        """Save learning state"""
        import pickle
        state = {
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'embeddings': self.embeddings,
            'context_weights': self.context_weights,
            'context_weight_strength': self.context_weight_strength,
            'token_counts': self.token_counts,
            'token_exposure_count': self.token_exposure_count,
            'token_last_seen': self.token_last_seen,
            'consolidated_tokens': self.consolidated_tokens,
            'total_tokens': self.total_tokens,
            'learning_step': self.learning_step,
            'next_idx': self.next_idx
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load learning state"""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.token_to_idx = state['token_to_idx']
        self.idx_to_token = state['idx_to_token']
        self.embeddings = state['embeddings']
        self.context_weights = state['context_weights']
        self.context_weight_strength = state.get('context_weight_strength', {})
        self.token_counts = state['token_counts']
        self.token_exposure_count = state.get('token_exposure_count', {})
        self.token_last_seen = state.get('token_last_seen', {})
        self.consolidated_tokens = state.get('consolidated_tokens', set())
        self.total_tokens = state['total_tokens']
        self.learning_step = state.get('learning_step', 0)
        self.next_idx = state['next_idx']
