"""
Text Learning Module for Atlas

Enables Atlas to learn from text input, similar to audio-visual learning.
Text is treated as a sequential sensory stream where:
- Characters/words are sensory tokens
- Context provides temporal structure
- Meaning emerges from statistical patterns

This allows Atlas to:
1. Learn language structure from text corpora
2. Ground words in conceptual space
3. Generate text responses
4. Communicate with humans
"""

import numpy as np
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class TextLearningModule:
    """
    Self-organizing text learning system.
    
    Learns from text streams using similar principles to AV learning:
    - Predictive coding: predict next token from context
    - Hebbian learning: co-occurring tokens strengthen connections
    - Structural plasticity: grow vocabulary as needed
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        max_vocabulary: int = 100000,
        context_window: int = 5,
        learning_rate: float = 0.05
    ):
        self.embedding_dim = embedding_dim
        self.max_vocabulary = max_vocabulary  # Increased to 100k - never saturate
        self.context_window = context_window
        self.learning_rate = learning_rate  # Increased for faster learning
        
        # Vocabulary: token -> index
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self.next_idx = 0
        
        # Embeddings: each token has a distributed representation
        self.embeddings: Dict[int, np.ndarray] = {}
        
        # Context weights: which tokens predict which others
        self.context_weights: Dict[Tuple[int, ...], np.ndarray] = {}
        
        # Token statistics
        self.token_counts = Counter()
        self.total_tokens = 0
        
        # Special tokens
        self._add_special_tokens()
        
    def _add_special_tokens(self):
        """Add special tokens for text structure"""
        special = ['<PAD>', '<UNK>', '<START>', '<END>', '<SPACE>']
        for token in special:
            self._get_or_add_token(token)
    
    def _get_or_add_token(self, token: str) -> int:
        """Get token index, adding to vocabulary if new"""
        if token not in self.token_to_idx:
            if len(self.token_to_idx) >= self.max_vocabulary:
                # Replace rare token
                rarest = min(self.token_counts, key=self.token_counts.get)
                idx = self.token_to_idx[rarest]
                del self.token_to_idx[rarest]
                del self.idx_to_token[idx]
                del self.embeddings[idx]
            else:
                idx = self.next_idx
                self.next_idx += 1
            
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            # Initialize embedding randomly
            self.embeddings[idx] = np.random.randn(self.embedding_dim) * 0.1
            
        return self.token_to_idx[token]
    
    def tokenize(self, text: str) -> List[str]:
        """Convert text to tokens"""
        # Simple tokenization: words, punctuation, numbers
        text = text.lower().strip()
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens if tokens else ['<UNK>']
    
    def learn_from_text(self, text: str) -> Dict[str, any]:
        """
        Learn from a text input.
        
        Similar to processing an AV frame:
        - Tokenize = sensory preprocessing
        - Predict next token = temporal prediction
        - Update weights = Hebbian learning
        """
        tokens = self.tokenize(text)
        token_indices = [self._get_or_add_token(t) for t in tokens]
        
        predictions = []
        errors = []
        
        # Learn from each position
        for i in range(len(token_indices)):
            # Get context window
            start = max(0, i - self.context_window)
            context = tuple(token_indices[start:i])
            target = token_indices[i]
            
            # Predict target from context
            if context:
                pred_embedding = self._predict_from_context(context)
                target_embedding = self.embeddings[target]
                
                # Prediction error (what we didn't predict)
                error = target_embedding - pred_embedding
                errors.append(np.linalg.norm(error))
                
                # Update context weights (Hebbian learning)
                self._update_context_weights(context, target, error)
                
                predictions.append({
                    'context': [self.idx_to_token[idx] for idx in context],
                    'predicted': self.idx_to_token[target],
                    'error': float(np.linalg.norm(error))
                })
            
            # Update token statistics
            self.token_counts[target] += 1
            self.total_tokens += 1
        
        return {
            'tokens_processed': len(tokens),
            'unique_tokens': len(set(tokens)),
            'vocabulary_size': len(self.token_to_idx),
            'avg_prediction_error': np.mean(errors) if errors else 0,
            'predictions': predictions[:5]  # First 5 for inspection
        }
    
    def _predict_from_context(self, context: Tuple[int, ...]) -> np.ndarray:
        """Predict token embedding from context"""
        if context not in self.context_weights:
            # Initialize random weights for new context
            self.context_weights[context] = np.random.randn(
                len(context), self.embedding_dim
            ) * 0.1
        
        # Weighted combination of context embeddings
        weights = self.context_weights[context]
        context_embs = np.array([self.embeddings[idx] for idx in context])
        
        # Simple linear prediction
        prediction = np.mean(context_embs * weights, axis=0)
        return prediction
    
    def _update_context_weights(self, context: Tuple[int, ...], 
                                target: int, 
                                error: np.ndarray):
        """Update weights based on prediction error (Hebbian/Oja's rule)"""
        # Hebbian update: strengthen connections that reduce error
        context_embs = np.array([self.embeddings[idx] for idx in context])
        
        # Weight update proportional to error and context activation
        delta = self.learning_rate * np.outer(
            np.ones(len(context)),  # Simple uniform weighting
            error
        )
        
        self.context_weights[context] += delta
        
        # Normalize to prevent explosion
        self.context_weights[context] = np.clip(
            self.context_weights[context], -1, 1
        )
    
    def generate_text(self, prompt: str = "", max_length: int = 50) -> str:
        """Generate text from learned patterns"""
        tokens = self.tokenize(prompt) if prompt else ['<START>']
        token_indices = [self.token_to_idx.get(t, self.token_to_idx['<UNK>']) 
                        for t in tokens]
        
        for _ in range(max_length):
            # Get context
            start = max(0, len(token_indices) - self.context_window)
            context = tuple(token_indices[start:])
            
            # Predict next token
            if context in self.context_weights:
                pred_emb = self._predict_from_context(context)
                
                # Find closest token in embedding space
                similarities = []
                for idx, emb in self.embeddings.items():
                    sim = np.dot(pred_emb, emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(emb) + 1e-8)
                    similarities.append((idx, sim))
                
                # Sample from top predictions
                similarities.sort(key=lambda x: x[1], reverse=True)
                next_idx = similarities[0][0] if similarities else self.token_to_idx['<UNK>']
            else:
                # Random token if no context
                next_idx = np.random.choice(list(self.embeddings.keys()))
            
            if next_idx == self.token_to_idx.get('<END>'):
                break
                
            token_indices.append(next_idx)
        
        # Convert back to text
        generated = ' '.join(self.idx_to_token[idx] for idx in token_indices)
        return generated.replace(' <SPACE> ', ' ').replace('<START> ', '').replace(' <END>', '')
    
    def get_stats(self) -> Dict[str, any]:
        """Get learning statistics"""
        return {
            'vocabulary_size': len(self.token_to_idx),
            'total_tokens_seen': self.total_tokens,
            'unique_contexts': len(self.context_weights),
            'most_common_tokens': self.token_counts.most_common(10)
        }
    
    def save_state(self, filepath: str):
        """Save learned state"""
        import pickle
        state = {
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'embeddings': self.embeddings,
            'context_weights': self.context_weights,
            'token_counts': self.token_counts,
            'total_tokens': self.total_tokens,
            'next_idx': self.next_idx
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load learned state"""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.token_to_idx = state['token_to_idx']
        self.idx_to_token = state['idx_to_token']
        self.embeddings = state['embeddings']
        self.context_weights = state['context_weights']
        self.token_counts = state['token_counts']
        self.total_tokens = state['total_tokens']
        self.next_idx = state['next_idx']
