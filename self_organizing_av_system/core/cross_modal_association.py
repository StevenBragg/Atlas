import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import time

from .backend import xp, to_cpu

logger = logging.getLogger(__name__)


class CrossModalAssociation:
    """
    Manages associations between different sensory modalities using Hebbian learning.
    
    This module implements cross-modal learning between visual and auditory pathways
    based on temporal coincidence and correlation. It supports:
    
    1. Bidirectional Hebbian learning between modalities
    2. Temporal association tracking with variable time windows
    3. Cross-modal prediction in both directions
    4. Association strength analysis and visualization
    5. Pruning of weak associations
    
    The class maintains association matrices that capture relationships between 
    neural activities across modalities and uses these to enable cross-modal
    prediction and completion.
    """
    
    def __init__(
        self,
        visual_size: int,
        audio_size: int,
        learning_rate: float = 0.01,
        decay_rate: float = 0.0001,
        association_threshold: float = 0.1,
        temporal_window_size: int = 5,
        bidirectional: bool = True,
        initial_weight_scale: float = 0.01,
        max_weight: float = 1.0,
        prune_threshold: float = 0.01,
        prediction_strength: float = 0.5,
        normalize_associations: bool = True,
        connection_sparsity: float = 1.0
    ):
        """
        Initialize cross-modal association learning.
        
        Args:
            visual_size: Size of visual representation
            audio_size: Size of audio representation
            learning_rate: Rate of Hebbian association learning
            decay_rate: Rate of association decay (forgetting)
            association_threshold: Threshold for considering an association valid
            temporal_window_size: Number of time steps to consider for temporal associations
            bidirectional: Whether to learn bidirectional associations
            initial_weight_scale: Scale for initializing association weights
            max_weight: Maximum allowed association weight
            prune_threshold: Threshold for pruning weak associations
            prediction_strength: Strength of cross-modal prediction signals
            normalize_associations: Whether to normalize association matrices
            connection_sparsity: Initial connection density (1.0 = fully connected)
        """
        self.visual_size = visual_size
        self.audio_size = audio_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.association_threshold = association_threshold
        self.temporal_window_size = temporal_window_size
        self.bidirectional = bidirectional
        self.max_weight = max_weight
        self.prune_threshold = prune_threshold
        self.prediction_strength = prediction_strength
        self.normalize_associations = normalize_associations
        
        # Initialize association matrices with random small weights
        # Visual to audio associations
        self.visual_to_audio = xp.random.normal(
            0, initial_weight_scale, (visual_size, audio_size)
        )
        
        # Audio to visual associations
        self.audio_to_visual = xp.random.normal(
            0, initial_weight_scale, (audio_size, visual_size)
        )
        
        # Apply connection sparsity if < 1.0
        if connection_sparsity < 1.0:
            # Create connection masks
            v2a_mask = xp.random.random((visual_size, audio_size)) < connection_sparsity
            a2v_mask = xp.random.random((audio_size, visual_size)) < connection_sparsity
            
            # Apply masks
            self.visual_to_audio *= v2a_mask
            self.audio_to_visual *= a2v_mask
        
        # Buffers for temporal association
        self.visual_buffer = []
        self.audio_buffer = []
        self.max_buffer_size = temporal_window_size * 2 + 1  # Center plus window in both directions
        
        # Statistics
        self.update_count = 0
        self.association_strength_history = []
        self.prediction_accuracy_history = {
            'visual_to_audio': [],
            'audio_to_visual': []
        }
        
        # Timestamp of last update
        self.last_update_time = time.time()
        
        logger.info(f"Initialized cross-modal association: {visual_size}x{audio_size}")
    
    def update(
        self,
        visual_activity: xp.ndarray,
        audio_activity: xp.ndarray,
        learn: bool = True
    ) -> Dict[str, xp.ndarray]:
        """
        Update cross-modal associations based on concurrent sensory input.
        
        Args:
            visual_activity: Activity pattern from visual pathway
            audio_activity: Activity pattern from audio pathway
            learn: Whether to update associations based on current input
            
        Returns:
            Dictionary with prediction results
        """
        # Verify input dimensions
        assert len(visual_activity) == self.visual_size, "Visual activity size mismatch"
        assert len(audio_activity) == self.audio_size, "Audio activity size mismatch"
        
        # Add current activities to temporal buffers
        self._update_buffers(visual_activity, audio_activity)
        
        # Generate predictions in both directions
        visual_prediction = self._predict_visual_from_audio(audio_activity)
        audio_prediction = self._predict_audio_from_visual(visual_activity)
        
        # Combine predictions with actual activities for output
        combined_visual = visual_activity + visual_prediction * self.prediction_strength
        combined_audio = audio_activity + audio_prediction * self.prediction_strength
        
        # Apply learning if enabled
        if learn:
            # Perform Hebbian updates using temporal information
            self._hebbian_update(visual_activity, audio_activity)
            
            # Apply association maintenance
            if self.update_count % 100 == 0:
                self._prune_weak_associations()
                
            self.update_count += 1
        
        # Update metrics
        if self.update_count % 50 == 0:
            self._update_metrics(visual_activity, audio_activity, 
                                visual_prediction, audio_prediction)
        
        # Update timestamp
        self.last_update_time = time.time()
        
        return {
            'visual_prediction': visual_prediction,
            'audio_prediction': audio_prediction,
            'combined_visual': combined_visual,
            'combined_audio': combined_audio
        }
    
    def _update_buffers(self, visual_activity: xp.ndarray, audio_activity: xp.ndarray) -> None:
        """
        Update temporal buffers with current activities.
        
        Args:
            visual_activity: Current visual activity
            audio_activity: Current audio activity
        """
        # Add new activities to buffers
        self.visual_buffer.append(visual_activity.copy())
        self.audio_buffer.append(audio_activity.copy())
        
        # Trim buffers if they exceed maximum size
        if len(self.visual_buffer) > self.max_buffer_size:
            self.visual_buffer.pop(0)
        if len(self.audio_buffer) > self.max_buffer_size:
            self.audio_buffer.pop(0)
    
    def _hebbian_update(self, visual_activity: xp.ndarray, audio_activity: xp.ndarray) -> None:
        """
        Apply Hebbian learning to update association matrices.
        
        Args:
            visual_activity: Current visual activity
            audio_activity: Current audio activity
        """
        # Basic Hebbian update: strengthen connections between co-active neurons
        # w_ij += lr * x_i * y_j
        
        # Update visual to audio associations
        # Outer product gives all pairwise activities
        hebbian_update = xp.outer(visual_activity, audio_activity)
        self.visual_to_audio += self.learning_rate * hebbian_update
        
        # Update audio to visual associations if bidirectional
        if self.bidirectional:
            hebbian_update = xp.outer(audio_activity, visual_activity)
            self.audio_to_visual += self.learning_rate * hebbian_update
        
        # Apply decay to all associations (forgetting)
        self.visual_to_audio -= self.decay_rate * self.visual_to_audio
        if self.bidirectional:
            self.audio_to_visual -= self.decay_rate * self.audio_to_visual
        
        # Apply temporal associations using buffered activities
        self._apply_temporal_associations()
        
        # Enforce constraints on association weights
        self._enforce_constraints()
    
    def _apply_temporal_associations(self) -> None:
        """
        Apply associations based on temporal proximity in the buffers.
        """
        if len(self.visual_buffer) <= 1 or len(self.audio_buffer) <= 1:
            return  # Need at least two time points
        
        # Get current time point indices
        current_idx = len(self.visual_buffer) - 1
        
        # Calculate temporal weightings based on distance from current time
        temporal_weights = self._get_temporal_weights(current_idx)
        
        # Apply temporal associations within the window
        for t, weight in temporal_weights.items():
            if t < 0 or t >= len(self.visual_buffer) or t >= len(self.audio_buffer):
                continue
                
            # Get activities at time t
            v_activity = self.visual_buffer[t]
            a_activity = self.audio_buffer[t]
            
            # Apply weighted Hebbian updates
            hebbian_update = xp.outer(v_activity, a_activity)
            self.visual_to_audio += self.learning_rate * weight * hebbian_update
            
            if self.bidirectional:
                hebbian_update = xp.outer(a_activity, v_activity)
                self.audio_to_visual += self.learning_rate * weight * hebbian_update
    
    def _get_temporal_weights(self, current_idx: int) -> Dict[int, float]:
        """
        Calculate temporal association weights based on distance from current time.
        
        Args:
            current_idx: Index of current time in buffer
            
        Returns:
            Dictionary mapping time indices to association weights
        """
        weights = {}
        
        # Calculate weights using a Gaussian-like falloff
        for t in range(max(0, current_idx - self.temporal_window_size),
                      min(len(self.visual_buffer), current_idx + self.temporal_window_size + 1)):
            if t == current_idx:
                weights[t] = 1.0  # Full weight for current time
            else:
                # Calculate temporal weight with distance-based falloff
                distance = abs(t - current_idx)
                weights[t] = xp.exp(-0.5 * (distance / (self.temporal_window_size / 2))**2)
        
        return weights
    
    def _enforce_constraints(self) -> None:
        """
        Enforce constraints on association matrices.
        """
        # Clip weights to maximum value
        self.visual_to_audio = xp.clip(self.visual_to_audio, 0, self.max_weight)
        self.audio_to_visual = xp.clip(self.audio_to_visual, 0, self.max_weight)
        
        # Normalize association matrices if enabled
        if self.normalize_associations:
            # Normalize by row (source neurons)
            row_sums_v2a = xp.sum(self.visual_to_audio, axis=1, keepdims=True)
            row_sums_a2v = xp.sum(self.audio_to_visual, axis=1, keepdims=True)
            
            # Avoid division by zero
            row_sums_v2a[row_sums_v2a == 0] = 1.0
            row_sums_a2v[row_sums_a2v == 0] = 1.0
            
            # Apply normalization
            self.visual_to_audio = self.visual_to_audio / row_sums_v2a
            self.audio_to_visual = self.audio_to_visual / row_sums_a2v
    
    def _predict_visual_from_audio(self, audio_activity: xp.ndarray) -> xp.ndarray:
        """
        Predict visual activity based on audio input.
        
        Args:
            audio_activity: Current audio activity
            
        Returns:
            Predicted visual activity
        """
        # Matrix multiplication: audio_activity.dot(audio_to_visual_matrix)
        prediction = xp.dot(audio_activity, self.audio_to_visual)
        
        # Only keep predictions above threshold
        prediction[prediction < self.association_threshold] = 0
        
        return prediction
    
    def _predict_audio_from_visual(self, visual_activity: xp.ndarray) -> xp.ndarray:
        """
        Predict audio activity based on visual input.
        
        Args:
            visual_activity: Current visual activity
            
        Returns:
            Predicted audio activity
        """
        # Matrix multiplication: visual_activity.dot(visual_to_audio_matrix)
        prediction = xp.dot(visual_activity, self.visual_to_audio)
        
        # Only keep predictions above threshold
        prediction[prediction < self.association_threshold] = 0
        
        return prediction
    
    def _prune_weak_associations(self) -> None:
        """
        Prune weak associations to maintain sparsity.
        """
        # Set weights below prune threshold to zero
        self.visual_to_audio[self.visual_to_audio < self.prune_threshold] = 0
        self.audio_to_visual[self.audio_to_visual < self.prune_threshold] = 0
        
        logger.debug(f"Pruned associations. Remaining: "
                     f"V→A: {xp.sum(self.visual_to_audio > 0)}, "
                     f"A→V: {xp.sum(self.audio_to_visual > 0)}")
    
    def _update_metrics(
        self,
        visual_activity: xp.ndarray,
        audio_activity: xp.ndarray,
        visual_prediction: xp.ndarray,
        audio_prediction: xp.ndarray
    ) -> None:
        """
        Update association metrics.
        
        Args:
            visual_activity: Actual visual activity
            audio_activity: Actual audio activity
            visual_prediction: Predicted visual activity
            audio_prediction: Predicted audio activity
        """
        # Calculate overall association strength
        v2a_strength = xp.mean(self.visual_to_audio)
        a2v_strength = xp.mean(self.audio_to_visual)
        
        # Record association strength
        self.association_strength_history.append({
            'time': self.update_count,
            'v2a': float(v2a_strength),
            'a2v': float(a2v_strength),
            'total': float(v2a_strength + a2v_strength) / 2
        })
        
        # Trim history if it gets too long
        if len(self.association_strength_history) > 1000:
            self.association_strength_history = self.association_strength_history[-1000:]
        
        # Calculate prediction accuracy
        if xp.sum(visual_activity) > 0 and xp.sum(audio_activity) > 0:
            # Cosine similarity between prediction and actual
            v_norm = xp.linalg.norm(visual_activity)
            a_norm = xp.linalg.norm(audio_activity)
            vp_norm = xp.linalg.norm(visual_prediction)
            ap_norm = xp.linalg.norm(audio_prediction)
            
            # Avoid division by zero
            if v_norm > 0 and vp_norm > 0:
                a2v_accuracy = xp.dot(visual_activity, visual_prediction) / (v_norm * vp_norm)
                self.prediction_accuracy_history['audio_to_visual'].append(float(a2v_accuracy))
            
            if a_norm > 0 and ap_norm > 0:
                v2a_accuracy = xp.dot(audio_activity, audio_prediction) / (a_norm * ap_norm)
                self.prediction_accuracy_history['visual_to_audio'].append(float(v2a_accuracy))
            
            # Trim history
            for key in self.prediction_accuracy_history:
                if len(self.prediction_accuracy_history[key]) > 1000:
                    self.prediction_accuracy_history[key] = self.prediction_accuracy_history[key][-1000:]
    
    def get_association_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current association matrices.
        
        Returns:
            Dictionary with association statistics
        """
        v2a_density = xp.mean(self.visual_to_audio > 0)
        a2v_density = xp.mean(self.audio_to_visual > 0)
        
        # Most strongly associated pairs
        top_v2a_idx = xp.unravel_index(
            xp.argmax(self.visual_to_audio), self.visual_to_audio.shape
        )
        top_a2v_idx = xp.unravel_index(
            xp.argmax(self.audio_to_visual), self.audio_to_visual.shape
        )
        
        # Get prediction accuracy metrics
        v2a_accuracy = 0.0
        a2v_accuracy = 0.0
        
        if self.prediction_accuracy_history['visual_to_audio']:
            v2a_accuracy = xp.mean(self.prediction_accuracy_history['visual_to_audio'][-20:])
        
        if self.prediction_accuracy_history['audio_to_visual']:
            a2v_accuracy = xp.mean(self.prediction_accuracy_history['audio_to_visual'][-20:])
        
        return {
            'v2a_density': float(v2a_density),
            'a2v_density': float(a2v_density),
            'v2a_strength': float(xp.mean(self.visual_to_audio)),
            'a2v_strength': float(xp.mean(self.audio_to_visual)),
            'top_v2a_weight': float(self.visual_to_audio[top_v2a_idx]),
            'top_a2v_weight': float(self.audio_to_visual[top_a2v_idx]),
            'top_v2a_indices': (int(top_v2a_idx[0]), int(top_v2a_idx[1])),
            'top_a2v_indices': (int(top_a2v_idx[0]), int(top_a2v_idx[1])),
            'v2a_accuracy': float(v2a_accuracy),
            'a2v_accuracy': float(a2v_accuracy),
            'update_count': self.update_count
        }
    
    def resize(self, new_visual_size: int = None, new_audio_size: int = None) -> None:
        """
        Resize association matrices when representation sizes change.
        
        Args:
            new_visual_size: New size for visual representation
            new_audio_size: New size for audio representation
        """
        if new_visual_size is None:
            new_visual_size = self.visual_size
        
        if new_audio_size is None:
            new_audio_size = self.audio_size
        
        if new_visual_size == self.visual_size and new_audio_size == self.audio_size:
            return  # No resize needed
        
        logger.info(f"Resizing cross-modal associations from {self.visual_size}x{self.audio_size} "
                    f"to {new_visual_size}x{new_audio_size}")
        
        # Store original matrices
        old_v2a = self.visual_to_audio
        old_a2v = self.audio_to_visual
        
        # Create new matrices
        new_v2a = xp.zeros((new_visual_size, new_audio_size))
        new_a2v = xp.zeros((new_audio_size, new_visual_size))
        
        # Copy existing weights
        min_visual = min(self.visual_size, new_visual_size)
        min_audio = min(self.audio_size, new_audio_size)
        
        new_v2a[:min_visual, :min_audio] = old_v2a[:min_visual, :min_audio]
        new_a2v[:min_audio, :min_visual] = old_a2v[:min_audio, :min_visual]
        
        # Initialize new regions with small random values
        if new_visual_size > self.visual_size:
            new_v2a[self.visual_size:, :min_audio] = xp.random.normal(
                0, 0.01, (new_visual_size - self.visual_size, min_audio)
            )
        
        if new_audio_size > self.audio_size:
            new_v2a[:min_visual, self.audio_size:] = xp.random.normal(
                0, 0.01, (min_visual, new_audio_size - self.audio_size)
            )
            new_a2v[self.audio_size:, :min_visual] = xp.random.normal(
                0, 0.01, (new_audio_size - self.audio_size, min_visual)
            )
        
        if new_visual_size > self.visual_size and new_audio_size > self.audio_size:
            new_v2a[self.visual_size:, self.audio_size:] = xp.random.normal(
                0, 0.01, (new_visual_size - self.visual_size, new_audio_size - self.audio_size)
            )
        
        # Update matrices and sizes
        self.visual_to_audio = new_v2a
        self.audio_to_visual = new_a2v
        self.visual_size = new_visual_size
        self.audio_size = new_audio_size
        
        # Reset buffers
        self.visual_buffer = []
        self.audio_buffer = []
    
    def get_top_associations(self, n: int = 5) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Get the top N strongest associations in both directions.
        
        Args:
            n: Number of top associations to retrieve
            
        Returns:
            Dictionary with lists of (source_idx, target_idx, weight) tuples
        """
        # Flatten matrices
        v2a_flat = self.visual_to_audio.flatten()
        a2v_flat = self.audio_to_visual.flatten()
        
        # Get indices of top n weights
        top_v2a_flat_idx = xp.argsort(v2a_flat)[-n:][::-1]
        top_a2v_flat_idx = xp.argsort(a2v_flat)[-n:][::-1]
        
        # Convert to 2D indices and weights
        top_v2a = []
        for idx in top_v2a_flat_idx:
            v_idx, a_idx = xp.unravel_index(idx, self.visual_to_audio.shape)
            weight = self.visual_to_audio[v_idx, a_idx]
            if weight > 0:  # Only include non-zero weights
                top_v2a.append((int(v_idx), int(a_idx), float(weight)))
        
        top_a2v = []
        for idx in top_a2v_flat_idx:
            a_idx, v_idx = xp.unravel_index(idx, self.audio_to_visual.shape)
            weight = self.audio_to_visual[a_idx, v_idx]
            if weight > 0:  # Only include non-zero weights
                top_a2v.append((int(a_idx), int(v_idx), float(weight)))
        
        return {
            'visual_to_audio': top_v2a,
            'audio_to_visual': top_a2v
        }
    
    def get_synced_activity_stats(self) -> Dict[str, float]:
        """
        Calculate statistics about synchronized activity between modalities.
        
        Returns:
            Dictionary with synchronization statistics
        """
        if not self.visual_buffer or not self.audio_buffer:
            return {
                'sync_correlation': 0.0,
                'sync_mutual_info': 0.0
            }
        
        # Use only the most recent activities
        v_activities = xp.array(self.visual_buffer[-min(10, len(self.visual_buffer)):])
        a_activities = xp.array(self.audio_buffer[-min(10, len(self.audio_buffer)):])
        
        # Ensure same number of samples
        min_samples = min(len(v_activities), len(a_activities))
        v_activities = v_activities[-min_samples:]
        a_activities = a_activities[-min_samples:]
        
        # Calculate correlation between overall activity levels
        v_activity_levels = xp.sum(v_activities, axis=1)
        a_activity_levels = xp.sum(a_activities, axis=1)
        
        # Calculate correlation coefficient
        if xp.std(v_activity_levels) > 0 and xp.std(a_activity_levels) > 0:
            sync_correlation = xp.corrcoef(v_activity_levels, a_activity_levels)[0, 1]
        else:
            sync_correlation = 0.0
        
        # Simplified mutual information estimation (coarse approximation)
        sync_mutual_info = max(0, sync_correlation**2)
        
        return {
            'sync_correlation': float(sync_correlation),
            'sync_mutual_info': float(sync_mutual_info)
        }
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the association state for saving.
        
        Returns:
            Dictionary with serialized state
        """
        return {
            'visual_size': self.visual_size,
            'audio_size': self.audio_size,
            'learning_rate': self.learning_rate,
            'decay_rate': self.decay_rate,
            'association_threshold': self.association_threshold,
            'temporal_window_size': self.temporal_window_size,
            'bidirectional': self.bidirectional,
            'max_weight': self.max_weight,
            'prune_threshold': self.prune_threshold,
            'prediction_strength': self.prediction_strength,
            'normalize_associations': self.normalize_associations,
            'visual_to_audio': self.visual_to_audio.tolist(),
            'audio_to_visual': self.audio_to_visual.tolist(),
            'update_count': self.update_count,
            'association_strength_history': self.association_strength_history[-100:] if self.association_strength_history else []
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'CrossModalAssociation':
        """
        Create an association instance from serialized data.
        
        Args:
            data: Dictionary with serialized state
            
        Returns:
            CrossModalAssociation instance
        """
        instance = cls(
            visual_size=data['visual_size'],
            audio_size=data['audio_size'],
            learning_rate=data['learning_rate'],
            decay_rate=data['decay_rate'],
            association_threshold=data['association_threshold'],
            temporal_window_size=data['temporal_window_size'],
            bidirectional=data['bidirectional'],
            max_weight=data['max_weight'],
            prune_threshold=data['prune_threshold'],
            prediction_strength=data['prediction_strength'],
            normalize_associations=data['normalize_associations']
        )
        
        # Load association matrices
        instance.visual_to_audio = xp.array(data['visual_to_audio'])
        instance.audio_to_visual = xp.array(data['audio_to_visual'])
        
        # Load history and state
        instance.update_count = data['update_count']
        instance.association_strength_history = data.get('association_strength_history', [])
        
        return instance 