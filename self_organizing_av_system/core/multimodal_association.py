import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum

from .backend import xp, to_cpu

logger = logging.getLogger(__name__)


class AssociationMode(Enum):
    """Association modes for multimodal learning"""
    HEBBIAN = "hebbian"             # Simple Hebbian learning
    COMPETITIVE = "competitive"     # Competitive learning with winner-take-all
    STDP = "stdp"                   # Spike-timing-dependent plasticity-inspired
    ADAPTIVE = "adaptive"           # Adaptive learning that combines multiple strategies


class MultimodalAssociation:
    """
    Implements multimodal association learning to establish connections
    between different sensory modalities (e.g., visual and auditory).
    
    This module enables:
    1. Learning cross-modal associations between representations
    2. Completing partial patterns through cross-modal reconstruction
    3. Attentional modulation and feature binding
    4. Bidirectional influence between modalities
    """
    
    def __init__(
        self,
        modality_sizes: Dict[str, int],
        association_size: int = None,
        learning_rate: float = 0.01,
        association_threshold: float = 0.1,
        association_mode: Union[str, AssociationMode] = AssociationMode.HEBBIAN,
        modality_weights: Dict[str, float] = None,
        normalization_mode: str = "softmax",
        lateral_inhibition: float = 0.2,
        use_sparse_coding: bool = True,
        stability_threshold: float = 0.1,
        regularization_strength: float = 0.001,
        decay_rate: float = 0.0005,
        enable_attention: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize multimodal association module.
        
        Args:
            modality_sizes: Dictionary mapping modality names to representation sizes
            association_size: Size of the multimodal association layer (None to auto-calculate)
            learning_rate: Learning rate for association weights
            association_threshold: Threshold for establishing an association
            association_mode: Mode of association learning
            modality_weights: Dictionary mapping modality names to their influence weights
            normalization_mode: How to normalize association activity ("softmax", "max", "none")
            lateral_inhibition: Strength of lateral inhibition within association layer
            use_sparse_coding: Whether to use sparse coding in the association layer
            stability_threshold: Threshold for determining stable associations
            regularization_strength: L2 regularization strength
            decay_rate: Weight decay rate to forget unused associations
            enable_attention: Whether to enable attentional modulation
            random_seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if random_seed is not None:
            xp.random.seed(random_seed)
            
        self.modality_sizes = modality_sizes
        
        # Auto-calculate association size if not provided
        if association_size is None:
            # Set to the maximum of input modality sizes
            association_size = max(modality_sizes.values())
        self.association_size = association_size
        
        self.learning_rate = learning_rate
        self.association_threshold = association_threshold
        
        # Convert string to enum if needed
        if isinstance(association_mode, str):
            association_mode = AssociationMode(association_mode)
        self.association_mode = association_mode
        
        # Set default modality weights if not provided
        if modality_weights is None:
            modality_weights = {modality: 1.0 / len(modality_sizes) for modality in modality_sizes}
        self.modality_weights = modality_weights
        
        self.normalization_mode = normalization_mode
        self.lateral_inhibition = lateral_inhibition
        self.use_sparse_coding = use_sparse_coding
        self.stability_threshold = stability_threshold
        self.regularization_strength = regularization_strength
        self.decay_rate = decay_rate
        self.enable_attention = enable_attention
        
        # Initialize weights from each modality to association layer
        self.forward_weights = {}
        self.backward_weights = {}
        self._weights = {}  # Storage for weights accessed through property
        
        for modality, size in modality_sizes.items():
            # Forward weights: modality -> association
            self.forward_weights[modality] = xp.random.normal(
                0, 0.01, (self.association_size, size)
            )
            
            # Backward weights: association -> modality (for reconstruction)
            self.backward_weights[modality] = xp.random.normal(
                0, 0.01, (size, self.association_size)
            )
            
            # Store in weights property
            self._weights[modality] = self.forward_weights[modality]
        
        # Initialize association layer state
        self.association_activity = xp.zeros(self.association_size)
        
        # Attention weights (modulates influence of each modality)
        self.attention_weights = {modality: 1.0 for modality in modality_sizes}
        
        # Stability tracking
        self.weight_deltas = {modality: 0.0 for modality in modality_sizes}
        self.stable_associations = set()
        
        # Performance tracking
        self.reconstruction_errors = {modality: [] for modality in modality_sizes}
        self.mean_reconstruction_error = {modality: 0.0 for modality in modality_sizes}
        self.update_count = 0
        
        # Cache for efficiency
        self.last_reconstruction = {}
        
        logger.info(f"Initialized multimodal association: "
                   f"modalities={list(modality_sizes.keys())}, "
                   f"association_size={association_size}, "
                   f"mode={association_mode.value}")
    
    @property
    def weights(self):
        """Get the association weights dictionary."""
        # Return forward weights by default
        return {modality: weights.copy() for modality, weights in self.forward_weights.items()}
        
    @weights.setter
    def weights(self, new_weights):
        """Set the association weights.
        
        Args:
            new_weights: Dictionary of weights for each modality
        """
        # Validate input
        if not isinstance(new_weights, dict):
            raise ValueError("Weights must be a dictionary")
        
        # Store weights
        self._weights = new_weights
        
        # Update internal state
        for modality, weights in new_weights.items():
            if modality in self.modality_sizes:
                # Make sure we have valid weights as numpy arrays
                if isinstance(weights, list):
                    weights = xp.array(weights)
                
                # Check if dimensions match current network
                current_size = self.modality_sizes[modality]
                association_size = self.association_size
                
                # Check if weights need resizing
                if isinstance(weights, xp.ndarray) and (len(weights.shape) != 2 or weights.shape[0] != association_size or weights.shape[1] != current_size):
                    logger.warning(f"Weight size mismatch for {modality}: saved={weights.shape}, expected=({association_size}, {current_size})")
                    logger.info(f"Creating new random weights for {modality}")
                    
                    # Create new random weights instead of trying to resize (safer option)
                    self.forward_weights[modality] = xp.random.normal(
                        0, 0.01, (association_size, current_size)
                    )
                    
                    # Also recreate backward weights
                    self.backward_weights[modality] = xp.random.normal(
                        0, 0.01, (current_size, association_size)
                    )
                    
                    continue  # Skip to next modality
                
                # Update forward weights
                if modality in self.forward_weights:
                    self.forward_weights[modality] = weights
        
        # Log the update
        logger.info(f"Updated association weights for modalities: {list(new_weights.keys())}")
    
    def update(
        self,
        modality_activities: Dict[str, xp.ndarray],
        learning_enabled: bool = True,
        attention_focus: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update associations based on current activity in modalities.
        
        Args:
            modality_activities: Dictionary mapping modality names to their activity vectors
            learning_enabled: Whether to update weights during this step
            attention_focus: Optional modality to focus attention on
            
        Returns:
            Dictionary with association results
        """
        self.update_count += 1
        
        # Apply attention focus if specified
        if attention_focus is not None and self.enable_attention:
            # Boost specified modality
            old_weights = self.attention_weights.copy()
            
            # Reset weights
            for modality in self.attention_weights:
                self.attention_weights[modality] = 0.5  # Base level
            
            # Boost focused modality
            self.attention_weights[attention_focus] = 2.0
            
            # Normalize
            total = sum(self.attention_weights.values())
            for modality in self.attention_weights:
                self.attention_weights[modality] /= total
        
        # Initialize result dictionary
        result = {
            "association_activity": None,
            "reconstructions": {},
            "reconstruction_errors": {},
            "weight_changes": {},
            "stability": {},
        }
        
        # Update association layer based on input activities
        self._update_association_layer(modality_activities)
        
        # Copy current association activity to result
        result["association_activity"] = self.association_activity.copy()
        
        # Generate reconstructions for each modality
        reconstructions = {}
        reconstruction_errors = {}
        
        for modality in self.modality_sizes:
            # Skip if this modality wasn't provided
            if modality not in modality_activities:
                continue
                
            # Reconstruct this modality from association layer
            reconstruction = self._reconstruct_modality(modality)
            reconstructions[modality] = reconstruction
            
            # Calculate reconstruction error
            orig = modality_activities[modality]
            error = xp.mean(xp.abs(orig - reconstruction))
            reconstruction_errors[modality] = error
            
            # Store in result
            result["reconstructions"][modality] = reconstruction
            result["reconstruction_errors"][modality] = float(error)
            
            # Update running average of reconstruction error
            alpha = 0.1  # Smoothing factor
            self.mean_reconstruction_error[modality] = (
                (1 - alpha) * self.mean_reconstruction_error[modality] + 
                alpha * error
            )
            
            # Store error
            self.reconstruction_errors[modality].append((self.update_count, error))
            if len(self.reconstruction_errors[modality]) > 1000:
                self.reconstruction_errors[modality] = self.reconstruction_errors[modality][-1000:]
        
        # Update weights if learning is enabled
        if learning_enabled:
            weight_changes = self._update_weights(modality_activities)
            
            # Store weight changes in result
            for modality, change in weight_changes.items():
                result["weight_changes"][modality] = float(change)
                
            # Update stability for each modality
            for modality in self.modality_sizes:
                if modality in weight_changes:
                    stability = 1.0 - min(1.0, weight_changes[modality] / self.stability_threshold)
                    result["stability"][modality] = float(stability)
                    
                    # Check if association is stable
                    if weight_changes[modality] < self.stability_threshold:
                        self.stable_associations.add(modality)
                    else:
                        if modality in self.stable_associations:
                            self.stable_associations.remove(modality)
        
        # Cache reconstructions for next update
        self.last_reconstruction = reconstructions
        
        return result
    
    def _update_association_layer(self, modality_activities: Dict[str, xp.ndarray]):
        """
        Update the association layer based on current activities.
        
        Args:
            modality_activities: Dictionary mapping modality names to their activities
        """
        # Initialize association activity
        self.association_activity = xp.zeros(self.association_size)
        
        # Combine contributions from each modality
        for modality, activity in modality_activities.items():
            if modality not in self.forward_weights:
                # Skip modalities without weights
                continue
            
            # Apply forward weights
            contribution = xp.dot(self.forward_weights[modality], activity)
            
            # Apply modality weight
            if modality in self.modality_weights:
                contribution *= self.modality_weights[modality]
            
            # Apply attention weight if enabled
            if self.enable_attention and modality in self.attention_weights:
                contribution *= self.attention_weights[modality]
            
            # Add to association activity
            self.association_activity += contribution
        
        # Apply lateral inhibition if enabled
        if self.lateral_inhibition > 0:
            # Simple k-winners-take-all approach
            if self.use_sparse_coding:
                # Determine activation threshold based on top-k neurons
                k = max(1, int(0.1 * self.association_size))  # Activate top 10%
                threshold = xp.sort(self.association_activity)[-k]
                
                # Apply threshold
                inhibition_mask = self.association_activity < threshold
                self.association_activity[inhibition_mask] *= (1 - self.lateral_inhibition)
        
        # Apply normalization
        if self.normalization_mode == "softmax":
            # Softmax normalization
            exp_act = xp.exp(self.association_activity - xp.max(self.association_activity))
            self.association_activity = exp_act / xp.sum(exp_act)
        elif self.normalization_mode == "max":
            # Max normalization
            max_act = xp.max(xp.abs(self.association_activity))
            if max_act > 0:
                self.association_activity /= max_act
    
    def _reconstruct_modality(self, modality: str) -> xp.ndarray:
        """
        Reconstruct activity for a modality based on association layer.
        
        Args:
            modality: Name of modality to reconstruct
            
        Returns:
            Reconstructed activity vector for the modality
        """
        if modality not in self.backward_weights:
            # Return zeros if modality doesn't exist
            return xp.zeros(self.modality_sizes.get(modality, 0))
        
        # Apply backward weights
        reconstruction = xp.dot(self.backward_weights[modality], self.association_activity)
        
        # Normalize reconstruction
        max_value = xp.max(xp.abs(reconstruction))
        if max_value > 0:
            reconstruction /= max_value
        
        return reconstruction
    
    def _update_weights(self, modality_activities: Dict[str, xp.ndarray]) -> Dict[str, float]:
        """
        Update association weights based on current activities.
        
        Args:
            modality_activities: Dictionary mapping modality names to their activities
            
        Returns:
            Dictionary mapping modality names to their weight change magnitudes
        """
        weight_changes = {}
        
        # Update weights for each modality
        for modality, activity in modality_activities.items():
            if modality not in self.forward_weights or modality not in self.backward_weights:
                # Skip modalities without weights
                continue
            
            # Calculate weight changes based on association mode
            if self.association_mode == AssociationMode.HEBBIAN:
                # Hebbian learning: increase weights between co-active neurons
                fw_delta = self.learning_rate * xp.outer(self.association_activity, activity)
                bw_delta = self.learning_rate * xp.outer(activity, self.association_activity)
            
            elif self.association_mode == AssociationMode.COMPETITIVE:
                # Competitive learning: strengthen weights for winning neurons
                # Find winning associations
                winners = self.association_activity > xp.mean(self.association_activity)
                
                # Calculate deltas for winning associations
                fw_delta = xp.zeros_like(self.forward_weights[modality])
                bw_delta = xp.zeros_like(self.backward_weights[modality])
                
                # Update only winning neurons
                for i in range(self.association_size):
                    if winners[i]:
                        # Direction is towards the input
                        error = activity - xp.dot(self.backward_weights[modality], 
                                                 xp.eye(self.association_size)[i])
                        
                        # Update forward and backward weights for this winner
                        fw_delta[i, :] = self.learning_rate * error
                        bw_delta[:, i] = self.learning_rate * error
            
            elif self.association_mode == AssociationMode.STDP:
                # STDP-inspired learning: timing matters
                # For simplicity, we'll use current activity and previous reconstructions
                if modality in self.last_reconstruction:
                    prev_reconstruction = self.last_reconstruction[modality]
                    
                    # Calculate prediction error
                    error = activity - prev_reconstruction
                    
                    # Causal update (input -> association)
                    fw_delta = self.learning_rate * xp.outer(self.association_activity, error)
                    
                    # Anticausal update (association -> reconstruction)
                    bw_delta = self.learning_rate * xp.outer(error, self.association_activity)
                else:
                    # Default to Hebbian for first update
                    fw_delta = self.learning_rate * xp.outer(self.association_activity, activity)
                    bw_delta = self.learning_rate * xp.outer(activity, self.association_activity)
            
            else:  # AssociationMode.ADAPTIVE or fallback
                # Combine strategies based on current performance
                # Check reconstruction error to determine approach
                if modality in self.mean_reconstruction_error:
                    error_level = self.mean_reconstruction_error[modality]
                    
                    if error_level > 0.5:  # High error, use competitive
                        # Find winning associations
                        winners = self.association_activity > xp.mean(self.association_activity)
                        
                        fw_delta = xp.zeros_like(self.forward_weights[modality])
                        bw_delta = xp.zeros_like(self.backward_weights[modality])
                        
                        # Update only winning neurons
                        for i in range(self.association_size):
                            if winners[i]:
                                error = activity - xp.dot(self.backward_weights[modality], 
                                                         xp.eye(self.association_size)[i])
                                fw_delta[i, :] = self.learning_rate * error
                                bw_delta[:, i] = self.learning_rate * error
                    else:  # Low error, use Hebbian
                        fw_delta = self.learning_rate * xp.outer(self.association_activity, activity)
                        bw_delta = self.learning_rate * xp.outer(activity, self.association_activity)
                else:
                    # Default to Hebbian
                    fw_delta = self.learning_rate * xp.outer(self.association_activity, activity)
                    bw_delta = self.learning_rate * xp.outer(activity, self.association_activity)
            
            # Apply regularization
            fw_delta -= self.regularization_strength * self.forward_weights[modality]
            bw_delta -= self.regularization_strength * self.backward_weights[modality]
            
            # Apply weight decay to forget unused connections
            if self.decay_rate > 0:
                # Decay weights proportional to their magnitude
                fw_decay = self.decay_rate * self.forward_weights[modality]
                bw_decay = self.decay_rate * self.backward_weights[modality]
                
                # Apply decay
                fw_delta -= fw_decay
                bw_delta -= bw_decay
            
            # Apply thresholding to encourage sparse connectivity
            fw_delta[xp.abs(fw_delta) < self.association_threshold] = 0
            bw_delta[xp.abs(bw_delta) < self.association_threshold] = 0
            
            # Apply weight updates
            self.forward_weights[modality] += fw_delta
            self.backward_weights[modality] += bw_delta
            
            # Calculate total weight change magnitude
            fw_change = xp.mean(xp.abs(fw_delta))
            bw_change = xp.mean(xp.abs(bw_delta))
            weight_change = (fw_change + bw_change) / 2
            
            weight_changes[modality] = weight_change
            self.weight_deltas[modality] = weight_change
        
        return weight_changes
    
    def get_cross_modal_prediction(
        self,
        source_modality: str,
        source_activity: xp.ndarray,
        target_modality: str,
    ) -> xp.ndarray:
        """
        Generate prediction for target modality given activity in source modality.
        
        Args:
            source_modality: Source modality name
            source_activity: Activity vector for source modality
            target_modality: Target modality to predict
            
        Returns:
            Predicted activity for target modality
        """
        if (source_modality not in self.forward_weights or
            target_modality not in self.backward_weights):
            # Return zeros if modalities don't exist
            return xp.zeros(self.modality_sizes.get(target_modality, 0))
        
        # Calculate association activity from source
        association = xp.dot(self.forward_weights[source_modality], source_activity)
        
        # Apply normalization
        if self.normalization_mode == "softmax":
            # Softmax normalization
            exp_act = xp.exp(association - xp.max(association))
            association = exp_act / xp.sum(exp_act)
        elif self.normalization_mode == "max":
            # Max normalization
            max_act = xp.max(xp.abs(association))
            if max_act > 0:
                association /= max_act
        
        # Generate prediction for target modality
        prediction = xp.dot(self.backward_weights[target_modality], association)
        
        # Normalize prediction
        max_value = xp.max(xp.abs(prediction))
        if max_value > 0:
            prediction /= max_value
        
        return prediction
    
    def integrate_multiple_modalities(
        self,
        modality_activities: Dict[str, xp.ndarray],
        target_modality: str,
    ) -> xp.ndarray:
        """
        Integrate multiple modalities to produce a prediction for target modality.
        
        Args:
            modality_activities: Dictionary of activity vectors for available modalities
            target_modality: Target modality to predict
            
        Returns:
            Predicted activity for target modality
        """
        # First update the association layer with available modalities
        self._update_association_layer(modality_activities)
        
        # Then reconstruct the target modality
        return self._reconstruct_modality(target_modality)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about association performance.
        
        Returns:
            Dictionary with statistics
        """
        # Calculate average reconstruction errors
        reconstruction_errors = {}
        for modality, errors in self.reconstruction_errors.items():
            recent_errors = [err for _, err in errors[-100:]] if errors else [0]
            reconstruction_errors[modality] = float(xp.mean(recent_errors))
        
        # Calculate weight sparsity (proportion of zero weights)
        weight_sparsity = {}
        for modality in self.forward_weights:
            fw_sparsity = xp.mean(xp.abs(self.forward_weights[modality]) < 0.01)
            bw_sparsity = xp.mean(xp.abs(self.backward_weights[modality]) < 0.01)
            weight_sparsity[modality] = float((fw_sparsity + bw_sparsity) / 2)
        
        stats = {
            'modalities': list(self.modality_sizes.keys()),
            'association_size': self.association_size,
            'association_mode': self.association_mode.value,
            'reconstruction_errors': reconstruction_errors,
            'weight_deltas': {m: float(d) for m, d in self.weight_deltas.items()},
            'stable_associations': list(self.stable_associations),
            'weight_sparsity': weight_sparsity,
            'update_count': self.update_count
        }
        
        return stats
    
    def get_multimodal_activity(self) -> Dict[str, Any]:
        """
        Get the current activity state of the multimodal association layer.
        
        Returns:
            Dictionary with multimodal activity statistics
        """
        # Calculate statistics
        sparsity = xp.mean(self.association_activity > 0)
        avg_activation = xp.mean(self.association_activity)
        active_count = xp.sum(self.association_activity > 0)
        
        return {
            'activations': self.association_activity,
            'sparsity': float(sparsity),
            'avg_activation': float(avg_activation),
            'active_count': int(active_count)
        }
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the multimodal association state for saving.
        
        Returns:
            Dictionary with serialized state
        """
        # Convert weights to lists
        forward_weights_serialized = {}
        for modality, weights in self.forward_weights.items():
            forward_weights_serialized[modality] = weights.tolist()
        
        backward_weights_serialized = {}
        for modality, weights in self.backward_weights.items():
            backward_weights_serialized[modality] = weights.tolist()
        
        # Create serialized data
        data = {
            'modality_sizes': self.modality_sizes,
            'association_size': self.association_size,
            'learning_rate': self.learning_rate,
            'association_threshold': self.association_threshold,
            'association_mode': self.association_mode.value,
            'modality_weights': self.modality_weights,
            'normalization_mode': self.normalization_mode,
            'lateral_inhibition': self.lateral_inhibition,
            'use_sparse_coding': self.use_sparse_coding,
            'stability_threshold': self.stability_threshold,
            'forward_weights': forward_weights_serialized,
            'backward_weights': backward_weights_serialized,
            'attention_weights': self.attention_weights,
            'stable_associations': list(self.stable_associations),
            'mean_reconstruction_error': self.mean_reconstruction_error,
            'update_count': self.update_count
        }
        
        return data
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'MultimodalAssociation':
        """
        Create a multimodal association instance from serialized data.
        
        Args:
            data: Dictionary with serialized state
            
        Returns:
            MultimodalAssociation instance
        """
        instance = cls(
            modality_sizes=data['modality_sizes'],
            association_size=data['association_size'],
            learning_rate=data['learning_rate'],
            association_threshold=data['association_threshold'],
            association_mode=data['association_mode'],
            modality_weights=data['modality_weights'],
            normalization_mode=data['normalization_mode'],
            lateral_inhibition=data['lateral_inhibition'],
            use_sparse_coding=data['use_sparse_coding'],
            stability_threshold=data['stability_threshold']
        )
        
        # Restore weights
        for modality, weights_list in data['forward_weights'].items():
            instance.forward_weights[modality] = xp.array(weights_list)
        
        for modality, weights_list in data['backward_weights'].items():
            instance.backward_weights[modality] = xp.array(weights_list)
        
        # Restore state
        instance.attention_weights = data['attention_weights']
        instance.stable_associations = set(data['stable_associations'])
        instance.mean_reconstruction_error = data['mean_reconstruction_error']
        instance.update_count = data['update_count']
        
        return instance 