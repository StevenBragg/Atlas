import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum

from .backend import xp, to_cpu

logger = logging.getLogger(__name__)


class InhibitionStrategy(Enum):
    """Strategies for lateral inhibition"""
    NONE = "none"                     # No inhibition
    WTA = "winner_take_all"           # Winner-take-all
    KWA = "k_winners_allowed"         # K-winners-allowed
    MEXICAN_HAT = "mexican_hat"       # Mexican hat function
    ADAPTIVE = "adaptive"             # Adaptive inhibition based on activity statistics


class StabilityMechanisms:
    """
    Implements neural network stability mechanisms.
    
    This module provides:
    1. Homeostatic plasticity to maintain target activity levels
    2. Lateral inhibition to promote sparse, distributed representations
    3. Adaptive thresholds for balancing activity
    4. Activity diversity regulation to prevent representational collapse
    5. Weight normalization to prevent unbounded growth
    
    These mechanisms ensure the stability and efficiency of neural representations
    throughout the learning process.
    """
    
    def __init__(
        self,
        representation_size: int,
        target_activity: float = 0.1,
        homeostatic_rate: float = 0.01,
        inhibition_strategy: Union[str, InhibitionStrategy] = InhibitionStrategy.KWA,
        inhibition_strength: float = 0.5,
        k_winners: int = None,
        normalization_threshold: float = 3.0,
        diversity_target: float = 0.8,
        diversity_rate: float = 0.01,
        adaptive_threshold_tau: float = 0.01,
        minimum_threshold: float = 0.01,
        maximum_threshold: float = 1.0,
        weight_scaling_enabled: bool = True,
        weight_decay: float = 0.0001,
    ):
        """
        Initialize stability mechanisms.
        
        Args:
            representation_size: Size of neural representation
            target_activity: Target fraction of active neurons
            homeostatic_rate: Learning rate for homeostatic adaptations
            inhibition_strategy: Strategy for lateral inhibition
            inhibition_strength: Strength of lateral inhibition
            k_winners: Number of winners for K-winners-allowed strategy
            normalization_threshold: Threshold for weight normalization
            diversity_target: Target diversity of neural representations
            diversity_rate: Learning rate for diversity adaptations
            adaptive_threshold_tau: Time constant for threshold adaptation
            minimum_threshold: Minimum value for adaptive thresholds
            maximum_threshold: Maximum value for adaptive thresholds
            weight_scaling_enabled: Whether weight scaling is enabled
            weight_decay: Rate of weight decay for regularization
        """
        self.representation_size = representation_size
        self.target_activity = target_activity
        self.homeostatic_rate = homeostatic_rate
        
        # Convert string to enum if needed
        if isinstance(inhibition_strategy, str):
            inhibition_strategy = InhibitionStrategy(inhibition_strategy)
        self.inhibition_strategy = inhibition_strategy
        
        self.inhibition_strength = inhibition_strength
        if k_winners is None:
            self.k_winners = max(1, int(representation_size * target_activity))
        else:
            self.k_winners = k_winners
        
        self.normalization_threshold = normalization_threshold
        self.diversity_target = diversity_target
        self.diversity_rate = diversity_rate
        self.adaptive_threshold_tau = adaptive_threshold_tau
        self.minimum_threshold = minimum_threshold
        self.maximum_threshold = maximum_threshold
        self.weight_scaling_enabled = weight_scaling_enabled
        self.weight_decay = weight_decay
        
        # Initialize adaptive thresholds
        self.thresholds = xp.ones(representation_size) * 0.5
        
        # Activity statistics tracking
        self.activity_history = xp.zeros((100, representation_size))
        self.activity_history_index = 0
        self.activity_history_filled = False
        
        # Neuron firing rates
        self.firing_rates = xp.zeros(representation_size)
        
        # Initialization tracking
        self.update_count = 0
        
        # Homeostatic factors (intrinsic excitability)
        self.homeostatic_factors = xp.ones(representation_size)
        
        # Weight scaling factors
        self.input_scaling_factors = xp.ones(representation_size)
        self.output_scaling_factors = xp.ones(representation_size)
        
        # Diversity tracking
        self.diversity_score = 1.0
        self.correlation_matrix = xp.zeros((representation_size, representation_size))
        
        # Adaptive inhibition parameters
        if self.inhibition_strategy == InhibitionStrategy.ADAPTIVE:
            # Create inhibition kernel for adaptive inhibition
            # Initially a Mexican hat function with adjustable parameters
            self.inhibition_kernel = self._create_mexican_hat_kernel(representation_size, 
                                                                    width_ratio=0.2)
        else:
            self.inhibition_kernel = None
        
        logger.info(f"Initialized stability mechanisms: size={representation_size}, "
                   f"target_activity={target_activity}, "
                   f"inhibition_strategy={inhibition_strategy.value}")
    
    def _create_mexican_hat_kernel(self, size: int, width_ratio: float = 0.15) -> xp.ndarray:
        """
        Create a Mexican hat kernel for lateral inhibition.
        
        Args:
            size: Size of the kernel
            width_ratio: Width of the central excitatory region as a fraction of size
            
        Returns:
            Mexican hat kernel
        """
        # Create distance matrix
        x = xp.arange(size)
        distances = xp.abs(x[:, xp.newaxis] - x[xp.newaxis, :])
        
        # Calculate width parameters
        sigma_e = width_ratio * size
        sigma_i = 2 * sigma_e
        
        # Mexican hat function
        excitatory = xp.exp(-(distances**2) / (2 * sigma_e**2))
        inhibitory = 0.5 * xp.exp(-(distances**2) / (2 * sigma_i**2))
        
        # Combine to create Mexican hat
        kernel = excitatory - inhibitory
        
        # Zero out the diagonal (self connections)
        xp.fill_diagonal(kernel, 0)
        
        # Normalize
        kernel = kernel / xp.max(xp.abs(kernel))
        
        return kernel
    
    def apply_inhibition(self, activity: xp.ndarray) -> xp.ndarray:
        """
        Apply lateral inhibition to neural activity.
        
        Args:
            activity: Neural activity pattern
            
        Returns:
            Inhibited activity pattern
        """
        # Make a copy to avoid modifying the original
        inhibited = activity.copy()
        
        # Apply the selected inhibition strategy
        if self.inhibition_strategy == InhibitionStrategy.NONE:
            # No inhibition, return original
            return inhibited
        
        elif self.inhibition_strategy == InhibitionStrategy.WTA:
            # Winner-take-all: only the strongest neuron remains active
            if xp.max(inhibited) > 0:
                winner_idx = xp.argmax(inhibited)
                mask = xp.zeros_like(inhibited)
                mask[winner_idx] = 1
                inhibited = inhibited * mask
        
        elif self.inhibition_strategy == InhibitionStrategy.KWA:
            # K-winners-allowed: only the k strongest neurons remain active
            if xp.sum(inhibited > 0) > self.k_winners:
                # Find the k largest activities
                threshold = xp.sort(inhibited)[-self.k_winners]
                # Create mask for values above threshold
                mask = (inhibited >= threshold).astype(float)
                # Apply mask
                inhibited = inhibited * mask
        
        elif self.inhibition_strategy == InhibitionStrategy.MEXICAN_HAT:
            # Mexican hat: local excitation, distant inhibition
            # Convolve with inhibition kernel
            inhibition = xp.zeros_like(inhibited)
            active_indices = xp.where(inhibited > 0)[0]
            
            for i in active_indices:
                # Distance-based inhibition
                distances = xp.abs(xp.arange(len(inhibited)) - i)
                # Mexican hat function: local excitation, distant inhibition
                sigma_e = int(0.05 * len(inhibited))
                sigma_i = int(0.15 * len(inhibited))
                excitatory = xp.exp(-(distances**2) / (2 * sigma_e**2))
                inhibitory = self.inhibition_strength * xp.exp(-(distances**2) / (2 * sigma_i**2))
                effect = inhibited[i] * (excitatory - inhibitory)
                # Accumulate effects (excitatory and inhibitory)
                inhibition += effect
            
            # Combine original activity with inhibition
            inhibited = inhibited + inhibition
            
            # Ensure non-negative values
            inhibited = xp.maximum(0, inhibited)
        
        elif self.inhibition_strategy == InhibitionStrategy.ADAPTIVE:
            # Adaptive inhibition based on learned correlation matrix
            if self.inhibition_kernel is not None:
                # Apply inhibition kernel: multiply current activity vector by kernel matrix
                inhibition = xp.dot(inhibited, self.inhibition_kernel)
                # Adjust inhibition strength
                inhibition *= self.inhibition_strength
                # Apply inhibition
                inhibited = inhibited + inhibition
                # Ensure non-negative values
                inhibited = xp.maximum(0, inhibited)
        
        # Normalize to preserve energy
        if xp.sum(inhibited) > 0:
            total_activity = xp.sum(activity)
            inhibited = inhibited * (total_activity / (xp.sum(inhibited) + 1e-10))
        
        return inhibited
    
    def apply_homeostasis(self, activity: xp.ndarray) -> xp.ndarray:
        """
        Apply homeostatic plasticity to neural activity.
        
        Args:
            activity: Neural activity pattern
            
        Returns:
            Activity after homeostatic adjustment
        """
        # Apply homeostatic factors to activity
        adjusted = activity * self.homeostatic_factors
        
        # Ensure non-negative values
        adjusted = xp.maximum(0, adjusted)
        
        return adjusted
    
    def apply_thresholds(self, activity: xp.ndarray) -> xp.ndarray:
        """
        Apply adaptive thresholds to neural activity.
        
        Args:
            activity: Neural activity pattern
            
        Returns:
            Thresholded activity pattern
        """
        # Apply thresholds
        thresholded = xp.maximum(0, activity - self.thresholds)
        
        return thresholded
    
    def update(
        self,
        pre_activity: xp.ndarray,
        post_activity: xp.ndarray,
        weights: Optional[xp.ndarray] = None
    ) -> Dict[str, xp.ndarray]:
        """
        Update stability mechanisms based on current activity.
        
        Args:
            pre_activity: Pre-synaptic activity
            post_activity: Post-synaptic activity
            weights: Weight matrix (optional)
            
        Returns:
            Dictionary with updated parameters
        """
        # Update activity history
        self._update_activity_history(post_activity)
        
        # Update firing rates
        self._update_firing_rates(post_activity)
        
        # Update adaptive thresholds
        self._update_thresholds(post_activity)
        
        # Update homeostatic factors
        self._update_homeostatic_factors()
        
        # Update correlation matrix and diversity score
        self._update_correlation_matrix()
        
        # Update weight scaling factors
        if weights is not None and self.weight_scaling_enabled:
            self._update_weight_scaling(weights)
        
        # Update inhibition kernel if using adaptive inhibition
        if self.inhibition_strategy == InhibitionStrategy.ADAPTIVE and self.update_count % 50 == 0:
            self._update_inhibition_kernel()
        
        # Increment update counter
        self.update_count += 1
        
        # Return updated parameters
        return {
            'homeostatic_factors': self.homeostatic_factors.copy(),
            'thresholds': self.thresholds.copy(),
            'diversity_score': self.diversity_score,
            'input_scaling_factors': self.input_scaling_factors.copy(),
            'output_scaling_factors': self.output_scaling_factors.copy()
        }
    
    def _update_activity_history(self, activity: xp.ndarray):
        """
        Update activity history with current activity.
        
        Args:
            activity: Current neural activity
        """
        # Store activity in circular buffer
        self.activity_history[self.activity_history_index] = activity.copy()
        self.activity_history_index = (self.activity_history_index + 1) % self.activity_history.shape[0]
        
        # Mark buffer as filled if we've gone through one cycle
        if self.activity_history_index == 0:
            self.activity_history_filled = True
    
    def _update_firing_rates(self, activity: xp.ndarray):
        """
        Update firing rate estimates.
        
        Args:
            activity: Current neural activity
        """
        # Exponential moving average of activity
        tau = 0.01 if self.update_count > 100 else 0.1  # faster initial adaptation
        self.firing_rates = (1 - tau) * self.firing_rates + tau * (activity > 0).astype(float)
    
    def _update_thresholds(self, activity: xp.ndarray):
        """
        Update adaptive thresholds based on recent activity.
        
        Args:
            activity: Current neural activity
        """
        # Adapt thresholds based on current activity
        # If neuron is too active, increase threshold
        # If neuron is not active enough, decrease threshold
        rate_diff = self.firing_rates - self.target_activity
        
        # Use sign of rate_diff to determine direction and magnitude for threshold change
        threshold_delta = self.adaptive_threshold_tau * rate_diff
        
        # Update thresholds
        self.thresholds += threshold_delta
        
        # Clip to min/max range
        self.thresholds = xp.clip(self.thresholds, self.minimum_threshold, self.maximum_threshold)
    
    def _update_homeostatic_factors(self):
        """Update homeostatic factors based on firing rates."""
        # Calculate multiplicative homeostatic factors
        # Neurons firing too much get decreased excitability
        # Neurons firing too little get increased excitability
        rate_ratio = self.target_activity / (self.firing_rates + 1e-10)
        
        # Clip ratio to prevent extreme adjustments
        rate_ratio = xp.clip(rate_ratio, 0.1, 10.0)
        
        # Gradually adjust homeostatic factors
        self.homeostatic_factors *= (1.0 - self.homeostatic_rate)
        self.homeostatic_factors += self.homeostatic_rate * rate_ratio
        
        # Normalize to keep overall activity level stable
        if xp.mean(self.homeostatic_factors) > 0:
            self.homeostatic_factors *= (1.0 / xp.mean(self.homeostatic_factors))
    
    def _update_correlation_matrix(self):
        """Update correlation matrix and diversity score."""
        # Only update if we have enough history
        if not self.activity_history_filled and self.activity_history_index < 10:
            return
        
        # Get active part of history buffer
        if self.activity_history_filled:
            history = self.activity_history
        else:
            history = self.activity_history[:self.activity_history_index]
        
        if len(history) < 2:
            return
        
        # Calculate correlation matrix
        # Normalize each activity vector
        norms = xp.sqrt(xp.sum(history**2, axis=1))
        normalized_history = history / (norms[:, xp.newaxis] + 1e-10)
        
        # Calculate correlation matrix
        corr_matrix = xp.zeros((self.representation_size, self.representation_size))
        for i in range(self.representation_size):
            for j in range(i, self.representation_size):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr = xp.mean(normalized_history[:, i] * normalized_history[:, j])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
        
        # Update correlation matrix with exponential decay
        alpha = 0.05 if self.update_count > 200 else 0.2  # faster initial adaptation
        if self.update_count == 0:
            self.correlation_matrix = corr_matrix
        else:
            self.correlation_matrix = (1 - alpha) * self.correlation_matrix + alpha * corr_matrix
        
        # Calculate diversity score
        # Lower absolute correlations mean higher diversity
        off_diagonal = self.correlation_matrix[~xp.eye(self.representation_size, dtype=bool)]
        mean_abs_corr = xp.mean(xp.abs(off_diagonal))
        
        # Convert to diversity score (1 = perfectly diverse, 0 = perfectly correlated)
        self.diversity_score = 1.0 - mean_abs_corr
    
    def _update_weight_scaling(self, weights: xp.ndarray):
        """
        Update weight scaling factors.
        
        Args:
            weights: Weight matrix
        """
        # Check if weights is 2D
        if weights.ndim != 2:
            return
        
        # Calculate row and column norms
        row_norms = xp.sqrt(xp.sum(weights**2, axis=1))
        col_norms = xp.sqrt(xp.sum(weights**2, axis=0))
        
        # Detect weights that are too strong
        strong_inputs = row_norms > self.normalization_threshold
        strong_outputs = col_norms > self.normalization_threshold
        
        # Adjust scaling factors for strong weights
        if xp.any(strong_inputs):
            # Scale down input weights that are too strong
            self.input_scaling_factors[strong_inputs] *= 0.95
        
        if xp.any(strong_outputs):
            # Scale down output weights that are too strong
            self.output_scaling_factors[strong_outputs] *= 0.95
        
        # Gradually restore scaling factors for weak weights
        self.input_scaling_factors[~strong_inputs] *= 1.01
        self.output_scaling_factors[~strong_outputs] *= 1.01
        
        # Clip scaling factors
        self.input_scaling_factors = xp.clip(self.input_scaling_factors, 0.1, 1.0)
        self.output_scaling_factors = xp.clip(self.output_scaling_factors, 0.1, 1.0)
    
    def _update_inhibition_kernel(self):
        """Update adaptive inhibition kernel based on correlation matrix."""
        if self.correlation_matrix is None:
            return
        
        # Use inverse of correlation matrix as basis for inhibition
        kernel = -self.correlation_matrix.copy()
        
        # Set diagonal to zero (no self-inhibition)
        xp.fill_diagonal(kernel, 0)
        
        # Scale kernel based on inhibition strength
        kernel *= self.inhibition_strength
        
        # Ensure kernel has zero mean
        kernel = kernel - xp.mean(kernel)
        
        # Update inhibition kernel
        if self.inhibition_kernel is None:
            self.inhibition_kernel = kernel
        else:
            # Gradually update
            self.inhibition_kernel = 0.9 * self.inhibition_kernel + 0.1 * kernel
    
    def normalize_weights(self, weights: xp.ndarray) -> xp.ndarray:
        """
        Apply weight normalization.
        
        Args:
            weights: Weight matrix
            
        Returns:
            Normalized weight matrix
        """
        # Apply input and output scaling factors
        if weights.ndim == 2:
            # Apply input scaling (rows)
            scaled_weights = weights * self.input_scaling_factors[:, xp.newaxis]
            # Apply output scaling (columns)
            scaled_weights = scaled_weights * self.output_scaling_factors[xp.newaxis, :]
            
            # Apply weight decay
            if self.weight_decay > 0:
                scaled_weights *= (1.0 - self.weight_decay)
            
            return scaled_weights
        else:
            # For non-2D weights, just return original
            return weights
    
    def adjust_diversity(
        self,
        weights: xp.ndarray,
        adjust_rate: float = None
    ) -> xp.ndarray:
        """
        Adjust weight matrix to promote representational diversity.
        
        Args:
            weights: Weight matrix
            adjust_rate: Rate of adjustment (if None, use self.diversity_rate)
            
        Returns:
            Adjusted weight matrix
        """
        if weights.ndim != 2:
            return weights
        
        if adjust_rate is None:
            adjust_rate = self.diversity_rate
        
        # Skip if diversity is already high enough
        if self.diversity_score >= self.diversity_target:
            return weights
        
        # Adjust weights to reduce correlations
        diversity_deficit = self.diversity_target - self.diversity_score
        
        # Calculate decorrelation adjustment
        adjustment = xp.zeros_like(weights)
        
        # If high positive correlation, reduce one of the weights
        high_corr_pairs = xp.where(self.correlation_matrix > 0.7)
        for i, j in zip(*high_corr_pairs):
            if i != j:
                # Reduce weights for one of the neurons
                j_weights = weights[j]
                adjustment[i] -= j_weights * adjust_rate * diversity_deficit
        
        # Apply adjustment
        adjusted_weights = weights + adjustment
        
        return adjusted_weights
    
    def enforce_sparse_activity(
        self,
        activity: xp.ndarray,
        sparsity: float = None
    ) -> xp.ndarray:
        """
        Enforce sparse activity by keeping only strongest activations.
        
        Args:
            activity: Activity pattern
            sparsity: Target sparsity level (if None, use self.target_activity)
            
        Returns:
            Sparse activity pattern
        """
        if sparsity is None:
            sparsity = self.target_activity
        
        # If activity is already sparse enough, return as is
        active_fraction = xp.mean(activity > 0)
        if active_fraction <= sparsity:
            return activity
        
        # Determine threshold to achieve target sparsity
        k = max(1, int(len(activity) * sparsity))
        threshold = xp.sort(activity)[-k]
        
        # Apply threshold
        sparse_activity = xp.zeros_like(activity)
        strong_indices = activity >= threshold
        sparse_activity[strong_indices] = activity[strong_indices]
        
        return sparse_activity
    
    def resize(self, new_size: int) -> Dict[str, Any]:
        """
        Resize the stability mechanisms.
        
        Args:
            new_size: New representation size
            
        Returns:
            Dictionary with resize information
        """
        if new_size == self.representation_size:
            return {"resized": False}
        
        old_size = self.representation_size
        
        # Resize thresholds
        if new_size > old_size:
            # Growing
            new_thresholds = xp.ones(new_size) * xp.mean(self.thresholds)
            new_thresholds[:old_size] = self.thresholds
            self.thresholds = new_thresholds
            
            # Resize homeostatic factors
            new_homeostatic = xp.ones(new_size)
            new_homeostatic[:old_size] = self.homeostatic_factors
            self.homeostatic_factors = new_homeostatic
            
            # Resize firing rates
            new_firing_rates = xp.zeros(new_size)
            new_firing_rates[:old_size] = self.firing_rates
            self.firing_rates = new_firing_rates
            
            # Resize scaling factors
            new_input_scaling = xp.ones(new_size)
            new_input_scaling[:old_size] = self.input_scaling_factors
            self.input_scaling_factors = new_input_scaling
            
            new_output_scaling = xp.ones(new_size)
            new_output_scaling[:old_size] = self.output_scaling_factors
            self.output_scaling_factors = new_output_scaling
            
            # Resize correlation matrix
            new_correlation = xp.identity(new_size)
            new_correlation[:old_size, :old_size] = self.correlation_matrix
            self.correlation_matrix = new_correlation
            
            # Reset activity history
            self.activity_history = xp.zeros((100, new_size))
            self.activity_history_index = 0
            self.activity_history_filled = False
            
            # Update k_winners if it was derived from representation size
            if self.k_winners == max(1, int(old_size * self.target_activity)):
                self.k_winners = max(1, int(new_size * self.target_activity))
            
            # Recreate inhibition kernel if needed
            if self.inhibition_strategy == InhibitionStrategy.ADAPTIVE:
                self.inhibition_kernel = self._create_mexican_hat_kernel(new_size, width_ratio=0.2)
        else:
            # Shrinking
            self.thresholds = self.thresholds[:new_size]
            self.homeostatic_factors = self.homeostatic_factors[:new_size]
            self.firing_rates = self.firing_rates[:new_size]
            self.input_scaling_factors = self.input_scaling_factors[:new_size]
            self.output_scaling_factors = self.output_scaling_factors[:new_size]
            self.correlation_matrix = self.correlation_matrix[:new_size, :new_size]
            
            # Reset activity history
            self.activity_history = xp.zeros((100, new_size))
            self.activity_history_index = 0
            self.activity_history_filled = False
            
            # Update k_winners if it was derived from representation size
            if self.k_winners == max(1, int(old_size * self.target_activity)):
                self.k_winners = max(1, int(new_size * self.target_activity))
            
            # Recreate inhibition kernel if needed
            if self.inhibition_strategy == InhibitionStrategy.ADAPTIVE:
                self.inhibition_kernel = self._create_mexican_hat_kernel(new_size, width_ratio=0.2)
        
        # Update representation size
        self.representation_size = new_size
        
        logger.info(f"Resized stability mechanisms from {old_size} to {new_size}")
        
        return {
            "resized": True,
            "old_size": old_size,
            "new_size": new_size
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stability mechanisms.
        
        Returns:
            Dictionary with stability statistics
        """
        return {
            'representation_size': self.representation_size,
            'diversity_score': float(self.diversity_score),
            'mean_threshold': float(xp.mean(self.thresholds)),
            'threshold_std': float(xp.std(self.thresholds)),
            'mean_homeostatic_factor': float(xp.mean(self.homeostatic_factors)),
            'homeostatic_range': float(xp.max(self.homeostatic_factors) - xp.min(self.homeostatic_factors)),
            'mean_firing_rate': float(xp.mean(self.firing_rates)),
            'active_neuron_fraction': float(xp.mean(self.firing_rates > 0.01)),
            'k_winners': self.k_winners,
            'inhibition_strategy': self.inhibition_strategy.value
        }
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the stability mechanisms state for saving.
        
        Returns:
            Dictionary with serialized state
        """
        data = {
            'representation_size': self.representation_size,
            'target_activity': self.target_activity,
            'homeostatic_rate': self.homeostatic_rate,
            'inhibition_strategy': self.inhibition_strategy.value,
            'inhibition_strength': self.inhibition_strength,
            'k_winners': self.k_winners,
            'normalization_threshold': self.normalization_threshold,
            'diversity_target': self.diversity_target,
            'diversity_rate': self.diversity_rate,
            'adaptive_threshold_tau': self.adaptive_threshold_tau,
            'minimum_threshold': self.minimum_threshold,
            'maximum_threshold': self.maximum_threshold,
            'weight_scaling_enabled': self.weight_scaling_enabled,
            'weight_decay': self.weight_decay,
            'thresholds': self.thresholds.tolist(),
            'homeostatic_factors': self.homeostatic_factors.tolist(),
            'diversity_score': self.diversity_score,
            'update_count': self.update_count
        }
        
        return data
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'StabilityMechanisms':
        """
        Create a stability mechanisms instance from serialized data.
        
        Args:
            data: Dictionary with serialized state
            
        Returns:
            StabilityMechanisms instance
        """
        instance = cls(
            representation_size=data['representation_size'],
            target_activity=data['target_activity'],
            homeostatic_rate=data['homeostatic_rate'],
            inhibition_strategy=data['inhibition_strategy'],
            inhibition_strength=data['inhibition_strength'],
            k_winners=data['k_winners'],
            normalization_threshold=data['normalization_threshold'],
            diversity_target=data['diversity_target'],
            diversity_rate=data['diversity_rate'],
            adaptive_threshold_tau=data['adaptive_threshold_tau'],
            minimum_threshold=data['minimum_threshold'],
            maximum_threshold=data['maximum_threshold'],
            weight_scaling_enabled=data['weight_scaling_enabled'],
            weight_decay=data['weight_decay']
        )
        
        # Restore state
        instance.thresholds = xp.array(data['thresholds'])
        instance.homeostatic_factors = xp.array(data['homeostatic_factors'])
        instance.diversity_score = data['diversity_score']
        instance.update_count = data['update_count']
        
        return instance 