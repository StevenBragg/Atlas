import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

from .backend import xp, to_cpu

logger = logging.getLogger(__name__)


class InhibitionStrategy(Enum):
    """Strategies for implementing lateral inhibition."""
    NONE = "none"
    WTA = "winner_take_all"  # Winner-take-all
    KWA = "k_winners_adaptive"  # K-winners adaptive
    SOFT = "soft_competition"  # Soft competition
    ADAPTIVE = "adaptive_threshold"  # Adaptive threshold


class HomeostaticPlasticity:
    """
    Implements homeostatic plasticity mechanisms to maintain stable neural activity.
    
    Homeostatic plasticity refers to the brain's ability to maintain stability
    while still allowing for learning and adaptation. This module implements several
    mechanisms:
    
    1. Intrinsic excitability regulation
    2. Synaptic scaling to normalize input strength
    3. Lateral inhibition for competition and sparse coding
    4. Target activity regulation to maintain desired activity levels
    5. Heterosynaptic competition to balance connection strengths
    
    These mechanisms work together to prevent runaway excitation, maintain
    representational stability, and encourage efficient sparse coding.
    """
    
    def __init__(
        self,
        layer_size: int,
        target_activity: float = 0.1,
        target_sparsity: float = 0.05,
        time_constant: float = 0.01,
        inhibition_strategy: str = "k_winners_adaptive",
        k_percent: float = 0.05,
        wta_strength: float = 1.0,
        adaptive_threshold_init: float = 0.5,
        threshold_adaptation_rate: float = 0.01,
        min_threshold: float = 0.01,
        max_threshold: float = 0.99,
        intrinsic_plasticity_rate: float = 0.01,
        synaptic_scaling_rate: float = 0.001,
        variance_target: float = 1.0,
        heterosynaptic_rate: float = 0.0005,
        activity_smoothing: float = 0.9,
        stability_weight: float = 0.5
    ):
        """
        Initialize homeostatic plasticity mechanisms.
        
        Args:
            layer_size: Size of the neural layer
            target_activity: Target average activity rate for neurons
            target_sparsity: Target proportion of active neurons
            time_constant: Time constant for adaptation
            inhibition_strategy: Lateral inhibition method to use
            k_percent: Percentage of neurons to keep active in k-winners
            wta_strength: Strength of winner-take-all inhibition
            adaptive_threshold_init: Initial value for adaptive thresholds
            threshold_adaptation_rate: Rate for threshold adaptation
            min_threshold: Minimum threshold value
            max_threshold: Maximum threshold value
            intrinsic_plasticity_rate: Learning rate for intrinsic plasticity
            synaptic_scaling_rate: Rate of synaptic scaling
            variance_target: Target variance for neuron activity
            heterosynaptic_rate: Rate of heterosynaptic competition
            activity_smoothing: Smoothing factor for activity history
            stability_weight: Weight given to stability mechanisms
        """
        self.layer_size = layer_size
        self.target_activity = target_activity
        self.target_sparsity = target_sparsity
        self.time_constant = time_constant
        self.inhibition_strategy = InhibitionStrategy(inhibition_strategy)
        self.k_percent = k_percent
        self.wta_strength = wta_strength
        self.threshold_adaptation_rate = threshold_adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.intrinsic_plasticity_rate = intrinsic_plasticity_rate
        self.synaptic_scaling_rate = synaptic_scaling_rate
        self.variance_target = variance_target
        self.heterosynaptic_rate = heterosynaptic_rate
        self.activity_smoothing = activity_smoothing
        self.stability_weight = stability_weight
        
        # Initialize adaptive thresholds for each neuron
        self.adaptive_thresholds = xp.ones(layer_size) * adaptive_threshold_init
        
        # Individual neuron excitability factors (intrinsic plasticity)
        self.excitability = xp.ones(layer_size)
        
        # Track average activity for each neuron
        self.avg_activity = xp.zeros(layer_size)
        
        # Track variance of activity for each neuron
        self.act_variance = xp.ones(layer_size)
        
        # Track active neuron indices for last update
        self.last_active = xp.zeros(layer_size, dtype=bool)
        
        # Activity history for stability analysis
        self.activity_history = []
        self.max_history_length = 100
        
        # Lateral inhibition matrix (for some inhibition strategies)
        if self.inhibition_strategy in [InhibitionStrategy.SOFT]:
            # Initialize lateral inhibition weights
            self.lateral_weights = xp.ones((layer_size, layer_size)) * -0.1
            xp.fill_diagonal(self.lateral_weights, 0)  # No self-inhibition
        else:
            self.lateral_weights = None
        
        # Track metrics
        self.stability_index = 1.0
        self.sparsity_index = 0.0
        self.avg_layer_activity = 0.0
        self.update_counter = 0
        
        logger.info(f"Initialized homeostatic plasticity with {inhibition_strategy} inhibition")
    
    def apply(self, activity: xp.ndarray) -> xp.ndarray:
        """
        Apply homeostatic plasticity mechanisms to neural activity.
        
        Args:
            activity: Pre-regulation neural activity
            
        Returns:
            Regulated neural activity
        """
        # Adjust layer size if needed
        if len(activity) != self.layer_size:
            self._resize(len(activity))
        
        # 1. Apply intrinsic excitability regulation
        regulated_activity = activity * self.excitability
        
        # 2. Apply lateral inhibition
        regulated_activity = self._apply_inhibition(regulated_activity)
        
        # 3. Update tracking of average activity (with smoothing)
        self.avg_activity = (
            self.activity_smoothing * self.avg_activity + 
            (1 - self.activity_smoothing) * regulated_activity
        )
        
        # Update activity variance
        self.act_variance = (
            self.activity_smoothing * self.act_variance + 
            (1 - self.activity_smoothing) * (regulated_activity - self.avg_activity)**2
        )
        
        # 4. Update active neuron tracking
        self.last_active = regulated_activity > self.adaptive_thresholds
        
        # 5. Store in activity history
        if len(self.activity_history) > self.max_history_length:
            self.activity_history.pop(0)
        self.activity_history.append(regulated_activity.copy())
        
        # 6. Update internal state and metrics
        self._update_internal_state(regulated_activity)
        
        return regulated_activity
    
    def update_thresholds(self) -> None:
        """
        Update adaptive thresholds based on average activity.
        """
        # Adjust thresholds to push neuron activity toward target
        delta = self.avg_activity - self.target_activity
        self.adaptive_thresholds += self.threshold_adaptation_rate * delta
        
        # Ensure thresholds stay within bounds
        self.adaptive_thresholds = xp.clip(
            self.adaptive_thresholds, 
            self.min_threshold, 
            self.max_threshold
        )
    
    def update_excitability(self) -> None:
        """
        Update intrinsic excitability based on average activity.
        """
        # Neurons with activity below target get more excitable
        # Neurons with activity above target get less excitable
        delta = self.target_activity - self.avg_activity
        self.excitability += self.intrinsic_plasticity_rate * delta
        
        # Ensure excitability stays positive
        self.excitability = xp.maximum(0.1, self.excitability)
        
        # Normalize excitability to prevent overall scaling issues
        if xp.mean(self.excitability) > 0:
            self.excitability = self.excitability / xp.mean(self.excitability)
    
    def apply_synaptic_scaling(self, weights: xp.ndarray) -> xp.ndarray:
        """
        Scale synaptic weights to maintain homeostasis.
        
        Args:
            weights: Weight matrix to scale
            
        Returns:
            Scaled weight matrix
        """
        if weights.ndim != 2:
            return weights  # Only support 2D weight matrices
            
        # Determine scaling axis (0 for input weights, 1 for output weights)
        if weights.shape[0] == self.layer_size:
            # These are output weights from this layer
            axis = 1
            activity_diff = self.avg_activity - self.target_activity
            scaling_factors = 1.0 - self.synaptic_scaling_rate * activity_diff
            
            # Apply scaling row-wise (each neuron's outputs)
            for i in range(weights.shape[0]):
                nonzero_mask = weights[i, :] != 0
                if xp.any(nonzero_mask):
                    weights[i, nonzero_mask] *= scaling_factors[i]
                    
        elif weights.shape[1] == self.layer_size:
            # These are input weights to this layer
            axis = 0
            activity_diff = self.avg_activity - self.target_activity
            scaling_factors = 1.0 - self.synaptic_scaling_rate * activity_diff
            
            # Apply scaling column-wise (each neuron's inputs)
            for i in range(weights.shape[1]):
                nonzero_mask = weights[:, i] != 0
                if xp.any(nonzero_mask):
                    weights[nonzero_mask, i] *= scaling_factors[i]
        
        # Limit weight changes
        weights = xp.clip(weights, -10.0, 10.0)
        
        return weights
    
    def apply_heterosynaptic_competition(self, weights: xp.ndarray) -> xp.ndarray:
        """
        Apply heterosynaptic competition to balance weights.
        
        Args:
            weights: Weight matrix to balance
            
        Returns:
            Balanced weight matrix
        """
        if weights.ndim != 2:
            return weights  # Only support 2D weight matrices
            
        # Normalize weights for each neuron
        if weights.shape[1] == self.layer_size:
            # Input weights
            for i in range(self.layer_size):
                w = weights[:, i]
                nonzero = w != 0
                if xp.sum(nonzero) > 1:  # Need at least 2 connections for competition
                    mean_weight = xp.mean(xp.abs(w[nonzero]))
                    # Shrink weights that are too strong, boost weights that are too weak
                    w[nonzero] -= self.heterosynaptic_rate * (xp.abs(w[nonzero]) - mean_weight) * xp.sign(w[nonzero])
                    weights[:, i] = w
        
        elif weights.shape[0] == self.layer_size:
            # Output weights
            for i in range(self.layer_size):
                w = weights[i, :]
                nonzero = w != 0
                if xp.sum(nonzero) > 1:
                    mean_weight = xp.mean(xp.abs(w[nonzero]))
                    w[nonzero] -= self.heterosynaptic_rate * (xp.abs(w[nonzero]) - mean_weight) * xp.sign(w[nonzero])
                    weights[i, :] = w
        
        return weights
    
    def _apply_inhibition(self, activity: xp.ndarray) -> xp.ndarray:
        """
        Apply lateral inhibition to activity pattern.
        
        Args:
            activity: Neural activity pattern
            
        Returns:
            Activity after inhibition
        """
        if self.inhibition_strategy == InhibitionStrategy.NONE:
            return activity
            
        elif self.inhibition_strategy == InhibitionStrategy.WTA:
            # Winner-take-all: Only keep the single highest activation
            regulated = xp.zeros_like(activity)
            if xp.max(activity) > 0:
                winner_idx = xp.argmax(activity)
                regulated[winner_idx] = activity[winner_idx]
            return regulated
            
        elif self.inhibition_strategy == InhibitionStrategy.KWA:
            # k-Winners-Adaptive: Keep top k% activations
            k = max(1, int(self.k_percent * len(activity)))
            
            if xp.sum(activity > 0) > 0:
                # Find top k indices
                indices = xp.argsort(activity)[-k:]
                
                # Create mask for top k
                mask = xp.zeros_like(activity, dtype=bool)
                mask[indices] = True
                
                # Keep only top k activations
                regulated = xp.zeros_like(activity)
                regulated[mask] = activity[mask]
                
                return regulated
            else:
                return activity
                
        elif self.inhibition_strategy == InhibitionStrategy.SOFT:
            # Soft competition: Apply inhibition matrix
            if self.lateral_weights is not None:
                # Apply lateral inhibition (simplified approximation)
                inhibition = xp.dot(activity, self.lateral_weights)
                return xp.maximum(0, activity + inhibition)
            return activity
            
        elif self.inhibition_strategy == InhibitionStrategy.ADAPTIVE:
            # Adaptive threshold: neurons compete with a threshold
            mask = activity > self.adaptive_thresholds
            regulated = xp.zeros_like(activity)
            regulated[mask] = activity[mask]
            
            # Update thresholds based on this activity
            self.update_thresholds()
            
            return regulated
        
        # Default case
        return activity
    
    def _update_internal_state(self, activity: xp.ndarray) -> None:
        """
        Update internal state and metrics.
        
        Args:
            activity: Current regulated activity
        """
        self.update_counter += 1
        
        # Calculate overall layer activity
        self.avg_layer_activity = xp.mean(activity)
        
        # Update sparsity index (proportion of active neurons)
        active_proportion = xp.mean(activity > 0)
        self.sparsity_index = 1.0 - (active_proportion / self.target_sparsity if self.target_sparsity > 0 else 0)
        self.sparsity_index = max(0, min(1, self.sparsity_index))
        
        # Periodically update homeostatic mechanisms
        if self.update_counter % 10 == 0:
            self.update_excitability()
            
        # Periodically calculate stability index
        if self.update_counter % 50 == 0 and len(self.activity_history) > 1:
            self._calculate_stability()
    
    def _calculate_stability(self) -> None:
        """
        Calculate stability index based on activity history.
        """
        if len(self.activity_history) < 2:
            self.stability_index = 1.0
            return
            
        # Calculate consistency of activations over time
        history_array = xp.array(self.activity_history[-10:])
        
        # Calculate variance across time for each neuron
        temporal_variance = xp.var(history_array, axis=0)
        
        # Calculate variance across neurons for each time point
        spatial_variance = xp.var(history_array, axis=1)
        
        # Stable networks have moderate temporal variance and high spatial variance
        temporal_var_score = xp.exp(-xp.mean(temporal_variance) / self.variance_target)
        spatial_var_score = 1.0 - xp.exp(-xp.mean(spatial_variance) / self.variance_target)
        
        # Combine into stability index
        self.stability_index = (temporal_var_score + spatial_var_score) / 2.0
    
    def _resize(self, new_size: int) -> None:
        """
        Resize homeostatic components when layer size changes.
        
        Args:
            new_size: New layer size
        """
        if new_size == self.layer_size:
            return
            
        logger.debug(f"Resizing homeostatic mechanisms from {self.layer_size} to {new_size}")
        
        # Handle size increase
        if new_size > self.layer_size:
            # Pad arrays with default values
            self.adaptive_thresholds = xp.pad(
                self.adaptive_thresholds, 
                (0, new_size - self.layer_size),
                constant_values=0.5
            )
            self.excitability = xp.pad(
                self.excitability, 
                (0, new_size - self.layer_size),
                constant_values=1.0
            )
            self.avg_activity = xp.pad(
                self.avg_activity, 
                (0, new_size - self.layer_size)
            )
            self.act_variance = xp.pad(
                self.act_variance, 
                (0, new_size - self.layer_size),
                constant_values=1.0
            )
            self.last_active = xp.pad(
                self.last_active, 
                (0, new_size - self.layer_size)
            )
            
            # Resize lateral inhibition matrix if used
            if self.lateral_weights is not None:
                new_lateral = xp.ones((new_size, new_size)) * -0.1
                xp.fill_diagonal(new_lateral, 0)
                new_lateral[:self.layer_size, :self.layer_size] = self.lateral_weights
                self.lateral_weights = new_lateral
                
        # Handle size decrease
        else:
            # Truncate arrays
            self.adaptive_thresholds = self.adaptive_thresholds[:new_size]
            self.excitability = self.excitability[:new_size]
            self.avg_activity = self.avg_activity[:new_size]
            self.act_variance = self.act_variance[:new_size]
            self.last_active = self.last_active[:new_size]
            
            # Resize lateral inhibition matrix if used
            if self.lateral_weights is not None:
                self.lateral_weights = self.lateral_weights[:new_size, :new_size]
        
        # Update layer size
        self.layer_size = new_size
    
    def set_target_activity(self, new_target: float) -> None:
        """
        Set a new target activity level.
        
        Args:
            new_target: New target average activity
        """
        self.target_activity = new_target
        logger.debug(f"Set target activity to {new_target:.4f}")
    
    def set_inhibition_strategy(self, strategy: str) -> None:
        """
        Change the inhibition strategy.
        
        Args:
            strategy: New inhibition strategy name
        """
        try:
            self.inhibition_strategy = InhibitionStrategy(strategy)
            
            # Initialize lateral weights if switching to soft competition
            if self.inhibition_strategy == InhibitionStrategy.SOFT and self.lateral_weights is None:
                self.lateral_weights = xp.ones((self.layer_size, self.layer_size)) * -0.1
                xp.fill_diagonal(self.lateral_weights, 0)
                
            logger.info(f"Changed inhibition strategy to {strategy}")
        except ValueError:
            logger.error(f"Invalid inhibition strategy: {strategy}")
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """
        Get homeostatic stability metrics.
        
        Returns:
            Dictionary of stability metrics
        """
        return {
            "stability_index": float(self.stability_index),
            "sparsity_index": float(self.sparsity_index),
            "avg_activity": float(self.avg_layer_activity),
            "target_activity": float(self.target_activity),
            "mean_excitability": float(xp.mean(self.excitability)),
            "mean_threshold": float(xp.mean(self.adaptive_thresholds)),
            "activity_variance": float(xp.mean(self.act_variance))
        }
        
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the homeostatic state.
        
        Returns:
            Dictionary with serialized state
        """
        serialized = {
            "layer_size": self.layer_size,
            "target_activity": self.target_activity,
            "target_sparsity": self.target_sparsity,
            "time_constant": self.time_constant,
            "inhibition_strategy": self.inhibition_strategy.value,
            "k_percent": self.k_percent,
            "wta_strength": self.wta_strength,
            "threshold_adaptation_rate": self.threshold_adaptation_rate,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "intrinsic_plasticity_rate": self.intrinsic_plasticity_rate,
            "synaptic_scaling_rate": self.synaptic_scaling_rate,
            "variance_target": self.variance_target,
            "heterosynaptic_rate": self.heterosynaptic_rate,
            "activity_smoothing": self.activity_smoothing,
            "stability_weight": self.stability_weight,
            "adaptive_thresholds": self.adaptive_thresholds.tolist(),
            "excitability": self.excitability.tolist(),
            "avg_activity": self.avg_activity.tolist(),
            "act_variance": self.act_variance.tolist(),
            "last_active": self.last_active.tolist(),
            "stability_index": self.stability_index,
            "sparsity_index": self.sparsity_index,
            "avg_layer_activity": self.avg_layer_activity,
            "update_counter": self.update_counter
        }
        
        # Serialize lateral weights if they exist
        if self.lateral_weights is not None:
            serialized["lateral_weights"] = self.lateral_weights.tolist()
        
        return serialized
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'HomeostaticPlasticity':
        """
        Create a homeostatic plasticity instance from serialized data.
        
        Args:
            data: Dictionary with serialized data
            
        Returns:
            HomeostaticPlasticity instance
        """
        instance = cls(
            layer_size=data["layer_size"],
            target_activity=data["target_activity"],
            target_sparsity=data["target_sparsity"],
            time_constant=data["time_constant"],
            inhibition_strategy=data["inhibition_strategy"],
            k_percent=data["k_percent"],
            wta_strength=data["wta_strength"],
            threshold_adaptation_rate=data["threshold_adaptation_rate"],
            min_threshold=data["min_threshold"],
            max_threshold=data["max_threshold"],
            intrinsic_plasticity_rate=data["intrinsic_plasticity_rate"],
            synaptic_scaling_rate=data["synaptic_scaling_rate"],
            variance_target=data["variance_target"],
            heterosynaptic_rate=data["heterosynaptic_rate"],
            activity_smoothing=data["activity_smoothing"],
            stability_weight=data["stability_weight"]
        )
        
        # Restore state
        instance.adaptive_thresholds = xp.array(data["adaptive_thresholds"])
        instance.excitability = xp.array(data["excitability"])
        instance.avg_activity = xp.array(data["avg_activity"])
        instance.act_variance = xp.array(data["act_variance"])
        instance.last_active = xp.array(data["last_active"])
        instance.stability_index = data["stability_index"]
        instance.sparsity_index = data["sparsity_index"]
        instance.avg_layer_activity = data["avg_layer_activity"]
        instance.update_counter = data["update_counter"]
        
        # Restore lateral weights if they exist in the data
        if "lateral_weights" in data:
            instance.lateral_weights = xp.array(data["lateral_weights"])
        
        return instance 