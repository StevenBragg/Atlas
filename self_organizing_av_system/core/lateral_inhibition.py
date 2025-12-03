import logging
from typing import List, Dict, Tuple, Optional, Any, Union, Callable

from .backend import xp, to_cpu
from .neuron import Neuron

logger = logging.getLogger(__name__)


class LateralInhibition:
    """
    Implements lateral inhibition mechanisms for competitive learning and representational stability.
    
    Lateral inhibition is a biological mechanism where active neurons suppress the activity of their
    neighbors, creating a competition that leads to sparse, efficient coding. This implementation
    supports different inhibition strategies including:
    
    1. Winner-Take-All (WTA): Only the most active neuron(s) can fire
    2. Local inhibition: Neurons compete with their nearest neighbors
    3. Global inhibition: All neurons compete with each other
    4. Soft competition: Activity is scaled by relative activation
    
    These mechanisms help stabilize learning and prevent representational collapse by
    enforcing sparsity and competition between neurons.
    """
    
    def __init__(
        self,
        inhibition_strategy: str = "soft_wta",
        inhibition_radius: float = 0.0,  # 0 means global inhibition
        inhibition_strength: float = 0.5,
        k_winners: int = 1,  # Number of winners for k-WTA
        sparsity_target: float = 0.1,  # Target activation sparsity (0-1)
        inhibition_decay: float = 0.99,  # Decay factor for inhibition
        excitation_strength: float = 0.1,  # Strength of lateral excitation
        adaptation_rate: float = 0.01,  # Rate of adaptive threshold change
        similarity_metric: str = "euclidean"  # For calculating neural similarity
    ):
        """
        Initialize the lateral inhibition module.
        
        Args:
            inhibition_strategy: Type of inhibition ("wta", "soft_wta", "local", "global")
            inhibition_radius: Radius of local inhibition (as fraction of weight space)
            inhibition_strength: Strength of inhibitory connections
            k_winners: Number of winners in k-WTA
            sparsity_target: Target activation sparsity (fraction of active neurons)
            inhibition_decay: Decay factor for inhibition
            excitation_strength: Strength of lateral excitation for similar neurons
            adaptation_rate: Rate at which neuron thresholds adapt
            similarity_metric: Metric for calculating neuron similarity
        """
        self.inhibition_strategy = inhibition_strategy
        self.inhibition_radius = inhibition_radius
        self.inhibition_strength = inhibition_strength
        self.k_winners = k_winners
        self.sparsity_target = sparsity_target
        self.inhibition_decay = inhibition_decay
        self.excitation_strength = excitation_strength
        self.adaptation_rate = adaptation_rate
        self.similarity_metric = similarity_metric
        
        # Internal state
        self.inhibition_matrix = None  # Will be initialized on first apply
        self.inhibition_state = None  # Current inhibition state
        self.prev_winners = None  # Previous winning neurons
        self.neuron_excitation = None  # Current excitation state
        
        # Adaptive parameters
        self.adaptive_threshold = None
        self.activity_history = []  # Track recent activity levels
        
        # Monitoring/stats
        self.active_count_history = []
        self.winner_history = []
        self.stability_metrics = {
            "redundancy": [],
            "selectivity": [],
            "activation_variance": []
        }
    
    def initialize(self, num_neurons: int, weight_vectors: List[xp.ndarray]) -> None:
        """
        Initialize or update internal structures based on neuron count and weights.
        
        Args:
            num_neurons: Number of neurons in the layer
            weight_vectors: Weight vectors of neurons (for similarity calculation)
        """
        # Initialize inhibition matrix if needed
        if (self.inhibition_matrix is None or 
            self.inhibition_matrix.shape[0] != num_neurons):
            
            # Create new matrix
            self.inhibition_matrix = xp.zeros((num_neurons, num_neurons))
            
            # Compute similarity/distance between neurons
            for i in range(num_neurons):
                for j in range(num_neurons):
                    if i == j:
                        # No self-inhibition by default
                        self.inhibition_matrix[i, j] = 0
                    else:
                        # Calculate similarity or distance between neurons
                        w_i = weight_vectors[i]
                        w_j = weight_vectors[j]
                        
                        if self.similarity_metric == "cosine":
                            # Cosine similarity
                            similarity = xp.dot(w_i, w_j) / (xp.linalg.norm(w_i) * xp.linalg.norm(w_j) + 1e-10)
                            # Convert similarity to inhibition strength (more similar = more inhibition)
                            self.inhibition_matrix[i, j] = similarity * self.inhibition_strength
                        elif self.similarity_metric == "euclidean":
                            # Euclidean distance
                            distance = xp.linalg.norm(w_i - w_j)
                            # Convert distance to inhibition (closer = more inhibition)
                            # Using a gaussian falloff
                            if self.inhibition_radius > 0:
                                # Local inhibition based on radius
                                radius = self.inhibition_radius * xp.sqrt(len(w_i))
                                self.inhibition_matrix[i, j] = (
                                    self.inhibition_strength * 
                                    xp.exp(-distance**2 / (2 * radius**2))
                                )
                            else:
                                # Global inhibition with distance modulation
                                self.inhibition_matrix[i, j] = (
                                    self.inhibition_strength * 
                                    (1.0 - xp.tanh(distance / 2.0))
                                )
            
            # Initialize inhibition state and adaptive threshold
            self.inhibition_state = xp.zeros(num_neurons)
            self.neuron_excitation = xp.zeros(num_neurons)
            
            if self.adaptive_threshold is None or len(self.adaptive_threshold) != num_neurons:
                self.adaptive_threshold = xp.ones(num_neurons) * 0.5
            
            # Truncate or extend previous winners list if needed
            if self.prev_winners is not None:
                if len(self.prev_winners) > num_neurons:
                    self.prev_winners = self.prev_winners[:num_neurons]
                elif len(self.prev_winners) < num_neurons:
                    self.prev_winners = xp.concatenate([
                        self.prev_winners,
                        xp.zeros(num_neurons - len(self.prev_winners))
                    ])
            else:
                self.prev_winners = xp.zeros(num_neurons)
    
    def apply(
        self, 
        activations: xp.ndarray, 
        weight_vectors: Optional[List[xp.ndarray]] = None,
        learning_enabled: bool = True
    ) -> xp.ndarray:
        """
        Apply lateral inhibition to neuron activations.
        
        Args:
            activations: Raw neuron activations
            weight_vectors: Optional updated weight vectors (for similarity recalculation)
            learning_enabled: Whether learning is enabled (affects adaptation)
            
        Returns:
            Inhibited activations
        """
        num_neurons = len(activations)
        
        # Initialize or update if needed
        if weight_vectors is not None:
            self.initialize(num_neurons, weight_vectors)
        elif (self.inhibition_matrix is None or 
              self.inhibition_matrix.shape[0] != num_neurons):
            # Can't calculate proper inhibition without weights, use default
            fake_weights = [xp.ones(10) for _ in range(num_neurons)]
            self.initialize(num_neurons, fake_weights)
        
        # Apply specific inhibition strategy
        if self.inhibition_strategy == "wta":
            inhibited_act = self._apply_wta(activations)
        elif self.inhibition_strategy == "soft_wta":
            inhibited_act = self._apply_soft_wta(activations)
        elif self.inhibition_strategy == "local":
            inhibited_act = self._apply_local_inhibition(activations)
        elif self.inhibition_strategy == "global":
            inhibited_act = self._apply_global_inhibition(activations)
        else:
            # Default to soft WTA
            inhibited_act = self._apply_soft_wta(activations)
        
        # Update the inhibition state (exponential decay)
        self.inhibition_state = self.inhibition_state * self.inhibition_decay
        
        # Record active neuron count
        active_count = xp.sum(inhibited_act > 0)
        self.active_count_history.append(active_count)
        if len(self.active_count_history) > 1000:
            self.active_count_history.pop(0)
        
        # Adapt thresholds if learning is enabled
        if learning_enabled:
            self._adapt_thresholds(activations, inhibited_act)
        
        # Record winners for temporal stability
        winners = (inhibited_act > 0).astype(float)
        self.winner_history.append(winners)
        if len(self.winner_history) > 100:
            self.winner_history.pop(0)
        
        # Update temporal stability metrics
        if len(self.winner_history) > 10:
            self._update_stability_metrics()
        
        # Update previous winners
        self.prev_winners = winners
        
        return inhibited_act
    
    def _apply_wta(self, activations: xp.ndarray) -> xp.ndarray:
        """
        Apply strict Winner-Take-All inhibition.
        
        Args:
            activations: Raw neuron activations
            
        Returns:
            Inhibited activations (only k winners active)
        """
        inhibited = xp.zeros_like(activations)
        
        if xp.max(activations) <= 0:
            # No active neurons
            return inhibited
        
        # Find top-k neurons
        k = min(self.k_winners, len(activations))
        top_k_indices = xp.argsort(activations)[-k:]
        
        # Only winners remain active
        inhibited[top_k_indices] = activations[top_k_indices]
        
        # Apply inhibition to these winners
        for i in top_k_indices:
            # Add to inhibition state for future timesteps
            self.inhibition_state += self.inhibition_matrix[i]
        
        return inhibited
    
    def _apply_soft_wta(self, activations: xp.ndarray) -> xp.ndarray:
        """
        Apply soft Winner-Take-All inhibition, allowing graded activations.
        
        Args:
            activations: Raw neuron activations
            
        Returns:
            Soft-inhibited activations
        """
        if xp.max(activations) <= 0:
            # No active neurons
            return xp.zeros_like(activations)
        
        # Start with original activations
        inhibited = activations.copy()
        
        # Apply current inhibition state
        inhibited = inhibited - self.inhibition_state
        
        # Apply adaptive thresholds
        if self.adaptive_threshold is not None:
            inhibited = inhibited - self.adaptive_threshold
        
        # Rectify negative values
        inhibited = xp.maximum(0, inhibited)
        
        # Calculate total inhibition to apply
        total_inhibition = xp.zeros_like(activations)
        for i in range(len(activations)):
            if inhibited[i] > 0:
                # Each neuron inhibits others proportionally to its activation
                inhibition = self.inhibition_matrix[i] * inhibited[i]
                total_inhibition += inhibition
        
        # Apply lateral excitation based on previous winners
        if xp.any(self.prev_winners > 0):
            # Each previously active neuron provides excitation
            excitation = self.prev_winners * self.excitation_strength
            inhibited += excitation
        
        # Apply the calculated inhibition
        inhibited = inhibited - total_inhibition
        
        # Ensure no negative values
        inhibited = xp.maximum(0, inhibited)
        
        # Update inhibition state for next timestep
        active_mask = inhibited > 0
        if xp.any(active_mask):
            self.inhibition_state += xp.sum(
                [self.inhibition_matrix[i] * inhibited[i] for i in xp.where(active_mask)[0]], 
                axis=0
            )
        
        return inhibited
    
    def _apply_local_inhibition(self, activations: xp.ndarray) -> xp.ndarray:
        """
        Apply local inhibition where neurons inhibit only their neighbors.
        
        Args:
            activations: Raw neuron activations
            
        Returns:
            Locally inhibited activations
        """
        # Start with original activations
        inhibited = activations.copy()
        
        # Apply current inhibition state
        inhibited = inhibited - self.inhibition_state
        
        # Apply adaptive thresholds
        if self.adaptive_threshold is not None:
            inhibited = inhibited - self.adaptive_threshold
        
        # Rectify negative values
        inhibited = xp.maximum(0, inhibited)
        
        # Each active neuron inhibits its neighbors based on inhibition matrix
        for i in range(len(activations)):
            if inhibited[i] > 0:
                # Inhibit neighbors based on their similarity/distance
                for j in range(len(activations)):
                    if i != j:
                        inhibited[j] -= inhibited[i] * self.inhibition_matrix[i, j]
        
        # Ensure no negative values
        inhibited = xp.maximum(0, inhibited)
        
        # Update inhibition state for next timestep
        active_mask = inhibited > 0
        if xp.any(active_mask):
            for i in xp.where(active_mask)[0]:
                self.inhibition_state += self.inhibition_matrix[i] * inhibited[i]
        
        return inhibited
    
    def _apply_global_inhibition(self, activations: xp.ndarray) -> xp.ndarray:
        """
        Apply global inhibition with sparsity target.
        
        Args:
            activations: Raw neuron activations
            
        Returns:
            Globally inhibited activations
        """
        # Start with original activations
        inhibited = activations.copy()
        
        # Apply current inhibition state
        inhibited = inhibited - self.inhibition_state
        
        # Apply adaptive thresholds
        if self.adaptive_threshold is not None:
            inhibited = inhibited - self.adaptive_threshold
        
        # Rectify
        inhibited = xp.maximum(0, inhibited)
        
        # Determine threshold to achieve target sparsity
        if xp.sum(inhibited > 0) > 0:
            target_active = max(1, int(self.sparsity_target * len(activations)))
            
            # Sort activations
            sorted_act = xp.sort(inhibited)
            
            # Find threshold that gives desired sparsity
            if len(sorted_act) > target_active:
                threshold = sorted_act[-(target_active+1)]
            else:
                threshold = 0
            
            # Apply threshold
            inhibited[inhibited <= threshold] = 0
            
            # Update inhibition state
            active_mask = inhibited > 0
            if xp.any(active_mask):
                # Global inhibition depends on total activity
                total_activity = xp.sum(inhibited)
                self.inhibition_state += self.inhibition_strength * (total_activity / len(activations))
        
        return inhibited
    
    def _adapt_thresholds(self, raw_activations: xp.ndarray, inhibited_activations: xp.ndarray) -> None:
        """
        Adapt neuron thresholds based on activity to maintain target sparsity.
        
        Args:
            raw_activations: Raw pre-inhibition activations
            inhibited_activations: Post-inhibition activations
        """
        if self.adaptive_threshold is None or len(self.adaptive_threshold) != len(raw_activations):
            self.adaptive_threshold = xp.ones(len(raw_activations)) * 0.5
        
        # Calculate current activity level
        activity_level = xp.mean(inhibited_activations > 0)
        self.activity_history.append(activity_level)
        if len(self.activity_history) > 100:
            self.activity_history.pop(0)
        
        # Calculate average recent activity
        avg_activity = xp.mean(self.activity_history) if self.activity_history else activity_level
        
        # Adjust thresholds based on difference from target sparsity
        activity_error = avg_activity - self.sparsity_target
        
        # Global threshold adjustment
        global_adjustment = activity_error * self.adaptation_rate
        
        # Individual neuron adjustments
        for i in range(len(raw_activations)):
            if inhibited_activations[i] > 0:
                # Active neurons: increase threshold if overall activity is too high
                self.adaptive_threshold[i] += global_adjustment
            else:
                # Inactive neurons: decrease threshold if activity is too low
                # But only if they had some raw activation
                if raw_activations[i] > 0:
                    self.adaptive_threshold[i] -= global_adjustment * 0.1
            
            # Ensure thresholds stay in reasonable range
            self.adaptive_threshold[i] = xp.clip(self.adaptive_threshold[i], 0.0, 1.0)
    
    def _update_stability_metrics(self) -> None:
        """Update metrics that track representational stability."""
        if len(self.winner_history) < 10:
            return
        
        # Calculate redundancy (how many neurons respond to multiple inputs)
        # Higher is more redundant
        recent_winners = xp.array(self.winner_history[-10:])
        winner_counts = xp.sum(recent_winners, axis=0)
        redundancy = xp.mean(winner_counts > 1) if len(winner_counts) > 0 else 0
        self.stability_metrics["redundancy"].append(redundancy)
        
        # Calculate selectivity (how selective neurons are to specific inputs)
        # Higher means more selective
        selectivity = 1.0 - xp.mean(winner_counts / 10) if len(winner_counts) > 0 else 0
        self.stability_metrics["selectivity"].append(selectivity)
        
        # Calculate activation variance (stability of individual neuron activations)
        # Lower means more stable
        if recent_winners.shape[1] > 0:
            variances = xp.var(recent_winners, axis=0)
            avg_variance = xp.mean(variances)
            self.stability_metrics["activation_variance"].append(avg_variance)
        
        # Keep metrics history limited
        for key in self.stability_metrics:
            if len(self.stability_metrics[key]) > 1000:
                self.stability_metrics[key] = self.stability_metrics[key][-1000:]
    
    def reset(self) -> None:
        """Reset inhibition state and history."""
        if self.inhibition_state is not None:
            self.inhibition_state = xp.zeros_like(self.inhibition_state)
        
        if self.prev_winners is not None:
            self.prev_winners = xp.zeros_like(self.prev_winners)
        
        self.activity_history = []
        self.active_count_history = []
        self.winner_history = []
        
        for key in self.stability_metrics:
            self.stability_metrics[key] = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about lateral inhibition.
        
        Returns:
            Dictionary with inhibition statistics
        """
        # Calculate average activity level
        avg_activity = (
            xp.mean(self.active_count_history) / len(self.inhibition_state) 
            if self.inhibition_state is not None and len(self.active_count_history) > 0
            else 0
        )
        
        # Calculate stability metrics
        stability = {}
        for key in self.stability_metrics:
            if len(self.stability_metrics[key]) > 0:
                stability[key] = xp.mean(self.stability_metrics[key][-100:])
            else:
                stability[key] = 0
        
        return {
            "average_activity": avg_activity,
            "stability_metrics": stability,
            "adaptive_thresholds": self.adaptive_threshold.tolist() if self.adaptive_threshold is not None else None,
            "inhibition_strength": self.inhibition_strength
        }
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the lateral inhibition state.
        
        Returns:
            Dictionary with serialized state
        """
        return {
            "inhibition_strategy": self.inhibition_strategy,
            "inhibition_radius": self.inhibition_radius,
            "inhibition_strength": self.inhibition_strength,
            "k_winners": self.k_winners,
            "sparsity_target": self.sparsity_target,
            "inhibition_decay": self.inhibition_decay,
            "excitation_strength": self.excitation_strength,
            "adaptation_rate": self.adaptation_rate,
            "similarity_metric": self.similarity_metric,
            "adaptive_threshold": self.adaptive_threshold.tolist() if self.adaptive_threshold is not None else None,
            "inhibition_state": self.inhibition_state.tolist() if self.inhibition_state is not None else None,
            "statistics": self.get_statistics()
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'LateralInhibition':
        """
        Create a lateral inhibition module from serialized data.
        
        Args:
            data: Dictionary with serialized data
            
        Returns:
            LateralInhibition instance
        """
        instance = cls(
            inhibition_strategy=data.get("inhibition_strategy", "soft_wta"),
            inhibition_radius=data.get("inhibition_radius", 0.0),
            inhibition_strength=data.get("inhibition_strength", 0.5),
            k_winners=data.get("k_winners", 1),
            sparsity_target=data.get("sparsity_target", 0.1),
            inhibition_decay=data.get("inhibition_decay", 0.99),
            excitation_strength=data.get("excitation_strength", 0.1),
            adaptation_rate=data.get("adaptation_rate", 0.01),
            similarity_metric=data.get("similarity_metric", "euclidean")
        )
        
        # Restore internal state
        if "adaptive_threshold" in data and data["adaptive_threshold"] is not None:
            instance.adaptive_threshold = xp.array(data["adaptive_threshold"])
        
        if "inhibition_state" in data and data["inhibition_state"] is not None:
            instance.inhibition_state = xp.array(data["inhibition_state"])
        
        return instance 