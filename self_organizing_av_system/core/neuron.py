from typing import List, Dict, Optional, Tuple, Union, Any

from .backend import xp, to_cpu


class Neuron:
    """
    A neuron with local learning rules, homeostatic plasticity, and 
    spike-timing dependent adaptation.
    
    This class implements a self-organizing neuron model with local learning
    rules and homeostatic mechanisms to maintain stable activity.
    """
    
    def __init__(
        self,
        input_size: int,
        neuron_id: int = 0,
        learning_rate: float = 0.01,
        threshold: float = 0.5,
        target_activation: float = 0.1,
        homeostatic_factor: float = 0.01,
        adaptation_time_constant: float = 0.01,
        refactory_period: int = 3,
        hebbian_learning_window: int = 5,
        eligibility_trace_decay: float = 0.95,
        initial_weights: Optional['xp.ndarray'] = None
    ):
        """
        Initialize neuron with given parameters.
        
        Args:
            input_size: Dimension of the input vector
            neuron_id: Unique identifier for this neuron
            learning_rate: How quickly weights adapt to input
            threshold: Minimum activation value to fire
            target_activation: Target frequency of activation for homeostasis
            homeostatic_factor: Strength of homeostatic adjustment
            adaptation_time_constant: Time constant for adaptation
            refactory_period: Minimum time steps between activations
            hebbian_learning_window: Time window for STDP learning
            eligibility_trace_decay: Decay rate for eligibility traces
            initial_weights: Optional initial weight values
        """
        self.id = neuron_id
        self.input_size = input_size
        self.learning_rate = learning_rate
        
        # Activation parameters
        self.threshold = threshold
        self.target_activation = target_activation
        self.homeostatic_factor = homeostatic_factor
        
        # Temporal dynamics parameters
        self.adaptation_time_constant = adaptation_time_constant
        self.refractory_period = refactory_period
        self.hebbian_learning_window = hebbian_learning_window
        self.eligibility_trace_decay = eligibility_trace_decay
        
        # Initialize weights with small random values if not provided
        if initial_weights is not None:
            self.weights = self._normalize_weights(initial_weights)
        else:
            raw_weights = xp.random.randn(input_size)
            self.weights = self._normalize_weights(raw_weights)
        
        # State variables
        self.activation = 0.0
        self.is_winner = False
        self.last_activation_time = -1000  # For refactory period
        self.current_time_step = 0  # Track time for dynamics
        
        # Temporal dynamics and learning
        self.activation_history = []  # Recent activation values
        self.recent_mean_activation = 0.0  # For homeostasis
        self.eligibility_traces = xp.zeros(input_size)  # For STDP
        self.input_history = []  # Recent input vectors (for STDP)
        self.activation_times = []  # When this neuron fired
        
        # Adaptation state (adaptation increases with activity, reducing sensitivity)
        self.adaptation = 0.0
        
        # For STDP trace-based learning
        self.pre_synaptic_traces = xp.zeros(input_size)
        self.post_synaptic_trace = 0.0
        
        # For BCM-like learning (activity-dependent thresholding)
        self.bcm_sliding_threshold = threshold
        
        # For debugging and analysis
        self.debug_info = {}
        
    def activate(self, inputs: xp.ndarray, time_step: Optional[int] = None) -> float:
        """
        Compute activation for given inputs.
        
        Args:
            inputs: Input activation vector
            time_step: Current simulation time step
            
        Returns:
            Neuron activation value
        """
        # Update time tracking
        if time_step is not None:
            self.current_time_step = time_step
            
        # Calculate activation (weighted sum of inputs)
        raw_activation = xp.dot(self.weights, inputs)
        
        # Apply threshold and adaptation
        if raw_activation > (self.threshold + self.adaptation):
            # Check refractory period
            if self.is_in_refractory_period():
                self.activation = 0.0
            else:
                # Neuron fires
                self.activation = raw_activation - self.threshold
                self.activation_times.append(self.current_time_step)
                self.last_activation_time = self.current_time_step
                
                # Increase adaptation (activity-dependent fatigue)
                self.adaptation += 0.1
        else:
            # No activation
            self.activation = 0.0
            
        # Update adaptation (decay over time)
        self._update_adaptation()
        
        # Update dynamics traces
        self._update_traces(inputs)
        
        # Update activation history
        self.activation_history.append(self.activation)
        if len(self.activation_history) > 100:
            self.activation_history.pop(0)
            
        # Update input history (for STDP)
        self.input_history.append(inputs.copy())
        if len(self.input_history) > self.hebbian_learning_window:
            self.input_history.pop(0)
            
        # Update recent activation mean
        if len(self.activation_history) > 0:
            self.recent_mean_activation = xp.mean(
                [a > 0 for a in self.activation_history]
            )
            
        return self.activation
    
    def _update_adaptation(self) -> None:
        """
        Update adaptation state (activity-dependent fatigue).
        """
        # Adaptation decays toward zero over time
        self.adaptation *= (1.0 - self.adaptation_time_constant)
        
    def _update_traces(self, inputs: xp.ndarray) -> None:
        """
        Update traces for STDP learning.
        
        Args:
            inputs: Current input vector
        """
        # Decay existing traces
        self.pre_synaptic_traces *= self.eligibility_trace_decay
        self.post_synaptic_trace *= self.eligibility_trace_decay
        
        # Update pre-synaptic traces (increases when input is active)
        self.pre_synaptic_traces += inputs
        
        # Update post-synaptic trace (increases when neuron fires)
        if self.activation > 0:
            self.post_synaptic_trace += 1.0
            
        # Update BCM sliding threshold
        target_threshold = 1.0 + 10.0 * self.recent_mean_activation
        self.bcm_sliding_threshold += 0.001 * (target_threshold - self.bcm_sliding_threshold)
    
    def is_in_refractory_period(self) -> bool:
        """
        Check if the neuron is currently in its refractory period.
        
        Returns:
            True if in refractory period, False otherwise
        """
        time_since_last_activation = self.current_time_step - self.last_activation_time
        return time_since_last_activation < self.refractory_period
    
    def update_weights_hebbian(self, inputs: xp.ndarray, use_oja: bool = True) -> None:
        """
        Update weights using Hebbian learning with optional Oja normalization.
        
        Args:
            inputs: Input activation vector
            use_oja: Whether to use Oja's rule for normalization
        """
        # Skip update if neuron didn't activate
        if self.activation <= 0:
            return
            
        # Calculate Hebbian weight change
        if use_oja:
            # Oja's rule: Hebbian update with normalization term
            # Δw = η * (y * x - y^2 * w)
            # This automatically normalizes weights over time
            delta_w = self.learning_rate * (
                self.activation * inputs - 
                self.activation * self.activation * self.weights
            )
        else:
            # Basic Hebbian: Δw = η * y * x
            delta_w = self.learning_rate * self.activation * inputs
            
        # Apply weight update
        self.weights += delta_w
        
        # Explicit normalization if not using Oja's rule
        if not use_oja:
            self.weights = self._normalize_weights(self.weights)
    
    def update_weights_stdp(
        self, 
        inputs: xp.ndarray, 
        pre_spike_times: Optional[List[Optional[int]]] = None,
        post_spike_time: Optional[int] = None
    ) -> None:
        """
        Update weights using Spike-Timing Dependent Plasticity.
        
        Args:
            inputs: Input activation vector
            pre_spike_times: When each input was active (None for no spike)
            post_spike_time: When this neuron fired (None for no spike)
        """
        # If no explicit spike times, use trace-based approach
        if pre_spike_times is None or post_spike_time is None:
            self._update_weights_stdp_traces()
            return
            
        # Skip if neuron didn't fire
        if post_spike_time is None:
            return
            
        # For each input (pre-synaptic neuron)
        for i in range(self.input_size):
            # Skip if this input didn't spike
            if pre_spike_times[i] is None:
                continue
                
            # Calculate time difference
            time_diff = post_spike_time - pre_spike_times[i]
            
            # Compute STDP weight change based on relative timing
            if time_diff > 0:
                # Pre-before-post: strengthen connection
                # (causal relationship, LTP - Long-Term Potentiation)
                delta_w = self.learning_rate * xp.exp(-time_diff / 20.0)
            else:
                # Post-before-pre: weaken connection
                # (acausal relationship, LTD - Long-Term Depression)
                delta_w = -self.learning_rate * 0.5 * xp.exp(time_diff / 20.0)
                
            # Apply weight change
            self.weights[i] += delta_w
            
        # Normalize weights
        self.weights = self._normalize_weights(self.weights)
    
    def _update_weights_stdp_traces(self) -> None:
        """
        Update weights using trace-based STDP approach.
        """
        # Skip if neuron didn't activate
        if self.activation <= 0:
            return
            
        # Calculate STDP update using eligibility traces
        # pre-before-post: pre_trace * post_spike (potentiation)
        # post-before-pre: post_trace * pre_spike (depression)
        
        # LTP component (strengthen synapses that were active before the neuron fired)
        ltp = self.learning_rate * self.pre_synaptic_traces
        
        # LTD component (weaken synapses that become active after the neuron fired)
        # New inputs potentiate the post-synaptic trace
        ltd = -0.5 * self.learning_rate * self.post_synaptic_trace * self.input_history[-1]
        
        # Combined STDP update
        delta_w = ltp + ltd
        
        # Apply weight update
        self.weights += delta_w
        
        # Normalize weights
        self.weights = self._normalize_weights(self.weights)
    
    def update_weights_bcm(self, inputs: xp.ndarray) -> None:
        """
        Update weights using BCM learning rule with sliding threshold.
        
        The BCM rule is activity-dependent:
        - If postsynaptic activity < threshold: weights decrease
        - If postsynaptic activity > threshold: weights increase
        
        Args:
            inputs: Input activation vector
        """
        # Skip if no inputs are active
        if xp.sum(inputs) <= 0:
            return
            
        # Calculate BCM weight change:
        # Δw = η * y * (y - θ) * x
        # Where θ is the sliding threshold based on recent activity
        
        activity_factor = self.activation * (self.activation - self.bcm_sliding_threshold)
        delta_w = self.learning_rate * activity_factor * inputs
        
        # Apply weight update
        self.weights += delta_w
        
        # Normalize weights
        self.weights = self._normalize_weights(self.weights)
    
    def update_threshold_homeostatic(self) -> None:
        """
        Update activation threshold using homeostatic plasticity.
        
        This helps maintain a target activation rate over time.
        """
        # Skip if not enough history
        if len(self.activation_history) < 10:
            return
            
        # Compare recent activity to target
        activity_error = self.recent_mean_activation - self.target_activation
        
        # Adjust threshold (higher if too active, lower if not active enough)
        self.threshold += self.homeostatic_factor * activity_error
        
        # Keep threshold in reasonable range
        self.threshold = max(0.01, self.threshold)
    
    def prune_weakest_connections(self, threshold: float = 0.01) -> List[int]:
        """
        Prune connections weaker than the threshold.
        
        Args:
            threshold: Weight magnitude threshold for pruning
            
        Returns:
            Indices of pruned connections
        """
        # Find weak connections (below threshold)
        weak_indices = xp.where(xp.abs(self.weights) < threshold)[0]
        
        # Set weak weights to zero
        self.weights[weak_indices] = 0.0
        
        return list(weak_indices)
    
    def get_receptive_field(self) -> xp.ndarray:
        """
        Get the neuron's weights (receptive field).
        
        Returns:
            Copy of weight vector
        """
        return self.weights.copy()
    
    def set_receptive_field(self, weights: xp.ndarray) -> None:
        """
        Set the neuron's weights to the specified values.
        
        Args:
            weights: New weight vector
        """
        if weights.shape[0] != self.input_size:
            raise ValueError(f"Weight size {weights.shape[0]} doesn't match input size {self.input_size}")
            
        self.weights = self._normalize_weights(weights)
        
    def reset_state(self) -> None:
        """Reset neuron state for a new forward pass."""
        self.activation = 0.0
        self.is_winner = False
        
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the neuron.

        Returns:
            Dictionary with neuron state information
        """
        # Convert GPU arrays to CPU scalars for serialization
        return {
            'id': self.id,
            'activation': self.activation,
            'threshold': self.threshold,
            'is_winner': self.is_winner,
            'recent_mean_activation': self.recent_mean_activation,
            'adaptation': self.adaptation,
            'weights_norm': float(xp.linalg.norm(self.weights)),
            'weights_min': float(xp.min(self.weights)),
            'weights_max': float(xp.max(self.weights)),
            'weights_mean': float(xp.mean(self.weights)),
            'weights_std': float(xp.std(self.weights)),
            'non_zero_weights': int(xp.sum(xp.abs(self.weights) > 0.001))
        }
    
    def _normalize_weights(self, weights: xp.ndarray) -> xp.ndarray:
        """
        Normalize weight vector to unit length.
        
        Args:
            weights: Weight vector to normalize
            
        Returns:
            Normalized weights
        """
        # Calculate norm (L2 norm/Euclidean length)
        norm = xp.linalg.norm(weights)
        
        # Avoid division by zero
        if norm < 1e-10:
            # Initialize with random unit vector if all weights are zero
            random_weights = xp.random.randn(self.input_size)
            return random_weights / (xp.linalg.norm(random_weights) + 1e-10)
        else:
            # Normalize to unit length
            return weights / norm
            
    def __repr__(self) -> str:
        return f"Neuron(id={self.id}, act={self.activation:.3f}, threshold={self.threshold:.3f})" 