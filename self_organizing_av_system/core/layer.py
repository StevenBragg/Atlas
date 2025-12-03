import logging
from typing import List, Dict, Optional, Tuple, Union, Any

from .backend import xp, to_cpu
from .neuron import Neuron


class NeuralLayer:
    """
    A layer of self-organizing neurons with local learning rules and competition.
    
    This implements a competitive layer with stability mechanisms including
    homeostatic plasticity, lateral inhibition, and sparsity constraints.
    """
    
    def __init__(
        self,
        input_size: int,
        layer_size: int,
        name: str = "layer",
        learning_rate: float = 0.01,
        initial_threshold: float = 0.5,
        target_activation: float = 0.1,
        homeostatic_factor: float = 0.01,
        k_winners: int = 0,  # 0 means no winner-take-all
        lateral_inhibition_strength: float = 0.1,
        lateral_excitation_strength: float = 0.05,
        homeostatic_adaptation_rate: float = 0.001,
        weight_regularization: float = 0.0001
    ):
        """
        Initialize a layer of self-organizing neurons.
        
        Args:
            input_size: Size of input vector
            layer_size: Number of neurons in the layer
            name: Name for the layer
            learning_rate: Rate of weight updates during learning
            initial_threshold: Starting activation threshold for neurons
            target_activation: Target average activation for homeostasis
            homeostatic_factor: Strength of homeostatic adjustment
            k_winners: Number of winners for WTA (0 for none)
            lateral_inhibition_strength: Strength of lateral inhibition
            lateral_excitation_strength: Strength of lateral excitation (for topography)
            homeostatic_adaptation_rate: Rate of adaptation for homeostasis target
            weight_regularization: Regularization strength to prevent overfitting
        """
        self.input_size = input_size
        self.layer_size = layer_size
        self.name = name
        self.learning_rate = learning_rate
        self.initial_threshold = initial_threshold
        self.target_activation = target_activation
        self.homeostatic_factor = homeostatic_factor
        self.k_winners = k_winners
        self.lateral_inhibition_strength = lateral_inhibition_strength
        self.lateral_excitation_strength = lateral_excitation_strength
        self.homeostatic_adaptation_rate = homeostatic_adaptation_rate
        self.weight_regularization = weight_regularization
        
        # Initialize neurons
        self.neurons = [
            Neuron(
                input_size=input_size,
                neuron_id=i,
                learning_rate=learning_rate,
                threshold=initial_threshold,
                target_activation=target_activation,
                homeostatic_factor=homeostatic_factor
            )
            for i in range(layer_size)
        ]
        
        # Initialize activations
        self.activations = xp.zeros(layer_size)
        self.activations_raw = xp.zeros(layer_size)  # Pre-competition activation
        
        # Lateral connectivity (for inhibition and excitation)
        if lateral_inhibition_strength > 0 or lateral_excitation_strength > 0:
            self.lateral_weights = self._initialize_lateral_weights()
        else:
            self.lateral_weights = None
        
        # For recurrent connections (temporal context)
        self.recurrent_weights = None
        
        # For layer-level adaptation and monitoring
        self.threshold = initial_threshold
        self.mean_activation_history = []
        self.sparsity_history = []
        self.last_winners = []  # Track which neurons won recently
        
        # Additional feedback signals (for reentrant/top-down processing)
        self.prediction_signal = None  # Signal from other modality
        self.top_down_signal = None  # Signal from higher layer
        
        # Momentum for weight updates (helps with stability)
        self.weight_momentum = 0.5
        self.momentum_buffer = [xp.zeros((input_size)) for _ in range(layer_size)]
        
        # For diversity maintenance
        self.correlation_matrix = xp.zeros((layer_size, layer_size))
        self.decorrelation_strength = 0.1  # Strength of decorrelation pressure
        
        # Logger setup
        self.logger = logging.getLogger(f'Layer.{name}')
    
    def _initialize_lateral_weights(self) -> xp.ndarray:
        """
        Initialize the lateral connectivity weights between neurons.
        
        Returns:
            Matrix of lateral weights
        """
        # Initialize with mostly inhibitory connections
        weights = -self.lateral_inhibition_strength * xp.ones((self.layer_size, self.layer_size))
        xp.fill_diagonal(weights, 0.0)  # No self-connections
        
        # Add local excitation (helps with topographic organization)
        if self.lateral_excitation_strength > 0:
            # Calculate distances between neuron indices
            # This is a simplified 1D topography - could be extended to 2D
            indices = xp.arange(self.layer_size)
            distances = xp.abs(indices[:, xp.newaxis] - indices[xp.newaxis, :])
            
            # Add excitation for nearby neurons (Mexican hat function)
            # Neurons close to each other excite, further ones inhibit
            proximity = xp.exp(-0.5 * (distances / 2.0) ** 2)
            excitation = self.lateral_excitation_strength * proximity
            
            # Combine excitation and inhibition
            weights += excitation
            
            # Ensure no self-connections
            xp.fill_diagonal(weights, 0.0)
        
        return weights
    
    def activate(self, inputs: xp.ndarray, time_step: Optional[int] = None) -> xp.ndarray:
        """
        Compute activations for all neurons given input.
        
        Args:
            inputs: Input activation vector
            time_step: Current simulation time step
            
        Returns:
            Vector of neuron activations
        """
        if inputs.shape[0] != self.input_size:
            raise ValueError(f"Input size {inputs.shape[0]} doesn't match expected {self.input_size}")
        
        # Compute initial activations for all neurons
        self.activations_raw = xp.zeros(self.layer_size)
        
        for i, neuron in enumerate(self.neurons):
            # Compute basic feedforward activation
            activation = neuron.activate(inputs, time_step)
            self.activations_raw[i] = activation
        
        # Apply recurrent influence if available
        if self.recurrent_weights is not None and xp.any(self.activations):
            recurrent_input = xp.dot(self.activations, self.recurrent_weights)
            # Blend with raw activations
            self.activations_raw = 0.7 * self.activations_raw + 0.3 * recurrent_input
        
        # Apply cross-modal prediction signal if available
        if self.prediction_signal is not None and len(self.prediction_signal) == self.layer_size:
            # Blend with raw activations (weak influence)
            self.activations_raw = 0.8 * self.activations_raw + 0.2 * self.prediction_signal
            # Clear for next round
            self.prediction_signal = None
        
        # Apply top-down signal if available
        if self.top_down_signal is not None and len(self.top_down_signal) == self.layer_size:
            # Blend with raw activations (weak influence)
            self.activations_raw = 0.8 * self.activations_raw + 0.2 * self.top_down_signal
            # Clear for next round
            self.top_down_signal = None
        
        # Apply competition
        if self.k_winners > 0:
            # k-Winners-Take-All
            self._apply_k_winners()
        elif self.lateral_weights is not None:
            # Lateral inhibition based on explicit weights
            self._apply_lateral_competition()
        else:
            # Simple threshold activation
            self.activations = xp.maximum(0, self.activations_raw - self.threshold)
        
        # Track sparsity and mean activation
        sparsity = xp.mean(self.activations > 0)
        mean_activation = xp.mean(self.activations)
        
        # Add to history
        self.sparsity_history.append(float(sparsity))
        self.mean_activation_history.append(float(mean_activation))
        
        # Limit history size
        if len(self.sparsity_history) > 1000:
            self.sparsity_history.pop(0)
            self.mean_activation_history.pop(0)
        
        # Adapt target activation based on history if needed
        if len(self.mean_activation_history) > 100:
            self._adapt_homeostatic_targets()
        
        return self.activations.copy()
    
    def _apply_k_winners(self) -> None:
        """Apply k-Winners-Take-All competition."""
        # Find top k indices
        k = min(self.k_winners, self.layer_size)
        
        if k <= 0:
            # No competition, just apply threshold
            self.activations = xp.maximum(0, self.activations_raw - self.threshold)
            return
        
        # Get top k indices
        top_k_indices = xp.argsort(self.activations_raw)[-k:]
        
        # Set activations to zero except for winners
        self.activations = xp.zeros_like(self.activations_raw)
        
        # Set winner activations to their raw values (optionally minus a threshold)
        for idx in top_k_indices:
            self.activations[idx] = max(0, self.activations_raw[idx] - self.threshold)
        
        # Mark winners for learning
        for i, neuron in enumerate(self.neurons):
            neuron.is_winner = i in top_k_indices
        
        # Keep track of winners for diversity
        self.last_winners = list(top_k_indices)
    
    def _apply_lateral_competition(self) -> None:
        """Apply lateral inhibition based on explicit weights."""
        # Apply lateral influence from current activations
        lateral_influence = xp.dot(self.activations_raw, self.lateral_weights)
        
        # Combine feedforward and lateral inputs
        combined_input = self.activations_raw + lateral_influence
        
        # Apply threshold activation
        self.activations = xp.maximum(0, combined_input - self.threshold)
        
        # Mark winners for learning
        winners = self.activations > 0
        for i, neuron in enumerate(self.neurons):
            neuron.is_winner = winners[i]
        
        # Keep track of winners for diversity
        self.last_winners = list(xp.where(winners)[0])
    
    def learn(self, inputs: xp.ndarray, learning_rule: str = 'oja') -> None:
        """
        Apply learning to all neurons in the layer.
        
        Args:
            inputs: Input activation vector
            learning_rule: Learning rule to use ('hebbian', 'oja', or 'stdp')
        """
        # Regulate learning rate based on stability
        effective_learning_rate = self._regulate_learning_rate()
        
        # Step 1: Update weight correlation matrix
        self._update_weight_correlations()
        
        for i, neuron in enumerate(self.neurons):
            # Skip if neuron didn't win competition (if using competition)
            if self.k_winners > 0 and not neuron.is_winner:
                continue
            
            # Adjust neuron's learning rate if needed
            if effective_learning_rate != self.learning_rate:
                neuron.learning_rate = effective_learning_rate
            
            # Apply selected learning rule
            if learning_rule == 'hebbian':
                neuron.update_weights_hebbian(inputs, use_oja=False)
            elif learning_rule == 'oja':
                neuron.update_weights_hebbian(inputs, use_oja=True)
            elif learning_rule == 'stdp':
                # Not fully implemented in this example
                neuron.update_weights_hebbian(inputs, use_oja=True)
            else:
                raise ValueError(f"Unknown learning rule: {learning_rule}")
            
            # Apply weight regularization to prevent overfitting
            if self.weight_regularization > 0:
                neuron.weights *= (1.0 - self.weight_regularization)
                
            # Apply weight decorrelation for diversity
            if self.decorrelation_strength > 0 and len(self.neurons) > 1:
                self._apply_weight_decorrelation(i)
    
    def _update_weight_correlations(self) -> None:
        """Update the correlation matrix between neuron weights."""
        if self.decorrelation_strength <= 0 or len(self.neurons) <= 1:
            return
            
        # Extract weight vectors
        weight_vectors = xp.array([n.weights for n in self.neurons])
        
        # Calculate correlation matrix
        self.correlation_matrix = xp.corrcoef(weight_vectors)
        
        # Ensure valid values (NaN can happen with constant weights)
        xp.fill_diagonal(self.correlation_matrix, 1.0)
        self.correlation_matrix = xp.nan_to_num(self.correlation_matrix)
    
    def _apply_weight_decorrelation(self, neuron_idx: int) -> None:
        """
        Apply decorrelation pressure to a neuron's weights to increase diversity.
        
        Args:
            neuron_idx: Index of the neuron to decorrelate
        """
        # Skip if no correlations or just one neuron
        if len(self.neurons) <= 1:
            return
            
        neuron = self.neurons[neuron_idx]
        
        # Calculate decorrelation term (away from correlated neurons)
        decorrelation_pressure = xp.zeros_like(neuron.weights)
        
        for j, other_neuron in enumerate(self.neurons):
            if j == neuron_idx:
                continue
                
            # Get correlation between this neuron and the other
            correlation = self.correlation_matrix[neuron_idx, j]
            
            # Only push away from positively correlated neurons
            if correlation > 0.5:  # Threshold for significant correlation
                # Add pressure away from the other neuron's weights
                # Stronger for higher correlations
                decorrelation_pressure -= correlation * other_neuron.weights
        
        # Normalize and apply decorrelation pressure
        norm = xp.linalg.norm(decorrelation_pressure)
        if norm > 0:
            decorrelation_pressure /= norm
            
            # Apply decorrelation (small adjustment)
            adjust_rate = self.learning_rate * self.decorrelation_strength
            neuron.weights += adjust_rate * decorrelation_pressure
            
            # Re-normalize weights
            neuron.weights = neuron._normalize_weights(neuron.weights)
    
    def _regulate_learning_rate(self) -> float:
        """
        Regulate learning rate based on stability metrics.
        
        Returns:
            Adjusted learning rate
        """
        # Default is to use the configured learning rate
        effective_rate = self.learning_rate
        
        # Check for stability issues
        if len(self.mean_activation_history) > 100:
            # Calculate recent activation variance
            recent_history = self.mean_activation_history[-100:]
            activation_variance = xp.var(recent_history)
            
            # If variance is high, reduce learning rate (instability)
            if activation_variance > 0.1:
                # Reduce learning rate proportional to variance
                stability_factor = max(0.1, 1.0 - 5.0 * activation_variance)
                effective_rate *= stability_factor
            
            # If activations are consistently too low, increase rate
            if xp.mean(recent_history) < 0.01:
                effective_rate *= 1.5
        
        return effective_rate
    
    def apply_homeostasis(self) -> None:
        """
        Apply homeostatic plasticity to maintain target activity levels.
        """
        # Apply homeostasis to each neuron
        for neuron in self.neurons:
            neuron.update_threshold_homeostatic()
        
        # Update layer-level threshold for new neurons
        if len(self.mean_activation_history) > 10:
            recent_mean = xp.mean(self.mean_activation_history[-10:])
            
            # Adjust layer threshold based on overall activity
            if recent_mean > self.target_activation * 1.5:
                # Too much activity, increase threshold
                self.threshold += self.homeostatic_factor * 0.1
            elif recent_mean < self.target_activation * 0.5:
                # Too little activity, decrease threshold
                self.threshold -= self.homeostatic_factor * 0.1
                
            # Ensure threshold stays in reasonable range
            self.threshold = max(0.1, min(2.0, self.threshold))
    
    def _adapt_homeostatic_targets(self) -> None:
        """
        Adaptively adjust target activation based on layer statistics.
        """
        # Only adapt if enough history
        if len(self.mean_activation_history) < 100:
            return
            
        # Calculate long-term and short-term statistics
        long_term_mean = xp.mean(self.mean_activation_history[-100:])
        long_term_sparsity = xp.mean(self.sparsity_history[-100:])
        
        # Calculate ideal sparsity based on layer size
        # Larger layers should have lower activation ratios
        ideal_sparsity = min(0.3, 2.0 / xp.sqrt(self.layer_size))
        
        # Adjust target activation if actual sparsity is far from ideal
        if long_term_sparsity > ideal_sparsity * 1.5:
            # Too many neurons active, decrease target
            new_target = self.target_activation * (1.0 - self.homeostatic_adaptation_rate)
        elif long_term_sparsity < ideal_sparsity * 0.5:
            # Too few neurons active, increase target
            new_target = self.target_activation * (1.0 + self.homeostatic_adaptation_rate)
        else:
            # Sparsity in good range, no change
            new_target = self.target_activation
        
        # Ensure target stays in reasonable range
        new_target = max(0.01, min(0.3, new_target))
        
        # Apply the change (if significant)
        if abs(new_target - self.target_activation) > 0.001:
            self.target_activation = new_target
            
            # Update individual neuron targets
            for neuron in self.neurons:
                neuron.target_activation = new_target
    
    def init_recurrent_connections(self) -> None:
        """
        Initialize recurrent connections for sequence learning.
        """
        # Create recurrent weight matrix (initially small random weights)
        self.recurrent_weights = xp.random.normal(0, 0.01, (self.layer_size, self.layer_size))
        
        # No self-recurrence initially
        xp.fill_diagonal(self.recurrent_weights, 0)
    
    def update_recurrent_connections(self) -> None:
        """
        Update recurrent connections based on temporal coincidence.
        """
        if self.recurrent_weights is None:
            return
            
        # Need sufficient history to learn meaningful sequences
        if len(self.mean_activation_history) < 2:
            return
            
        # Apply temporal Hebbian learning
        # This is a very simplified version of sequence learning
        # In practice, STDP or more sophisticated rules should be used
        
        # Get current activations
        current_act = self.activations
        
        # No updates if not enough activity
        if xp.sum(current_act > 0) < 2:
            return
        
        # Get previous layer state (assuming we recorded it)
        if not hasattr(self, 'previous_activations') or self.previous_activations is None:
            self.previous_activations = current_act
            return
        
        # Apply Hebbian update: if neuron A fired before neuron B,
        # strengthen the connection A→B (but not B→A)
        prev_act = self.previous_activations
        
        # Create outer product of previous with current
        temporal_coincidence = xp.outer(prev_act, current_act)
        
        # Update recurrent weights
        recurrent_lr = self.learning_rate * 0.1  # Typically slower than feedforward
        self.recurrent_weights += recurrent_lr * temporal_coincidence
        
        # Apply simple normalization to keep weights bounded
        # Row-wise normalization
        for i in range(self.layer_size):
            row_sum = xp.sum(xp.abs(self.recurrent_weights[i, :]))
            if row_sum > 0.001:
                self.recurrent_weights[i, :] /= row_sum
        
        # Ensure no self-recurrence (optional, depends on model)
        xp.fill_diagonal(self.recurrent_weights, 0)
        
        # Update previous activations
        self.previous_activations = current_act.copy()
    
    def predict_next_activation(self) -> xp.ndarray:
        """
        Predict the next expected activations based on current state.
        
        Returns:
            Predicted next activation vector
        """
        if self.recurrent_weights is None or not xp.any(self.activations):
            return xp.zeros(self.layer_size)
            
        # Simple linear prediction using recurrent weights
        predicted_activations = xp.dot(self.activations, self.recurrent_weights)
        
        # Apply threshold for realistic prediction
        return xp.maximum(0, predicted_activations - self.threshold * 0.5)
    
    def add_neuron(self, initial_weights: Optional[xp.ndarray] = None) -> int:
        """
        Add a new neuron to the layer.
        
        Args:
            initial_weights: Optional initial weights for the new neuron
            
        Returns:
            Index of the newly added neuron
        """
        # Create new neuron
        new_idx = self.layer_size
        new_neuron = Neuron(
            input_size=self.input_size,
            neuron_id=new_idx,
            learning_rate=self.learning_rate,
            threshold=self.threshold,  # Use current layer threshold
            target_activation=self.target_activation,
            homeostatic_factor=self.homeostatic_factor,
            initial_weights=initial_weights
        )
        
        # Add to layer
        self.neurons.append(new_neuron)
        self.layer_size += 1
        
        # Extend activations array
        self.activations = xp.append(self.activations, 0.0)
        self.activations_raw = xp.append(self.activations_raw, 0.0)
        
        # Update lateral weights if using inhibition
        if self.lateral_weights is not None:
            # Expand matrix with new row and column
            new_row = -self.lateral_inhibition_strength * xp.ones(self.layer_size - 1)
            new_col = -self.lateral_inhibition_strength * xp.ones((self.layer_size - 1, 1))
            
            # Combine
            self.lateral_weights = xp.vstack((self.lateral_weights, new_row))
            self.lateral_weights = xp.hstack((self.lateral_weights, xp.zeros((self.layer_size, 1))))
            self.lateral_weights[-1, -1] = 0.0  # No self-inhibition
            
            # Add local excitation if used
            if self.lateral_excitation_strength > 0:
                # Recompute the entire matrix
                self.lateral_weights = self._initialize_lateral_weights()
        
        # Extend recurrent weights if using them
        if self.recurrent_weights is not None:
            # Add zero row and column
            self.recurrent_weights = xp.vstack((
                self.recurrent_weights, 
                xp.zeros(self.layer_size - 1)
            ))
            self.recurrent_weights = xp.hstack((
                self.recurrent_weights, 
                xp.zeros((self.layer_size, 1))
            ))
        
        # Extend correlation matrix for weight diversity
        if self.decorrelation_strength > 0:
            # Expand with new row and column
            new_size = self.correlation_matrix.shape[0] + 1
            new_matrix = xp.ones((new_size, new_size))
            new_matrix[:-1, :-1] = self.correlation_matrix
            new_matrix[-1, :] = 0.0
            new_matrix[:, -1] = 0.0
            new_matrix[-1, -1] = 1.0
            self.correlation_matrix = new_matrix
        
        # Add a momentum buffer entry
        if self.weight_momentum > 0:
            self.momentum_buffer.append(xp.zeros(self.input_size))
        
        self.logger.debug(f"Added neuron {new_idx} to layer {self.name}")
        return new_idx
    
    def replace_neuron(self, neuron_idx: int, initial_weights: Optional[xp.ndarray] = None) -> None:
        """
        Replace the weights of an existing neuron.
        
        Args:
            neuron_idx: Index of the neuron to replace
            initial_weights: New weights for the neuron
        """
        if neuron_idx < 0 or neuron_idx >= self.layer_size:
            raise ValueError(f"Invalid neuron index {neuron_idx} for layer size {self.layer_size}")
            
        # Create replacement neuron
        replacement = Neuron(
            input_size=self.input_size,
            neuron_id=neuron_idx,
            learning_rate=self.learning_rate,
            threshold=self.threshold,  # Use current layer threshold
            target_activation=self.target_activation,
            homeostatic_factor=self.homeostatic_factor,
            initial_weights=initial_weights
        )
        
        # Replace in layer
        self.neurons[neuron_idx] = replacement
        
        # Reset activations for this neuron
        self.activations[neuron_idx] = 0.0
        self.activations_raw[neuron_idx] = 0.0
        
        # Reset momentum buffer if using momentum
        if self.weight_momentum > 0:
            self.momentum_buffer[neuron_idx] = xp.zeros(self.input_size)
            
        self.logger.debug(f"Replaced neuron {neuron_idx} in layer {self.name}")
    
    def prune_connections(self, threshold: float = 0.01) -> int:
        """
        Prune weak connections across all neurons.
        
        Args:
            threshold: Weight threshold for pruning
            
        Returns:
            Number of pruned connections
        """
        total_pruned = 0
        
        for neuron in self.neurons:
            # Find weights smaller than threshold
            small_weights = xp.abs(neuron.weights) < threshold
            
            # Count them
            pruned = xp.sum(small_weights)
            
            # Zero them out
            if pruned > 0:
                neuron.weights[small_weights] = 0.0
                
                # Re-normalize weights
                neuron.weights = neuron._normalize_weights(neuron.weights)
                
                total_pruned += pruned
        
        if total_pruned > 0:
            self.logger.debug(f"Pruned {total_pruned} weak connections in layer {self.name}")
            
        return int(total_pruned)
    
    def get_layer_state(self) -> Dict[str, Any]:
        """
        Get the current state of the layer.

        Returns:
            Dictionary with layer state information
        """
        # Convert GPU arrays to CPU for serialization
        return {
            'name': self.name,
            'input_size': self.input_size,
            'layer_size': self.layer_size,
            'learning_rate': self.learning_rate,
            'threshold': self.threshold,
            'target_activation': self.target_activation,
            'activations': to_cpu(self.activations).copy(),
            'activations_raw': to_cpu(self.activations_raw).copy(),
            'mean_activation': float(xp.mean(self.activations)),
            'sparsity': float(xp.mean(self.activations > 0)),
            'neuron_states': [
                {
                    'threshold': neuron.threshold,
                    'recent_activation': neuron.recent_mean_activation
                }
                for neuron in self.neurons
            ]
        }
    
    def get_all_receptive_fields(self) -> xp.ndarray:
        """
        Get receptive fields (weights) of all neurons in the layer.
        
        Returns:
            Matrix of receptive fields, one per row
        """
        fields = xp.zeros((self.layer_size, self.input_size))
        for i, neuron in enumerate(self.neurons):
            fields[i] = neuron.get_receptive_field()
        return fields
    
    def get_weight_matrix(self) -> xp.ndarray:
        """
        Get the weight matrix for the layer.
        
        Returns:
            Weight matrix of shape (layer_size, input_size)
        """
        weights = xp.zeros((self.layer_size, self.input_size))
        for i, neuron in enumerate(self.neurons):
            weights[i] = neuron.weights
        return weights 