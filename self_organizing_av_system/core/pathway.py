import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from .layer import NeuralLayer


class NeuralPathway:
    """
    A hierarchical stack of neural layers forming a processing pathway.
    
    This manages feedforward, feedback, and temporal connections between layers,
    implementing a complete sensory processing stream.
    """
    
    def __init__(
        self,
        name: str,
        input_size: int,
        layer_sizes: List[int],
        layer_k_winners: Optional[List[int]] = None,
        learning_rates: Optional[List[float]] = None,
        use_recurrent: bool = True
    ):
        """
        Initialize a neural pathway with a stack of layers.
        
        Args:
            name: Unique name for the pathway
            input_size: Size of raw input vector
            layer_sizes: List of neuron counts for each layer
            layer_k_winners: List of k-winner counts for each layer
            learning_rates: List of learning rates for each layer
            use_recurrent: Whether to use recurrent connections for sequence learning
        """
        self.name = name
        self.input_size = input_size
        self.num_layers = len(layer_sizes)
        
        # Default parameters if not provided
        if layer_k_winners is None:
            # Set k-winners to about 10% of layer size, minimum 1
            layer_k_winners = [max(1, int(0.1 * size)) for size in layer_sizes]
        
        if learning_rates is None:
            # Decreasing learning rates for higher layers
            learning_rates = [0.01 / (i + 1) for i in range(self.num_layers)]
        
        # Create layers
        self.layers = []
        prev_size = input_size
        
        for i in range(self.num_layers):
            layer = NeuralLayer(
                input_size=prev_size,
                layer_size=layer_sizes[i],
                name=f"{name}_layer_{i}",
                learning_rate=learning_rates[i],
                initial_threshold=0.5,
                target_activation=0.1,
                homeostatic_factor=0.01,
                k_winners=layer_k_winners[i]
            )
            
            self.layers.append(layer)
            prev_size = layer_sizes[i]
            
            # Initialize recurrent connections for sequence learning
            if use_recurrent:
                layer.init_recurrent_connections()
        
        # Pathway state tracking
        self.current_input = None
        self.previous_layer_activations = [None] * self.num_layers
        
        # Prediction state
        self.predicted_next_input = None
        self.error_history = []
    
    def process(self, inputs: np.ndarray, time_step: Optional[int] = None) -> np.ndarray:
        """
        Process an input through the full pathway.
        
        Args:
            inputs: Raw input vector
            time_step: Current simulation time step
            
        Returns:
            Activations of the final layer
        """
        if inputs.shape[0] != self.input_size:
            raise ValueError(f"Input size {inputs.shape[0]} doesn't match expected {self.input_size}")
        
        self.current_input = inputs.copy()
        current_activations = inputs
        
        # Feedforward pass through all layers
        for i, layer in enumerate(self.layers):
            # Store previous activations for temporal learning
            self.previous_layer_activations[i] = layer.activations.copy() if layer.activations.any() else None
            
            # Process through current layer
            current_activations = layer.activate(current_activations, time_step)
        
        return current_activations
    
    def learn(self, learning_rule: str = 'oja') -> None:
        """
        Apply learning to all layers in the pathway.
        
        Args:
            learning_rule: Learning rule to use ('hebbian', 'oja', or 'stdp')
        """
        if self.current_input is None:
            return  # No input has been processed yet
        
        # Start with the raw input
        current_input = self.current_input
        
        # Learn layer by layer
        for i, layer in enumerate(self.layers):
            # Apply learning rule to adapt weights
            layer.learn(current_input, learning_rule)
            
            # Skip recurrent updates - not implemented in NeuralLayer yet
            # This will be addressed in a future update with proper sequence learning
            
            # Apply homeostatic plasticity
            layer.apply_homeostasis()
            
            # The output of this layer becomes the input to the next
            if i < self.num_layers - 1:
                current_input = layer.activations
    
    def predict_next(self) -> List[np.ndarray]:
        """
        Predict the next activations for each layer.
        
        Returns:
            List of predicted activation vectors for each layer
        """
        predictions = []
        
        for layer in self.layers:
            pred = layer.predict_next_activation()
            predictions.append(pred)
        
        return predictions
    
    def generate_from_top(self, top_layer_activation: np.ndarray) -> np.ndarray:
        """
        Generate expected lower-layer activations from top-layer state.
        
        This implements top-down generation (like imagination or expectation).
        
        Args:
            top_layer_activation: Activation pattern for the top layer
            
        Returns:
            Predicted input vector
        """
        # Start with the top layer activation
        current_activation = top_layer_activation.copy()
        
        # Propagate activation down through the layers in reverse
        for layer in reversed(self.layers):
            # Use the transpose of the weight matrix for top-down generation
            # This approximates the inverse mapping
            weights = layer.get_weight_matrix()
            
            # Generate lower layer activation
            # Add bias to ensure non-zero activation
            lower_activation = np.dot(weights.T, current_activation)
            
            # Apply activation function (ReLU with small leak)
            lower_activation = np.where(lower_activation > 0, lower_activation, 0.01 * lower_activation)
            
            # Normalize to prevent explosion/vanishing
            norm = np.linalg.norm(lower_activation)
            if norm > 0:
                lower_activation = lower_activation / norm * np.sqrt(len(lower_activation))
            
            current_activation = lower_activation
        
        # The final activation is the predicted input
        # Ensure it's in valid range [0, 1] for input data
        predicted_input = np.clip(current_activation, 0, 1)
        
        return predicted_input
    
    def prune_pathway(self, threshold: float = 0.01) -> Dict[str, int]:
        """
        Prune weak connections across all layers.
        
        Args:
            threshold: Weight threshold for pruning
            
        Returns:
            Dictionary with pruning statistics per layer
        """
        pruning_stats = {}
        
        for i, layer in enumerate(self.layers):
            pruned = layer.prune_connections(threshold)
            pruning_stats[f"layer_{i}"] = pruned
        
        return pruning_stats
    
    def add_neurons_where_needed(self, max_new_per_layer: int = 1) -> Dict[str, int]:
        """
        Add neurons to layers with consistently high activation.
        
        This implements structural plasticity by growing the network
        in response to input complexity.
        
        Args:
            max_new_per_layer: Maximum number of neurons to add per layer
            
        Returns:
            Dictionary with counts of new neurons per layer
        """
        neuron_additions = {}
        
        for i, layer in enumerate(self.layers):
            # Check if this layer is consistently highly active
            recent_mean = np.mean(layer.mean_activation_history[-100:]) if len(layer.mean_activation_history) >= 100 else 0
            recent_sparsity = np.mean(layer.sparsity_history[-100:]) if len(layer.sparsity_history) >= 100 else 0
            
            # If many neurons are consistently active, we need more capacity
            if recent_sparsity > 0.3 and recent_mean > 0.2:
                new_count = 0
                for _ in range(max_new_per_layer):
                    layer.add_neuron()
                    new_count += 1
                
                neuron_additions[f"layer_{i}"] = new_count
            else:
                neuron_additions[f"layer_{i}"] = 0
        
        return neuron_additions
    
    def replace_dead_neurons(self, min_activation_threshold: float = 0.01) -> Dict[str, int]:
        """
        Replace neurons that are consistently inactive.
        
        Args:
            min_activation_threshold: Minimum acceptable activity level
            
        Returns:
            Dictionary with counts of replaced neurons per layer
        """
        replacements = {}
        
        for i, layer in enumerate(self.layers):
            replaced = 0
            
            for j, neuron in enumerate(layer.neurons):
                # Check if this neuron is dead (never activates)
                if neuron.recent_mean_activation < min_activation_threshold:
                    # Replace with a new randomly initialized neuron
                    if self.current_input is not None:
                        # Use random input or the current input as initialization
                        if i == 0:
                            # First layer - use random subset of current raw input
                            random_indices = np.random.choice(self.input_size, size=int(0.2 * self.input_size))
                            new_input = np.zeros_like(self.current_input)
                            new_input[random_indices] = self.current_input[random_indices]
                            layer.replace_neuron(j, new_input)
                        else:
                            # Higher layers - use random subset of current layer input
                            prev_layer_act = self.layers[i-1].activations
                            random_indices = np.random.choice(len(prev_layer_act), size=int(0.2 * len(prev_layer_act)))
                            new_input = np.zeros_like(prev_layer_act)
                            new_input[random_indices] = prev_layer_act[random_indices]
                            layer.replace_neuron(j, new_input)
                        
                        replaced += 1
            
            replacements[f"layer_{i}"] = replaced
        
        return replacements
    
    def get_pathway_state(self) -> Dict:
        """
        Get the current state of the entire pathway.
        
        Returns:
            Dictionary with pathway state information
        """
        state = {
            'name': self.name,
            'num_layers': self.num_layers,
            'layers': [layer.get_layer_state() for layer in self.layers]
        }
        return state
    
    def get_all_layer_activations(self) -> List[np.ndarray]:
        """
        Get activations for all layers.
        
        Returns:
            List of activation vectors for each layer
        """
        return [layer.activations.copy() for layer in self.layers]
    
    def get_prediction_error(self, actual_next_input: np.ndarray) -> float:
        """
        Calculate the prediction error between predicted and actual next input.
        
        Args:
            actual_next_input: Actual next input observed
            
        Returns:
            Mean squared error of the prediction
        """
        # Get the current top layer activation
        if len(self.layers) == 0:
            return 0.0
            
        top_layer_activation = self.layers[-1].activations
        
        # Generate predicted input from top layer
        predicted_input = self.generate_from_top(top_layer_activation)
        
        # Ensure both arrays have the same shape
        if predicted_input.shape != actual_next_input.shape:
            # Resize if needed
            min_size = min(len(predicted_input), len(actual_next_input))
            predicted_input = predicted_input[:min_size]
            actual_next_input = actual_next_input[:min_size]
        
        # Calculate mean squared error
        error = np.mean((predicted_input - actual_next_input) ** 2)
        
        # Store for learning adjustments
        self._last_prediction_error = error
        
        return float(error)
    
    def __repr__(self) -> str:
        layer_info = ", ".join([f"{layer.layer_size}" for layer in self.layers])
        return f"NeuralPathway(name='{self.name}', layers=[{layer_info}])" 