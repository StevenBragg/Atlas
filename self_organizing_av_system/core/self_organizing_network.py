"""
Self-Organizing Neural Network with Dynamic Topology

This class wraps NeuralLayer and StructuralPlasticity to create a truly
self-organizing network where:
- Neurons are added when learning demands it
- Neurons are pruned when they become redundant
- Layers can grow or shrink dynamically
- All structural changes are tracked for visualization
"""

import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .backend import xp, to_cpu, to_gpu
from .neuron import Neuron
from .layer import NeuralLayer
from .structural_plasticity import StructuralPlasticity, GrowthStrategy, PruningStrategy

logger = logging.getLogger(__name__)


class StructuralEventType(Enum):
    """Types of structural changes in the network."""
    NEURON_ADDED = "neuron_added"
    NEURON_PRUNED = "neuron_pruned"
    LAYER_ADDED = "layer_added"
    LAYER_REMOVED = "layer_removed"
    CONNECTION_SPROUTED = "connection_sprouted"
    CONNECTION_PRUNED = "connection_pruned"
    WEIGHT_UPDATED = "weight_updated"


@dataclass
class StructuralEvent:
    """Record of a structural change in the network."""
    event_type: StructuralEventType
    timestamp: float
    layer_index: int
    neuron_index: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkSnapshot:
    """Snapshot of network structure for visualization."""
    layers: List[Dict[str, Any]]
    total_neurons: int
    total_connections: int
    recent_events: List[StructuralEvent]
    timestamp: float


class SelfOrganizingNetwork:
    """
    A self-organizing neural network with dynamic topology.

    This network uses NeuralLayer for computation and StructuralPlasticity
    for managing when to grow and prune. The actual structural changes
    (adding/removing neurons) happen here.

    Key features:
    - Dynamic neuron addition based on learning needs
    - Pruning of inactive or redundant neurons
    - Layer-wise structural plasticity management
    - Complete event tracking for visualization
    """

    def __init__(
        self,
        input_dim: int,
        initial_layer_sizes: List[int],
        output_dim: int,
        learning_rate: float = 0.01,
        growth_threshold: float = 0.7,
        prune_threshold: float = 0.01,
        max_neurons_per_layer: int = 256,
        min_neurons_per_layer: int = 8,
        enable_structural_plasticity: bool = True,
    ):
        """
        Initialize the self-organizing network.

        Args:
            input_dim: Dimension of input vectors
            initial_layer_sizes: List of initial sizes for hidden layers
            output_dim: Dimension of output vectors
            learning_rate: Base learning rate
            growth_threshold: Activity threshold to trigger growth
            prune_threshold: Utility threshold for pruning
            max_neurons_per_layer: Maximum neurons allowed per layer
            min_neurons_per_layer: Minimum neurons required per layer
            enable_structural_plasticity: Whether to allow structural changes
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.growth_threshold = growth_threshold
        self.prune_threshold = prune_threshold
        self.max_neurons_per_layer = max_neurons_per_layer
        self.min_neurons_per_layer = min_neurons_per_layer
        self.enable_structural_plasticity = enable_structural_plasticity

        # Build the network layers
        self.layers: List[NeuralLayer] = []
        self.plasticity_managers: List[StructuralPlasticity] = []
        self._build_network(initial_layer_sizes)

        # Output weights (from last hidden layer to output)
        last_hidden_size = self.layers[-1].layer_size if self.layers else input_dim
        self.output_weights = xp.random.randn(last_hidden_size, output_dim).astype(xp.float32) * 0.1
        self.output_bias = xp.zeros(output_dim, dtype=xp.float32)

        # Event tracking for visualization
        self.structural_events: List[StructuralEvent] = []
        self.max_events = 1000

        # Statistics
        self.forward_count = 0
        self.learn_count = 0
        self.total_neurons_added = 0
        self.total_neurons_pruned = 0
        self.total_layers_added = 0
        self.total_layers_removed = 0

        # Internal state for learning
        self.last_activations: List[xp.ndarray] = []
        self.last_input: Optional[xp.ndarray] = None
        self.reconstruction_error = 0.0

        logger.info(
            f"SelfOrganizingNetwork initialized: input={input_dim}, "
            f"layers={[l.layer_size for l in self.layers]}, output={output_dim}"
        )

    def _build_network(self, layer_sizes: List[int]) -> None:
        """Build the initial network structure."""
        prev_size = self.input_dim

        for i, size in enumerate(layer_sizes):
            # Create layer
            layer = NeuralLayer(
                input_size=prev_size,
                layer_size=size,
                name=f"hidden_{i}",
                learning_rate=self.learning_rate,
                k_winners=max(1, size // 4),  # k-WTA with 25% winners
            )
            self.layers.append(layer)

            # Create plasticity manager for this layer
            plasticity = StructuralPlasticity(
                initial_size=size,
                max_size=self.max_neurons_per_layer,
                min_size=self.min_neurons_per_layer,
                growth_threshold=self.growth_threshold,
                prune_threshold=self.prune_threshold,
                growth_strategy=GrowthStrategy.HYBRID,
                pruning_strategy=PruningStrategy.UTILITY_BASED,
                growth_cooldown=50,
                check_interval=20,
                max_growth_per_step=3,
                max_prune_per_step=2,
            )
            self.plasticity_managers.append(plasticity)

            prev_size = size

    def forward(self, x: xp.ndarray) -> xp.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Input array of shape (input_dim,) or (batch, input_dim)

        Returns:
            Output array of shape (output_dim,) or (batch, output_dim)
        """
        self.forward_count += 1

        # Handle batched vs single input
        is_batch = len(x.shape) > 1
        if not is_batch:
            x = x.reshape(1, -1)

        batch_size = x.shape[0]
        self.last_input = x
        self.last_activations = []

        # Process through each layer
        current = x
        for layer in self.layers:
            # Process each sample in batch
            layer_outputs = []
            for i in range(batch_size):
                activation = layer.activate(current[i])
                layer_outputs.append(activation)

            current = xp.array(layer_outputs)
            self.last_activations.append(current.copy())

        # Output layer (linear)
        output = current @ self.output_weights + self.output_bias

        if not is_batch:
            output = output.squeeze(0)

        return output

    def learn(
        self,
        x: xp.ndarray,
        target: Optional[xp.ndarray] = None,
        reward: float = 0.0,
        learning_rule: str = 'oja'
    ) -> Dict[str, Any]:
        """
        Apply learning and potentially structural changes.

        Args:
            x: Input array
            target: Optional target output for supervised learning
            reward: Reward signal for reinforcement-style learning
            learning_rule: Learning rule to use ('oja', 'hebbian', 'stdp')

        Returns:
            Dictionary with learning results and structural changes
        """
        self.learn_count += 1

        # Forward pass if not already done
        if self.last_input is None or not xp.allclose(self.last_input.flatten()[:len(x.flatten())], x.flatten()):
            output = self.forward(x)
        else:
            output = self.last_activations[-1] @ self.output_weights + self.output_bias

        # Calculate reconstruction error if target provided
        if target is not None:
            self.reconstruction_error = float(xp.mean((output - target) ** 2))
        else:
            self.reconstruction_error = 0.0

        result = {
            'reconstruction_error': self.reconstruction_error,
            'neurons_added': [],
            'neurons_pruned': [],
            'layers_changed': [],
        }

        # Handle batched input
        if len(x.shape) > 1:
            x = x[0]  # Use first sample for learning

        # Apply learning to each layer
        current_input = x
        for layer_idx, layer in enumerate(self.layers):
            # Get this layer's activation
            if layer_idx < len(self.last_activations):
                activation = self.last_activations[layer_idx]
                if len(activation.shape) > 1:
                    activation = activation[0]
            else:
                activation = layer.activations

            # Apply the learning rule
            layer.learn(current_input, learning_rule=learning_rule)

            # Apply homeostasis
            layer.apply_homeostasis()

            # Check for structural changes if enabled
            if self.enable_structural_plasticity:
                structural_result = self._apply_structural_plasticity(
                    layer_idx, activation, reward
                )

                if structural_result['neurons_added'] > 0:
                    result['neurons_added'].append({
                        'layer': layer_idx,
                        'count': structural_result['neurons_added']
                    })

                if structural_result['neurons_pruned'] > 0:
                    result['neurons_pruned'].append({
                        'layer': layer_idx,
                        'count': structural_result['neurons_pruned'],
                        'indices': structural_result.get('pruned_indices', [])
                    })

                if structural_result['size_changed']:
                    result['layers_changed'].append(layer_idx)

            # Use this layer's output as next layer's input
            current_input = layer.activations

        # Update output weights if target provided
        if target is not None:
            self._update_output_weights(target, reward)

        return result

    def _apply_structural_plasticity(
        self,
        layer_idx: int,
        activation: xp.ndarray,
        reward: float
    ) -> Dict[str, Any]:
        """
        Apply structural plasticity to a layer.

        Args:
            layer_idx: Index of the layer
            activation: Current activation pattern
            reward: Reward signal

        Returns:
            Dictionary with structural changes
        """
        layer = self.layers[layer_idx]
        plasticity = self.plasticity_managers[layer_idx]

        # Get layer weights for plasticity analysis
        weights = layer.get_weight_matrix()

        # Update plasticity manager
        plasticity_result = plasticity.update(
            activity=activation,
            weights=weights,
            reconstruction_error=self.reconstruction_error * (1.0 - reward)
        )

        result = {
            'size_changed': False,
            'neurons_added': 0,
            'neurons_pruned': 0,
            'pruned_indices': []
        }

        # Handle neuron growth
        if plasticity_result['grown'] > 0:
            neurons_to_add = plasticity_result['grown']

            for _ in range(neurons_to_add):
                new_idx = layer.add_neuron()

                # Record event
                self._record_event(StructuralEvent(
                    event_type=StructuralEventType.NEURON_ADDED,
                    timestamp=time.time(),
                    layer_index=layer_idx,
                    neuron_index=new_idx,
                    details={'reason': plasticity_result.get('reason', 'growth')}
                ))

            result['neurons_added'] = neurons_to_add
            result['size_changed'] = True
            self.total_neurons_added += neurons_to_add

            # Update output weights if this is the last layer
            if layer_idx == len(self.layers) - 1:
                self._resize_output_weights(layer.layer_size)

            # Update next layer's input size if not the last layer
            if layer_idx < len(self.layers) - 1:
                self._resize_layer_input(layer_idx + 1, layer.layer_size)

            logger.info(f"Layer {layer_idx}: Added {neurons_to_add} neurons, new size: {layer.layer_size}")

        # Handle neuron pruning
        if plasticity_result['pruned'] > 0:
            prune_indices = plasticity_result.get('prune_indices', [])

            # Actually remove neurons (by replacing with fresh ones)
            for idx in sorted(prune_indices, reverse=True):
                # Record event before pruning
                self._record_event(StructuralEvent(
                    event_type=StructuralEventType.NEURON_PRUNED,
                    timestamp=time.time(),
                    layer_index=layer_idx,
                    neuron_index=idx,
                    details={'utility': float(plasticity.utility_scores[idx]) if idx < len(plasticity.utility_scores) else 0}
                ))

                # Replace with a fresh neuron (effective pruning + regrowth)
                layer.replace_neuron(idx)

            result['neurons_pruned'] = len(prune_indices)
            result['pruned_indices'] = prune_indices
            result['size_changed'] = True
            self.total_neurons_pruned += len(prune_indices)

            logger.info(f"Layer {layer_idx}: Pruned {len(prune_indices)} neurons")

        return result

    def _resize_output_weights(self, new_hidden_size: int) -> None:
        """Resize output weights when last hidden layer changes size."""
        old_size = self.output_weights.shape[0]

        if new_hidden_size == old_size:
            return

        new_weights = xp.random.randn(new_hidden_size, self.output_dim).astype(xp.float32) * 0.1

        # Copy existing weights
        copy_size = min(old_size, new_hidden_size)
        new_weights[:copy_size, :] = self.output_weights[:copy_size, :]

        self.output_weights = new_weights

    def _resize_layer_input(self, layer_idx: int, new_input_size: int) -> None:
        """Resize a layer's input connections when previous layer changes."""
        layer = self.layers[layer_idx]

        if layer.input_size == new_input_size:
            return

        # Create new neurons with correct input size
        old_neurons = layer.neurons
        layer.input_size = new_input_size

        for i, neuron in enumerate(old_neurons):
            # Create new neuron with new input size
            new_weights = xp.random.randn(new_input_size).astype(xp.float32) * 0.1

            # Copy existing weights where possible
            copy_size = min(neuron.input_size, new_input_size)
            new_weights[:copy_size] = neuron.weights[:copy_size]

            layer.neurons[i] = Neuron(
                input_size=new_input_size,
                neuron_id=i,
                learning_rate=neuron.learning_rate,
                threshold=neuron.threshold,
                initial_weights=new_weights
            )

        # Update lateral weights
        if layer.lateral_weights is not None:
            layer.lateral_weights = layer._initialize_lateral_weights()

    def _update_output_weights(self, target: xp.ndarray, reward: float) -> None:
        """Update output weights using delta rule with reward modulation."""
        if len(self.last_activations) == 0:
            return

        # Get last hidden activation
        hidden = self.last_activations[-1]
        if len(hidden.shape) > 1:
            hidden = hidden[0]

        # Calculate output
        output = hidden @ self.output_weights + self.output_bias

        # Calculate error
        error = target - output

        # Modulate learning by reward (higher reward = more learning)
        effective_lr = self.learning_rate * (0.5 + 0.5 * max(0, reward))

        # Delta rule update
        delta_w = effective_lr * xp.outer(hidden, error)
        delta_b = effective_lr * error

        self.output_weights += delta_w
        self.output_bias += delta_b

    def _record_event(self, event: StructuralEvent) -> None:
        """Record a structural event for visualization."""
        self.structural_events.append(event)

        # Limit event history
        if len(self.structural_events) > self.max_events:
            self.structural_events = self.structural_events[-self.max_events:]

    def add_layer(self, size: int, position: int = -1) -> int:
        """
        Add a new layer to the network.

        Args:
            size: Number of neurons in the new layer
            position: Position to insert (-1 for end)

        Returns:
            Index of the new layer
        """
        if position == -1:
            position = len(self.layers)

        # Determine input and output sizes
        if position == 0:
            input_size = self.input_dim
        else:
            input_size = self.layers[position - 1].layer_size

        # Create new layer
        new_layer = NeuralLayer(
            input_size=input_size,
            layer_size=size,
            name=f"hidden_{position}",
            learning_rate=self.learning_rate,
            k_winners=max(1, size // 4),
        )

        # Create plasticity manager
        new_plasticity = StructuralPlasticity(
            initial_size=size,
            max_size=self.max_neurons_per_layer,
            min_size=self.min_neurons_per_layer,
            growth_threshold=self.growth_threshold,
            prune_threshold=self.prune_threshold,
        )

        # Insert
        self.layers.insert(position, new_layer)
        self.plasticity_managers.insert(position, new_plasticity)

        # Update subsequent layers' input sizes
        if position < len(self.layers) - 1:
            self._resize_layer_input(position + 1, size)

        # Update output weights if added at end
        if position == len(self.layers) - 1:
            self._resize_output_weights(size)

        # Record event
        self._record_event(StructuralEvent(
            event_type=StructuralEventType.LAYER_ADDED,
            timestamp=time.time(),
            layer_index=position,
            details={'size': size}
        ))

        self.total_layers_added += 1

        # Rename layers
        for i, layer in enumerate(self.layers):
            layer.name = f"hidden_{i}"

        logger.info(f"Added layer at position {position} with {size} neurons")

        return position

    def get_structure(self) -> Dict[str, Any]:
        """
        Get complete current network structure for visualization.

        Returns:
            Dictionary with full network structure
        """
        layers_info = []

        for i, layer in enumerate(self.layers):
            plasticity = self.plasticity_managers[i]

            # Get neuron-level information
            neurons_info = []
            for j, neuron in enumerate(layer.neurons):
                neuron_state = neuron.get_state()
                neurons_info.append({
                    'id': j,
                    'activation': float(neuron.activation),
                    'threshold': float(neuron.threshold),
                    'weight_norm': neuron_state['weights_norm'],
                    'is_winner': neuron.is_winner,
                    'recent_activity': float(neuron.recent_mean_activation),
                })

            # Get connections (weight matrix)
            weight_matrix = layer.get_weight_matrix()

            layers_info.append({
                'index': i,
                'name': layer.name,
                'size': layer.layer_size,
                'input_size': layer.input_size,
                'neurons': neurons_info,
                'mean_activation': float(xp.mean(layer.activations)),
                'sparsity': float(xp.mean(layer.activations > 0)),
                'weight_stats': {
                    'mean': float(xp.mean(weight_matrix)),
                    'std': float(xp.std(weight_matrix)),
                    'min': float(xp.min(weight_matrix)),
                    'max': float(xp.max(weight_matrix)),
                },
                'plasticity_stats': plasticity.get_stats(),
            })

        # Count total connections
        total_connections = sum(
            layer.layer_size * layer.input_size for layer in self.layers
        )
        total_connections += self.output_weights.shape[0] * self.output_weights.shape[1]

        # Get recent events (last 50)
        recent_events = self.structural_events[-50:]

        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_layers': len(self.layers),
            'layers': layers_info,
            'total_neurons': sum(l.layer_size for l in self.layers),
            'total_connections': total_connections,
            'recent_events': [
                {
                    'type': e.event_type.value,
                    'timestamp': e.timestamp,
                    'layer': e.layer_index,
                    'neuron': e.neuron_index,
                    'details': e.details
                }
                for e in recent_events
            ],
            'stats': {
                'forward_count': self.forward_count,
                'learn_count': self.learn_count,
                'total_neurons_added': self.total_neurons_added,
                'total_neurons_pruned': self.total_neurons_pruned,
                'total_layers_added': self.total_layers_added,
                'reconstruction_error': self.reconstruction_error,
            }
        }

    def get_recent_structural_changes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent structural changes for animation.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of recent structural events
        """
        recent = self.structural_events[-limit:]
        return [
            {
                'type': e.event_type.value,
                'timestamp': e.timestamp,
                'layer': e.layer_index,
                'neuron': e.neuron_index,
                'details': e.details
            }
            for e in recent
        ]

    def get_layer_activations(self) -> List[xp.ndarray]:
        """Get the activations from the last forward pass."""
        return [to_cpu(a) for a in self.last_activations]

    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            'num_layers': len(self.layers),
            'layer_sizes': [l.layer_size for l in self.layers],
            'total_neurons': sum(l.layer_size for l in self.layers),
            'forward_count': self.forward_count,
            'learn_count': self.learn_count,
            'total_neurons_added': self.total_neurons_added,
            'total_neurons_pruned': self.total_neurons_pruned,
            'total_layers_added': self.total_layers_added,
            'reconstruction_error': self.reconstruction_error,
            'structural_plasticity_enabled': self.enable_structural_plasticity,
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize the network for saving."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'learning_rate': self.learning_rate,
            'growth_threshold': self.growth_threshold,
            'prune_threshold': self.prune_threshold,
            'max_neurons_per_layer': self.max_neurons_per_layer,
            'min_neurons_per_layer': self.min_neurons_per_layer,
            'layer_sizes': [l.layer_size for l in self.layers],
            'output_weights': to_cpu(self.output_weights).tolist(),
            'output_bias': to_cpu(self.output_bias).tolist(),
            'stats': self.get_stats(),
            'plasticity_states': [p.serialize() for p in self.plasticity_managers],
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'SelfOrganizingNetwork':
        """Create a network from serialized data."""
        network = cls(
            input_dim=data['input_dim'],
            initial_layer_sizes=data['layer_sizes'],
            output_dim=data['output_dim'],
            learning_rate=data['learning_rate'],
            growth_threshold=data['growth_threshold'],
            prune_threshold=data['prune_threshold'],
            max_neurons_per_layer=data['max_neurons_per_layer'],
            min_neurons_per_layer=data['min_neurons_per_layer'],
        )

        # Restore output weights
        network.output_weights = xp.array(data['output_weights'], dtype=xp.float32)
        network.output_bias = xp.array(data['output_bias'], dtype=xp.float32)

        # Restore plasticity states
        for i, pstate in enumerate(data.get('plasticity_states', [])):
            if i < len(network.plasticity_managers):
                network.plasticity_managers[i] = StructuralPlasticity.deserialize(pstate)

        return network
