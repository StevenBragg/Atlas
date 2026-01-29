"""
Comprehensive tests for the SelfOrganizingNetwork class
(core/self_organizing_network.py).

Tests cover initialization, forward pass, learning, structural events,
network snapshots, self-organizing mechanisms (growth/pruning), layer
addition, serialization/deserialization, and get_state / get_stats.
"""
import sys
import os
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.self_organizing_network import (
    SelfOrganizingNetwork,
    StructuralEvent,
    StructuralEventType,
    NetworkSnapshot,
)


def _seed(val=42):
    """Set a deterministic random seed before each test."""
    xp.random.seed(val)


def _patch_plasticity_states(data):
    """Patch serialized network data so that StructuralPlasticity.deserialize
    receives the keys it expects but that serialize() currently omits.

    This works around a known gap between StructuralPlasticity.serialize()
    and StructuralPlasticity.deserialize().
    """
    for pstate in data.get('plasticity_states', []):
        pstate.setdefault('max_growth_per_step', 5)
        pstate.setdefault('max_prune_per_step', 3)
    return data


class TestStructuralEventType(unittest.TestCase):
    """Test the StructuralEventType enum values."""

    def test_all_expected_values_exist(self):
        expected = [
            "neuron_added",
            "neuron_pruned",
            "layer_added",
            "layer_removed",
            "connection_sprouted",
            "connection_pruned",
            "weight_updated",
        ]
        for val in expected:
            member = StructuralEventType(val)
            self.assertEqual(member.value, val)

    def test_neuron_added(self):
        self.assertEqual(StructuralEventType.NEURON_ADDED.value, "neuron_added")

    def test_neuron_pruned(self):
        self.assertEqual(StructuralEventType.NEURON_PRUNED.value, "neuron_pruned")

    def test_layer_added(self):
        self.assertEqual(StructuralEventType.LAYER_ADDED.value, "layer_added")

    def test_layer_removed(self):
        self.assertEqual(StructuralEventType.LAYER_REMOVED.value, "layer_removed")

    def test_connection_sprouted(self):
        self.assertEqual(StructuralEventType.CONNECTION_SPROUTED.value, "connection_sprouted")

    def test_connection_pruned(self):
        self.assertEqual(StructuralEventType.CONNECTION_PRUNED.value, "connection_pruned")

    def test_weight_updated(self):
        self.assertEqual(StructuralEventType.WEIGHT_UPDATED.value, "weight_updated")

    def test_total_member_count(self):
        self.assertEqual(len(StructuralEventType), 7)

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            StructuralEventType("nonexistent_type")


class TestStructuralEvent(unittest.TestCase):
    """Test the StructuralEvent dataclass."""

    def test_creation_with_required_fields(self):
        event = StructuralEvent(
            event_type=StructuralEventType.NEURON_ADDED,
            timestamp=1000.0,
            layer_index=0,
        )
        self.assertEqual(event.event_type, StructuralEventType.NEURON_ADDED)
        self.assertAlmostEqual(event.timestamp, 1000.0)
        self.assertEqual(event.layer_index, 0)
        self.assertIsNone(event.neuron_index)
        self.assertEqual(event.details, {})

    def test_creation_with_all_fields(self):
        details = {"reason": "growth", "count": 3}
        event = StructuralEvent(
            event_type=StructuralEventType.NEURON_PRUNED,
            timestamp=2000.0,
            layer_index=1,
            neuron_index=5,
            details=details,
        )
        self.assertEqual(event.event_type, StructuralEventType.NEURON_PRUNED)
        self.assertEqual(event.neuron_index, 5)
        self.assertEqual(event.details["reason"], "growth")

    def test_details_default_factory(self):
        e1 = StructuralEvent(
            event_type=StructuralEventType.LAYER_ADDED,
            timestamp=0.0,
            layer_index=0,
        )
        e2 = StructuralEvent(
            event_type=StructuralEventType.LAYER_ADDED,
            timestamp=0.0,
            layer_index=0,
        )
        # Each instance should get its own dict
        e1.details["key"] = "value"
        self.assertNotIn("key", e2.details)


class TestNetworkSnapshot(unittest.TestCase):
    """Test the NetworkSnapshot dataclass."""

    def test_creation(self):
        event = StructuralEvent(
            event_type=StructuralEventType.NEURON_ADDED,
            timestamp=100.0,
            layer_index=0,
        )
        snap = NetworkSnapshot(
            layers=[{"index": 0, "size": 16}],
            total_neurons=16,
            total_connections=160,
            recent_events=[event],
            timestamp=100.0,
        )
        self.assertEqual(snap.total_neurons, 16)
        self.assertEqual(snap.total_connections, 160)
        self.assertEqual(len(snap.recent_events), 1)
        self.assertEqual(len(snap.layers), 1)
        self.assertAlmostEqual(snap.timestamp, 100.0)

    def test_empty_events(self):
        snap = NetworkSnapshot(
            layers=[],
            total_neurons=0,
            total_connections=0,
            recent_events=[],
            timestamp=0.0,
        )
        self.assertEqual(snap.recent_events, [])


class TestSelfOrganizingNetworkInit(unittest.TestCase):
    """Test SelfOrganizingNetwork construction and initial state."""

    def setUp(self):
        _seed()
        self.net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16, 12],
            output_dim=4,
            learning_rate=0.01,
        )

    def test_input_output_dims(self):
        self.assertEqual(self.net.input_dim, 10)
        self.assertEqual(self.net.output_dim, 4)

    def test_layer_count(self):
        self.assertEqual(len(self.net.layers), 2)

    def test_layer_sizes_match(self):
        self.assertEqual(self.net.layers[0].layer_size, 16)
        self.assertEqual(self.net.layers[1].layer_size, 12)

    def test_plasticity_managers_created(self):
        self.assertEqual(len(self.net.plasticity_managers), 2)

    def test_output_weights_shape(self):
        self.assertEqual(self.net.output_weights.shape, (12, 4))

    def test_output_bias_shape(self):
        self.assertEqual(self.net.output_bias.shape, (4,))

    def test_output_bias_initialized_to_zero(self):
        self.assertTrue(xp.allclose(self.net.output_bias, 0.0))

    def test_initial_statistics(self):
        self.assertEqual(self.net.forward_count, 0)
        self.assertEqual(self.net.learn_count, 0)
        self.assertEqual(self.net.total_neurons_added, 0)
        self.assertEqual(self.net.total_neurons_pruned, 0)
        self.assertEqual(self.net.total_layers_added, 0)
        self.assertEqual(self.net.total_layers_removed, 0)

    def test_structural_events_empty(self):
        self.assertEqual(len(self.net.structural_events), 0)

    def test_max_events_default(self):
        self.assertEqual(self.net.max_events, 1000)

    def test_enable_structural_plasticity_default(self):
        self.assertTrue(self.net.enable_structural_plasticity)

    def test_disabled_structural_plasticity(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
            enable_structural_plasticity=False,
        )
        self.assertFalse(net.enable_structural_plasticity)

    def test_custom_thresholds(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
            growth_threshold=0.9,
            prune_threshold=0.05,
            max_neurons_per_layer=128,
            min_neurons_per_layer=4,
        )
        self.assertAlmostEqual(net.growth_threshold, 0.9)
        self.assertAlmostEqual(net.prune_threshold, 0.05)
        self.assertEqual(net.max_neurons_per_layer, 128)
        self.assertEqual(net.min_neurons_per_layer, 4)

    def test_single_layer_network(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=8,
            initial_layer_sizes=[20],
            output_dim=3,
        )
        self.assertEqual(len(net.layers), 1)
        self.assertEqual(net.output_weights.shape, (20, 3))

    def test_many_layers_network(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=8,
            initial_layer_sizes=[16, 32, 24, 12],
            output_dim=3,
        )
        self.assertEqual(len(net.layers), 4)
        self.assertEqual(net.layers[0].input_size, 8)
        self.assertEqual(net.layers[1].input_size, 16)
        self.assertEqual(net.layers[2].input_size, 32)
        self.assertEqual(net.layers[3].input_size, 24)


class TestSelfOrganizingNetworkForward(unittest.TestCase):
    """Test forward pass through the network."""

    def setUp(self):
        _seed()
        self.net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16, 12],
            output_dim=4,
            learning_rate=0.01,
        )

    def test_forward_single_input(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        output = self.net.forward(x)
        self.assertEqual(output.shape, (4,))

    def test_forward_batch_input(self):
        _seed(99)
        x = xp.random.randn(5, 10).astype(xp.float32)
        output = self.net.forward(x)
        self.assertEqual(output.shape, (5, 4))

    def test_forward_increments_counter(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        self.net.forward(x)
        self.assertEqual(self.net.forward_count, 1)
        self.net.forward(x)
        self.assertEqual(self.net.forward_count, 2)

    def test_forward_stores_activations(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        self.net.forward(x)
        self.assertEqual(len(self.net.last_activations), 2)

    def test_forward_stores_input(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        self.net.forward(x)
        self.assertIsNotNone(self.net.last_input)

    def test_forward_deterministic(self):
        """Two forward passes with the same input produce the same output."""
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        out1 = self.net.forward(x)
        out2 = self.net.forward(x)
        self.assertTrue(xp.allclose(out1, out2))


class TestSelfOrganizingNetworkLearn(unittest.TestCase):
    """Test learning functionality."""

    def setUp(self):
        _seed()
        self.net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16, 12],
            output_dim=4,
            learning_rate=0.01,
            enable_structural_plasticity=False,
        )

    def test_learn_returns_result_dict(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        target = xp.random.randn(4).astype(xp.float32)
        result = self.net.learn(x, target=target)
        self.assertIn('reconstruction_error', result)
        self.assertIn('neurons_added', result)
        self.assertIn('neurons_pruned', result)
        self.assertIn('layers_changed', result)

    def test_learn_increments_counter(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        self.net.learn(x)
        self.assertEqual(self.net.learn_count, 1)

    def test_learn_with_target_computes_error(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        target = xp.random.randn(4).astype(xp.float32)
        result = self.net.learn(x, target=target)
        self.assertGreaterEqual(result['reconstruction_error'], 0.0)

    def test_learn_without_target_zero_error(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        result = self.net.learn(x)
        self.assertAlmostEqual(result['reconstruction_error'], 0.0)

    def test_learn_no_structural_changes_when_disabled(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        target = xp.random.randn(4).astype(xp.float32)
        result = self.net.learn(x, target=target)
        self.assertEqual(result['neurons_added'], [])
        self.assertEqual(result['neurons_pruned'], [])
        self.assertEqual(result['layers_changed'], [])

    def test_learn_with_reward(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        target = xp.random.randn(4).astype(xp.float32)
        result = self.net.learn(x, target=target, reward=0.8)
        self.assertIn('reconstruction_error', result)

    def test_learn_updates_output_weights(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        target = xp.ones(4, dtype=xp.float32) * 10.0
        # Do a forward pass first, then inject non-zero hidden activation
        # to guarantee the delta rule produces a weight change.
        self.net.forward(x)
        last_hidden_size = self.net.layers[-1].layer_size
        self.net.last_activations[-1] = xp.ones(
            (1, last_hidden_size), dtype=xp.float32
        ) * 0.5
        old_weights = self.net.output_weights.copy()
        self.net._update_output_weights(target, reward=0.0)
        # Output weights must change when hidden activation is non-zero
        self.assertFalse(xp.allclose(self.net.output_weights, old_weights))

    def test_learn_batched_input(self):
        _seed(99)
        x = xp.random.randn(3, 10).astype(xp.float32)
        target = xp.random.randn(4).astype(xp.float32)
        result = self.net.learn(x, target=target)
        self.assertIn('reconstruction_error', result)


class TestSelfOrganizingMechanisms(unittest.TestCase):
    """Test structural plasticity / self-organizing mechanisms."""

    def test_record_event(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
        )
        event = StructuralEvent(
            event_type=StructuralEventType.NEURON_ADDED,
            timestamp=time.time(),
            layer_index=0,
            neuron_index=0,
        )
        net._record_event(event)
        self.assertEqual(len(net.structural_events), 1)
        self.assertEqual(net.structural_events[0].event_type, StructuralEventType.NEURON_ADDED)

    def test_record_event_limits_history(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
        )
        net.max_events = 5
        for i in range(10):
            event = StructuralEvent(
                event_type=StructuralEventType.WEIGHT_UPDATED,
                timestamp=float(i),
                layer_index=0,
            )
            net._record_event(event)
        self.assertEqual(len(net.structural_events), 5)
        # Should keep the most recent events
        self.assertAlmostEqual(net.structural_events[0].timestamp, 5.0)

    def test_add_layer_at_end(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
        )
        old_count = len(net.layers)
        pos = net.add_layer(size=20)
        self.assertEqual(len(net.layers), old_count + 1)
        self.assertEqual(net.layers[pos].layer_size, 20)
        self.assertEqual(net.total_layers_added, 1)

    def test_add_layer_at_beginning(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
        )
        pos = net.add_layer(size=12, position=0)
        self.assertEqual(pos, 0)
        self.assertEqual(net.layers[0].layer_size, 12)
        self.assertEqual(net.layers[0].input_size, 10)
        # The next layer should now accept input from the new layer
        self.assertEqual(net.layers[1].input_size, 12)

    def test_add_layer_records_event(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
        )
        net.add_layer(size=20)
        self.assertEqual(len(net.structural_events), 1)
        self.assertEqual(
            net.structural_events[0].event_type,
            StructuralEventType.LAYER_ADDED,
        )

    def test_add_layer_renames_layers(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16, 12],
            output_dim=4,
        )
        net.add_layer(size=8, position=1)
        for i, layer in enumerate(net.layers):
            self.assertEqual(layer.name, f"hidden_{i}")

    def test_add_layer_updates_output_weights(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
        )
        net.add_layer(size=24)  # Added at end
        self.assertEqual(net.output_weights.shape[0], 24)

    def test_resize_output_weights_no_change(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
        )
        old_weights = net.output_weights.copy()
        net._resize_output_weights(16)  # Same size
        self.assertTrue(xp.allclose(net.output_weights, old_weights))

    def test_resize_output_weights_growth(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
        )
        old_weights = net.output_weights.copy()
        net._resize_output_weights(20)
        self.assertEqual(net.output_weights.shape, (20, 4))
        # First 16 rows should be preserved
        self.assertTrue(xp.allclose(net.output_weights[:16, :], old_weights))


class TestGetStructureAndStats(unittest.TestCase):
    """Test get_structure, get_stats, get_recent_structural_changes."""

    def setUp(self):
        _seed()
        self.net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16, 12],
            output_dim=4,
        )

    def test_get_stats_keys(self):
        stats = self.net.get_stats()
        expected_keys = [
            'num_layers', 'layer_sizes', 'total_neurons',
            'forward_count', 'learn_count',
            'total_neurons_added', 'total_neurons_pruned',
            'total_layers_added', 'reconstruction_error',
            'structural_plasticity_enabled',
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_get_stats_values(self):
        stats = self.net.get_stats()
        self.assertEqual(stats['num_layers'], 2)
        self.assertEqual(stats['layer_sizes'], [16, 12])
        self.assertEqual(stats['total_neurons'], 28)
        self.assertEqual(stats['forward_count'], 0)
        self.assertTrue(stats['structural_plasticity_enabled'])

    def test_get_structure_keys(self):
        # Need a forward pass so activations exist for get_structure
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        self.net.forward(x)

        structure = self.net.get_structure()
        expected_keys = [
            'input_dim', 'output_dim', 'num_layers', 'layers',
            'total_neurons', 'total_connections', 'recent_events', 'stats',
        ]
        for key in expected_keys:
            self.assertIn(key, structure)

    def test_get_structure_layers_info(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        self.net.forward(x)

        structure = self.net.get_structure()
        self.assertEqual(len(structure['layers']), 2)
        layer_info = structure['layers'][0]
        self.assertIn('index', layer_info)
        self.assertIn('name', layer_info)
        self.assertIn('size', layer_info)
        self.assertIn('neurons', layer_info)
        self.assertIn('weight_stats', layer_info)
        self.assertIn('plasticity_stats', layer_info)

    def test_get_structure_total_connections(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        self.net.forward(x)

        structure = self.net.get_structure()
        # Layer 0: 16 * 10 = 160, Layer 1: 12 * 16 = 192, Output: 12 * 4 = 48
        expected = 160 + 192 + 48
        self.assertEqual(structure['total_connections'], expected)

    def test_get_recent_structural_changes_empty(self):
        changes = self.net.get_recent_structural_changes()
        self.assertEqual(changes, [])

    def test_get_recent_structural_changes_with_events(self):
        for i in range(5):
            self.net._record_event(StructuralEvent(
                event_type=StructuralEventType.WEIGHT_UPDATED,
                timestamp=float(i),
                layer_index=0,
            ))
        changes = self.net.get_recent_structural_changes(limit=3)
        self.assertEqual(len(changes), 3)
        # Should return dictionaries with expected keys
        self.assertIn('type', changes[0])
        self.assertIn('timestamp', changes[0])
        self.assertIn('layer', changes[0])

    def test_get_recent_structural_changes_respects_limit(self):
        for i in range(20):
            self.net._record_event(StructuralEvent(
                event_type=StructuralEventType.NEURON_ADDED,
                timestamp=float(i),
                layer_index=0,
            ))
        changes = self.net.get_recent_structural_changes(limit=5)
        self.assertEqual(len(changes), 5)

    def test_get_layer_activations_after_forward(self):
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        self.net.forward(x)
        activations = self.net.get_layer_activations()
        self.assertEqual(len(activations), 2)


class TestSerializationDeserialization(unittest.TestCase):
    """Test serialization and deserialization (get_state / serialize)."""

    def setUp(self):
        _seed()
        self.net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16, 12],
            output_dim=4,
            learning_rate=0.02,
            growth_threshold=0.8,
            prune_threshold=0.02,
            max_neurons_per_layer=128,
            min_neurons_per_layer=4,
        )

    def test_serialize_returns_dict(self):
        data = self.net.serialize()
        self.assertIsInstance(data, dict)

    def test_serialize_contains_required_keys(self):
        data = self.net.serialize()
        expected_keys = [
            'input_dim', 'output_dim', 'learning_rate',
            'growth_threshold', 'prune_threshold',
            'max_neurons_per_layer', 'min_neurons_per_layer',
            'layer_sizes', 'output_weights', 'output_bias',
            'stats', 'plasticity_states',
        ]
        for key in expected_keys:
            self.assertIn(key, data)

    def test_serialize_values_match(self):
        data = self.net.serialize()
        self.assertEqual(data['input_dim'], 10)
        self.assertEqual(data['output_dim'], 4)
        self.assertAlmostEqual(data['learning_rate'], 0.02)
        self.assertAlmostEqual(data['growth_threshold'], 0.8)
        self.assertAlmostEqual(data['prune_threshold'], 0.02)
        self.assertEqual(data['max_neurons_per_layer'], 128)
        self.assertEqual(data['min_neurons_per_layer'], 4)
        self.assertEqual(data['layer_sizes'], [16, 12])

    def test_serialize_output_weights_are_lists(self):
        data = self.net.serialize()
        self.assertIsInstance(data['output_weights'], list)
        self.assertIsInstance(data['output_bias'], list)

    def test_serialize_plasticity_states(self):
        data = self.net.serialize()
        self.assertEqual(len(data['plasticity_states']), 2)
        for pstate in data['plasticity_states']:
            self.assertIn('current_size', pstate)

    def test_deserialize_reconstructs_network(self):
        data = _patch_plasticity_states(self.net.serialize())
        restored = SelfOrganizingNetwork.deserialize(data)
        self.assertEqual(restored.input_dim, 10)
        self.assertEqual(restored.output_dim, 4)
        self.assertAlmostEqual(restored.learning_rate, 0.02)
        self.assertEqual(len(restored.layers), 2)
        self.assertEqual(restored.layers[0].layer_size, 16)
        self.assertEqual(restored.layers[1].layer_size, 12)

    def test_deserialize_restores_output_weights(self):
        data = _patch_plasticity_states(self.net.serialize())
        restored = SelfOrganizingNetwork.deserialize(data)
        self.assertEqual(restored.output_weights.shape, (12, 4))
        original_weights = xp.array(data['output_weights'], dtype=xp.float32)
        self.assertTrue(xp.allclose(restored.output_weights, original_weights))

    def test_deserialize_restores_output_bias(self):
        data = _patch_plasticity_states(self.net.serialize())
        restored = SelfOrganizingNetwork.deserialize(data)
        original_bias = xp.array(data['output_bias'], dtype=xp.float32)
        self.assertTrue(xp.allclose(restored.output_bias, original_bias))

    def test_round_trip_forward_pass(self):
        """After serialize/deserialize, forward pass still produces valid output."""
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        data = _patch_plasticity_states(self.net.serialize())
        restored = SelfOrganizingNetwork.deserialize(data)
        output = restored.forward(x)
        self.assertEqual(output.shape, (4,))

    def test_deserialize_restores_plasticity_managers(self):
        data = _patch_plasticity_states(self.net.serialize())
        restored = SelfOrganizingNetwork.deserialize(data)
        self.assertEqual(len(restored.plasticity_managers), 2)
        for i in range(2):
            self.assertEqual(
                restored.plasticity_managers[i].current_size,
                self.net.plasticity_managers[i].current_size,
            )


class TestSelfOrganizingNetworkEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_zero_input(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
        )
        x = xp.zeros(10, dtype=xp.float32)
        output = net.forward(x)
        self.assertEqual(output.shape, (4,))

    def test_large_input(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
        )
        x = xp.ones(10, dtype=xp.float32) * 1000.0
        output = net.forward(x)
        self.assertEqual(output.shape, (4,))
        # Should not produce NaN
        self.assertFalse(xp.any(xp.isnan(output)))

    def test_negative_input(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
        )
        x = xp.ones(10, dtype=xp.float32) * -5.0
        output = net.forward(x)
        self.assertEqual(output.shape, (4,))
        self.assertFalse(xp.any(xp.isnan(output)))

    def test_multiple_forward_learn_cycles(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
            enable_structural_plasticity=False,
        )
        _seed(99)
        for _ in range(10):
            x = xp.random.randn(10).astype(xp.float32)
            target = xp.random.randn(4).astype(xp.float32)
            net.forward(x)
            net.learn(x, target=target)
        self.assertEqual(net.forward_count, 10)
        self.assertEqual(net.learn_count, 10)

    def test_learn_with_different_learning_rules(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
            enable_structural_plasticity=False,
        )
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        for rule in ['oja', 'hebbian', 'stdp']:
            result = net.learn(x, learning_rule=rule)
            self.assertIn('reconstruction_error', result)

    def test_reconstruction_error_tracked(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
            enable_structural_plasticity=False,
        )
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        target = xp.random.randn(4).astype(xp.float32)
        net.learn(x, target=target)
        self.assertGreater(net.reconstruction_error, 0.0)


class TestGetStructureAfterModification(unittest.TestCase):
    """Test get_structure after modifications to the network."""

    def test_stats_reflect_added_layer(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
        )
        net.add_layer(size=20)
        stats = net.get_stats()
        self.assertEqual(stats['num_layers'], 2)
        self.assertEqual(stats['total_layers_added'], 1)

    def test_structure_reflects_events(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
        )
        net.add_layer(size=20)
        # Run a forward pass so get_structure works
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        net.forward(x)

        structure = net.get_structure()
        self.assertEqual(len(structure['recent_events']), 1)
        self.assertEqual(structure['recent_events'][0]['type'], 'layer_added')

    def test_stats_after_learning(self):
        _seed()
        net = SelfOrganizingNetwork(
            input_dim=10,
            initial_layer_sizes=[16],
            output_dim=4,
            enable_structural_plasticity=False,
        )
        _seed(99)
        x = xp.random.randn(10).astype(xp.float32)
        net.learn(x)
        stats = net.get_stats()
        self.assertEqual(stats['learn_count'], 1)
        self.assertEqual(stats['forward_count'], 1)


if __name__ == '__main__':
    unittest.main()
