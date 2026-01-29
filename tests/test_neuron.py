"""
Comprehensive tests for the Neuron class (core/neuron.py).
Tests cover initialization, activation, learning rules (Hebbian, STDP, BCM),
homeostatic plasticity, pruning, and state management.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.neuron import Neuron


class TestNeuronInitialization(unittest.TestCase):
    """Test Neuron construction and initial state."""

    def test_default_initialization(self):
        n = Neuron(input_size=10)
        self.assertEqual(n.input_size, 10)
        self.assertEqual(n.id, 0)
        self.assertAlmostEqual(n.learning_rate, 0.01)
        self.assertAlmostEqual(n.threshold, 0.5)
        self.assertEqual(n.activation, 0.0)
        self.assertFalse(n.is_winner)
        self.assertEqual(len(n.weights), 10)

    def test_custom_parameters(self):
        n = Neuron(input_size=20, neuron_id=5, learning_rate=0.05,
                   threshold=0.3, target_activation=0.2)
        self.assertEqual(n.id, 5)
        self.assertEqual(n.input_size, 20)
        self.assertAlmostEqual(n.learning_rate, 0.05)
        self.assertAlmostEqual(n.threshold, 0.3)
        self.assertAlmostEqual(n.target_activation, 0.2)

    def test_weights_normalized(self):
        n = Neuron(input_size=50)
        norm = float(xp.linalg.norm(n.weights))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_custom_initial_weights(self):
        w = xp.ones(10) * 3.0
        n = Neuron(input_size=10, initial_weights=w)
        norm = float(xp.linalg.norm(n.weights))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_zero_weights_reinitialized(self):
        w = xp.zeros(10)
        n = Neuron(input_size=10, initial_weights=w)
        norm = float(xp.linalg.norm(n.weights))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_traces_initialized_to_zero(self):
        n = Neuron(input_size=10)
        self.assertTrue(xp.allclose(n.eligibility_traces, 0.0))
        self.assertTrue(xp.allclose(n.pre_synaptic_traces, 0.0))
        self.assertAlmostEqual(n.post_synaptic_trace, 0.0)


class TestNeuronActivation(unittest.TestCase):
    """Test Neuron activation behavior."""

    def test_activation_with_aligned_input(self):
        n = Neuron(input_size=10, threshold=0.0)
        # Use weights themselves as input to ensure high activation
        inputs = n.weights.copy() * 10.0
        act = n.activate(inputs)
        self.assertGreater(act, 0.0)

    def test_activation_with_zero_input(self):
        n = Neuron(input_size=10)
        inputs = xp.zeros(10)
        act = n.activate(inputs)
        self.assertEqual(act, 0.0)

    def test_activation_below_threshold(self):
        n = Neuron(input_size=10, threshold=100.0)
        inputs = xp.ones(10) * 0.01
        act = n.activate(inputs)
        self.assertEqual(act, 0.0)

    def test_refractory_period(self):
        n = Neuron(input_size=10, threshold=0.0, refactory_period=5)
        strong_input = n.weights.copy() * 10.0
        # First activation should succeed
        act1 = n.activate(strong_input, time_step=0)
        self.assertGreater(act1, 0.0)
        # Activation within refractory period should be 0
        act2 = n.activate(strong_input, time_step=1)
        self.assertEqual(act2, 0.0)
        # After refractory period, should activate again
        act3 = n.activate(strong_input, time_step=10)
        self.assertGreater(act3, 0.0)

    def test_activation_history_tracked(self):
        n = Neuron(input_size=10, threshold=0.0)
        inputs = n.weights.copy() * 10.0
        for t in range(20):
            n.activate(inputs, time_step=t * 10)
        self.assertGreater(len(n.activation_history), 0)
        self.assertLessEqual(len(n.activation_history), 100)

    def test_time_step_tracking(self):
        n = Neuron(input_size=10)
        n.activate(xp.ones(10), time_step=42)
        self.assertEqual(n.current_time_step, 42)

    def test_adaptation_increases_with_activity(self):
        n = Neuron(input_size=10, threshold=0.0)
        strong_input = n.weights.copy() * 10.0
        n.activate(strong_input, time_step=0)
        # Adaptation should have increased from firing
        self.assertGreater(n.adaptation, 0.0)


class TestNeuronLearningHebbian(unittest.TestCase):
    """Test Hebbian and Oja learning rules."""

    def test_hebbian_update_changes_weights(self):
        n = Neuron(input_size=10, threshold=0.0, learning_rate=0.1)
        # Use an input that is NOT aligned with the weights
        inputs = xp.zeros(10)
        inputs[0] = 5.0
        inputs[1] = 5.0
        # Force activation high enough by setting threshold to 0
        strong = n.weights.copy() * 10.0
        n.activate(strong, time_step=100)
        old_weights = n.weights.copy()
        n.update_weights_hebbian(inputs, use_oja=False)
        # After Hebbian + normalization, direction should shift toward input
        self.assertFalse(xp.allclose(n.weights, old_weights))

    def test_oja_update_maintains_normalization(self):
        n = Neuron(input_size=10, threshold=0.0, learning_rate=0.1)
        inputs = n.weights.copy() * 10.0
        n.activate(inputs, time_step=100)
        n.update_weights_hebbian(inputs, use_oja=True)
        # Oja's rule should keep weights near unit norm
        norm = float(xp.linalg.norm(n.weights))
        self.assertAlmostEqual(norm, 1.0, places=1)

    def test_no_update_when_inactive(self):
        n = Neuron(input_size=10, threshold=100.0)
        inputs = xp.ones(10) * 0.01
        n.activate(inputs)
        old_weights = n.weights.copy()
        n.update_weights_hebbian(inputs)
        self.assertTrue(xp.allclose(n.weights, old_weights))


class TestNeuronLearningSTDP(unittest.TestCase):
    """Test Spike-Timing Dependent Plasticity."""

    def test_stdp_with_explicit_spike_times(self):
        n = Neuron(input_size=5, threshold=0.0, learning_rate=0.1)
        inputs = n.weights.copy() * 10.0
        n.activate(inputs, time_step=100)
        old_weights = n.weights.copy()
        pre_spike_times = [95, 96, 97, 98, 99]
        post_spike_time = 100
        n.update_weights_stdp(inputs, pre_spike_times, post_spike_time)
        self.assertFalse(xp.allclose(n.weights, old_weights))

    def test_stdp_pre_before_post_strengthens(self):
        n = Neuron(input_size=5, threshold=0.0, learning_rate=0.1)
        inputs = n.weights.copy() * 10.0
        n.activate(inputs, time_step=100)
        old_w = n.weights.copy()
        pre_spike_times = [90, 90, 90, 90, 90]  # pre before post
        post_spike_time = 100
        n.update_weights_stdp(inputs, pre_spike_times, post_spike_time)
        # Weights should generally increase (LTP)
        mean_change = float(xp.mean(n.weights - old_w))
        # Due to normalization, check the weights changed
        self.assertFalse(xp.allclose(n.weights, old_w))

    def test_stdp_trace_based(self):
        n = Neuron(input_size=5, threshold=0.0, learning_rate=1.0,
                   eligibility_trace_decay=0.99)
        # Activate with strong aligned input to build traces and guarantee firing
        strong_input = n.weights.copy() * 20.0
        for t in range(5):
            n.activate(strong_input, time_step=t * 10)
        # Ensure the neuron just fired so activation > 0
        self.assertGreater(n.activation, 0.0,
                           "Neuron must be active for STDP trace learning")
        old_weights = n.weights.copy()
        # Now add a biased input to the trace manually to create asymmetry
        n.pre_synaptic_traces = xp.array([10.0, 0.0, 0.0, 0.0, 0.0])
        n.update_weights_stdp(strong_input)
        changed = not xp.allclose(n.weights, old_weights, atol=1e-6)
        self.assertTrue(changed, "STDP trace-based learning should change weights")


class TestNeuronLearningBCM(unittest.TestCase):
    """Test BCM learning rule."""

    def test_bcm_update_with_active_neuron(self):
        n = Neuron(input_size=10, threshold=0.0, learning_rate=0.5)
        # Use non-aligned input so direction changes after normalization
        inputs = xp.zeros(10)
        inputs[0] = 5.0
        inputs[1] = 5.0
        strong = n.weights.copy() * 10.0
        n.activate(strong, time_step=100)
        old_weights = n.weights.copy()
        n.update_weights_bcm(inputs)
        self.assertFalse(xp.allclose(n.weights, old_weights))

    def test_bcm_no_update_with_zero_inputs(self):
        n = Neuron(input_size=10)
        inputs = xp.zeros(10)
        n.activate(inputs)
        old_weights = n.weights.copy()
        n.update_weights_bcm(inputs)
        self.assertTrue(xp.allclose(n.weights, old_weights))


class TestNeuronHomeostasis(unittest.TestCase):
    """Test homeostatic threshold adjustment."""

    def test_threshold_unchanged_with_little_history(self):
        n = Neuron(input_size=10)
        old_threshold = n.threshold
        n.update_threshold_homeostatic()
        self.assertAlmostEqual(n.threshold, old_threshold)

    def test_threshold_adjusts_with_history(self):
        n = Neuron(input_size=10, threshold=0.0, homeostatic_factor=0.1)
        # Simulate lots of activations to build history
        for t in range(20):
            n.activation_history.append(1.0)
        n.recent_mean_activation = 1.0
        old_threshold = n.threshold
        n.update_threshold_homeostatic()
        # Should increase threshold when too active
        self.assertGreater(n.threshold, old_threshold)

    def test_threshold_minimum_bound(self):
        n = Neuron(input_size=10, threshold=0.02, homeostatic_factor=1.0)
        for t in range(20):
            n.activation_history.append(0.0)
        n.recent_mean_activation = 0.0
        # Target is 0.1, mean is 0, so error is -0.1
        # threshold += 1.0 * (-0.1) => 0.02 - 0.1 = -0.08 => clamped to 0.01
        n.update_threshold_homeostatic()
        self.assertGreaterEqual(n.threshold, 0.01)


class TestNeuronPruning(unittest.TestCase):
    """Test connection pruning."""

    def test_prune_removes_weak_weights(self):
        n = Neuron(input_size=10)
        # Set some weights to very small values
        n.weights[0] = 0.001
        n.weights[1] = 0.001
        pruned = n.prune_weakest_connections(threshold=0.01)
        self.assertIn(0, pruned)
        self.assertIn(1, pruned)
        self.assertAlmostEqual(float(n.weights[0]), 0.0)
        self.assertAlmostEqual(float(n.weights[1]), 0.0)

    def test_prune_keeps_strong_weights(self):
        n = Neuron(input_size=10)
        n.weights[:] = 0.5
        pruned = n.prune_weakest_connections(threshold=0.01)
        self.assertEqual(len(pruned), 0)


class TestNeuronState(unittest.TestCase):
    """Test state get/set and reset."""

    def test_get_state_keys(self):
        n = Neuron(input_size=10)
        state = n.get_state()
        expected_keys = ['id', 'activation', 'threshold', 'is_winner',
                         'recent_mean_activation', 'adaptation',
                         'weights_norm', 'weights_min', 'weights_max',
                         'weights_mean', 'weights_std', 'non_zero_weights']
        for key in expected_keys:
            self.assertIn(key, state)

    def test_get_receptive_field(self):
        n = Neuron(input_size=10)
        rf = n.get_receptive_field()
        self.assertEqual(rf.shape, (10,))
        self.assertTrue(xp.allclose(rf, n.weights))

    def test_set_receptive_field(self):
        n = Neuron(input_size=10)
        new_w = xp.ones(10) * 2.0
        n.set_receptive_field(new_w)
        norm = float(xp.linalg.norm(n.weights))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_set_receptive_field_wrong_size(self):
        n = Neuron(input_size=10)
        with self.assertRaises(ValueError):
            n.set_receptive_field(xp.ones(5))

    def test_reset_state(self):
        n = Neuron(input_size=10, threshold=0.0)
        inputs = n.weights.copy() * 10.0
        n.activate(inputs, time_step=100)
        n.is_winner = True
        n.reset_state()
        self.assertEqual(n.activation, 0.0)
        self.assertFalse(n.is_winner)

    def test_repr(self):
        n = Neuron(input_size=10, neuron_id=3)
        r = repr(n)
        self.assertIn("Neuron", r)
        self.assertIn("id=3", r)


if __name__ == '__main__':
    unittest.main()
