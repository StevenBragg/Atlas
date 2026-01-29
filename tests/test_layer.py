"""
Comprehensive tests for the NeuralLayer class (core/layer.py).
Tests cover initialization, activation (feedforward, k-WTA, lateral competition),
learning rules (Hebbian, Oja, decorrelation, learning rate regulation),
homeostatic plasticity, structural operations (add/replace/prune),
recurrent connections, and state/inspection methods.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.layer import NeuralLayer


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
class TestNeuralLayerInitialization(unittest.TestCase):
    """Test NeuralLayer construction and initial state."""

    def setUp(self):
        xp.random.seed(42)

    def test_basic_params(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5, name="test_layer",
            learning_rate=0.02, initial_threshold=0.3,
            target_activation=0.15, homeostatic_factor=0.02,
            k_winners=3,
            lateral_inhibition_strength=0.2,
            lateral_excitation_strength=0.08,
            homeostatic_adaptation_rate=0.005,
            weight_regularization=0.001,
        )
        self.assertEqual(layer.input_size, 10)
        self.assertEqual(layer.layer_size, 5)
        self.assertEqual(layer.name, "test_layer")
        self.assertAlmostEqual(layer.learning_rate, 0.02)
        self.assertAlmostEqual(layer.threshold, 0.3)
        self.assertAlmostEqual(layer.target_activation, 0.15)
        self.assertAlmostEqual(layer.homeostatic_factor, 0.02)
        self.assertEqual(layer.k_winners, 3)
        self.assertAlmostEqual(layer.lateral_inhibition_strength, 0.2)
        self.assertAlmostEqual(layer.lateral_excitation_strength, 0.08)
        self.assertAlmostEqual(layer.homeostatic_adaptation_rate, 0.005)
        self.assertAlmostEqual(layer.weight_regularization, 0.001)

    def test_neuron_creation_count_and_input_size(self):
        layer = NeuralLayer(input_size=8, layer_size=4)
        self.assertEqual(len(layer.neurons), 4)
        for neuron in layer.neurons:
            self.assertEqual(neuron.input_size, 8)
            self.assertEqual(len(neuron.weights), 8)

    def test_neuron_ids_sequential(self):
        layer = NeuralLayer(input_size=6, layer_size=4)
        ids = [n.id for n in layer.neurons]
        self.assertEqual(ids, [0, 1, 2, 3])

    def test_neurons_inherit_layer_params(self):
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            learning_rate=0.05, initial_threshold=0.4,
            target_activation=0.2, homeostatic_factor=0.03,
        )
        for neuron in layer.neurons:
            self.assertAlmostEqual(neuron.learning_rate, 0.05)
            self.assertAlmostEqual(neuron.threshold, 0.4)
            self.assertAlmostEqual(neuron.target_activation, 0.2)
            self.assertAlmostEqual(neuron.homeostatic_factor, 0.03)

    def test_lateral_weights_shape_and_zero_diagonal(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            lateral_inhibition_strength=0.1,
            lateral_excitation_strength=0.05,
        )
        self.assertIsNotNone(layer.lateral_weights)
        self.assertEqual(layer.lateral_weights.shape, (5, 5))
        for i in range(5):
            self.assertAlmostEqual(float(layer.lateral_weights[i, i]), 0.0)

    def test_lateral_weights_inhibitory_off_diagonal(self):
        layer = NeuralLayer(
            input_size=10, layer_size=4,
            lateral_inhibition_strength=0.2,
            lateral_excitation_strength=0.0,
        )
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertLess(float(layer.lateral_weights[i, j]), 0.0)

    def test_lateral_weights_disabled_when_strengths_zero(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        self.assertIsNone(layer.lateral_weights)

    def test_initial_activations_zero(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        self.assertTrue(xp.allclose(layer.activations, 0.0))
        self.assertTrue(xp.allclose(layer.activations_raw, 0.0))

    def test_correlation_matrix_initialized(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        self.assertEqual(layer.correlation_matrix.shape, (5, 5))

    def test_momentum_buffer_created(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        self.assertEqual(len(layer.momentum_buffer), 5)
        for buf in layer.momentum_buffer:
            self.assertEqual(len(buf), 10)

    def test_recurrent_weights_initially_none(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        self.assertIsNone(layer.recurrent_weights)

    def test_default_k_winners_zero(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        self.assertEqual(layer.k_winners, 0)

    def test_empty_histories(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        self.assertEqual(len(layer.mean_activation_history), 0)
        self.assertEqual(len(layer.sparsity_history), 0)
        self.assertEqual(layer.last_winners, [])


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------
class TestNeuralLayerActivation(unittest.TestCase):
    """Test feedforward activation, k-WTA, lateral competition, validation."""

    def setUp(self):
        xp.random.seed(42)

    # -- helpers --
    def _make_simple_layer(self, layer_size=5, threshold=0.0):
        """Layer with no competition (pure threshold activation)."""
        return NeuralLayer(
            input_size=10, layer_size=layer_size,
            initial_threshold=threshold,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
            k_winners=0,
        )

    def _make_kwta_layer(self, layer_size=5, k=2, threshold=0.0):
        """Layer with k-winners-take-all competition."""
        return NeuralLayer(
            input_size=10, layer_size=layer_size,
            initial_threshold=threshold,
            k_winners=k,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )

    def _set_distinct_weights(self, layer):
        """Give each neuron a unique 2-hot weight vector."""
        for i, neuron in enumerate(layer.neurons):
            w = xp.zeros(10)
            w[i * 2] = 1.0
            w[i * 2 + 1] = 1.0
            neuron.weights = neuron._normalize_weights(w)

    # -- tests --
    def test_feedforward_activation_returns_correct_shape(self):
        layer = self._make_simple_layer()
        inputs = layer.neurons[0].weights.copy()
        result = layer.activate(inputs)
        self.assertEqual(len(result), 5)

    def test_feedforward_activation_aligned_input(self):
        layer = self._make_simple_layer()
        # Neuron 0's own weight vector guarantees dot-product = 1.0
        inputs = layer.neurons[0].weights.copy()
        result = layer.activate(inputs)
        self.assertGreater(float(result[0]), 0.0)

    def test_feedforward_activation_scaled_input(self):
        layer = self._make_simple_layer()
        inputs = layer.neurons[0].weights.copy() * 5.0
        result = layer.activate(inputs)
        # Dot product = 5.0 > threshold 0.0
        self.assertGreater(float(result[0]), 0.0)

    def test_k_winners_take_all_exactly_k_active(self):
        k = 2
        layer = self._make_kwta_layer(k=k)
        self._set_distinct_weights(layer)
        # Input graded so neuron 0 > neuron 1 > ... > neuron 4
        inputs = xp.array([5.0, 5.0, 3.0, 3.0, 1.0, 1.0, 0.5, 0.5, 0.1, 0.1])
        result = layer.activate(inputs)
        num_active = int(xp.sum(result > 0))
        self.assertEqual(num_active, k)

    def test_k_winners_selects_strongest(self):
        k = 2
        layer = self._make_kwta_layer(k=k)
        self._set_distinct_weights(layer)
        inputs = xp.array([5.0, 5.0, 3.0, 3.0, 1.0, 1.0, 0.5, 0.5, 0.1, 0.1])
        result = layer.activate(inputs)
        # Neurons 0 and 1 should be the winners
        self.assertGreater(float(result[0]), 0.0)
        self.assertGreater(float(result[1]), 0.0)
        # All others should be zero
        for idx in [2, 3, 4]:
            self.assertAlmostEqual(float(result[idx]), 0.0)

    def test_k_winners_tracks_winners(self):
        k = 2
        layer = self._make_kwta_layer(k=k)
        self._set_distinct_weights(layer)
        inputs = xp.array([5.0, 5.0, 3.0, 3.0, 1.0, 1.0, 0.5, 0.5, 0.1, 0.1])
        layer.activate(inputs)
        self.assertEqual(len(layer.last_winners), k)
        self.assertIn(0, layer.last_winners)
        self.assertIn(1, layer.last_winners)

    def test_k_winners_marks_neuron_is_winner(self):
        k = 2
        layer = self._make_kwta_layer(k=k)
        self._set_distinct_weights(layer)
        inputs = xp.array([5.0, 5.0, 3.0, 3.0, 1.0, 1.0, 0.5, 0.5, 0.1, 0.1])
        layer.activate(inputs)
        self.assertTrue(layer.neurons[0].is_winner)
        self.assertTrue(layer.neurons[1].is_winner)
        self.assertFalse(layer.neurons[4].is_winner)

    def test_lateral_competition_produces_output(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            initial_threshold=0.0,
            lateral_inhibition_strength=0.1,
            lateral_excitation_strength=0.0,
            k_winners=0,
        )
        # Strong aligned input ensures at least some survive inhibition
        inputs = layer.neurons[0].weights.copy() * 10.0
        result = layer.activate(inputs)
        self.assertEqual(len(result), 5)
        self.assertTrue(xp.any(result > 0))

    def test_lateral_competition_marks_winners(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            initial_threshold=0.0,
            lateral_inhibition_strength=0.1,
            lateral_excitation_strength=0.0,
            k_winners=0,
        )
        inputs = layer.neurons[0].weights.copy() * 10.0
        layer.activate(inputs)
        self.assertGreater(len(layer.last_winners), 0)

    def test_input_size_validation(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        with self.assertRaises(ValueError):
            layer.activate(xp.ones(7))

    def test_activation_history_tracking(self):
        layer = self._make_simple_layer()
        inputs = layer.neurons[0].weights.copy()
        layer.activate(inputs)
        self.assertEqual(len(layer.sparsity_history), 1)
        self.assertEqual(len(layer.mean_activation_history), 1)
        layer.activate(inputs)
        self.assertEqual(len(layer.sparsity_history), 2)
        self.assertEqual(len(layer.mean_activation_history), 2)

    def test_high_threshold_blocks_activation(self):
        layer = self._make_simple_layer(threshold=100.0)
        inputs = xp.ones(10) * 0.1
        result = layer.activate(inputs)
        self.assertAlmostEqual(float(xp.sum(result)), 0.0)

    def test_activations_raw_stored(self):
        layer = self._make_simple_layer()
        inputs = layer.neurons[0].weights.copy()
        layer.activate(inputs)
        # activations_raw is populated from neuron activations
        self.assertEqual(len(layer.activations_raw), 5)

    def test_return_is_copy(self):
        """Returned array should be a copy, not a reference to internal state."""
        layer = self._make_simple_layer()
        inputs = layer.neurons[0].weights.copy()
        result = layer.activate(inputs)
        result[:] = -999.0
        # Internal state should be unaffected
        self.assertFalse(xp.any(layer.activations == -999.0))


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------
class TestNeuralLayerLearning(unittest.TestCase):
    """Test Hebbian/Oja learning, decorrelation, learning rate regulation."""

    def setUp(self):
        xp.random.seed(42)

    def _make_active_layer(self, input_size=10, layer_size=3):
        """Create a layer, activate it with an aligned input, and return
        (layer, learning_inputs) where learning_inputs is a *different*
        direction so that Hebbian/Oja updates actually shift the weight
        vector."""
        layer = NeuralLayer(
            input_size=input_size, layer_size=layer_size,
            initial_threshold=0.0,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
            k_winners=0,
            learning_rate=0.1,
            weight_regularization=0.0,
        )
        # Activate with aligned input so neurons fire
        aligned = layer.neurons[0].weights.copy() * 5.0
        layer.activate(aligned)
        # Return a *non-proportional* input for learning so that the
        # weight update does not just rescale the existing direction.
        learn_inputs = xp.zeros(input_size)
        learn_inputs[0] = 5.0
        learn_inputs[1] = 5.0
        return layer, learn_inputs

    def test_hebbian_learning_changes_weights(self):
        layer, learn_inputs = self._make_active_layer()
        weights_before = [n.weights.copy() for n in layer.neurons]
        layer.learn(learn_inputs, learning_rule='hebbian')
        any_changed = False
        for i, neuron in enumerate(layer.neurons):
            if float(layer.activations[i]) > 0:
                if not xp.allclose(neuron.weights, weights_before[i], atol=1e-10):
                    any_changed = True
                    break
        self.assertTrue(any_changed, "Hebbian learning should modify active neurons")

    def test_oja_learning_changes_weights(self):
        layer, learn_inputs = self._make_active_layer()
        weights_before = [n.weights.copy() for n in layer.neurons]
        layer.learn(learn_inputs, learning_rule='oja')
        any_changed = False
        for i, neuron in enumerate(layer.neurons):
            if float(layer.activations[i]) > 0:
                if not xp.allclose(neuron.weights, weights_before[i], atol=1e-10):
                    any_changed = True
                    break
        self.assertTrue(any_changed, "Oja learning should modify active neurons")

    def test_oja_keeps_near_unit_norm(self):
        """With a small learning rate, Oja's rule keeps weights near unit norm."""
        xp.random.seed(42)
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            initial_threshold=0.0,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
            k_winners=0,
            learning_rate=0.001,   # small rate so single step stays near 1.0
            weight_regularization=0.0,
        )
        aligned = layer.neurons[0].weights.copy() * 5.0
        layer.activate(aligned)
        learn_inputs = xp.zeros(10)
        learn_inputs[0] = 5.0
        learn_inputs[1] = 5.0
        layer.learn(learn_inputs, learning_rule='oja')
        for neuron in layer.neurons:
            norm = float(xp.linalg.norm(neuron.weights))
            self.assertAlmostEqual(norm, 1.0, places=1)

    def test_unknown_learning_rule_raises(self):
        layer, learn_inputs = self._make_active_layer()
        with self.assertRaises(ValueError):
            layer.learn(learn_inputs, learning_rule='unknown_rule')

    def test_weight_regularization_shrinks_weights(self):
        xp.random.seed(42)
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            initial_threshold=0.0,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
            k_winners=0,
            learning_rate=0.0001,
            weight_regularization=0.5,
        )
        inputs = layer.neurons[0].weights.copy() * 5.0
        layer.activate(inputs)
        norms_before = [float(xp.linalg.norm(n.weights)) for n in layer.neurons]
        layer.learn(inputs, learning_rule='oja')
        # With heavy regularization, norms should shrink for active neurons
        for i, neuron in enumerate(layer.neurons):
            if float(layer.activations[i]) > 0:
                norm_after = float(xp.linalg.norm(neuron.weights))
                self.assertLess(norm_after, norms_before[i] + 1e-6)

    def test_weight_decorrelation_modifies_correlated_weights(self):
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            initial_threshold=0.0,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
            learning_rate=0.1,
            weight_regularization=0.0,
        )
        # Use non-linear transforms of the same base sequence so vectors
        # have high Pearson correlation (>0.5) but are NOT related by a
        # constant offset.  Constant-offset vectors (e.g. arange(k, k+10))
        # have Pearson r = 1.0, making the decorrelation pressure exactly
        # anti-parallel to the weight vector, which vanishes after
        # re-normalization.  Non-linear transforms ensure a significant
        # orthogonal component that survives re-normalization.
        w0 = xp.arange(1.0, 11.0)
        w1 = xp.arange(1.0, 11.0) ** 2
        w2 = xp.sqrt(xp.arange(1.0, 11.0))
        layer.neurons[0].weights = w0 / xp.linalg.norm(w0)
        layer.neurons[1].weights = w1 / xp.linalg.norm(w1)
        layer.neurons[2].weights = w2 / xp.linalg.norm(w2)
        layer._update_weight_correlations()
        # Off-diagonal correlations should be very high
        self.assertGreater(float(layer.correlation_matrix[0, 1]), 0.5)
        weights_before = layer.neurons[0].weights.copy()
        layer._apply_weight_decorrelation(0)
        diff = float(xp.max(xp.abs(layer.neurons[0].weights - weights_before)))
        self.assertGreater(diff, 1e-10,
                           "Decorrelation should push apart correlated neurons")

    def test_decorrelation_skipped_for_single_neuron(self):
        layer = NeuralLayer(
            input_size=10, layer_size=1,
            initial_threshold=0.0,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        weights_before = layer.neurons[0].weights.copy()
        layer._apply_weight_decorrelation(0)
        self.assertTrue(xp.allclose(layer.neurons[0].weights, weights_before))

    def test_learning_rate_regulation_stable_returns_base(self):
        layer = NeuralLayer(input_size=10, layer_size=3, learning_rate=0.05)
        # No history yet -- should return base rate
        rate = layer._regulate_learning_rate()
        self.assertAlmostEqual(rate, 0.05)

    def test_learning_rate_regulation_high_variance_reduces(self):
        layer = NeuralLayer(input_size=10, layer_size=3, learning_rate=0.05)
        layer.mean_activation_history = [0.0, 1.0] * 55  # 110 entries, var > 0.1
        rate = layer._regulate_learning_rate()
        self.assertLess(rate, 0.05)

    def test_learning_rate_regulation_low_mean_increases(self):
        layer = NeuralLayer(input_size=10, layer_size=3, learning_rate=0.05)
        layer.mean_activation_history = [0.001] * 110
        rate = layer._regulate_learning_rate()
        self.assertGreater(rate, 0.05)

    def test_k_winners_skips_non_winners(self):
        """Only winner neurons get weight updates under k-WTA."""
        xp.random.seed(42)
        k = 1
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            initial_threshold=0.0,
            k_winners=k,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
            learning_rate=0.1,
            weight_regularization=0.0,
        )
        # Set distinct weights
        for i, neuron in enumerate(layer.neurons):
            w = xp.zeros(10)
            w[i * 2] = 1.0
            w[i * 2 + 1] = 1.0
            neuron.weights = neuron._normalize_weights(w)
        inputs = xp.array([5.0, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        layer.activate(inputs)
        # Neuron 0 is the winner
        self.assertTrue(layer.neurons[0].is_winner)
        loser_weights_before = [n.weights.copy() for n in layer.neurons[1:]]
        layer.learn(inputs, learning_rule='oja')
        # Non-winners should be unchanged (no regularization, no learning)
        for idx, neuron in enumerate(layer.neurons[1:]):
            self.assertTrue(
                xp.allclose(neuron.weights, loser_weights_before[idx]),
                f"Non-winner neuron {idx+1} should not have updated weights"
            )


# ---------------------------------------------------------------------------
# Homeostasis
# ---------------------------------------------------------------------------
class TestNeuralLayerHomeostasis(unittest.TestCase):
    """Test homeostatic plasticity and target adaptation."""

    def setUp(self):
        xp.random.seed(42)

    def test_threshold_increases_when_too_active(self):
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            initial_threshold=0.5,
            target_activation=0.1,
            homeostatic_factor=0.01,
        )
        # Simulate high activity in layer history
        layer.mean_activation_history = [0.5] * 20  # >> target * 1.5
        # Also give neurons enough history so their homeostasis can run
        for neuron in layer.neurons:
            neuron.activation_history = [1.0] * 20
            neuron.recent_mean_activation = 1.0
        old_threshold = layer.threshold
        layer.apply_homeostasis()
        self.assertGreater(layer.threshold, old_threshold)

    def test_threshold_decreases_when_too_inactive(self):
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            initial_threshold=0.5,
            target_activation=0.1,
            homeostatic_factor=0.01,
        )
        # Simulate very low activity
        layer.mean_activation_history = [0.001] * 20  # << target * 0.5
        for neuron in layer.neurons:
            neuron.activation_history = [0.0] * 20
            neuron.recent_mean_activation = 0.0
        old_threshold = layer.threshold
        layer.apply_homeostasis()
        self.assertLess(layer.threshold, old_threshold)

    def test_threshold_clamped_to_valid_range(self):
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            initial_threshold=0.11,
            target_activation=0.1,
            homeostatic_factor=5.0,
        )
        # Force strong decrease
        layer.mean_activation_history = [0.0] * 20
        for neuron in layer.neurons:
            neuron.activation_history = [0.0] * 20
            neuron.recent_mean_activation = 0.0
        layer.apply_homeostasis()
        self.assertGreaterEqual(layer.threshold, 0.1)
        self.assertLessEqual(layer.threshold, 2.0)

    def test_neuron_thresholds_also_adjusted(self):
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            initial_threshold=0.5,
            target_activation=0.1,
            homeostatic_factor=0.1,
        )
        for neuron in layer.neurons:
            neuron.activation_history = [1.0] * 20
            neuron.recent_mean_activation = 1.0
        old_neuron_thresholds = [n.threshold for n in layer.neurons]
        layer.apply_homeostasis()
        # At least one neuron threshold should have increased
        any_increased = any(
            layer.neurons[i].threshold > old_neuron_thresholds[i]
            for i in range(len(layer.neurons))
        )
        self.assertTrue(any_increased)

    def test_target_adaptation_decreases_when_too_many_active(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            target_activation=0.2,
            homeostatic_adaptation_rate=0.1,
        )
        # Sparsity way above ideal => decrease target
        # For layer_size=5: ideal_sparsity = min(0.3, 2/sqrt(5)) = 0.3
        # Need long_term_sparsity > 0.3 * 1.5 = 0.45
        layer.mean_activation_history = [0.5] * 110
        layer.sparsity_history = [0.9] * 110
        old_target = layer.target_activation
        layer._adapt_homeostatic_targets()
        self.assertLess(layer.target_activation, old_target)

    def test_target_adaptation_increases_when_too_few_active(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            target_activation=0.2,
            homeostatic_adaptation_rate=0.1,
        )
        # Sparsity far below ideal => increase target
        # Need long_term_sparsity < 0.3 * 0.5 = 0.15
        layer.mean_activation_history = [0.001] * 110
        layer.sparsity_history = [0.01] * 110
        old_target = layer.target_activation
        layer._adapt_homeostatic_targets()
        self.assertGreater(layer.target_activation, old_target)

    def test_target_adaptation_updates_neuron_targets(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            target_activation=0.2,
            homeostatic_adaptation_rate=0.1,
        )
        layer.mean_activation_history = [0.001] * 110
        layer.sparsity_history = [0.01] * 110
        layer._adapt_homeostatic_targets()
        for neuron in layer.neurons:
            self.assertAlmostEqual(neuron.target_activation, layer.target_activation)

    def test_target_adaptation_no_change_when_insufficient_history(self):
        layer = NeuralLayer(input_size=10, layer_size=5, target_activation=0.2)
        layer.mean_activation_history = [0.5] * 50  # only 50, need > 100
        layer.sparsity_history = [0.9] * 50
        old_target = layer.target_activation
        layer._adapt_homeostatic_targets()
        self.assertAlmostEqual(layer.target_activation, old_target)


# ---------------------------------------------------------------------------
# Structural operations
# ---------------------------------------------------------------------------
class TestNeuralLayerStructural(unittest.TestCase):
    """Test add_neuron, replace_neuron, prune_connections."""

    def setUp(self):
        xp.random.seed(42)

    def test_add_neuron_increases_size(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        old_size = layer.layer_size
        new_idx = layer.add_neuron()
        self.assertEqual(layer.layer_size, old_size + 1)
        self.assertEqual(len(layer.neurons), old_size + 1)
        self.assertEqual(new_idx, old_size)

    def test_add_neuron_extends_activations(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        layer.add_neuron()
        self.assertEqual(len(layer.activations), 6)
        self.assertEqual(len(layer.activations_raw), 6)

    def test_add_neuron_extends_lateral_weights(self):
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            lateral_inhibition_strength=0.1,
            lateral_excitation_strength=0.0,
        )
        layer.add_neuron()
        self.assertEqual(layer.lateral_weights.shape, (4, 4))
        # Diagonal still zero
        for i in range(4):
            self.assertAlmostEqual(float(layer.lateral_weights[i, i]), 0.0)

    def test_add_neuron_extends_recurrent_weights(self):
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        layer.init_recurrent_connections()
        layer.add_neuron()
        self.assertEqual(layer.recurrent_weights.shape, (4, 4))

    def test_add_neuron_extends_correlation_matrix(self):
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        layer.add_neuron()
        self.assertEqual(layer.correlation_matrix.shape, (4, 4))

    def test_add_neuron_extends_momentum_buffer(self):
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        layer.add_neuron()
        self.assertEqual(len(layer.momentum_buffer), 4)

    def test_add_neuron_with_initial_weights(self):
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        custom_w = xp.ones(10) * 2.0
        new_idx = layer.add_neuron(initial_weights=custom_w)
        # Weights should be normalised
        norm = float(xp.linalg.norm(layer.neurons[new_idx].weights))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_replace_neuron_preserves_size(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        old_size = layer.layer_size
        layer.replace_neuron(2)
        self.assertEqual(layer.layer_size, old_size)
        self.assertEqual(len(layer.neurons), old_size)

    def test_replace_neuron_resets_activation(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            initial_threshold=0.0,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        inputs = layer.neurons[0].weights.copy()
        layer.activate(inputs)
        layer.replace_neuron(0)
        self.assertAlmostEqual(float(layer.activations[0]), 0.0)
        self.assertAlmostEqual(float(layer.activations_raw[0]), 0.0)

    def test_replace_neuron_with_custom_weights(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        custom_w = xp.ones(10) * 3.0
        layer.replace_neuron(1, initial_weights=custom_w)
        norm = float(xp.linalg.norm(layer.neurons[1].weights))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_replace_neuron_invalid_index_raises(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        with self.assertRaises(ValueError):
            layer.replace_neuron(-1)
        with self.assertRaises(ValueError):
            layer.replace_neuron(5)

    def test_replace_neuron_resets_momentum_buffer(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        # Dirty the buffer
        layer.momentum_buffer[2] = xp.ones(10)
        layer.replace_neuron(2)
        self.assertTrue(xp.allclose(layer.momentum_buffer[2], 0.0))

    def test_prune_connections_removes_weak_weights(self):
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        # Craft weights with a mix of large and tiny components
        for neuron in layer.neurons:
            w = xp.array([1.0, 0.001, 0.5, 0.002, 0.3,
                          0.001, 0.4, 0.001, 0.2, 0.001])
            neuron.weights = neuron._normalize_weights(w)
        total_pruned = layer.prune_connections(threshold=0.01)
        self.assertGreater(total_pruned, 0)

    def test_prune_connections_zeros_out_pruned(self):
        layer = NeuralLayer(
            input_size=10, layer_size=1,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        w = xp.array([1.0, 0.001, 0.5, 0.002, 0.3,
                      0.001, 0.4, 0.001, 0.2, 0.001])
        layer.neurons[0].weights = layer.neurons[0]._normalize_weights(w)
        layer.prune_connections(threshold=0.01)
        # Originally-small positions should now be 0
        weights = layer.neurons[0].weights
        for idx in [1, 3, 5, 7, 9]:
            self.assertAlmostEqual(float(weights[idx]), 0.0)

    def test_prune_no_pruning_when_all_strong(self):
        layer = NeuralLayer(
            input_size=10, layer_size=3,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        # Uniform weights -- all components equal 1/sqrt(10) ~ 0.316
        for neuron in layer.neurons:
            neuron.weights = xp.ones(10) / xp.sqrt(10.0)
        total_pruned = layer.prune_connections(threshold=0.01)
        self.assertEqual(total_pruned, 0)


# ---------------------------------------------------------------------------
# Recurrent connections
# ---------------------------------------------------------------------------
class TestNeuralLayerRecurrent(unittest.TestCase):
    """Test recurrent connection init, update, and prediction."""

    def setUp(self):
        xp.random.seed(42)

    def _make_recurrent_layer(self):
        """Create layer with recurrent connections and deterministic weights."""
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            initial_threshold=0.0,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
            k_winners=0,
        )
        # All neurons same positive weights so all fire on positive input
        w = xp.ones(10) / xp.sqrt(10.0)
        for neuron in layer.neurons:
            neuron.weights = w.copy()
        return layer

    def test_init_recurrent_connections_shape(self):
        layer = self._make_recurrent_layer()
        layer.init_recurrent_connections()
        self.assertIsNotNone(layer.recurrent_weights)
        self.assertEqual(layer.recurrent_weights.shape, (5, 5))

    def test_init_recurrent_connections_zero_diagonal(self):
        layer = self._make_recurrent_layer()
        layer.init_recurrent_connections()
        for i in range(5):
            self.assertAlmostEqual(float(layer.recurrent_weights[i, i]), 0.0)

    def test_update_recurrent_connections_modifies_weights(self):
        layer = self._make_recurrent_layer()
        inputs = xp.ones(10)
        # Activate twice with distinct time_steps that are far apart
        # to avoid the refractory period (default = 3 steps).
        layer.activate(inputs, time_step=0)
        layer.activate(inputs, time_step=100)
        # After two activations, all neurons should have fired
        self.assertGreater(float(xp.sum(layer.activations > 0)), 1)
        layer.init_recurrent_connections()
        weights_before = layer.recurrent_weights.copy()
        # First call sets previous_activations
        layer.update_recurrent_connections()
        # Second call performs the actual update
        layer.update_recurrent_connections()
        diff = float(xp.max(xp.abs(layer.recurrent_weights - weights_before)))
        self.assertGreater(diff, 0.0)

    def test_update_recurrent_no_op_without_init(self):
        layer = self._make_recurrent_layer()
        # recurrent_weights is None -- should silently return
        layer.update_recurrent_connections()
        self.assertIsNone(layer.recurrent_weights)

    def test_update_recurrent_no_op_insufficient_history(self):
        layer = self._make_recurrent_layer()
        layer.init_recurrent_connections()
        weights_before = layer.recurrent_weights.copy()
        # No activations yet -- history length < 2
        layer.update_recurrent_connections()
        self.assertTrue(xp.allclose(layer.recurrent_weights, weights_before))

    def test_update_recurrent_preserves_zero_diagonal(self):
        layer = self._make_recurrent_layer()
        inputs = xp.ones(10)
        layer.activate(inputs, time_step=0)
        layer.activate(inputs, time_step=100)
        layer.init_recurrent_connections()
        layer.update_recurrent_connections()
        layer.update_recurrent_connections()
        for i in range(5):
            self.assertAlmostEqual(float(layer.recurrent_weights[i, i]), 0.0)

    def test_predict_next_activation_with_recurrent(self):
        layer = self._make_recurrent_layer()
        layer.init_recurrent_connections()
        # Set known recurrent weights and activations
        layer.recurrent_weights = xp.ones((5, 5)) * 0.1
        xp.fill_diagonal(layer.recurrent_weights, 0.0)
        layer.activations = xp.array([1.0, 1.0, 0.0, 0.0, 0.0])
        layer.threshold = 0.0
        prediction = layer.predict_next_activation()
        self.assertEqual(len(prediction), 5)
        # With threshold=0, prediction = max(0, dot(activations, weights))
        # prediction[0] = 0*0.1 + 1*0.1 + 0 + 0 + 0 = 0.1
        self.assertAlmostEqual(float(prediction[0]), 0.1, places=5)
        # prediction[2] = 1*0.1 + 1*0.1 = 0.2
        self.assertAlmostEqual(float(prediction[2]), 0.2, places=5)

    def test_predict_next_activation_without_recurrent(self):
        layer = self._make_recurrent_layer()
        # No recurrent init -- should return zeros
        prediction = layer.predict_next_activation()
        self.assertTrue(xp.allclose(prediction, 0.0))

    def test_predict_next_activation_zero_activations(self):
        layer = self._make_recurrent_layer()
        layer.init_recurrent_connections()
        # Activations are all zero (initial state)
        prediction = layer.predict_next_activation()
        self.assertTrue(xp.allclose(prediction, 0.0))


# ---------------------------------------------------------------------------
# State / inspection methods
# ---------------------------------------------------------------------------
class TestNeuralLayerState(unittest.TestCase):
    """Test get_layer_state, get_all_receptive_fields, get_weight_matrix."""

    def setUp(self):
        xp.random.seed(42)

    def test_get_layer_state_keys(self):
        layer = NeuralLayer(input_size=10, layer_size=5, name="s_layer")
        state = layer.get_layer_state()
        expected_keys = [
            'name', 'input_size', 'layer_size', 'learning_rate',
            'threshold', 'target_activation', 'activations',
            'activations_raw', 'mean_activation', 'sparsity',
            'neuron_states',
        ]
        for key in expected_keys:
            self.assertIn(key, state)

    def test_get_layer_state_values(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5, name="s_layer",
            learning_rate=0.03, initial_threshold=0.4,
            target_activation=0.12,
        )
        state = layer.get_layer_state()
        self.assertEqual(state['name'], "s_layer")
        self.assertEqual(state['input_size'], 10)
        self.assertEqual(state['layer_size'], 5)
        self.assertAlmostEqual(state['learning_rate'], 0.03)
        self.assertAlmostEqual(state['threshold'], 0.4)
        self.assertAlmostEqual(state['target_activation'], 0.12)

    def test_get_layer_state_neuron_states(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        state = layer.get_layer_state()
        self.assertEqual(len(state['neuron_states']), 5)
        for ns in state['neuron_states']:
            self.assertIn('threshold', ns)
            self.assertIn('recent_activation', ns)

    def test_get_layer_state_activations_shape(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        state = layer.get_layer_state()
        self.assertEqual(len(state['activations']), 5)
        self.assertEqual(len(state['activations_raw']), 5)

    def test_get_layer_state_after_activation(self):
        layer = NeuralLayer(
            input_size=10, layer_size=5,
            initial_threshold=0.0,
            lateral_inhibition_strength=0.0,
            lateral_excitation_strength=0.0,
        )
        inputs = layer.neurons[0].weights.copy() * 5.0
        layer.activate(inputs)
        state = layer.get_layer_state()
        self.assertGreater(state['mean_activation'], 0.0)

    def test_get_all_receptive_fields_shape(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        fields = layer.get_all_receptive_fields()
        self.assertEqual(fields.shape, (5, 10))

    def test_get_all_receptive_fields_matches_neurons(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        fields = layer.get_all_receptive_fields()
        for i, neuron in enumerate(layer.neurons):
            self.assertTrue(xp.allclose(fields[i], neuron.weights))

    def test_get_all_receptive_fields_are_copies(self):
        layer = NeuralLayer(input_size=10, layer_size=3)
        fields = layer.get_all_receptive_fields()
        fields[0, :] = 999.0
        # Original weights should be unaffected
        self.assertFalse(xp.any(layer.neurons[0].weights == 999.0))

    def test_get_weight_matrix_shape(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        W = layer.get_weight_matrix()
        self.assertEqual(W.shape, (5, 10))

    def test_get_weight_matrix_matches_neurons(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        W = layer.get_weight_matrix()
        for i, neuron in enumerate(layer.neurons):
            self.assertTrue(xp.allclose(W[i], neuron.weights))

    def test_get_weight_matrix_rows_unit_norm(self):
        layer = NeuralLayer(input_size=10, layer_size=5)
        W = layer.get_weight_matrix()
        for i in range(5):
            norm = float(xp.linalg.norm(W[i]))
            self.assertAlmostEqual(norm, 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
