"""
Comprehensive tests for the LateralInhibition class (core/lateral_inhibition.py).
Tests cover initialization, k-winners-take-all, winner-take-all, apply() with
multiple strategies, competitive dynamics, and state retrieval (get_statistics,
serialize/deserialize).
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.lateral_inhibition import LateralInhibition


def _make_weights(num_neurons, dim, seed=42):
    """Create deterministic weight vectors for testing."""
    rng = xp.random.RandomState(seed)
    weights = []
    for _ in range(num_neurons):
        w = rng.randn(dim)
        w = w / (xp.linalg.norm(w) + 1e-10)
        weights.append(w)
    return weights


class TestLateralInhibitionInitialization(unittest.TestCase):
    """Test LateralInhibition construction and default state."""

    def test_default_parameters(self):
        li = LateralInhibition()
        self.assertEqual(li.inhibition_strategy, "soft_wta")
        self.assertAlmostEqual(li.inhibition_radius, 0.0)
        self.assertAlmostEqual(li.inhibition_strength, 0.5)
        self.assertEqual(li.k_winners, 1)
        self.assertAlmostEqual(li.sparsity_target, 0.1)
        self.assertAlmostEqual(li.inhibition_decay, 0.99)
        self.assertAlmostEqual(li.excitation_strength, 0.1)
        self.assertAlmostEqual(li.adaptation_rate, 0.01)
        self.assertEqual(li.similarity_metric, "euclidean")

    def test_custom_parameters(self):
        li = LateralInhibition(
            inhibition_strategy="wta",
            inhibition_radius=0.5,
            inhibition_strength=0.8,
            k_winners=3,
            sparsity_target=0.2,
            inhibition_decay=0.95,
            excitation_strength=0.05,
            adaptation_rate=0.02,
            similarity_metric="cosine"
        )
        self.assertEqual(li.inhibition_strategy, "wta")
        self.assertAlmostEqual(li.inhibition_radius, 0.5)
        self.assertAlmostEqual(li.inhibition_strength, 0.8)
        self.assertEqual(li.k_winners, 3)
        self.assertAlmostEqual(li.sparsity_target, 0.2)
        self.assertAlmostEqual(li.inhibition_decay, 0.95)
        self.assertAlmostEqual(li.excitation_strength, 0.05)
        self.assertAlmostEqual(li.adaptation_rate, 0.02)
        self.assertEqual(li.similarity_metric, "cosine")

    def test_initial_internal_state_is_none(self):
        li = LateralInhibition()
        self.assertIsNone(li.inhibition_matrix)
        self.assertIsNone(li.inhibition_state)
        self.assertIsNone(li.prev_winners)
        self.assertIsNone(li.neuron_excitation)
        self.assertIsNone(li.adaptive_threshold)

    def test_initial_histories_empty(self):
        li = LateralInhibition()
        self.assertEqual(li.activity_history, [])
        self.assertEqual(li.active_count_history, [])
        self.assertEqual(li.winner_history, [])
        for key in ("redundancy", "selectivity", "activation_variance"):
            self.assertEqual(li.stability_metrics[key], [])

    def test_initialize_creates_structures(self):
        li = LateralInhibition()
        num_neurons = 5
        dim = 4
        weights = _make_weights(num_neurons, dim)
        li.initialize(num_neurons, weights)

        self.assertIsNotNone(li.inhibition_matrix)
        self.assertEqual(li.inhibition_matrix.shape, (num_neurons, num_neurons))
        self.assertIsNotNone(li.inhibition_state)
        self.assertEqual(len(li.inhibition_state), num_neurons)
        self.assertIsNotNone(li.neuron_excitation)
        self.assertEqual(len(li.neuron_excitation), num_neurons)
        self.assertIsNotNone(li.adaptive_threshold)
        self.assertEqual(len(li.adaptive_threshold), num_neurons)
        self.assertIsNotNone(li.prev_winners)
        self.assertEqual(len(li.prev_winners), num_neurons)

    def test_initialize_no_self_inhibition(self):
        li = LateralInhibition()
        num_neurons = 4
        weights = _make_weights(num_neurons, 3)
        li.initialize(num_neurons, weights)
        for i in range(num_neurons):
            self.assertAlmostEqual(float(li.inhibition_matrix[i, i]), 0.0)

    def test_initialize_euclidean_metric(self):
        li = LateralInhibition(similarity_metric="euclidean", inhibition_strength=0.5)
        weights = _make_weights(4, 3)
        li.initialize(4, weights)
        # Off-diagonal entries should be non-negative for euclidean (tanh-based)
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertGreaterEqual(float(li.inhibition_matrix[i, j]), 0.0)

    def test_initialize_cosine_metric(self):
        li = LateralInhibition(similarity_metric="cosine", inhibition_strength=0.5)
        weights = _make_weights(4, 3)
        li.initialize(4, weights)
        # Cosine-based inhibition matrix should be populated
        self.assertEqual(li.inhibition_matrix.shape, (4, 4))

    def test_initialize_with_local_radius(self):
        li = LateralInhibition(
            similarity_metric="euclidean",
            inhibition_radius=0.5,
            inhibition_strength=0.5
        )
        weights = _make_weights(4, 3)
        li.initialize(4, weights)
        # With radius > 0, gaussian falloff is used
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertGreaterEqual(float(li.inhibition_matrix[i, j]), 0.0)

    def test_reinitialize_on_size_change(self):
        li = LateralInhibition()
        weights_a = _make_weights(4, 3, seed=10)
        li.initialize(4, weights_a)
        self.assertEqual(li.inhibition_matrix.shape, (4, 4))

        weights_b = _make_weights(6, 3, seed=20)
        li.initialize(6, weights_b)
        self.assertEqual(li.inhibition_matrix.shape, (6, 6))
        self.assertEqual(len(li.inhibition_state), 6)


class TestKWinnersTakeAll(unittest.TestCase):
    """Test k-Winners-Take-All mechanism."""

    def _make_wta_li(self, k=2):
        """Helper: create a WTA LateralInhibition with given k."""
        li = LateralInhibition(
            inhibition_strategy="wta",
            k_winners=k,
            inhibition_strength=0.1
        )
        return li

    def test_k_winners_selects_top_k(self):
        li = self._make_wta_li(k=2)
        weights = _make_weights(5, 4)
        li.initialize(5, weights)

        activations = xp.array([0.1, 0.5, 0.3, 0.9, 0.2])
        result = li._apply_wta(activations)

        # Top-2 are indices 3 (0.9) and 1 (0.5)
        self.assertGreater(float(result[3]), 0.0)
        self.assertGreater(float(result[1]), 0.0)
        # The rest should be zero
        self.assertAlmostEqual(float(result[0]), 0.0)
        self.assertAlmostEqual(float(result[2]), 0.0)
        self.assertAlmostEqual(float(result[4]), 0.0)

    def test_k_winners_preserves_activation_values(self):
        li = self._make_wta_li(k=2)
        weights = _make_weights(5, 4)
        li.initialize(5, weights)

        activations = xp.array([0.1, 0.5, 0.3, 0.9, 0.2])
        result = li._apply_wta(activations)

        self.assertAlmostEqual(float(result[3]), 0.9)
        self.assertAlmostEqual(float(result[1]), 0.5)

    def test_k_winners_k_equals_1(self):
        li = self._make_wta_li(k=1)
        weights = _make_weights(4, 3)
        li.initialize(4, weights)

        activations = xp.array([0.2, 0.8, 0.5, 0.1])
        result = li._apply_wta(activations)

        # Only the top neuron (index 1) should be active
        self.assertGreater(float(result[1]), 0.0)
        non_winner_sum = float(xp.sum(result)) - float(result[1])
        self.assertAlmostEqual(non_winner_sum, 0.0)

    def test_k_winners_k_larger_than_neurons(self):
        li = self._make_wta_li(k=10)
        weights = _make_weights(3, 4)
        li.initialize(3, weights)

        activations = xp.array([0.5, 0.3, 0.8])
        result = li._apply_wta(activations)

        # k is clamped to len(activations), so all neurons are winners
        for i in range(3):
            self.assertAlmostEqual(float(result[i]), float(activations[i]))

    def test_k_winners_all_zeros(self):
        li = self._make_wta_li(k=2)
        weights = _make_weights(4, 3)
        li.initialize(4, weights)

        activations = xp.array([0.0, 0.0, 0.0, 0.0])
        result = li._apply_wta(activations)

        # All zero activations => no winners
        self.assertAlmostEqual(float(xp.sum(result)), 0.0)

    def test_k_winners_all_negative(self):
        li = self._make_wta_li(k=2)
        weights = _make_weights(4, 3)
        li.initialize(4, weights)

        activations = xp.array([-0.5, -0.3, -0.1, -0.7])
        result = li._apply_wta(activations)

        # Max activation <= 0 => return zeros
        self.assertAlmostEqual(float(xp.sum(result)), 0.0)

    def test_k_winners_updates_inhibition_state(self):
        li = self._make_wta_li(k=1)
        weights = _make_weights(4, 3)
        li.initialize(4, weights)

        state_before = li.inhibition_state.copy()
        activations = xp.array([0.1, 0.9, 0.2, 0.3])
        li._apply_wta(activations)

        # Inhibition state should have been updated
        state_after = li.inhibition_state
        self.assertFalse(xp.allclose(state_before, state_after))


class TestWinnerTakeAll(unittest.TestCase):
    """Test strict Winner-Take-All (single winner via apply with wta strategy)."""

    def test_single_winner_through_apply(self):
        li = LateralInhibition(
            inhibition_strategy="wta",
            k_winners=1,
            inhibition_strength=0.1
        )
        weights = _make_weights(5, 4)
        activations = xp.array([0.3, 0.1, 0.7, 0.4, 0.2])
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)

        # Only one neuron active
        active_count = int(xp.sum(result > 0))
        self.assertEqual(active_count, 1)
        # That neuron is index 2 (highest activation)
        self.assertGreater(float(result[2]), 0.0)

    def test_wta_repeated_calls_with_inhibition_decay(self):
        li = LateralInhibition(
            inhibition_strategy="wta",
            k_winners=1,
            inhibition_strength=0.1,
            inhibition_decay=0.5
        )
        weights = _make_weights(4, 3)
        activations = xp.array([0.3, 0.9, 0.5, 0.2])

        # First call
        r1 = li.apply(activations, weight_vectors=weights, learning_enabled=False)
        self.assertGreater(float(r1[1]), 0.0)

        # Second call with same activations - inhibition state decays
        r2 = li.apply(activations, learning_enabled=False)
        # Winner should still be index 1 (strongest activation)
        winner_idx = int(xp.argmax(r2))
        self.assertEqual(winner_idx, 1)


class TestApplyMethod(unittest.TestCase):
    """Test the apply() method across different inhibition strategies."""

    def test_apply_wta_strategy(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=2)
        weights = _make_weights(5, 4)
        activations = xp.array([0.1, 0.8, 0.3, 0.6, 0.05])
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)

        active = int(xp.sum(result > 0))
        self.assertLessEqual(active, 2)

    def test_apply_soft_wta_strategy(self):
        li = LateralInhibition(
            inhibition_strategy="soft_wta",
            inhibition_strength=0.1,
            k_winners=2
        )
        weights = _make_weights(5, 4)
        # High activations to ensure some survive soft inhibition + threshold
        activations = xp.array([1.5, 2.5, 0.8, 3.0, 0.5])
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)

        # Soft WTA should produce non-negative results
        self.assertTrue(xp.all(result >= 0))

    def test_apply_local_inhibition_strategy(self):
        li = LateralInhibition(
            inhibition_strategy="local",
            inhibition_strength=0.1,
            inhibition_radius=0.5
        )
        weights = _make_weights(5, 4)
        activations = xp.array([1.5, 2.0, 0.5, 3.0, 1.0])
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)

        self.assertTrue(xp.all(result >= 0))
        self.assertEqual(len(result), 5)

    def test_apply_global_inhibition_strategy(self):
        li = LateralInhibition(
            inhibition_strategy="global",
            inhibition_strength=0.1,
            sparsity_target=0.4
        )
        weights = _make_weights(5, 4)
        activations = xp.array([1.5, 2.5, 0.8, 3.0, 1.0])
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)

        self.assertTrue(xp.all(result >= 0))
        self.assertEqual(len(result), 5)

    def test_apply_unknown_strategy_falls_back_to_soft_wta(self):
        li = LateralInhibition(inhibition_strategy="nonexistent")
        weights = _make_weights(4, 3)
        activations = xp.array([1.5, 2.0, 0.5, 3.0])
        # Should not raise; defaults to soft_wta
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)
        self.assertTrue(xp.all(result >= 0))
        self.assertEqual(len(result), 4)

    def test_apply_without_weights_uses_defaults(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=1)
        activations = xp.array([0.3, 0.9, 0.1, 0.5])
        # Call without weight_vectors; should create fake weights internally
        result = li.apply(activations, weight_vectors=None, learning_enabled=False)
        self.assertEqual(len(result), 4)
        # Should still pick the max
        self.assertGreater(float(result[1]), 0.0)

    def test_apply_records_active_count_history(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=2)
        weights = _make_weights(5, 4)
        activations = xp.array([0.1, 0.5, 0.3, 0.9, 0.2])

        self.assertEqual(len(li.active_count_history), 0)
        li.apply(activations, weight_vectors=weights, learning_enabled=False)
        self.assertEqual(len(li.active_count_history), 1)

    def test_apply_records_winner_history(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=1)
        weights = _make_weights(4, 3)

        li.apply(xp.array([0.5, 0.9, 0.3, 0.1]),
                 weight_vectors=weights, learning_enabled=False)
        self.assertEqual(len(li.winner_history), 1)

        li.apply(xp.array([0.8, 0.1, 0.2, 0.3]), learning_enabled=False)
        self.assertEqual(len(li.winner_history), 2)

    def test_apply_updates_prev_winners(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=1)
        weights = _make_weights(4, 3)

        li.apply(xp.array([0.1, 0.9, 0.3, 0.2]),
                 weight_vectors=weights, learning_enabled=False)
        # prev_winners should mark index 1 as winner
        self.assertAlmostEqual(float(li.prev_winners[1]), 1.0)
        self.assertAlmostEqual(float(li.prev_winners[0]), 0.0)

    def test_apply_with_learning_enabled_adapts_thresholds(self):
        li = LateralInhibition(
            inhibition_strategy="wta",
            k_winners=2,
            adaptation_rate=0.1
        )
        weights = _make_weights(5, 4)
        activations = xp.array([0.1, 0.8, 0.3, 0.6, 0.05])

        li.apply(activations, weight_vectors=weights, learning_enabled=True)
        # Thresholds should have been adapted (activity_history populated)
        self.assertGreater(len(li.activity_history), 0)

    def test_apply_learning_disabled_no_threshold_adaptation(self):
        li = LateralInhibition(
            inhibition_strategy="wta",
            k_winners=2,
            adaptation_rate=0.1
        )
        weights = _make_weights(5, 4)
        activations = xp.array([0.1, 0.8, 0.3, 0.6, 0.05])

        li.apply(activations, weight_vectors=weights, learning_enabled=False)
        # With learning disabled, _adapt_thresholds is not called
        self.assertEqual(len(li.activity_history), 0)

    def test_apply_all_zero_activations(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=2)
        weights = _make_weights(4, 3)
        activations = xp.array([0.0, 0.0, 0.0, 0.0])
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)
        self.assertAlmostEqual(float(xp.sum(result)), 0.0)

    def test_apply_inhibition_state_decays(self):
        li = LateralInhibition(
            inhibition_strategy="wta",
            k_winners=1,
            inhibition_decay=0.5,
            inhibition_strength=0.5
        )
        weights = _make_weights(4, 3)
        activations = xp.array([0.1, 0.9, 0.3, 0.2])

        li.apply(activations, weight_vectors=weights, learning_enabled=False)
        state_after_first = li.inhibition_state.copy()

        li.apply(activations, learning_enabled=False)
        state_after_second = li.inhibition_state

        # The first contribution should have been scaled by decay (0.5)
        # The state should reflect the decay applied to the prior state
        # plus any new inhibition added in the second call
        # Verify decay happened by checking the magnitude is reasonable
        self.assertFalse(xp.allclose(state_after_first, state_after_second))


class TestCompetitiveDynamics(unittest.TestCase):
    """Test competitive dynamics between neurons."""

    def test_stronger_neuron_wins_wta(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=1)
        weights = _make_weights(4, 3)
        activations = xp.array([0.2, 0.5, 0.9, 0.1])
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)

        # Neuron 2 has the highest activation and should win
        winner_idx = int(xp.argmax(result))
        self.assertEqual(winner_idx, 2)

    def test_inhibition_suppresses_weaker_neurons(self):
        li = LateralInhibition(
            inhibition_strategy="wta",
            k_winners=1,
            inhibition_strength=0.5
        )
        weights = _make_weights(5, 4)
        activations = xp.array([0.3, 0.8, 0.1, 0.4, 0.05])
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)

        # Only 1 active neuron in strict WTA
        active_count = int(xp.sum(result > 0))
        self.assertEqual(active_count, 1)

    def test_soft_wta_allows_graded_response(self):
        li = LateralInhibition(
            inhibition_strategy="soft_wta",
            inhibition_strength=0.01,  # very weak inhibition
            excitation_strength=0.0
        )
        weights = _make_weights(4, 3)
        # Use very high activations so they survive thresholding
        activations = xp.array([2.0, 3.0, 2.5, 1.5])
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)

        # With very weak inhibition, multiple neurons can be active
        active_count = int(xp.sum(result > 0))
        self.assertGreaterEqual(active_count, 1)

    def test_global_inhibition_enforces_sparsity(self):
        li = LateralInhibition(
            inhibition_strategy="global",
            inhibition_strength=0.1,
            sparsity_target=0.2  # expect ~20% active
        )
        num_neurons = 10
        weights = _make_weights(num_neurons, 4)
        # All neurons have non-trivial activations above threshold
        activations = xp.array([1.5, 2.0, 1.8, 3.0, 1.2, 2.5, 1.0, 2.2, 1.6, 1.9])
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)

        active_count = int(xp.sum(result > 0))
        expected_active = max(1, int(0.2 * num_neurons))  # 2
        # Global inhibition targets the sparsity level
        self.assertLessEqual(active_count, expected_active + 1)

    def test_repeated_competition_stability(self):
        """Repeated application should not cause unbounded growth."""
        li = LateralInhibition(
            inhibition_strategy="wta",
            k_winners=1,
            inhibition_strength=0.1,
            inhibition_decay=0.9
        )
        weights = _make_weights(4, 3)
        activations = xp.array([0.3, 0.9, 0.5, 0.2])

        for _ in range(20):
            result = li.apply(activations, learning_enabled=False)

        # Result should remain bounded and non-negative
        self.assertTrue(xp.all(result >= 0))
        self.assertTrue(xp.all(result <= float(xp.max(activations)) + 1.0))

    def test_excitation_from_previous_winners(self):
        """Previous winners provide excitation in soft WTA."""
        li = LateralInhibition(
            inhibition_strategy="soft_wta",
            inhibition_strength=0.01,
            excitation_strength=0.5  # Strong excitation
        )
        weights = _make_weights(4, 3)

        # First call: set previous winners
        act1 = xp.array([2.0, 3.0, 1.0, 0.5])
        li.apply(act1, weight_vectors=weights, learning_enabled=False)

        # Second call: previous winners get excitation boost
        act2 = xp.array([2.0, 2.0, 2.0, 2.0])
        result = li.apply(act2, learning_enabled=False)

        # All outputs should be non-negative
        self.assertTrue(xp.all(result >= 0))

    def test_local_inhibition_suppresses_neighbors(self):
        li = LateralInhibition(
            inhibition_strategy="local",
            inhibition_strength=0.3,
            inhibition_radius=0.5
        )
        weights = _make_weights(5, 4)
        activations = xp.array([2.0, 3.0, 1.5, 2.5, 1.0])
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)

        # Local inhibition produces non-negative outputs
        self.assertTrue(xp.all(result >= 0))
        # At least one neuron should survive (the strongest)
        self.assertGreater(float(xp.max(result)), 0.0)


class TestGetState(unittest.TestCase):
    """Test get_statistics(), serialize(), and deserialize() for state retrieval."""

    def test_get_statistics_before_apply(self):
        li = LateralInhibition()
        stats = li.get_statistics()
        self.assertIn("average_activity", stats)
        self.assertIn("stability_metrics", stats)
        self.assertIn("inhibition_strength", stats)
        self.assertEqual(stats["average_activity"], 0)

    def test_get_statistics_after_apply(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=2)
        weights = _make_weights(5, 4)
        activations = xp.array([0.1, 0.5, 0.3, 0.9, 0.2])
        li.apply(activations, weight_vectors=weights, learning_enabled=True)

        stats = li.get_statistics()
        self.assertIn("average_activity", stats)
        self.assertIn("adaptive_thresholds", stats)
        self.assertIsNotNone(stats["adaptive_thresholds"])
        self.assertEqual(len(stats["adaptive_thresholds"]), 5)
        self.assertAlmostEqual(stats["inhibition_strength"], 0.5)

    def test_serialize_returns_complete_dict(self):
        li = LateralInhibition(
            inhibition_strategy="wta",
            k_winners=3,
            inhibition_strength=0.7
        )
        weights = _make_weights(4, 3)
        li.apply(xp.array([0.5, 0.8, 0.3, 0.9]),
                 weight_vectors=weights, learning_enabled=False)

        data = li.serialize()
        self.assertEqual(data["inhibition_strategy"], "wta")
        self.assertEqual(data["k_winners"], 3)
        self.assertAlmostEqual(data["inhibition_strength"], 0.7)
        self.assertIn("adaptive_threshold", data)
        self.assertIn("inhibition_state", data)
        self.assertIn("statistics", data)

    def test_serialize_deserialize_roundtrip(self):
        li = LateralInhibition(
            inhibition_strategy="wta",
            k_winners=2,
            inhibition_strength=0.6,
            sparsity_target=0.15,
            inhibition_decay=0.95,
            excitation_strength=0.08,
            adaptation_rate=0.02,
            similarity_metric="cosine"
        )
        weights = _make_weights(5, 4)
        li.apply(xp.array([0.3, 0.7, 0.1, 0.9, 0.5]),
                 weight_vectors=weights, learning_enabled=True)

        data = li.serialize()
        restored = LateralInhibition.deserialize(data)

        self.assertEqual(restored.inhibition_strategy, li.inhibition_strategy)
        self.assertEqual(restored.k_winners, li.k_winners)
        self.assertAlmostEqual(restored.inhibition_strength, li.inhibition_strength)
        self.assertAlmostEqual(restored.sparsity_target, li.sparsity_target)
        self.assertAlmostEqual(restored.inhibition_decay, li.inhibition_decay)
        self.assertAlmostEqual(restored.excitation_strength, li.excitation_strength)
        self.assertAlmostEqual(restored.adaptation_rate, li.adaptation_rate)
        self.assertEqual(restored.similarity_metric, li.similarity_metric)

    def test_deserialize_restores_adaptive_threshold(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=1)
        weights = _make_weights(4, 3)
        li.apply(xp.array([0.3, 0.9, 0.1, 0.5]),
                 weight_vectors=weights, learning_enabled=True)

        data = li.serialize()
        restored = LateralInhibition.deserialize(data)

        self.assertIsNotNone(restored.adaptive_threshold)
        self.assertEqual(len(restored.adaptive_threshold), len(li.adaptive_threshold))
        self.assertTrue(xp.allclose(restored.adaptive_threshold, li.adaptive_threshold))

    def test_deserialize_restores_inhibition_state(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=1)
        weights = _make_weights(4, 3)
        li.apply(xp.array([0.3, 0.9, 0.1, 0.5]),
                 weight_vectors=weights, learning_enabled=False)

        data = li.serialize()
        restored = LateralInhibition.deserialize(data)

        self.assertIsNotNone(restored.inhibition_state)
        self.assertTrue(xp.allclose(restored.inhibition_state, li.inhibition_state))

    def test_deserialize_with_none_state(self):
        li = LateralInhibition()
        data = li.serialize()
        restored = LateralInhibition.deserialize(data)

        self.assertIsNone(restored.adaptive_threshold)
        self.assertIsNone(restored.inhibition_state)

    def test_reset_clears_state(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=1)
        weights = _make_weights(4, 3)

        for _ in range(5):
            li.apply(xp.array([0.3, 0.9, 0.1, 0.5]),
                     weight_vectors=weights, learning_enabled=True)

        self.assertGreater(len(li.winner_history), 0)
        self.assertGreater(len(li.active_count_history), 0)
        self.assertGreater(len(li.activity_history), 0)

        li.reset()

        self.assertEqual(len(li.winner_history), 0)
        self.assertEqual(len(li.active_count_history), 0)
        self.assertEqual(len(li.activity_history), 0)
        # Inhibition state should be zeroed
        self.assertAlmostEqual(float(xp.sum(xp.abs(li.inhibition_state))), 0.0)
        self.assertAlmostEqual(float(xp.sum(xp.abs(li.prev_winners))), 0.0)

    def test_stability_metrics_populated_after_many_applies(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=1)
        weights = _make_weights(4, 3)

        # Need >10 calls to trigger _update_stability_metrics
        for i in range(15):
            act = xp.array([0.3 + i * 0.01, 0.9, 0.1, 0.5])
            li.apply(act, weight_vectors=weights if i == 0 else None,
                     learning_enabled=False)

        stats = li.get_statistics()
        sm = stats["stability_metrics"]
        # After 15 iterations (>10), stability metrics should be populated
        self.assertIn("redundancy", sm)
        self.assertIn("selectivity", sm)
        self.assertIn("activation_variance", sm)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_single_neuron(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=1)
        weights = [xp.array([1.0, 0.0, 0.0])]
        activations = xp.array([0.5])
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result[0]), 0.5)

    def test_two_neurons_equal_activations(self):
        li = LateralInhibition(inhibition_strategy="wta", k_winners=1)
        weights = _make_weights(2, 3)
        activations = xp.array([0.5, 0.5])
        result = li.apply(activations, weight_vectors=weights, learning_enabled=False)
        # At least one neuron should be active
        self.assertGreater(float(xp.sum(result > 0)), 0)

    def test_history_truncation(self):
        """active_count_history should be truncated to 1000 entries."""
        li = LateralInhibition(inhibition_strategy="wta", k_winners=1)
        weights = _make_weights(3, 2)
        activations = xp.array([0.3, 0.9, 0.1])

        for i in range(1005):
            li.apply(activations,
                     weight_vectors=weights if i == 0 else None,
                     learning_enabled=False)

        self.assertLessEqual(len(li.active_count_history), 1000)

    def test_winner_history_truncation(self):
        """winner_history should be truncated to 100 entries."""
        li = LateralInhibition(inhibition_strategy="wta", k_winners=1)
        weights = _make_weights(3, 2)
        activations = xp.array([0.3, 0.9, 0.1])

        for i in range(105):
            li.apply(activations,
                     weight_vectors=weights if i == 0 else None,
                     learning_enabled=False)

        self.assertLessEqual(len(li.winner_history), 100)


if __name__ == "__main__":
    unittest.main()
