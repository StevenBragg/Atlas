"""
Comprehensive tests for the StabilityMechanisms class (core/stability.py).
Tests cover initialization, inhibition strategies, homeostatic plasticity,
adaptive thresholds, activity monitoring, weight normalization, diversity
adjustment, sparse activity enforcement, resize, serialization, and state
retrieval.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.stability import StabilityMechanisms, InhibitionStrategy


class TestStabilityInitialization(unittest.TestCase):
    """Test StabilityMechanisms construction and initial state."""

    def test_default_initialization(self):
        sm = StabilityMechanisms(representation_size=16)
        self.assertEqual(sm.representation_size, 16)
        self.assertAlmostEqual(sm.target_activity, 0.1)
        self.assertAlmostEqual(sm.homeostatic_rate, 0.01)
        self.assertEqual(sm.inhibition_strategy, InhibitionStrategy.KWA)
        self.assertAlmostEqual(sm.inhibition_strength, 0.5)
        # k_winners derived: max(1, int(16 * 0.1)) = 1
        self.assertEqual(sm.k_winners, max(1, int(16 * 0.1)))
        self.assertAlmostEqual(sm.normalization_threshold, 3.0)
        self.assertAlmostEqual(sm.diversity_target, 0.8)
        self.assertAlmostEqual(sm.diversity_rate, 0.01)
        self.assertAlmostEqual(sm.weight_decay, 0.0001)
        self.assertTrue(sm.weight_scaling_enabled)
        self.assertEqual(sm.update_count, 0)

    def test_custom_parameters(self):
        sm = StabilityMechanisms(
            representation_size=32,
            target_activity=0.2,
            homeostatic_rate=0.05,
            inhibition_strategy=InhibitionStrategy.WTA,
            inhibition_strength=0.8,
            k_winners=5,
            normalization_threshold=2.0,
            diversity_target=0.9,
            diversity_rate=0.02,
            adaptive_threshold_tau=0.05,
            minimum_threshold=0.05,
            maximum_threshold=2.0,
            weight_scaling_enabled=False,
            weight_decay=0.001,
        )
        self.assertEqual(sm.representation_size, 32)
        self.assertAlmostEqual(sm.target_activity, 0.2)
        self.assertAlmostEqual(sm.homeostatic_rate, 0.05)
        self.assertEqual(sm.inhibition_strategy, InhibitionStrategy.WTA)
        self.assertAlmostEqual(sm.inhibition_strength, 0.8)
        self.assertEqual(sm.k_winners, 5)
        self.assertAlmostEqual(sm.normalization_threshold, 2.0)
        self.assertAlmostEqual(sm.diversity_target, 0.9)
        self.assertAlmostEqual(sm.diversity_rate, 0.02)
        self.assertAlmostEqual(sm.adaptive_threshold_tau, 0.05)
        self.assertAlmostEqual(sm.minimum_threshold, 0.05)
        self.assertAlmostEqual(sm.maximum_threshold, 2.0)
        self.assertFalse(sm.weight_scaling_enabled)
        self.assertAlmostEqual(sm.weight_decay, 0.001)

    def test_inhibition_strategy_from_string(self):
        sm = StabilityMechanisms(
            representation_size=8,
            inhibition_strategy="winner_take_all",
        )
        self.assertEqual(sm.inhibition_strategy, InhibitionStrategy.WTA)

    def test_inhibition_strategy_from_enum(self):
        sm = StabilityMechanisms(
            representation_size=8,
            inhibition_strategy=InhibitionStrategy.MEXICAN_HAT,
        )
        self.assertEqual(sm.inhibition_strategy, InhibitionStrategy.MEXICAN_HAT)

    def test_k_winners_default_derivation(self):
        # k_winners = max(1, int(size * target_activity))
        sm = StabilityMechanisms(representation_size=100, target_activity=0.05)
        self.assertEqual(sm.k_winners, 5)

    def test_k_winners_explicit_override(self):
        sm = StabilityMechanisms(representation_size=100, target_activity=0.05, k_winners=10)
        self.assertEqual(sm.k_winners, 10)

    def test_initial_thresholds(self):
        sm = StabilityMechanisms(representation_size=10)
        expected = xp.ones(10) * 0.5
        self.assertTrue(xp.allclose(sm.thresholds, expected))

    def test_initial_homeostatic_factors(self):
        sm = StabilityMechanisms(representation_size=10)
        self.assertTrue(xp.allclose(sm.homeostatic_factors, xp.ones(10)))

    def test_initial_firing_rates_zero(self):
        sm = StabilityMechanisms(representation_size=10)
        self.assertTrue(xp.allclose(sm.firing_rates, xp.zeros(10)))

    def test_initial_scaling_factors(self):
        sm = StabilityMechanisms(representation_size=10)
        self.assertTrue(xp.allclose(sm.input_scaling_factors, xp.ones(10)))
        self.assertTrue(xp.allclose(sm.output_scaling_factors, xp.ones(10)))

    def test_initial_diversity_score(self):
        sm = StabilityMechanisms(representation_size=10)
        self.assertAlmostEqual(sm.diversity_score, 1.0)

    def test_activity_history_shape(self):
        sm = StabilityMechanisms(representation_size=12)
        self.assertEqual(sm.activity_history.shape, (100, 12))
        self.assertEqual(sm.activity_history_index, 0)
        self.assertFalse(sm.activity_history_filled)

    def test_adaptive_strategy_creates_kernel(self):
        sm = StabilityMechanisms(
            representation_size=10,
            inhibition_strategy=InhibitionStrategy.ADAPTIVE,
        )
        self.assertIsNotNone(sm.inhibition_kernel)
        self.assertEqual(sm.inhibition_kernel.shape, (10, 10))

    def test_non_adaptive_strategy_no_kernel(self):
        sm = StabilityMechanisms(
            representation_size=10,
            inhibition_strategy=InhibitionStrategy.KWA,
        )
        self.assertIsNone(sm.inhibition_kernel)


class TestInhibitionStrategies(unittest.TestCase):
    """Test the apply_inhibition method across all strategies."""

    def _make_activity(self, size=16):
        """Create a deterministic activity pattern."""
        activity = xp.zeros(size)
        # Set specific neurons active with decreasing values
        activity[0] = 1.0
        activity[3] = 0.8
        activity[7] = 0.6
        activity[11] = 0.4
        activity[14] = 0.2
        return activity

    def test_none_strategy_preserves_activity(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.NONE,
        )
        activity = self._make_activity()
        result = sm.apply_inhibition(activity)
        self.assertTrue(xp.allclose(result, activity))

    def test_wta_single_winner(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.WTA,
        )
        activity = self._make_activity()
        result = sm.apply_inhibition(activity)
        # Only the strongest neuron (index 0, value 1.0) should remain
        nonzero_count = int(xp.sum(result > 0))
        self.assertEqual(nonzero_count, 1)
        self.assertEqual(int(xp.argmax(result)), 0)

    def test_wta_preserves_energy(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.WTA,
        )
        activity = self._make_activity()
        total_before = float(xp.sum(activity))
        result = sm.apply_inhibition(activity)
        total_after = float(xp.sum(result))
        self.assertAlmostEqual(total_before, total_after, places=5)

    def test_wta_with_all_zeros(self):
        sm = StabilityMechanisms(
            representation_size=8,
            inhibition_strategy=InhibitionStrategy.WTA,
        )
        activity = xp.zeros(8)
        result = sm.apply_inhibition(activity)
        self.assertTrue(xp.allclose(result, xp.zeros(8)))

    def test_kwa_limits_active_neurons(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.KWA,
            k_winners=3,
        )
        activity = self._make_activity()
        result = sm.apply_inhibition(activity)
        active_count = int(xp.sum(result > 0))
        self.assertLessEqual(active_count, 3)

    def test_kwa_keeps_strongest(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.KWA,
            k_winners=2,
        )
        activity = self._make_activity()
        result = sm.apply_inhibition(activity)
        # The two strongest (indices 0 and 3) should be nonzero
        self.assertGreater(float(result[0]), 0)
        self.assertGreater(float(result[3]), 0)

    def test_kwa_preserves_energy(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.KWA,
            k_winners=3,
        )
        activity = self._make_activity()
        total_before = float(xp.sum(activity))
        result = sm.apply_inhibition(activity)
        total_after = float(xp.sum(result))
        self.assertAlmostEqual(total_before, total_after, places=4)

    def test_kwa_no_suppression_when_few_active(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.KWA,
            k_winners=10,
        )
        # Only 5 active neurons, below k=10 threshold
        activity = self._make_activity()
        result = sm.apply_inhibition(activity)
        # No suppression should occur, energy is normalized though
        active_before = int(xp.sum(activity > 0))
        active_after = int(xp.sum(result > 0))
        self.assertEqual(active_before, active_after)

    def test_mexican_hat_nonnegative(self):
        # Use size >= 20 so int(0.05 * size) > 0, avoiding sigma_e = 0
        size = 32
        sm = StabilityMechanisms(
            representation_size=size,
            inhibition_strategy=InhibitionStrategy.MEXICAN_HAT,
        )
        activity = xp.zeros(size)
        activity[0] = 1.0
        activity[5] = 0.8
        activity[15] = 0.6
        activity[25] = 0.4
        activity[30] = 0.2
        result = sm.apply_inhibition(activity)
        self.assertTrue(xp.all(result >= 0))

    def test_mexican_hat_preserves_energy(self):
        size = 32
        sm = StabilityMechanisms(
            representation_size=size,
            inhibition_strategy=InhibitionStrategy.MEXICAN_HAT,
        )
        activity = xp.zeros(size)
        activity[0] = 1.0
        activity[5] = 0.8
        activity[15] = 0.6
        activity[25] = 0.4
        activity[30] = 0.2
        total_before = float(xp.sum(activity))
        result = sm.apply_inhibition(activity)
        total_after = float(xp.sum(result))
        self.assertAlmostEqual(total_before, total_after, places=4)

    def test_adaptive_nonnegative(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.ADAPTIVE,
        )
        activity = self._make_activity()
        result = sm.apply_inhibition(activity)
        self.assertTrue(xp.all(result >= 0))

    def test_adaptive_preserves_energy(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.ADAPTIVE,
        )
        activity = self._make_activity()
        total_before = float(xp.sum(activity))
        result = sm.apply_inhibition(activity)
        total_after = float(xp.sum(result))
        self.assertAlmostEqual(total_before, total_after, places=4)

    def test_inhibition_does_not_mutate_input(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.KWA,
            k_winners=2,
        )
        activity = self._make_activity()
        original = activity.copy()
        sm.apply_inhibition(activity)
        self.assertTrue(xp.allclose(activity, original))


class TestApplyMethods(unittest.TestCase):
    """Test homeostasis, thresholds, and sparse activity enforcement."""

    def test_apply_homeostasis_default(self):
        sm = StabilityMechanisms(representation_size=8)
        activity = xp.array([0.5, 0.3, 0.0, 0.7, 0.1, 0.0, 0.9, 0.2])
        result = sm.apply_homeostasis(activity)
        # With default homeostatic factors of 1.0, result == activity (clipped >= 0)
        self.assertTrue(xp.allclose(result, xp.maximum(0, activity)))

    def test_apply_homeostasis_with_modified_factors(self):
        sm = StabilityMechanisms(representation_size=4)
        sm.homeostatic_factors = xp.array([2.0, 0.5, 1.0, 0.0])
        activity = xp.array([0.3, 0.6, 0.4, 0.8])
        result = sm.apply_homeostasis(activity)
        expected = xp.maximum(0, activity * sm.homeostatic_factors)
        self.assertTrue(xp.allclose(result, expected))

    def test_apply_homeostasis_nonnegative(self):
        sm = StabilityMechanisms(representation_size=4)
        sm.homeostatic_factors = xp.array([1.0, -1.0, 0.5, 2.0])
        activity = xp.array([0.5, 0.5, 0.5, 0.5])
        result = sm.apply_homeostasis(activity)
        self.assertTrue(xp.all(result >= 0))

    def test_apply_thresholds_default(self):
        sm = StabilityMechanisms(representation_size=6)
        # Thresholds start at 0.5
        activity = xp.array([1.0, 0.6, 0.5, 0.4, 0.3, 0.0])
        result = sm.apply_thresholds(activity)
        expected = xp.maximum(0, activity - 0.5)
        self.assertTrue(xp.allclose(result, expected))

    def test_apply_thresholds_zeros_below(self):
        sm = StabilityMechanisms(representation_size=4)
        sm.thresholds = xp.array([0.3, 0.3, 0.3, 0.3])
        activity = xp.array([0.1, 0.2, 0.3, 0.5])
        result = sm.apply_thresholds(activity)
        # First three are at or below threshold
        self.assertAlmostEqual(float(result[0]), 0.0)
        self.assertAlmostEqual(float(result[1]), 0.0)
        self.assertAlmostEqual(float(result[2]), 0.0)
        self.assertAlmostEqual(float(result[3]), 0.2, places=5)

    def test_enforce_sparse_activity(self):
        sm = StabilityMechanisms(representation_size=10, target_activity=0.2)
        activity = xp.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        result = sm.enforce_sparse_activity(activity)
        active_count = int(xp.sum(result > 0))
        # target_activity=0.2 => k=max(1,int(10*0.2))=2
        self.assertLessEqual(active_count, 2)

    def test_enforce_sparse_activity_custom_sparsity(self):
        sm = StabilityMechanisms(representation_size=10)
        activity = xp.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        result = sm.enforce_sparse_activity(activity, sparsity=0.3)
        active_count = int(xp.sum(result > 0))
        # sparsity=0.3 => k=max(1,int(10*0.3))=3
        self.assertLessEqual(active_count, 3)

    def test_enforce_sparse_already_sparse(self):
        sm = StabilityMechanisms(representation_size=10, target_activity=0.5)
        activity = xp.zeros(10)
        activity[0] = 0.5
        activity[1] = 0.3
        # Only 20% active, well below target of 50%
        result = sm.enforce_sparse_activity(activity)
        self.assertTrue(xp.allclose(result, activity))


class TestUpdate(unittest.TestCase):
    """Test the update method and internal state transitions."""

    def setUp(self):
        self.sm = StabilityMechanisms(representation_size=8, target_activity=0.25)

    def test_update_returns_expected_keys(self):
        pre = xp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        post = xp.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.8, 0.0, 0.3])
        result = self.sm.update(pre, post)
        self.assertIn('homeostatic_factors', result)
        self.assertIn('thresholds', result)
        self.assertIn('diversity_score', result)
        self.assertIn('input_scaling_factors', result)
        self.assertIn('output_scaling_factors', result)

    def test_update_increments_counter(self):
        pre = xp.ones(8) * 0.5
        post = xp.ones(8) * 0.1
        self.assertEqual(self.sm.update_count, 0)
        self.sm.update(pre, post)
        self.assertEqual(self.sm.update_count, 1)
        self.sm.update(pre, post)
        self.assertEqual(self.sm.update_count, 2)

    def test_update_modifies_firing_rates(self):
        pre = xp.ones(8) * 0.5
        post = xp.zeros(8)
        post[0] = 1.0
        post[4] = 0.5
        initial_rates = self.sm.firing_rates.copy()
        self.sm.update(pre, post)
        # Firing rates should have changed for active neurons
        self.assertFalse(xp.allclose(self.sm.firing_rates, initial_rates))

    def test_update_modifies_thresholds(self):
        pre = xp.ones(8) * 0.5
        post = xp.zeros(8)
        post[0] = 1.0
        # Run multiple updates so firing rates build up
        for _ in range(5):
            self.sm.update(pre, post)
        # Thresholds should no longer be uniform 0.5
        self.assertFalse(xp.allclose(self.sm.thresholds, xp.ones(8) * 0.5))

    def test_update_with_weights_modifies_scaling(self):
        sm = StabilityMechanisms(
            representation_size=8,
            normalization_threshold=0.5,
            weight_scaling_enabled=True,
        )
        pre = xp.ones(8) * 0.5
        post = xp.ones(8) * 0.1
        # Create a weight matrix with some large norms
        weights = xp.ones((8, 8)) * 5.0
        sm.update(pre, post, weights=weights)
        # Scaling factors for strong weights should have decreased
        self.assertTrue(xp.any(sm.input_scaling_factors < 1.0))

    def test_update_without_weights_preserves_scaling(self):
        sm = StabilityMechanisms(representation_size=8)
        pre = xp.ones(8) * 0.5
        post = xp.ones(8) * 0.1
        initial_input_scaling = sm.input_scaling_factors.copy()
        initial_output_scaling = sm.output_scaling_factors.copy()
        sm.update(pre, post, weights=None)
        self.assertTrue(xp.allclose(sm.input_scaling_factors, initial_input_scaling))
        self.assertTrue(xp.allclose(sm.output_scaling_factors, initial_output_scaling))

    def test_activity_history_recording(self):
        pre = xp.ones(8) * 0.5
        post = xp.array([1.0, 0.0, 0.5, 0.0, 0.3, 0.0, 0.0, 0.7])
        self.sm.update(pre, post)
        self.assertEqual(self.sm.activity_history_index, 1)
        stored = self.sm.activity_history[0]
        self.assertTrue(xp.allclose(stored, post))

    def test_activity_history_circular_buffer(self):
        pre = xp.ones(8) * 0.5
        post = xp.ones(8) * 0.1
        # Fill the circular buffer (100 entries)
        for _ in range(100):
            self.sm.update(pre, post)
        self.assertTrue(self.sm.activity_history_filled)
        self.assertEqual(self.sm.activity_history_index, 0)


class TestStabilityMonitoringMetrics(unittest.TestCase):
    """Test get_stats and metric tracking."""

    def test_get_stats_keys(self):
        sm = StabilityMechanisms(representation_size=10)
        stats = sm.get_stats()
        expected_keys = [
            'representation_size', 'diversity_score', 'mean_threshold',
            'threshold_std', 'mean_homeostatic_factor', 'homeostatic_range',
            'mean_firing_rate', 'active_neuron_fraction', 'k_winners',
            'inhibition_strategy',
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_get_stats_initial_values(self):
        sm = StabilityMechanisms(representation_size=10, target_activity=0.1)
        stats = sm.get_stats()
        self.assertEqual(stats['representation_size'], 10)
        self.assertAlmostEqual(stats['diversity_score'], 1.0)
        self.assertAlmostEqual(stats['mean_threshold'], 0.5, places=5)
        self.assertAlmostEqual(stats['mean_homeostatic_factor'], 1.0, places=5)
        self.assertAlmostEqual(stats['mean_firing_rate'], 0.0, places=5)
        self.assertAlmostEqual(stats['active_neuron_fraction'], 0.0, places=5)
        self.assertEqual(stats['inhibition_strategy'], 'k_winners_allowed')

    def test_get_stats_values_are_floats(self):
        sm = StabilityMechanisms(representation_size=10)
        stats = sm.get_stats()
        for key in ['diversity_score', 'mean_threshold', 'threshold_std',
                     'mean_homeostatic_factor', 'homeostatic_range',
                     'mean_firing_rate', 'active_neuron_fraction']:
            self.assertIsInstance(stats[key], float)

    def test_stats_reflect_updates(self):
        sm = StabilityMechanisms(representation_size=8, target_activity=0.25)
        pre = xp.ones(8) * 0.5
        # Neuron 0 always active
        post = xp.zeros(8)
        post[0] = 1.0
        for _ in range(20):
            sm.update(pre, post)
        stats = sm.get_stats()
        # Mean firing rate should be nonzero now
        self.assertGreater(stats['mean_firing_rate'], 0.0)


class TestGetState(unittest.TestCase):
    """Test serialize and deserialize (get_state equivalent)."""

    def test_serialize_returns_dict(self):
        sm = StabilityMechanisms(representation_size=8)
        data = sm.serialize()
        self.assertIsInstance(data, dict)

    def test_serialize_contains_all_parameters(self):
        sm = StabilityMechanisms(representation_size=8)
        data = sm.serialize()
        expected_keys = [
            'representation_size', 'target_activity', 'homeostatic_rate',
            'inhibition_strategy', 'inhibition_strength', 'k_winners',
            'normalization_threshold', 'diversity_target', 'diversity_rate',
            'adaptive_threshold_tau', 'minimum_threshold', 'maximum_threshold',
            'weight_scaling_enabled', 'weight_decay', 'thresholds',
            'homeostatic_factors', 'diversity_score', 'update_count',
        ]
        for key in expected_keys:
            self.assertIn(key, data)

    def test_serialize_values_match(self):
        sm = StabilityMechanisms(
            representation_size=8,
            target_activity=0.15,
            inhibition_strategy=InhibitionStrategy.WTA,
        )
        data = sm.serialize()
        self.assertEqual(data['representation_size'], 8)
        self.assertAlmostEqual(data['target_activity'], 0.15)
        self.assertEqual(data['inhibition_strategy'], 'winner_take_all')
        self.assertEqual(data['update_count'], 0)

    def test_serialize_thresholds_as_list(self):
        sm = StabilityMechanisms(representation_size=4)
        data = sm.serialize()
        self.assertIsInstance(data['thresholds'], list)
        self.assertEqual(len(data['thresholds']), 4)

    def test_deserialize_roundtrip(self):
        sm = StabilityMechanisms(
            representation_size=10,
            target_activity=0.2,
            homeostatic_rate=0.05,
            inhibition_strategy=InhibitionStrategy.KWA,
            inhibition_strength=0.7,
            k_winners=3,
            normalization_threshold=2.5,
            diversity_target=0.85,
            diversity_rate=0.03,
            adaptive_threshold_tau=0.02,
            minimum_threshold=0.02,
            maximum_threshold=1.5,
            weight_scaling_enabled=True,
            weight_decay=0.0005,
        )
        # Simulate some updates
        pre = xp.ones(10) * 0.5
        post = xp.zeros(10)
        post[0] = 1.0
        post[5] = 0.5
        for _ in range(5):
            sm.update(pre, post)

        data = sm.serialize()
        sm_restored = StabilityMechanisms.deserialize(data)

        self.assertEqual(sm_restored.representation_size, sm.representation_size)
        self.assertAlmostEqual(sm_restored.target_activity, sm.target_activity)
        self.assertAlmostEqual(sm_restored.homeostatic_rate, sm.homeostatic_rate)
        self.assertEqual(sm_restored.inhibition_strategy, sm.inhibition_strategy)
        self.assertAlmostEqual(sm_restored.inhibition_strength, sm.inhibition_strength)
        self.assertEqual(sm_restored.k_winners, sm.k_winners)
        self.assertAlmostEqual(sm_restored.normalization_threshold, sm.normalization_threshold)
        self.assertAlmostEqual(sm_restored.diversity_target, sm.diversity_target)
        self.assertAlmostEqual(sm_restored.diversity_rate, sm.diversity_rate)
        self.assertAlmostEqual(sm_restored.weight_decay, sm.weight_decay)
        self.assertEqual(sm_restored.update_count, sm.update_count)
        self.assertTrue(xp.allclose(
            xp.array(sm_restored.thresholds), xp.array(sm.thresholds), atol=1e-6))
        self.assertTrue(xp.allclose(
            xp.array(sm_restored.homeostatic_factors),
            xp.array(sm.homeostatic_factors), atol=1e-6))
        self.assertAlmostEqual(sm_restored.diversity_score, sm.diversity_score, places=5)


class TestWeightNormalization(unittest.TestCase):
    """Test normalize_weights method."""

    def test_normalize_2d_weights(self):
        sm = StabilityMechanisms(representation_size=4, weight_decay=0.0)
        weights = xp.ones((4, 4))
        result = sm.normalize_weights(weights)
        # With unit scaling factors and zero decay, result should equal input
        self.assertTrue(xp.allclose(result, weights))

    def test_normalize_applies_weight_decay(self):
        sm = StabilityMechanisms(representation_size=4, weight_decay=0.01)
        weights = xp.ones((4, 4))
        result = sm.normalize_weights(weights)
        expected = weights * (1.0 - 0.01)
        self.assertTrue(xp.allclose(result, expected, atol=1e-6))

    def test_normalize_applies_scaling_factors(self):
        sm = StabilityMechanisms(representation_size=4, weight_decay=0.0)
        sm.input_scaling_factors = xp.array([0.5, 1.0, 0.5, 1.0])
        sm.output_scaling_factors = xp.array([1.0, 0.5, 1.0, 0.5])
        weights = xp.ones((4, 4))
        result = sm.normalize_weights(weights)
        # Row 0 scaled by 0.5 input, so result[0,:] = 0.5 * output_scaling
        self.assertAlmostEqual(float(result[0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(result[0, 1]), 0.25, places=5)

    def test_normalize_non_2d_returns_original(self):
        sm = StabilityMechanisms(representation_size=4)
        weights_1d = xp.array([1.0, 2.0, 3.0, 4.0])
        result = sm.normalize_weights(weights_1d)
        self.assertTrue(xp.allclose(result, weights_1d))


class TestDiversityAdjustment(unittest.TestCase):
    """Test the adjust_diversity method."""

    def test_no_adjustment_when_diversity_high(self):
        sm = StabilityMechanisms(representation_size=4, diversity_target=0.8)
        sm.diversity_score = 0.9  # Above target
        weights = xp.ones((4, 4))
        result = sm.adjust_diversity(weights)
        self.assertTrue(xp.allclose(result, weights))

    def test_adjustment_when_diversity_low(self):
        sm = StabilityMechanisms(representation_size=4, diversity_target=0.8,
                                  diversity_rate=0.1)
        sm.diversity_score = 0.3  # Below target
        # Create a high-correlation matrix to trigger adjustment
        sm.correlation_matrix = xp.ones((4, 4)) * 0.9
        xp.fill_diagonal(sm.correlation_matrix, 1.0)
        weights = xp.ones((4, 4))
        result = sm.adjust_diversity(weights)
        # Some weights should have changed
        self.assertFalse(xp.allclose(result, weights))

    def test_adjust_diversity_non_2d_returns_original(self):
        sm = StabilityMechanisms(representation_size=4)
        sm.diversity_score = 0.3
        weights_1d = xp.array([1.0, 2.0, 3.0, 4.0])
        result = sm.adjust_diversity(weights_1d)
        self.assertTrue(xp.allclose(result, weights_1d))


class TestMexicanHatKernel(unittest.TestCase):
    """Test the Mexican hat kernel creation."""

    def test_kernel_shape(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.ADAPTIVE,
        )
        self.assertEqual(sm.inhibition_kernel.shape, (16, 16))

    def test_kernel_diagonal_zero(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.ADAPTIVE,
        )
        diag = xp.diag(sm.inhibition_kernel)
        self.assertTrue(xp.allclose(diag, xp.zeros(16)))

    def test_kernel_symmetric(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.ADAPTIVE,
        )
        kernel = sm.inhibition_kernel
        self.assertTrue(xp.allclose(kernel, kernel.T, atol=1e-10))

    def test_kernel_normalized(self):
        sm = StabilityMechanisms(
            representation_size=16,
            inhibition_strategy=InhibitionStrategy.ADAPTIVE,
        )
        max_abs = float(xp.max(xp.abs(sm.inhibition_kernel)))
        self.assertAlmostEqual(max_abs, 1.0, places=5)


class TestResize(unittest.TestCase):
    """Test the resize method."""

    def test_resize_grow(self):
        sm = StabilityMechanisms(representation_size=8)
        result = sm.resize(16)
        self.assertTrue(result['resized'])
        self.assertEqual(result['old_size'], 8)
        self.assertEqual(result['new_size'], 16)
        self.assertEqual(sm.representation_size, 16)
        self.assertEqual(len(sm.thresholds), 16)
        self.assertEqual(len(sm.homeostatic_factors), 16)
        self.assertEqual(len(sm.firing_rates), 16)
        self.assertEqual(sm.activity_history.shape, (100, 16))

    def test_resize_shrink(self):
        sm = StabilityMechanisms(representation_size=16)
        result = sm.resize(8)
        self.assertTrue(result['resized'])
        self.assertEqual(result['old_size'], 16)
        self.assertEqual(result['new_size'], 8)
        self.assertEqual(sm.representation_size, 8)
        self.assertEqual(len(sm.thresholds), 8)
        self.assertEqual(len(sm.homeostatic_factors), 8)
        self.assertEqual(sm.correlation_matrix.shape, (8, 8))

    def test_resize_same_size_noop(self):
        sm = StabilityMechanisms(representation_size=10)
        result = sm.resize(10)
        self.assertFalse(result['resized'])
        self.assertEqual(sm.representation_size, 10)

    def test_resize_grow_preserves_old_data(self):
        sm = StabilityMechanisms(representation_size=4)
        sm.thresholds = xp.array([0.1, 0.2, 0.3, 0.4])
        sm.resize(8)
        # First 4 elements should be preserved
        self.assertAlmostEqual(float(sm.thresholds[0]), 0.1, places=5)
        self.assertAlmostEqual(float(sm.thresholds[1]), 0.2, places=5)
        self.assertAlmostEqual(float(sm.thresholds[2]), 0.3, places=5)
        self.assertAlmostEqual(float(sm.thresholds[3]), 0.4, places=5)

    def test_resize_resets_activity_history(self):
        sm = StabilityMechanisms(representation_size=8)
        sm.activity_history_index = 5
        sm.activity_history_filled = True
        sm.resize(12)
        self.assertEqual(sm.activity_history_index, 0)
        self.assertFalse(sm.activity_history_filled)
        self.assertEqual(sm.activity_history.shape, (100, 12))

    def test_resize_updates_k_winners(self):
        sm = StabilityMechanisms(representation_size=10, target_activity=0.2)
        # Default k_winners = max(1, int(10*0.2)) = 2
        self.assertEqual(sm.k_winners, 2)
        sm.resize(20)
        # After resize: max(1, int(20*0.2)) = 4
        self.assertEqual(sm.k_winners, 4)

    def test_resize_adaptive_recreates_kernel(self):
        sm = StabilityMechanisms(
            representation_size=8,
            inhibition_strategy=InhibitionStrategy.ADAPTIVE,
        )
        self.assertEqual(sm.inhibition_kernel.shape, (8, 8))
        sm.resize(16)
        self.assertEqual(sm.inhibition_kernel.shape, (16, 16))


class TestInhibitionStrategyEnum(unittest.TestCase):
    """Test the InhibitionStrategy enum."""

    def test_enum_values(self):
        self.assertEqual(InhibitionStrategy.NONE.value, "none")
        self.assertEqual(InhibitionStrategy.WTA.value, "winner_take_all")
        self.assertEqual(InhibitionStrategy.KWA.value, "k_winners_allowed")
        self.assertEqual(InhibitionStrategy.MEXICAN_HAT.value, "mexican_hat")
        self.assertEqual(InhibitionStrategy.ADAPTIVE.value, "adaptive")

    def test_enum_from_value(self):
        self.assertEqual(InhibitionStrategy("none"), InhibitionStrategy.NONE)
        self.assertEqual(InhibitionStrategy("winner_take_all"), InhibitionStrategy.WTA)
        self.assertEqual(InhibitionStrategy("k_winners_allowed"), InhibitionStrategy.KWA)
        self.assertEqual(InhibitionStrategy("mexican_hat"), InhibitionStrategy.MEXICAN_HAT)
        self.assertEqual(InhibitionStrategy("adaptive"), InhibitionStrategy.ADAPTIVE)

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            InhibitionStrategy("invalid_strategy")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_all_zero_activity_inhibition(self):
        for strategy in InhibitionStrategy:
            sm = StabilityMechanisms(
                representation_size=8,
                inhibition_strategy=strategy,
                k_winners=2,
            )
            activity = xp.zeros(8)
            result = sm.apply_inhibition(activity)
            self.assertTrue(xp.allclose(result, xp.zeros(8)),
                            f"Failed for strategy {strategy.value}")

    def test_single_active_neuron_wta(self):
        sm = StabilityMechanisms(
            representation_size=8,
            inhibition_strategy=InhibitionStrategy.WTA,
        )
        activity = xp.zeros(8)
        activity[3] = 0.5
        result = sm.apply_inhibition(activity)
        self.assertAlmostEqual(float(result[3]), 0.5, places=5)

    def test_uniform_activity_kwa(self):
        sm = StabilityMechanisms(
            representation_size=8,
            inhibition_strategy=InhibitionStrategy.KWA,
            k_winners=4,
        )
        activity = xp.ones(8) * 0.5
        result = sm.apply_inhibition(activity)
        # All values are equal so >= threshold includes all of them
        # No suppression should happen since all are equal
        total_before = float(xp.sum(activity))
        total_after = float(xp.sum(result))
        self.assertAlmostEqual(total_before, total_after, places=4)

    def test_very_small_representation(self):
        sm = StabilityMechanisms(representation_size=2)
        self.assertEqual(sm.representation_size, 2)
        self.assertEqual(sm.k_winners, 1)
        activity = xp.array([0.7, 0.3])
        result = sm.apply_inhibition(activity)
        total = float(xp.sum(result))
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_update_weight_scaling_non_2d_ignored(self):
        sm = StabilityMechanisms(representation_size=4, weight_scaling_enabled=True)
        pre = xp.ones(4) * 0.5
        post = xp.ones(4) * 0.1
        initial_input_scaling = sm.input_scaling_factors.copy()
        # Passing a 1D weights array - should not affect scaling
        sm.update(pre, post, weights=xp.ones(4))
        self.assertTrue(xp.allclose(sm.input_scaling_factors, initial_input_scaling))

    def test_homeostatic_factors_normalized_mean(self):
        sm = StabilityMechanisms(representation_size=8, target_activity=0.25)
        pre = xp.ones(8) * 0.5
        post = xp.zeros(8)
        post[0] = 1.0
        post[4] = 0.5
        for _ in range(10):
            sm.update(pre, post)
        # After updates, homeostatic factors should be normalized to mean ~1.0
        mean_factor = float(xp.mean(sm.homeostatic_factors))
        self.assertAlmostEqual(mean_factor, 1.0, places=4)

    def test_thresholds_clipped_to_range(self):
        sm = StabilityMechanisms(
            representation_size=4,
            minimum_threshold=0.05,
            maximum_threshold=0.9,
        )
        pre = xp.ones(4) * 0.5
        post = xp.ones(4) * 0.1
        # Run many updates to push thresholds
        for _ in range(200):
            sm.update(pre, post)
        self.assertTrue(xp.all(sm.thresholds >= 0.05))
        self.assertTrue(xp.all(sm.thresholds <= 0.9))


if __name__ == '__main__':
    unittest.main()
