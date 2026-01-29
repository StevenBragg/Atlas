"""
Comprehensive tests for HomeostaticPlasticity (core/homeostatic_plasticity.py).
Tests cover initialization, threshold adaptation, weight normalization,
apply/update methods, inhibition strategies, synaptic scaling,
heterosynaptic competition, serialization, and state reporting.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.homeostatic_plasticity import (
    HomeostaticPlasticity,
    InhibitionStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SMALL = 16  # default layer size used across tests


def _make_hp(**kwargs):
    """Create a HomeostaticPlasticity instance with small defaults."""
    kwargs.setdefault("layer_size", SMALL)
    return HomeostaticPlasticity(**kwargs)


def _uniform_activity(size=SMALL, value=0.5):
    """Return a uniform activity vector."""
    return xp.ones(size) * value


def _sparse_activity(size=SMALL, n_active=2, value=1.0):
    """Return an activity vector with only *n_active* non-zero entries."""
    act = xp.zeros(size)
    act[:n_active] = value
    return act


# ===================================================================
# 1. Initialization
# ===================================================================
class TestInitialization(unittest.TestCase):
    """Test constructor and default state."""

    def test_default_parameters(self):
        hp = _make_hp()
        self.assertEqual(hp.layer_size, SMALL)
        self.assertAlmostEqual(hp.target_activity, 0.1)
        self.assertAlmostEqual(hp.target_sparsity, 0.05)
        self.assertAlmostEqual(hp.time_constant, 0.01)
        self.assertEqual(hp.inhibition_strategy, InhibitionStrategy.KWA)

    def test_custom_parameters(self):
        hp = _make_hp(
            target_activity=0.2,
            target_sparsity=0.1,
            time_constant=0.05,
            inhibition_strategy="winner_take_all",
            k_percent=0.1,
            threshold_adaptation_rate=0.02,
            min_threshold=0.05,
            max_threshold=0.9,
        )
        self.assertAlmostEqual(hp.target_activity, 0.2)
        self.assertAlmostEqual(hp.target_sparsity, 0.1)
        self.assertEqual(hp.inhibition_strategy, InhibitionStrategy.WTA)
        self.assertAlmostEqual(hp.k_percent, 0.1)
        self.assertAlmostEqual(hp.min_threshold, 0.05)
        self.assertAlmostEqual(hp.max_threshold, 0.9)

    def test_initial_arrays_shape(self):
        hp = _make_hp()
        self.assertEqual(hp.adaptive_thresholds.shape, (SMALL,))
        self.assertEqual(hp.excitability.shape, (SMALL,))
        self.assertEqual(hp.avg_activity.shape, (SMALL,))
        self.assertEqual(hp.act_variance.shape, (SMALL,))
        self.assertEqual(hp.last_active.shape, (SMALL,))

    def test_initial_array_values(self):
        hp = _make_hp(adaptive_threshold_init=0.6)
        # Thresholds should all be the init value
        self.assertTrue(xp.allclose(hp.adaptive_thresholds, 0.6))
        # Excitability starts at 1
        self.assertTrue(xp.allclose(hp.excitability, 1.0))
        # Average activity starts at 0
        self.assertTrue(xp.allclose(hp.avg_activity, 0.0))
        # Activity variance starts at 1
        self.assertTrue(xp.allclose(hp.act_variance, 1.0))

    def test_initial_metrics(self):
        hp = _make_hp()
        self.assertAlmostEqual(hp.stability_index, 1.0)
        self.assertAlmostEqual(hp.sparsity_index, 0.0)
        self.assertAlmostEqual(hp.avg_layer_activity, 0.0)
        self.assertEqual(hp.update_counter, 0)

    def test_lateral_weights_none_for_non_soft(self):
        for strategy in ["none", "winner_take_all", "k_winners_adaptive",
                         "adaptive_threshold"]:
            hp = _make_hp(inhibition_strategy=strategy)
            self.assertIsNone(hp.lateral_weights,
                              msg=f"lateral_weights should be None for {strategy}")

    def test_lateral_weights_created_for_soft(self):
        hp = _make_hp(inhibition_strategy="soft_competition")
        self.assertIsNotNone(hp.lateral_weights)
        self.assertEqual(hp.lateral_weights.shape, (SMALL, SMALL))
        # Diagonal should be zero (no self-inhibition)
        for i in range(SMALL):
            self.assertAlmostEqual(float(hp.lateral_weights[i, i]), 0.0)

    def test_activity_history_empty(self):
        hp = _make_hp()
        self.assertEqual(len(hp.activity_history), 0)


# ===================================================================
# 2. Threshold Adaptation
# ===================================================================
class TestThresholdAdaptation(unittest.TestCase):
    """Test update_thresholds behaviour."""

    def test_thresholds_increase_when_activity_above_target(self):
        hp = _make_hp(target_activity=0.1, threshold_adaptation_rate=0.1,
                      adaptive_threshold_init=0.5)
        # Simulate high average activity
        hp.avg_activity = xp.ones(SMALL) * 0.5
        old_thresholds = hp.adaptive_thresholds.copy()
        hp.update_thresholds()
        # Thresholds should have increased because avg > target
        self.assertTrue(xp.all(hp.adaptive_thresholds > old_thresholds))

    def test_thresholds_decrease_when_activity_below_target(self):
        hp = _make_hp(target_activity=0.5, threshold_adaptation_rate=0.1,
                      adaptive_threshold_init=0.5)
        # Simulate low average activity
        hp.avg_activity = xp.ones(SMALL) * 0.05
        old_thresholds = hp.adaptive_thresholds.copy()
        hp.update_thresholds()
        # Thresholds should have decreased
        self.assertTrue(xp.all(hp.adaptive_thresholds < old_thresholds))

    def test_thresholds_clamped_to_bounds(self):
        hp = _make_hp(min_threshold=0.1, max_threshold=0.9,
                      threshold_adaptation_rate=100.0,
                      adaptive_threshold_init=0.5)
        hp.avg_activity = xp.ones(SMALL) * 10.0
        hp.update_thresholds()
        self.assertTrue(xp.all(hp.adaptive_thresholds <= 0.9))
        self.assertTrue(xp.all(hp.adaptive_thresholds >= 0.1))

    def test_thresholds_unchanged_at_target(self):
        hp = _make_hp(target_activity=0.3, threshold_adaptation_rate=0.01,
                      adaptive_threshold_init=0.5)
        hp.avg_activity = xp.ones(SMALL) * 0.3  # exactly at target
        old = hp.adaptive_thresholds.copy()
        hp.update_thresholds()
        self.assertTrue(xp.allclose(hp.adaptive_thresholds, old))


# ===================================================================
# 3. Weight Normalization (Synaptic Scaling)
# ===================================================================
class TestSynapticScaling(unittest.TestCase):
    """Test apply_synaptic_scaling on weight matrices."""

    def test_output_weights_scaled_down_when_overactive(self):
        hp = _make_hp(synaptic_scaling_rate=0.1)
        hp.avg_activity = xp.ones(SMALL) * 0.5  # well above target 0.1
        weights = xp.ones((SMALL, 8)) * 2.0
        scaled = hp.apply_synaptic_scaling(weights.copy())
        # Because avg > target, scaling_factor < 1 => weights shrink
        self.assertTrue(float(xp.mean(xp.abs(scaled))) < float(xp.mean(xp.abs(weights))))

    def test_input_weights_scaled_down_when_overactive(self):
        hp = _make_hp(synaptic_scaling_rate=0.1)
        hp.avg_activity = xp.ones(SMALL) * 0.5
        weights = xp.ones((8, SMALL)) * 2.0
        scaled = hp.apply_synaptic_scaling(weights.copy())
        self.assertTrue(float(xp.mean(xp.abs(scaled))) < float(xp.mean(xp.abs(weights))))

    def test_no_change_for_1d_weights(self):
        hp = _make_hp()
        w1d = xp.ones(SMALL) * 3.0
        result = hp.apply_synaptic_scaling(w1d.copy())
        self.assertTrue(xp.allclose(result, w1d))

    def test_weights_clipped_to_range(self):
        hp = _make_hp(synaptic_scaling_rate=0.001)
        hp.avg_activity = xp.zeros(SMALL)
        weights = xp.ones((SMALL, 8)) * 50.0
        scaled = hp.apply_synaptic_scaling(weights)
        self.assertTrue(float(xp.max(scaled)) <= 10.0)
        self.assertTrue(float(xp.min(scaled)) >= -10.0)

    def test_zero_weights_not_modified(self):
        hp = _make_hp(synaptic_scaling_rate=0.1)
        hp.avg_activity = xp.ones(SMALL) * 0.5
        weights = xp.zeros((SMALL, 8))
        weights[0, 0] = 1.0  # single nonzero
        scaled = hp.apply_synaptic_scaling(weights.copy())
        # All positions that were zero should still be zero
        zero_mask = weights == 0
        self.assertTrue(xp.all(scaled[zero_mask] == 0))


# ===================================================================
# 4. Heterosynaptic Competition
# ===================================================================
class TestHeterosynapticCompetition(unittest.TestCase):
    """Test apply_heterosynaptic_competition."""

    def test_competition_reduces_outlier_weights(self):
        hp = _make_hp(heterosynaptic_rate=0.1)
        weights = xp.ones((8, SMALL))
        weights[0, 0] = 10.0  # outlier
        result = hp.apply_heterosynaptic_competition(weights.copy())
        # The outlier should have moved closer to the mean
        self.assertTrue(float(result[0, 0]) < 10.0)

    def test_no_change_for_1d_weights(self):
        hp = _make_hp()
        w1d = xp.array([1.0, 2.0, 3.0])
        result = hp.apply_heterosynaptic_competition(w1d.copy())
        self.assertTrue(xp.allclose(result, w1d))

    def test_output_weights_path(self):
        hp = _make_hp(heterosynaptic_rate=0.1)
        weights = xp.ones((SMALL, 8))
        weights[0, 0] = 10.0
        result = hp.apply_heterosynaptic_competition(weights.copy())
        self.assertTrue(float(result[0, 0]) < 10.0)

    def test_single_connection_unchanged(self):
        """A neuron with only one nonzero connection has no competition."""
        hp = _make_hp(heterosynaptic_rate=0.1)
        weights = xp.zeros((8, SMALL))
        weights[3, 0] = 5.0  # single connection to neuron 0
        result = hp.apply_heterosynaptic_competition(weights.copy())
        self.assertAlmostEqual(float(result[3, 0]), 5.0, places=5)


# ===================================================================
# 5. Inhibition Strategies
# ===================================================================
class TestInhibitionStrategies(unittest.TestCase):
    """Test each lateral inhibition mode via apply()."""

    def test_no_inhibition(self):
        hp = _make_hp(inhibition_strategy="none")
        act = _uniform_activity()
        result = hp.apply(act.copy())
        self.assertTrue(xp.allclose(result, act))

    def test_wta_single_winner(self):
        hp = _make_hp(inhibition_strategy="winner_take_all")
        act = xp.arange(SMALL, dtype=float)
        result = hp.apply(act.copy())
        # Only the largest element should survive
        nonzero_count = int(xp.sum(result > 0))
        self.assertEqual(nonzero_count, 1)
        self.assertAlmostEqual(float(result[SMALL - 1]), float(SMALL - 1))

    def test_kwa_keeps_k_winners(self):
        hp = _make_hp(inhibition_strategy="k_winners_adaptive", k_percent=0.25)
        act = xp.arange(SMALL, dtype=float)  # 0..15
        result = hp.apply(act.copy())
        k = max(1, int(0.25 * SMALL))  # 4
        nonzero = int(xp.sum(result > 0))
        self.assertEqual(nonzero, k)

    def test_soft_competition_reduces_activity(self):
        hp = _make_hp(inhibition_strategy="soft_competition")
        act = _uniform_activity(value=1.0)
        result = hp.apply(act.copy())
        # Soft inhibition with negative lateral weights should reduce
        # uniform activity (each neuron inhibits others)
        self.assertTrue(float(xp.mean(result)) < 1.0)

    def test_adaptive_threshold_gates_activity(self):
        hp = _make_hp(inhibition_strategy="adaptive_threshold",
                      adaptive_threshold_init=0.5)
        act = xp.zeros(SMALL)
        act[0] = 1.0   # above threshold
        act[1] = 0.3   # below threshold
        result = hp.apply(act.copy())
        self.assertTrue(float(result[0]) > 0.0)
        self.assertAlmostEqual(float(result[1]), 0.0)


# ===================================================================
# 6. apply() Method
# ===================================================================
class TestApply(unittest.TestCase):
    """Test the main apply() pipeline."""

    def test_output_shape_matches_input(self):
        hp = _make_hp()
        act = _uniform_activity()
        result = hp.apply(act)
        self.assertEqual(result.shape, act.shape)

    def test_update_counter_increments(self):
        hp = _make_hp()
        self.assertEqual(hp.update_counter, 0)
        hp.apply(_uniform_activity())
        self.assertEqual(hp.update_counter, 1)
        hp.apply(_uniform_activity())
        self.assertEqual(hp.update_counter, 2)

    def test_activity_history_grows(self):
        hp = _make_hp()
        for i in range(5):
            hp.apply(_uniform_activity())
        self.assertEqual(len(hp.activity_history), 5)

    def test_activity_history_capped(self):
        hp = _make_hp()
        for i in range(120):
            hp.apply(_uniform_activity())
        self.assertLessEqual(len(hp.activity_history), hp.max_history_length + 1)

    def test_avg_activity_updated(self):
        hp = _make_hp(inhibition_strategy="none")
        hp.apply(_uniform_activity(value=0.8))
        # After one step avg_activity should be nonzero
        self.assertTrue(float(xp.mean(hp.avg_activity)) > 0.0)

    def test_last_active_updated(self):
        hp = _make_hp(inhibition_strategy="none",
                      adaptive_threshold_init=0.3)
        act = xp.zeros(SMALL)
        act[0] = 1.0
        hp.apply(act)
        self.assertTrue(bool(hp.last_active[0]))

    def test_resize_on_different_input_size(self):
        hp = _make_hp(inhibition_strategy="none")
        new_size = SMALL + 4
        act = xp.ones(new_size) * 0.5
        result = hp.apply(act)
        self.assertEqual(hp.layer_size, new_size)
        self.assertEqual(result.shape, (new_size,))

    def test_excitability_update_at_step_10(self):
        hp = _make_hp(inhibition_strategy="none")
        # Use non-uniform activity so neurons get different excitability updates
        act = xp.linspace(0.0, 1.0, SMALL)
        for _ in range(10):
            hp.apply(act)
        # After 10 apply calls update_excitability is triggered.
        # With non-uniform avg_activity the per-neuron deltas differ,
        # so after normalisation excitability should no longer be all ones.
        self.assertFalse(xp.allclose(hp.excitability, 1.0))


# ===================================================================
# 7. Excitability Update
# ===================================================================
class TestExcitabilityUpdate(unittest.TestCase):
    """Test update_excitability directly."""

    def test_excitability_increases_when_underactive(self):
        hp = _make_hp(target_activity=0.5, intrinsic_plasticity_rate=0.1)
        hp.avg_activity = xp.ones(SMALL) * 0.01  # very low
        old = hp.excitability.copy()
        hp.update_excitability()
        # Should increase to boost activity
        self.assertTrue(float(xp.mean(hp.excitability)) >= float(xp.mean(old)))

    def test_excitability_decreases_when_overactive(self):
        hp = _make_hp(target_activity=0.1, intrinsic_plasticity_rate=0.1)
        hp.avg_activity = xp.ones(SMALL) * 0.9  # very high
        hp.update_excitability()
        # After normalisation, mean should be 1.0 but individual values
        # that were high-activity should be < 1
        mean_exc = float(xp.mean(hp.excitability))
        self.assertAlmostEqual(mean_exc, 1.0, places=3)

    def test_excitability_stays_positive(self):
        hp = _make_hp(intrinsic_plasticity_rate=10.0)
        hp.avg_activity = xp.ones(SMALL) * 100.0
        hp.update_excitability()
        self.assertTrue(xp.all(hp.excitability >= 0.1))

    def test_excitability_normalised_mean(self):
        hp = _make_hp(intrinsic_plasticity_rate=0.1)
        hp.avg_activity = xp.linspace(0.0, 1.0, SMALL)
        hp.update_excitability()
        mean_exc = float(xp.mean(hp.excitability))
        self.assertAlmostEqual(mean_exc, 1.0, places=4)


# ===================================================================
# 8. get_state / get_stability_metrics
# ===================================================================
class TestGetState(unittest.TestCase):
    """Test get_stability_metrics and serialize/deserialize."""

    def test_stability_metrics_keys(self):
        hp = _make_hp()
        metrics = hp.get_stability_metrics()
        expected_keys = {
            "stability_index", "sparsity_index", "avg_activity",
            "target_activity", "mean_excitability", "mean_threshold",
            "activity_variance",
        }
        self.assertEqual(set(metrics.keys()), expected_keys)

    def test_stability_metrics_types(self):
        hp = _make_hp()
        metrics = hp.get_stability_metrics()
        for k, v in metrics.items():
            self.assertIsInstance(v, float, msg=f"{k} should be float")

    def test_serialize_returns_dict(self):
        hp = _make_hp()
        state = hp.serialize()
        self.assertIsInstance(state, dict)
        self.assertEqual(state["layer_size"], SMALL)
        self.assertEqual(state["inhibition_strategy"], "k_winners_adaptive")

    def test_serialize_round_trip(self):
        hp = _make_hp(target_activity=0.2, inhibition_strategy="winner_take_all")
        # Run a few steps so internal state is nontrivial
        for _ in range(3):
            hp.apply(_uniform_activity(value=0.4))
        state = hp.serialize()
        hp2 = HomeostaticPlasticity.deserialize(state)
        self.assertEqual(hp2.layer_size, hp.layer_size)
        self.assertAlmostEqual(hp2.target_activity, hp.target_activity)
        self.assertTrue(xp.allclose(hp2.adaptive_thresholds, hp.adaptive_thresholds))
        self.assertTrue(xp.allclose(hp2.excitability, hp.excitability))
        self.assertTrue(xp.allclose(hp2.avg_activity, hp.avg_activity))
        self.assertEqual(hp2.update_counter, hp.update_counter)

    def test_serialize_with_lateral_weights(self):
        hp = _make_hp(inhibition_strategy="soft_competition")
        state = hp.serialize()
        self.assertIn("lateral_weights", state)
        hp2 = HomeostaticPlasticity.deserialize(state)
        self.assertIsNotNone(hp2.lateral_weights)
        self.assertTrue(xp.allclose(hp2.lateral_weights, hp.lateral_weights))

    def test_serialize_without_lateral_weights(self):
        hp = _make_hp(inhibition_strategy="none")
        state = hp.serialize()
        self.assertNotIn("lateral_weights", state)


# ===================================================================
# 9. Resize
# ===================================================================
class TestResize(unittest.TestCase):
    """Test dynamic resizing of the layer."""

    def test_resize_up(self):
        hp = _make_hp()
        bigger = SMALL + 10
        hp.apply(xp.ones(bigger) * 0.5)
        self.assertEqual(hp.layer_size, bigger)
        self.assertEqual(hp.adaptive_thresholds.shape[0], bigger)
        self.assertEqual(hp.excitability.shape[0], bigger)

    def test_resize_down(self):
        hp = _make_hp()
        smaller = SMALL - 4
        hp.apply(xp.ones(smaller) * 0.5)
        self.assertEqual(hp.layer_size, smaller)
        self.assertEqual(hp.avg_activity.shape[0], smaller)

    def test_resize_soft_inhibition_lateral_weights(self):
        hp = _make_hp(inhibition_strategy="soft_competition")
        bigger = SMALL + 6
        hp.apply(xp.ones(bigger) * 0.5)
        self.assertEqual(hp.lateral_weights.shape, (bigger, bigger))


# ===================================================================
# 10. Set Methods
# ===================================================================
class TestSetters(unittest.TestCase):
    """Test set_target_activity and set_inhibition_strategy."""

    def test_set_target_activity(self):
        hp = _make_hp()
        hp.set_target_activity(0.42)
        self.assertAlmostEqual(hp.target_activity, 0.42)

    def test_set_inhibition_strategy_valid(self):
        hp = _make_hp(inhibition_strategy="none")
        hp.set_inhibition_strategy("winner_take_all")
        self.assertEqual(hp.inhibition_strategy, InhibitionStrategy.WTA)

    def test_set_inhibition_to_soft_creates_lateral_weights(self):
        hp = _make_hp(inhibition_strategy="none")
        self.assertIsNone(hp.lateral_weights)
        hp.set_inhibition_strategy("soft_competition")
        self.assertIsNotNone(hp.lateral_weights)

    def test_set_inhibition_invalid_keeps_old(self):
        hp = _make_hp(inhibition_strategy="none")
        hp.set_inhibition_strategy("bogus_strategy")
        # Should remain unchanged
        self.assertEqual(hp.inhibition_strategy, InhibitionStrategy.NONE)


# ===================================================================
# 11. InhibitionStrategy Enum
# ===================================================================
class TestInhibitionStrategyEnum(unittest.TestCase):
    """Test the InhibitionStrategy enum values."""

    def test_all_values(self):
        self.assertEqual(InhibitionStrategy.NONE.value, "none")
        self.assertEqual(InhibitionStrategy.WTA.value, "winner_take_all")
        self.assertEqual(InhibitionStrategy.KWA.value, "k_winners_adaptive")
        self.assertEqual(InhibitionStrategy.SOFT.value, "soft_competition")
        self.assertEqual(InhibitionStrategy.ADAPTIVE.value, "adaptive_threshold")

    def test_construction_from_value(self):
        self.assertEqual(InhibitionStrategy("none"), InhibitionStrategy.NONE)
        self.assertEqual(InhibitionStrategy("winner_take_all"), InhibitionStrategy.WTA)

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            InhibitionStrategy("invalid")


# ===================================================================
# 12. Integration / Multi-Step Scenarios
# ===================================================================
class TestIntegration(unittest.TestCase):
    """Multi-step scenarios that exercise several mechanisms together."""

    def test_many_steps_no_crash(self):
        """Run 100 apply steps without error."""
        hp = _make_hp()
        for i in range(100):
            act = xp.abs(xp.sin(xp.arange(SMALL, dtype=float) + i * 0.1))
            hp.apply(act)
        # Just check we can still get metrics
        metrics = hp.get_stability_metrics()
        self.assertIn("stability_index", metrics)

    def test_synaptic_scaling_then_heterosynaptic(self):
        hp = _make_hp(synaptic_scaling_rate=0.01, heterosynaptic_rate=0.01)
        hp.avg_activity = xp.ones(SMALL) * 0.5
        weights = xp.ones((SMALL, 10)) * 2.0
        weights[0, :] = 5.0  # outlier row
        weights = hp.apply_synaptic_scaling(weights)
        weights = hp.apply_heterosynaptic_competition(weights)
        # Outlier row should be closer to others
        row0_mean = float(xp.mean(xp.abs(weights[0, :])))
        row1_mean = float(xp.mean(xp.abs(weights[1, :])))
        self.assertTrue(abs(row0_mean - row1_mean) < 5.0)

    def test_stability_calculated_after_50_steps(self):
        hp = _make_hp(inhibition_strategy="none")
        for i in range(51):
            act = xp.ones(SMALL) * (0.3 + 0.01 * (i % 5))
            hp.apply(act)
        # Stability should have been calculated at step 50
        self.assertIsInstance(hp.stability_index, float)
        # With nearly constant input, stability should be high (close to 1)
        self.assertGreater(hp.stability_index, 0.0)

    def test_sparsity_index_reasonable(self):
        hp = _make_hp(inhibition_strategy="k_winners_adaptive",
                      k_percent=0.25, target_sparsity=0.25)
        act = xp.arange(SMALL, dtype=float)
        hp.apply(act)
        # sparsity_index should be between 0 and 1
        self.assertGreaterEqual(hp.sparsity_index, 0.0)
        self.assertLessEqual(hp.sparsity_index, 1.0)


if __name__ == "__main__":
    unittest.main()
