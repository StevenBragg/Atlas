"""
Comprehensive tests for the structural plasticity module.

Tests cover initialization, plasticity modes, neuron growth, connection pruning,
update mechanics, resize operations, and state serialization.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.structural_plasticity import (
    StructuralPlasticity,
    PlasticityMode,
    GrowthStrategy,
    PruningStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_activity(size, value=0.5):
    """Create a uniform activity pattern."""
    return xp.full(size, value, dtype=float)


def _make_peaked_activity(size, index, base=0.15, peak=1.0):
    """Create an activity pattern with a peak at *index* and a low base elsewhere.

    The default *base* of 0.15 ensures every element exceeds the 0.1 threshold
    used by the novelty-detection sparsity check (requires >= 2 elements > 0.1).
    """
    a = xp.full(size, base, dtype=float)
    a[index % size] = peak
    return a


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums(unittest.TestCase):
    """Verify enum members and their string values."""

    def test_plasticity_mode_values(self):
        self.assertEqual(PlasticityMode.STABLE.value, "stable")
        self.assertEqual(PlasticityMode.GROWING.value, "growing")
        self.assertEqual(PlasticityMode.PRUNING.value, "pruning")
        self.assertEqual(PlasticityMode.ADAPTIVE.value, "adaptive")
        self.assertEqual(PlasticityMode.SPROUTING.value, "sprouting")

    def test_growth_strategy_values(self):
        self.assertEqual(GrowthStrategy.ACTIVITY_BASED.value, "activity_based")
        self.assertEqual(GrowthStrategy.NOVELTY_BASED.value, "novelty_based")
        self.assertEqual(GrowthStrategy.ERROR_BASED.value, "error_based")
        self.assertEqual(GrowthStrategy.HYBRID.value, "hybrid")

    def test_pruning_strategy_values(self):
        self.assertEqual(PruningStrategy.WEIGHT_BASED.value, "weight_based")
        self.assertEqual(PruningStrategy.ACTIVITY_BASED.value, "activity_based")
        self.assertEqual(PruningStrategy.CORRELATION_BASED.value, "correlation_based")
        self.assertEqual(PruningStrategy.UTILITY_BASED.value, "utility_based")

    def test_enum_round_trip_from_value(self):
        """Creating an enum member from its .value should return the same member."""
        self.assertIs(PlasticityMode("stable"), PlasticityMode.STABLE)
        self.assertIs(GrowthStrategy("hybrid"), GrowthStrategy.HYBRID)
        self.assertIs(PruningStrategy("weight_based"), PruningStrategy.WEIGHT_BASED)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInitializationDefaults(unittest.TestCase):
    """Test that default parameter values are applied correctly."""

    def setUp(self):
        self.sp = StructuralPlasticity(initial_size=20, random_seed=42)

    def test_current_size(self):
        self.assertEqual(self.sp.current_size, 20)

    def test_max_size_is_five_times_initial(self):
        self.assertEqual(self.sp.max_size, 100)

    def test_min_size(self):
        self.assertEqual(self.sp.min_size, 10)

    def test_growth_threshold(self):
        self.assertAlmostEqual(self.sp.growth_threshold, 0.8)

    def test_growth_rate(self):
        self.assertAlmostEqual(self.sp.growth_rate, 0.1)

    def test_prune_threshold(self):
        self.assertAlmostEqual(self.sp.prune_threshold, 0.01)

    def test_utility_window(self):
        self.assertEqual(self.sp.utility_window, 100)

    def test_growth_strategy(self):
        self.assertEqual(self.sp.growth_strategy, GrowthStrategy.HYBRID)

    def test_pruning_strategy(self):
        self.assertEqual(self.sp.pruning_strategy, PruningStrategy.WEIGHT_BASED)

    def test_growth_cooldown(self):
        self.assertEqual(self.sp.growth_cooldown, 50)

    def test_check_interval(self):
        self.assertEqual(self.sp.check_interval, 100)

    def test_novelty_threshold(self):
        self.assertAlmostEqual(self.sp.novelty_threshold, 0.3)

    def test_redundancy_threshold(self):
        self.assertAlmostEqual(self.sp.redundancy_threshold, 0.85)

    def test_max_growth_per_step(self):
        self.assertEqual(self.sp.max_growth_per_step, 5)

    def test_max_prune_per_step(self):
        self.assertEqual(self.sp.max_prune_per_step, 3)

    def test_enable_consolidation(self):
        self.assertTrue(self.sp.enable_consolidation)

    def test_structural_plasticity_mode(self):
        self.assertEqual(self.sp.structural_plasticity_mode, PlasticityMode.ADAPTIVE)

    def test_fan_limits_none(self):
        self.assertIsNone(self.sp.max_fan_in)
        self.assertIsNone(self.sp.max_fan_out)

    def test_weight_init_params(self):
        self.assertAlmostEqual(self.sp.weight_init_mean, 0.0)
        self.assertAlmostEqual(self.sp.weight_init_std, 0.1)

    def test_enable_flags(self):
        self.assertTrue(self.sp.enable_neuron_growth)
        self.assertTrue(self.sp.enable_connection_pruning)
        self.assertTrue(self.sp.enable_connection_sprouting)


class TestInitializationCustom(unittest.TestCase):
    """Test initialization with non-default parameters."""

    def test_custom_sizes(self):
        sp = StructuralPlasticity(
            initial_size=30, max_size=200, min_size=5, random_seed=42,
        )
        self.assertEqual(sp.current_size, 30)
        self.assertEqual(sp.max_size, 200)
        self.assertEqual(sp.min_size, 5)

    def test_custom_strategies(self):
        sp = StructuralPlasticity(
            initial_size=10,
            growth_strategy=GrowthStrategy.NOVELTY_BASED,
            pruning_strategy=PruningStrategy.ACTIVITY_BASED,
            random_seed=42,
        )
        self.assertEqual(sp.growth_strategy, GrowthStrategy.NOVELTY_BASED)
        self.assertEqual(sp.pruning_strategy, PruningStrategy.ACTIVITY_BASED)

    def test_string_enum_conversion_for_strategies(self):
        """growth_strategy and pruning_strategy accept string values and convert to enums."""
        sp = StructuralPlasticity(
            initial_size=10,
            growth_strategy="error_based",
            pruning_strategy="utility_based",
            random_seed=42,
        )
        self.assertEqual(sp.growth_strategy, GrowthStrategy.ERROR_BASED)
        self.assertEqual(sp.pruning_strategy, PruningStrategy.UTILITY_BASED)

    def test_structural_plasticity_mode_stored_as_given(self):
        """structural_plasticity_mode stores whatever type is passed (enum or string)."""
        sp_enum = StructuralPlasticity(
            initial_size=10,
            structural_plasticity_mode=PlasticityMode.GROWING,
            random_seed=42,
        )
        self.assertEqual(sp_enum.structural_plasticity_mode, PlasticityMode.GROWING)

        sp_str = StructuralPlasticity(
            initial_size=10,
            structural_plasticity_mode="growing",
            random_seed=42,
        )
        self.assertEqual(sp_str.structural_plasticity_mode, "growing")

    def test_explicit_max_size_overrides_default(self):
        sp = StructuralPlasticity(initial_size=10, max_size=20, random_seed=42)
        self.assertEqual(sp.max_size, 20)

    def test_max_size_default_scales_with_initial(self):
        for n in (8, 12, 25):
            sp = StructuralPlasticity(initial_size=n, random_seed=42)
            self.assertEqual(sp.max_size, n * 5)

    def test_custom_fan_limits(self):
        sp = StructuralPlasticity(
            initial_size=10, max_fan_in=4, max_fan_out=6, random_seed=42,
        )
        self.assertEqual(sp.max_fan_in, 4)
        self.assertEqual(sp.max_fan_out, 6)

    def test_disable_flags(self):
        sp = StructuralPlasticity(
            initial_size=10,
            enable_neuron_growth=False,
            enable_connection_pruning=False,
            enable_connection_sprouting=False,
            random_seed=42,
        )
        self.assertFalse(sp.enable_neuron_growth)
        self.assertFalse(sp.enable_connection_pruning)
        self.assertFalse(sp.enable_connection_sprouting)


class TestInitialInternalState(unittest.TestCase):
    """Verify the internal bookkeeping arrays after construction."""

    def setUp(self):
        self.sp = StructuralPlasticity(initial_size=15, random_seed=42)

    def test_update_count_starts_at_zero(self):
        self.assertEqual(self.sp.update_count, 0)

    def test_empty_histories(self):
        self.assertEqual(len(self.sp.activity_history), 0)
        self.assertEqual(len(self.sp.reconstruction_errors), 0)
        self.assertEqual(len(self.sp.growth_events), 0)
        self.assertEqual(len(self.sp.prune_events), 0)
        self.assertEqual(len(self.sp.prototype_patterns), 0)

    def test_utility_scores_shape_and_value(self):
        self.assertEqual(len(self.sp.utility_scores), 15)
        np.testing.assert_array_equal(
            np.asarray(self.sp.utility_scores), np.ones(15),
        )

    def test_activation_frequency_shape_and_value(self):
        self.assertEqual(len(self.sp.activation_frequency), 15)
        np.testing.assert_array_equal(
            np.asarray(self.sp.activation_frequency), np.zeros(15),
        )


# ---------------------------------------------------------------------------
# PlasticityMode tests
# ---------------------------------------------------------------------------

class TestPlasticityModes(unittest.TestCase):
    """Ensure every PlasticityMode can be set and read back."""

    def test_each_mode_settable(self):
        for mode in PlasticityMode:
            sp = StructuralPlasticity(
                initial_size=10,
                structural_plasticity_mode=mode,
                random_seed=42,
            )
            self.assertEqual(sp.structural_plasticity_mode, mode)

    def test_each_mode_settable_by_string(self):
        """Passing a string stores the string; the code does not auto-convert it."""
        for mode in PlasticityMode:
            sp = StructuralPlasticity(
                initial_size=10,
                structural_plasticity_mode=mode.value,
                random_seed=42,
            )
            self.assertEqual(sp.structural_plasticity_mode, mode.value)


# ---------------------------------------------------------------------------
# Update method tests
# ---------------------------------------------------------------------------

class TestUpdateMethod(unittest.TestCase):
    """Tests for the update() method mechanics."""

    def setUp(self):
        self.sp = StructuralPlasticity(
            initial_size=10,
            max_size=10,          # Prevent growth
            check_interval=1,
            utility_window=10,
            random_seed=42,
        )

    def test_returns_dict(self):
        result = self.sp.update(_make_activity(10))
        self.assertIsInstance(result, dict)

    def test_result_has_required_keys(self):
        result = self.sp.update(_make_activity(10))
        for key in ('size_changed', 'grown', 'pruned',
                     'is_novel', 'current_size', 'consolidated'):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_update_increments_count(self):
        for i in range(5):
            self.sp.update(_make_activity(10))
            self.assertEqual(self.sp.update_count, i + 1)

    def test_activity_size_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            self.sp.update(_make_activity(5))

    def test_activity_history_grows(self):
        for i in range(1, 6):
            self.sp.update(_make_activity(10))
            self.assertEqual(len(self.sp.activity_history), i)

    def test_activity_history_capped_at_utility_window(self):
        for _ in range(15):
            self.sp.update(_make_activity(10))
        self.assertLessEqual(len(self.sp.activity_history), self.sp.utility_window)

    def test_reconstruction_error_tracked(self):
        for i in range(5):
            self.sp.update(_make_activity(10), reconstruction_error=0.1 * i)
        self.assertEqual(len(self.sp.reconstruction_errors), 5)

    def test_reconstruction_error_capped(self):
        for _ in range(15):
            self.sp.update(_make_activity(10), reconstruction_error=0.5)
        self.assertLessEqual(
            len(self.sp.reconstruction_errors), self.sp.utility_window,
        )

    def test_force_check(self):
        """force_check=True should trigger structural evaluation even outside interval."""
        sp = StructuralPlasticity(
            initial_size=10,
            max_size=10,
            check_interval=99999,
            random_seed=42,
        )
        # First update: 0 % 99999 == 0 => check happens automatically
        sp.update(_make_activity(10))
        # Second update: update_count=1, 1%99999 != 0, but force_check overrides
        result = sp.update(_make_activity(10), force_check=True)
        self.assertIn('size_changed', result)

    def test_stable_input_no_size_change(self):
        """Identical stable inputs with no room for growth/pruning should leave size unchanged."""
        for _ in range(8):
            result = self.sp.update(_make_activity(10))
        self.assertFalse(result['size_changed'])
        self.assertEqual(self.sp.current_size, 10)

    def test_first_update_is_novel(self):
        """The very first pattern is always flagged as novel."""
        result = self.sp.update(_make_activity(10, 0.6))
        self.assertTrue(result['is_novel'])

    def test_repeated_pattern_not_novel(self):
        """A repeated identical pattern should not be novel after the first time."""
        # First: novel (no prototypes yet)
        self.sp.update(_make_activity(10, 0.6))
        # Second: same pattern, similarity = 1.0, not novel
        result = self.sp.update(_make_activity(10, 0.6))
        self.assertFalse(result['is_novel'])


# ---------------------------------------------------------------------------
# Novelty detection tests
# ---------------------------------------------------------------------------

class TestNoveltyDetection(unittest.TestCase):
    """Test novelty detection through the update interface."""

    def test_orthogonal_patterns_are_novel(self):
        sp = StructuralPlasticity(
            initial_size=10,
            max_size=10,
            check_interval=1,
            novelty_threshold=0.3,
            random_seed=42,
        )
        # Pattern 1: peak at positions 0-1
        p1 = xp.zeros(10, dtype=float)
        p1[0] = 1.0
        p1[1] = 0.8
        sp.update(p1)  # Novel (first pattern)

        # Pattern 2: peak at positions 8-9 (orthogonal to p1)
        p2 = xp.zeros(10, dtype=float)
        p2[8] = 1.0
        p2[9] = 0.8
        result = sp.update(p2)
        self.assertTrue(result['is_novel'])

    def test_sparse_activity_not_novel(self):
        """Fewer than 2 elements above 0.1 should not be considered novel."""
        sp = StructuralPlasticity(initial_size=10, random_seed=42)
        sp.update(_make_activity(10, 0.6))  # seed a prototype

        sparse = xp.zeros(10, dtype=float)
        sparse[0] = 0.5  # only one element > 0.1
        result = sp.update(sparse)
        self.assertFalse(result['is_novel'])

    def test_prototype_accumulation(self):
        """Each novel pattern with max>0.5 adds a prototype."""
        sp = StructuralPlasticity(
            initial_size=10,
            max_size=10,
            check_interval=1,
            novelty_threshold=0.99,
            random_seed=42,
        )
        for i in range(6):
            sp.update(_make_peaked_activity(10, i))
        self.assertGreaterEqual(len(sp.prototype_patterns), 6)


# ---------------------------------------------------------------------------
# Neuron growth via resize
# ---------------------------------------------------------------------------

class TestResizeGrow(unittest.TestCase):
    """Test explicit growth through resize()."""

    def setUp(self):
        self.sp = StructuralPlasticity(initial_size=10, random_seed=42)

    def test_resize_increases_size(self):
        result = self.sp.resize(15)
        self.assertEqual(result['previous_size'], 10)
        self.assertEqual(result['new_size'], 15)
        self.assertEqual(result['change'], 5)
        self.assertEqual(self.sp.current_size, 15)

    def test_resize_extends_utility_scores(self):
        self.sp.resize(15)
        self.assertEqual(len(self.sp.utility_scores), 15)

    def test_resize_extends_activation_frequency(self):
        self.sp.resize(15)
        self.assertEqual(len(self.sp.activation_frequency), 15)

    def test_resize_records_growth_event(self):
        self.sp.resize(15)
        self.assertEqual(len(self.sp.growth_events), 1)
        evt = self.sp.growth_events[0]
        self.assertEqual(evt['grown'], 5)
        self.assertEqual(evt['new_size'], 15)
        self.assertEqual(evt['strategy'], 'manual')
        self.assertEqual(evt['reason'], 'explicit_resize')


class TestResizeShrink(unittest.TestCase):
    """Test explicit shrinking through resize()."""

    def setUp(self):
        self.sp = StructuralPlasticity(initial_size=20, random_seed=42)

    def test_resize_decreases_size(self):
        result = self.sp.resize(15)
        self.assertEqual(result['previous_size'], 20)
        self.assertEqual(result['new_size'], 15)
        self.assertEqual(result['change'], -5)
        self.assertEqual(self.sp.current_size, 15)

    def test_resize_shrinks_utility_scores(self):
        self.sp.resize(15)
        self.assertEqual(len(self.sp.utility_scores), 15)

    def test_resize_shrinks_activation_frequency(self):
        self.sp.resize(15)
        self.assertEqual(len(self.sp.activation_frequency), 15)

    def test_resize_returns_pruned_indices(self):
        result = self.sp.resize(15)
        self.assertIn('pruned_indices', result)
        self.assertEqual(len(result['pruned_indices']), 5)

    def test_resize_records_prune_event(self):
        self.sp.resize(15)
        self.assertEqual(len(self.sp.prune_events), 1)
        evt = self.sp.prune_events[0]
        self.assertEqual(evt['pruned'], 5)


class TestResizeClamping(unittest.TestCase):
    """Resize should clamp to [min_size, max_size]."""

    def test_clamp_to_min(self):
        sp = StructuralPlasticity(
            initial_size=20, min_size=10, random_seed=42,
        )
        result = sp.resize(5)
        self.assertEqual(result['new_size'], 10)
        self.assertEqual(sp.current_size, 10)

    def test_clamp_to_max(self):
        sp = StructuralPlasticity(
            initial_size=20, max_size=30, random_seed=42,
        )
        result = sp.resize(50)
        self.assertEqual(result['new_size'], 30)
        self.assertEqual(sp.current_size, 30)

    def test_no_change_same_size(self):
        sp = StructuralPlasticity(initial_size=20, random_seed=42)
        result = sp.resize(20)
        self.assertEqual(result['change'], 0)
        self.assertEqual(sp.current_size, 20)
        self.assertEqual(len(sp.growth_events), 0)
        self.assertEqual(len(sp.prune_events), 0)


# ---------------------------------------------------------------------------
# Neuron growth triggered through update()
# ---------------------------------------------------------------------------

class TestNeuronGrowthViaUpdate(unittest.TestCase):
    """Test automatic neuron recruitment driven by update()."""

    def test_novelty_based_growth(self):
        """Enough novel prototypes should trigger novelty-based growth."""
        sp = StructuralPlasticity(
            initial_size=10,
            growth_strategy=GrowthStrategy.NOVELTY_BASED,
            growth_cooldown=0,
            check_interval=1,
            novelty_threshold=0.99,  # Almost everything is novel
            utility_window=100,
            random_seed=42,
        )
        grew = False
        for i in range(12):
            act = _make_peaked_activity(sp.current_size, i)
            result = sp.update(act)
            if result['grown'] > 0:
                grew = True
                break
        self.assertTrue(grew, "Expected novelty-based growth to trigger")
        self.assertGreater(sp.current_size, 10)

    def test_error_based_growth(self):
        """A spike in reconstruction error should trigger error-based growth."""
        sp = StructuralPlasticity(
            initial_size=10,
            growth_strategy=GrowthStrategy.ERROR_BASED,
            growth_cooldown=0,
            check_interval=1,
            utility_window=20,
            random_seed=42,
        )
        # Build baseline with moderate error
        for _ in range(10):
            sp.update(_make_activity(10, 0.6), reconstruction_error=0.3)

        # Spike error well above baseline * 1.2
        result = sp.update(
            _make_activity(10, 0.6), reconstruction_error=0.5,
        )
        self.assertGreater(result['grown'], 0)
        self.assertEqual(sp.current_size, 11)

    def test_activity_based_growth(self):
        """Concentrated activity should trigger activity-based growth."""
        sp = StructuralPlasticity(
            initial_size=10,
            growth_strategy=GrowthStrategy.ACTIVITY_BASED,
            growth_threshold=0.05,   # Low threshold to make test feasible
            growth_cooldown=0,
            check_interval=1,
            random_seed=42,
        )
        # Feed concentrated pattern: all energy in one neuron
        concentrated = xp.zeros(10, dtype=float)
        concentrated[0] = 5.0

        grew = False
        for _ in range(11):
            result = sp.update(concentrated)
            if result['grown'] > 0:
                grew = True
                break
        self.assertTrue(grew, "Expected activity-based growth to trigger")

    def test_no_growth_at_max_size(self):
        """Growth must not happen when current_size >= max_size."""
        sp = StructuralPlasticity(
            initial_size=10,
            max_size=10,
            growth_strategy=GrowthStrategy.NOVELTY_BASED,
            growth_cooldown=0,
            check_interval=1,
            novelty_threshold=0.99,
            random_seed=42,
        )
        for i in range(15):
            act = _make_peaked_activity(10, i)
            result = sp.update(act)
            self.assertEqual(result['grown'], 0)
        self.assertEqual(sp.current_size, 10)

    def test_growth_cooldown_blocks_rapid_growth(self):
        """After growth the cooldown state should prevent immediate re-growth."""
        sp = StructuralPlasticity(
            initial_size=10,
            growth_strategy=GrowthStrategy.ERROR_BASED,
            growth_cooldown=1000,  # Very large cooldown
            check_interval=1,
            utility_window=20,
            random_seed=42,
        )
        # Build baseline
        for _ in range(10):
            sp.update(_make_activity(10, 0.6), reconstruction_error=0.3)

        # Spike -> growth
        result = sp.update(
            _make_activity(10, 0.6), reconstruction_error=0.5,
        )
        self.assertGreater(result['grown'], 0)

        # Verify that the cooldown state would block a second growth event:
        # elapsed time since growth must be less than cooldown
        elapsed = sp.update_count - sp.last_growth_update
        self.assertLess(elapsed, sp.growth_cooldown)

    def test_growth_records_event(self):
        """Growth through update should be recorded in growth_events."""
        sp = StructuralPlasticity(
            initial_size=10,
            growth_strategy=GrowthStrategy.ERROR_BASED,
            growth_cooldown=0,
            check_interval=1,
            utility_window=20,
            random_seed=42,
        )
        for _ in range(10):
            sp.update(_make_activity(10, 0.6), reconstruction_error=0.3)
        sp.update(_make_activity(10, 0.6), reconstruction_error=0.5)

        self.assertGreater(len(sp.growth_events), 0)
        evt = sp.growth_events[-1]
        self.assertIn('grown', evt)
        self.assertIn('new_size', evt)
        self.assertIn('strategy', evt)

    def test_growth_extends_internal_arrays(self):
        """After growth, utility_scores and activation_frequency must match new size."""
        sp = StructuralPlasticity(
            initial_size=10,
            growth_strategy=GrowthStrategy.ERROR_BASED,
            growth_cooldown=0,
            check_interval=1,
            utility_window=20,
            random_seed=42,
        )
        for _ in range(10):
            sp.update(_make_activity(10, 0.6), reconstruction_error=0.3)
        sp.update(_make_activity(10, 0.6), reconstruction_error=0.5)

        self.assertEqual(len(sp.utility_scores), sp.current_size)
        self.assertEqual(len(sp.activation_frequency), sp.current_size)

    def test_max_growth_per_step_cap(self):
        """Growth in a single step must not exceed max_growth_per_step."""
        sp = StructuralPlasticity(
            initial_size=10,
            growth_strategy=GrowthStrategy.ERROR_BASED,
            growth_cooldown=0,
            check_interval=1,
            utility_window=20,
            max_growth_per_step=2,
            growth_rate=0.5,  # Would want 5 neurons but cap is 2
            random_seed=42,
        )
        for _ in range(10):
            sp.update(_make_activity(10, 0.6), reconstruction_error=0.3)
        result = sp.update(
            _make_activity(10, 0.6), reconstruction_error=0.5,
        )
        self.assertLessEqual(result['grown'], 2)


# ---------------------------------------------------------------------------
# Connection pruning tests
# ---------------------------------------------------------------------------

class TestWeightBasedPruning(unittest.TestCase):
    """Test WEIGHT_BASED pruning through update()."""

    def test_weak_rows_are_pruned(self):
        """Rows with negligible weight norms should be pruned."""
        size = 20
        sp = StructuralPlasticity(
            initial_size=size,
            max_size=size,        # No growth
            min_size=10,          # Allow pruning
            pruning_strategy=PruningStrategy.WEIGHT_BASED,
            prune_threshold=0.01,
            check_interval=1,
            utility_window=10,
            random_seed=42,
        )

        # Build up enough activity history (need >= utility_window//2 = 5)
        weights = xp.ones((size, size), dtype=float) * 0.5
        weights[0, :] = 1e-4
        weights[1, :] = 1e-4

        pruned = False
        for _ in range(6):
            result = sp.update(_make_activity(size), weights=weights)
            if result['pruned'] > 0:
                pruned = True
                break

        self.assertTrue(pruned, "Expected weight-based pruning to occur")
        self.assertIn('prune_indices', result)
        self.assertTrue(result['size_changed'])

    def test_no_pruning_without_weights(self):
        """WEIGHT_BASED pruning returns empty when weights=None."""
        size = 20
        sp = StructuralPlasticity(
            initial_size=size,
            max_size=size,
            min_size=10,
            pruning_strategy=PruningStrategy.WEIGHT_BASED,
            check_interval=1,
            utility_window=10,
            random_seed=42,
        )
        for _ in range(6):
            result = sp.update(_make_activity(size))
        self.assertEqual(result['pruned'], 0)


class TestActivityBasedPruning(unittest.TestCase):
    """Test ACTIVITY_BASED pruning through update()."""

    def test_dead_neurons_are_pruned(self):
        """Neurons with consistently zero activity should be pruned."""
        size = 20
        sp = StructuralPlasticity(
            initial_size=size,
            max_size=size,
            min_size=10,
            pruning_strategy=PruningStrategy.ACTIVITY_BASED,
            prune_threshold=0.1,
            check_interval=1,
            utility_window=10,
            random_seed=42,
        )

        activity = xp.ones(size, dtype=float) * 0.5
        activity[0] = 0.0  # Dead neuron
        activity[1] = 0.0  # Dead neuron

        pruned = False
        for _ in range(6):
            result = sp.update(activity)
            if result['pruned'] > 0:
                pruned = True
                break

        self.assertTrue(pruned, "Expected activity-based pruning to occur")
        self.assertTrue(result['size_changed'])
        self.assertLess(sp.current_size, size)


class TestPruningEdgeCases(unittest.TestCase):
    """Edge-case pruning scenarios."""

    def test_no_pruning_below_min_size(self):
        """Pruning must not reduce size below min_size."""
        size = 10
        sp = StructuralPlasticity(
            initial_size=size,
            max_size=size,
            min_size=size,  # min == current => no pruning allowed
            pruning_strategy=PruningStrategy.ACTIVITY_BASED,
            prune_threshold=0.5,
            check_interval=1,
            utility_window=10,
            random_seed=42,
        )
        activity = xp.ones(size, dtype=float) * 0.5
        activity[0] = 0.0
        for _ in range(10):
            result = sp.update(activity)
        self.assertEqual(result['pruned'], 0)
        self.assertEqual(sp.current_size, size)

    def test_no_pruning_without_sufficient_history(self):
        """Pruning should not happen with too few activity history entries."""
        size = 20
        sp = StructuralPlasticity(
            initial_size=size,
            max_size=size,
            min_size=10,
            pruning_strategy=PruningStrategy.WEIGHT_BASED,
            check_interval=1,
            utility_window=100,  # Need >= 50 history entries
            random_seed=42,
        )
        weights = xp.ones((size, size), dtype=float) * 0.5
        weights[0, :] = 1e-6
        # Only 3 updates -- well below 50
        for _ in range(3):
            result = sp.update(_make_activity(size), weights=weights)
        self.assertEqual(result['pruned'], 0)

    def test_prune_events_recorded(self):
        """Pruning through update should add to prune_events."""
        size = 20
        sp = StructuralPlasticity(
            initial_size=size,
            max_size=size,
            min_size=10,
            pruning_strategy=PruningStrategy.ACTIVITY_BASED,
            prune_threshold=0.1,
            check_interval=1,
            utility_window=10,
            random_seed=42,
        )
        activity = xp.ones(size, dtype=float) * 0.5
        activity[0] = 0.0
        activity[1] = 0.0
        for _ in range(6):
            result = sp.update(activity)
            if result['pruned'] > 0:
                break

        self.assertGreater(len(sp.prune_events), 0)
        evt = sp.prune_events[-1]
        self.assertIn('pruned', evt)
        self.assertIn('strategy', evt)
        self.assertEqual(evt['strategy'], PruningStrategy.ACTIVITY_BASED.value)


# ---------------------------------------------------------------------------
# get_stats / get_state tests
# ---------------------------------------------------------------------------

class TestGetStats(unittest.TestCase):
    """Test the get_stats() method."""

    def setUp(self):
        self.sp = StructuralPlasticity(initial_size=10, random_seed=42)

    def test_returns_dict(self):
        stats = self.sp.get_stats()
        self.assertIsInstance(stats, dict)

    def test_expected_keys(self):
        expected = {
            'current_size', 'min_size', 'max_size', 'update_count',
            'total_grown', 'total_pruned', 'net_growth',
            'utility_mean', 'utility_std', 'utility_min', 'utility_max',
            'stability', 'prototype_count',
            'growth_strategy', 'pruning_strategy',
            'recent_grown', 'recent_pruned',
        }
        stats = self.sp.get_stats()
        for key in expected:
            self.assertIn(key, stats, f"Missing stats key: {key}")

    def test_initial_stats_values(self):
        stats = self.sp.get_stats()
        self.assertEqual(stats['current_size'], 10)
        self.assertEqual(stats['update_count'], 0)
        self.assertEqual(stats['total_grown'], 0)
        self.assertEqual(stats['total_pruned'], 0)
        self.assertEqual(stats['net_growth'], 0)
        self.assertEqual(stats['prototype_count'], 0)
        self.assertEqual(stats['growth_strategy'], GrowthStrategy.HYBRID.value)
        self.assertEqual(stats['pruning_strategy'], PruningStrategy.WEIGHT_BASED.value)

    def test_stats_types(self):
        stats = self.sp.get_stats()
        self.assertIsInstance(stats['current_size'], int)
        self.assertIsInstance(stats['utility_mean'], float)
        self.assertIsInstance(stats['stability'], float)

    def test_stats_after_updates(self):
        for _ in range(5):
            self.sp.update(_make_activity(10, 0.6))
        stats = self.sp.get_stats()
        self.assertEqual(stats['update_count'], 5)

    def test_stats_after_resize(self):
        self.sp.resize(15)
        stats = self.sp.get_stats()
        self.assertEqual(stats['current_size'], 15)
        self.assertEqual(stats['total_grown'], 5)

    def test_stability_metric_bounded(self):
        """Stability should be near 1.0 when no changes have occurred."""
        stats = self.sp.get_stats()
        self.assertGreater(stats['stability'], 0)
        self.assertLessEqual(stats['stability'], 1.0)


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerialize(unittest.TestCase):
    """Test the serialize() method."""

    def test_returns_dict(self):
        sp = StructuralPlasticity(initial_size=10, random_seed=42)
        data = sp.serialize()
        self.assertIsInstance(data, dict)

    def test_expected_keys_present(self):
        sp = StructuralPlasticity(initial_size=10, random_seed=42)
        data = sp.serialize()
        expected = {
            'current_size', 'max_size', 'min_size',
            'growth_threshold', 'growth_rate', 'prune_threshold',
            'utility_window', 'growth_strategy', 'pruning_strategy',
            'growth_cooldown', 'check_interval',
            'novelty_threshold', 'redundancy_threshold',
            'enable_consolidation', 'update_count',
            'last_growth_update', 'last_prune_update',
            'utility_scores', 'activation_frequency',
            'growth_events', 'prune_events', 'prototype_count',
            'min_age_for_pruning', 'structural_plasticity_mode',
            'max_fan_in', 'max_fan_out',
            'weight_init_mean', 'weight_init_std',
            'growth_increment', 'update_interval',
            'consolidation_threshold',
            'enable_neuron_growth', 'enable_connection_pruning',
            'enable_connection_sprouting',
        }
        for key in expected:
            self.assertIn(key, data, f"Missing serialize key: {key}")

    def test_utility_scores_are_list(self):
        sp = StructuralPlasticity(initial_size=10, random_seed=42)
        data = sp.serialize()
        self.assertIsInstance(data['utility_scores'], list)
        self.assertEqual(len(data['utility_scores']), 10)

    def test_strategy_values_are_strings(self):
        sp = StructuralPlasticity(initial_size=10, random_seed=42)
        data = sp.serialize()
        self.assertIsInstance(data['growth_strategy'], str)
        self.assertIsInstance(data['pruning_strategy'], str)
        self.assertIsInstance(data['structural_plasticity_mode'], str)

    def test_serialized_values_match_instance(self):
        sp = StructuralPlasticity(
            initial_size=10, max_size=40, min_size=5,
            growth_strategy=GrowthStrategy.ERROR_BASED,
            pruning_strategy=PruningStrategy.UTILITY_BASED,
            random_seed=42,
        )
        data = sp.serialize()
        self.assertEqual(data['current_size'], 10)
        self.assertEqual(data['max_size'], 40)
        self.assertEqual(data['min_size'], 5)
        self.assertEqual(data['growth_strategy'], 'error_based')
        self.assertEqual(data['pruning_strategy'], 'utility_based')


class TestDeserialize(unittest.TestCase):
    """Test the deserialize() class method."""

    def _make_complete_data(self, **overrides):
        """Return a minimal valid dictionary for deserialize()."""
        data = {
            'current_size': 10,
            'max_size': 50,
            'min_size': 5,
            'growth_threshold': 0.8,
            'growth_rate': 0.1,
            'prune_threshold': 0.01,
            'utility_window': 100,
            'growth_strategy': 'hybrid',
            'pruning_strategy': 'weight_based',
            'growth_cooldown': 50,
            'check_interval': 100,
            'novelty_threshold': 0.3,
            'redundancy_threshold': 0.85,
            'max_growth_per_step': 5,
            'max_prune_per_step': 3,
            'enable_consolidation': True,
            'update_count': 7,
            'last_growth_update': 3,
            'last_prune_update': 0,
            'utility_scores': [1.0] * 10,
            'activation_frequency': [0.0] * 10,
            'growth_events': [],
            'prune_events': [],
            'min_age_for_pruning': 100,
            'structural_plasticity_mode': 'adaptive',
            'max_fan_in': None,
            'max_fan_out': None,
            'weight_init_mean': 0.0,
            'weight_init_std': 0.1,
            'growth_increment': 1,
            'update_interval': 10,
            'consolidation_threshold': 0.5,
            'enable_neuron_growth': True,
            'enable_connection_pruning': True,
            'enable_connection_sprouting': True,
        }
        data.update(overrides)
        return data

    def test_creates_instance(self):
        data = self._make_complete_data()
        sp = StructuralPlasticity.deserialize(data)
        self.assertIsInstance(sp, StructuralPlasticity)

    def test_restores_current_size(self):
        data = self._make_complete_data(current_size=12,
                                        utility_scores=[1.0]*12,
                                        activation_frequency=[0.0]*12)
        sp = StructuralPlasticity.deserialize(data)
        self.assertEqual(sp.current_size, 12)

    def test_restores_update_count(self):
        data = self._make_complete_data(update_count=42)
        sp = StructuralPlasticity.deserialize(data)
        self.assertEqual(sp.update_count, 42)

    def test_restores_last_growth_update(self):
        data = self._make_complete_data(last_growth_update=10)
        sp = StructuralPlasticity.deserialize(data)
        self.assertEqual(sp.last_growth_update, 10)

    def test_restores_utility_scores(self):
        scores = [0.5, 1.0, 0.3, 0.8, 0.2, 0.9, 0.4, 0.7, 0.6, 0.1]
        data = self._make_complete_data(utility_scores=scores)
        sp = StructuralPlasticity.deserialize(data)
        np.testing.assert_array_almost_equal(
            np.asarray(sp.utility_scores), np.array(scores),
        )

    def test_restores_growth_events(self):
        events = [{'update': 5, 'grown': 2, 'new_size': 12,
                    'strategy': 'hybrid', 'reason': 'utility'}]
        data = self._make_complete_data(growth_events=events)
        sp = StructuralPlasticity.deserialize(data)
        self.assertEqual(sp.growth_events, events)

    def test_restores_strategies_as_enums(self):
        data = self._make_complete_data(
            growth_strategy='novelty_based',
            pruning_strategy='activity_based',
        )
        sp = StructuralPlasticity.deserialize(data)
        self.assertEqual(sp.growth_strategy, GrowthStrategy.NOVELTY_BASED)
        self.assertEqual(sp.pruning_strategy, PruningStrategy.ACTIVITY_BASED)

    def test_deserialized_instance_functional(self):
        """A deserialized instance should accept update() calls."""
        data = self._make_complete_data()
        sp = StructuralPlasticity.deserialize(data)
        result = sp.update(_make_activity(10, 0.5))
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# Consolidation (light test)
# ---------------------------------------------------------------------------

class TestConsolidation(unittest.TestCase):
    """Light test for the consolidation pathway."""

    def test_consolidation_with_weights_and_enough_history(self):
        """When conditions are met, consolidated should become True."""
        sp = StructuralPlasticity(
            initial_size=10,
            max_size=10,
            min_size=10,
            check_interval=2,
            utility_window=100,
            enable_consolidation=True,
            random_seed=42,
        )
        # Feed 11 updates to build history >= 10 entries
        for _ in range(10):
            sp.update(_make_activity(10, 0.6))

        # The 11th update: update_count == 10, check_interval*5 == 10
        # 10 % 10 == 0 and no size change => consolidation path runs
        weights = xp.ones((10, 10), dtype=float)
        result = sp.update(_make_activity(10, 0.6), weights=weights)
        self.assertTrue(result['consolidated'])

    def test_consolidation_skipped_when_disabled(self):
        """With enable_consolidation=False, consolidated stays False."""
        sp = StructuralPlasticity(
            initial_size=10,
            max_size=10,
            min_size=10,
            check_interval=2,
            utility_window=100,
            enable_consolidation=False,
            random_seed=42,
        )
        for _ in range(10):
            sp.update(_make_activity(10, 0.6))
        weights = xp.ones((10, 10), dtype=float)
        result = sp.update(_make_activity(10, 0.6), weights=weights)
        self.assertFalse(result['consolidated'])


# ---------------------------------------------------------------------------
# Combined growth + stats integration test
# ---------------------------------------------------------------------------

class TestGrowthStatsIntegration(unittest.TestCase):
    """Verify that stats reflect growth that occurred via update."""

    def test_stats_reflect_growth(self):
        sp = StructuralPlasticity(
            initial_size=10,
            growth_strategy=GrowthStrategy.ERROR_BASED,
            growth_cooldown=0,
            check_interval=1,
            utility_window=20,
            random_seed=42,
        )
        for _ in range(10):
            sp.update(_make_activity(10, 0.6), reconstruction_error=0.3)
        sp.update(_make_activity(10, 0.6), reconstruction_error=0.5)

        stats = sp.get_stats()
        self.assertGreater(stats['total_grown'], 0)
        self.assertEqual(stats['current_size'], sp.current_size)
        self.assertGreater(stats['update_count'], 0)


if __name__ == '__main__':
    unittest.main()
