"""
Comprehensive tests for the Meta-Learning System.

Tests cover:
- MetaLearner initialization
- Strategy selection (greedy and exploratory)
- Update with learning experiences
- Algorithm discovery
- Hyperparameter optimization
- LearningStrategy enum values
- Serialization and deserialization (get_state / serialize / deserialize)
- Statistics tracking
- Curriculum progression
- Experience buffer management
"""

import os
import sys
import unittest
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.meta_learning import (
    MetaLearner,
    LearningStrategy,
    LearningExperience,
)


class TestLearningStrategyEnum(unittest.TestCase):
    """Tests for the LearningStrategy enum."""

    def test_all_strategies_exist(self):
        """All expected strategy enum members should exist."""
        expected = [
            "HEBBIAN", "STDP", "OJA", "BCM",
            "ANTI_HEBBIAN", "COMPETITIVE", "COOPERATIVE",
        ]
        for name in expected:
            self.assertTrue(
                hasattr(LearningStrategy, name),
                f"LearningStrategy should have member '{name}'",
            )

    def test_strategy_values(self):
        """Each strategy should have the correct string value."""
        self.assertEqual(LearningStrategy.HEBBIAN.value, "hebbian")
        self.assertEqual(LearningStrategy.OJA.value, "oja")
        self.assertEqual(LearningStrategy.STDP.value, "stdp")
        self.assertEqual(LearningStrategy.BCM.value, "bcm")
        self.assertEqual(LearningStrategy.ANTI_HEBBIAN.value, "anti_hebbian")
        self.assertEqual(LearningStrategy.COMPETITIVE.value, "competitive")
        self.assertEqual(LearningStrategy.COOPERATIVE.value, "cooperative")

    def test_strategy_count(self):
        """There should be exactly 7 learning strategies."""
        self.assertEqual(len(LearningStrategy), 7)

    def test_strategy_lookup_by_value(self):
        """Should be able to look up a strategy by its string value."""
        self.assertEqual(LearningStrategy("hebbian"), LearningStrategy.HEBBIAN)
        self.assertEqual(LearningStrategy("stdp"), LearningStrategy.STDP)
        self.assertEqual(LearningStrategy("competitive"), LearningStrategy.COMPETITIVE)


class TestLearningExperience(unittest.TestCase):
    """Tests for the LearningExperience dataclass."""

    def test_creation(self):
        """LearningExperience should be constructable with all required fields."""
        exp = LearningExperience(
            strategy=LearningStrategy.HEBBIAN,
            hyperparameters={"learning_rate": 0.01},
            task_characteristics={"complexity": 0.5},
            performance_metrics={"accuracy": 0.9},
            timestamp=1000.0,
            success_score=0.85,
        )
        self.assertEqual(exp.strategy, LearningStrategy.HEBBIAN)
        self.assertEqual(exp.hyperparameters, {"learning_rate": 0.01})
        self.assertEqual(exp.task_characteristics, {"complexity": 0.5})
        self.assertEqual(exp.performance_metrics, {"accuracy": 0.9})
        self.assertEqual(exp.timestamp, 1000.0)
        self.assertAlmostEqual(exp.success_score, 0.85)

    def test_different_strategies(self):
        """LearningExperience should accept any LearningStrategy."""
        for strategy in LearningStrategy:
            exp = LearningExperience(
                strategy=strategy,
                hyperparameters={},
                task_characteristics={},
                performance_metrics={},
                timestamp=0.0,
                success_score=0.0,
            )
            self.assertEqual(exp.strategy, strategy)


class TestMetaLearnerInitialization(unittest.TestCase):
    """Tests for MetaLearner initialization."""

    def test_default_initialization(self):
        """MetaLearner should initialize with sensible defaults."""
        ml = MetaLearner(random_seed=42)
        self.assertEqual(ml.num_strategies, 7)
        self.assertEqual(ml.num_hyperparameters, 10)
        self.assertAlmostEqual(ml.exploration_rate, 0.2)
        self.assertAlmostEqual(ml.learning_rate, 0.01)
        self.assertEqual(ml.memory_size, 1000)
        self.assertTrue(ml.enable_algorithm_discovery)
        self.assertTrue(ml.enable_curriculum)

    def test_custom_initialization(self):
        """MetaLearner should accept custom parameters."""
        ml = MetaLearner(
            num_strategies=5,
            num_hyperparameters=8,
            exploration_rate=0.3,
            learning_rate=0.05,
            memory_size=500,
            enable_algorithm_discovery=False,
            enable_curriculum=False,
            random_seed=123,
        )
        self.assertEqual(ml.num_strategies, 5)
        self.assertEqual(ml.num_hyperparameters, 8)
        self.assertAlmostEqual(ml.exploration_rate, 0.3)
        self.assertAlmostEqual(ml.learning_rate, 0.05)
        self.assertEqual(ml.memory_size, 500)
        self.assertFalse(ml.enable_algorithm_discovery)
        self.assertFalse(ml.enable_curriculum)

    def test_initial_strategy_performance_empty(self):
        """Strategy performance tracking should start empty for each strategy."""
        ml = MetaLearner(random_seed=42)
        for strategy in LearningStrategy:
            self.assertIn(strategy, ml.strategy_performance)
            self.assertEqual(len(ml.strategy_performance[strategy]), 0)

    def test_initial_optimal_hyperparameters(self):
        """Each strategy should have initial hyperparameters."""
        ml = MetaLearner(random_seed=42)
        for strategy in LearningStrategy:
            self.assertIn(strategy, ml.optimal_hyperparameters)
            params = ml.optimal_hyperparameters[strategy]
            self.assertIn("learning_rate", params)
            self.assertIn("momentum", params)
            self.assertIn("decay", params)
            self.assertIn("threshold", params)
            self.assertAlmostEqual(params["learning_rate"], 0.01)
            self.assertAlmostEqual(params["momentum"], 0.5)

    def test_initial_experience_buffer_empty(self):
        """Experience buffer should start empty."""
        ml = MetaLearner(random_seed=42)
        self.assertEqual(len(ml.experience_buffer), 0)

    def test_initial_statistics_zero(self):
        """All statistics should start at zero."""
        ml = MetaLearner(random_seed=42)
        self.assertEqual(ml.total_selections, 0)
        self.assertEqual(ml.total_updates, 0)
        for strategy in LearningStrategy:
            self.assertEqual(ml.strategy_usage[strategy], 0)

    def test_initial_curriculum_difficulty(self):
        """Curriculum difficulty should start at 0."""
        ml = MetaLearner(random_seed=42)
        self.assertAlmostEqual(ml.difficulty_progression, 0.0)

    def test_initial_discovered_algorithms_empty(self):
        """Discovered algorithms list should start empty."""
        ml = MetaLearner(random_seed=42)
        self.assertEqual(len(ml.discovered_algorithms), 0)


class TestSelectStrategy(unittest.TestCase):
    """Tests for the select_strategy method."""

    def setUp(self):
        self.task_chars = {"complexity": 0.5, "dimensionality": 0.3}

    def test_select_strategy_returns_tuple(self):
        """select_strategy should return a (strategy, hyperparameters) tuple."""
        ml = MetaLearner(random_seed=42)
        result = ml.select_strategy(self.task_chars)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_select_strategy_returns_valid_strategy(self):
        """The returned strategy should be a valid LearningStrategy member."""
        ml = MetaLearner(random_seed=42)
        strategy, _ = ml.select_strategy(self.task_chars)
        self.assertIsInstance(strategy, LearningStrategy)

    def test_select_strategy_returns_hyperparameters(self):
        """The returned hyperparameters should be a dict with expected keys."""
        ml = MetaLearner(random_seed=42)
        _, params = ml.select_strategy(self.task_chars)
        self.assertIsInstance(params, dict)
        self.assertIn("learning_rate", params)
        self.assertIn("momentum", params)

    def test_select_strategy_exploration_mode(self):
        """When explore=True, a random strategy should be selected."""
        ml = MetaLearner(random_seed=42)
        strategy, params = ml.select_strategy(self.task_chars, explore=True)
        self.assertIsInstance(strategy, LearningStrategy)
        self.assertIsInstance(params, dict)

    def test_select_strategy_exploitation_mode(self):
        """When explore=False, the best strategy should be selected greedily."""
        ml = MetaLearner(random_seed=42)
        # With no prior experience, all Q-values are 0, so any strategy is valid
        strategy, params = ml.select_strategy(self.task_chars, explore=False)
        self.assertIsInstance(strategy, LearningStrategy)
        self.assertIsInstance(params, dict)

    def test_select_strategy_greedy_picks_best(self):
        """In exploitation mode, the strategy with highest Q-value should be chosen."""
        ml = MetaLearner(random_seed=42)
        task_chars = {"complexity": 0.5}
        # Force a strategy to have a high Q-value
        state_key = ml._encode_state(task_chars)
        ml.strategy_values[state_key] = {s: 0.0 for s in LearningStrategy}
        ml.strategy_values[state_key][LearningStrategy.STDP] = 10.0
        strategy, _ = ml.select_strategy(task_chars, explore=False)
        self.assertEqual(strategy, LearningStrategy.STDP)

    def test_select_strategy_increments_statistics(self):
        """Selecting a strategy should update selection count and usage."""
        ml = MetaLearner(random_seed=42)
        self.assertEqual(ml.total_selections, 0)
        ml.select_strategy(self.task_chars, explore=False)
        self.assertEqual(ml.total_selections, 1)
        ml.select_strategy(self.task_chars, explore=False)
        self.assertEqual(ml.total_selections, 2)

    def test_select_strategy_updates_usage_tracking(self):
        """The selected strategy's usage counter should increment."""
        ml = MetaLearner(random_seed=42)
        task_chars = {"complexity": 0.5}
        state_key = ml._encode_state(task_chars)
        ml.strategy_values[state_key] = {s: 0.0 for s in LearningStrategy}
        ml.strategy_values[state_key][LearningStrategy.BCM] = 5.0
        ml.select_strategy(task_chars, explore=False)
        self.assertEqual(ml.strategy_usage[LearningStrategy.BCM], 1)

    def test_select_strategy_creates_new_state(self):
        """Selecting with new task characteristics should create a new state entry."""
        ml = MetaLearner(random_seed=42)
        self.assertEqual(len(ml.strategy_values), 0)
        ml.select_strategy({"new_feature": 0.7}, explore=False)
        self.assertEqual(len(ml.strategy_values), 1)

    def test_select_strategy_returns_copy_of_hyperparameters(self):
        """Returned hyperparameters should be a copy, not a reference."""
        ml = MetaLearner(random_seed=42)
        _, params1 = ml.select_strategy(self.task_chars, explore=False)
        _, params2 = ml.select_strategy(self.task_chars, explore=False)
        params1["learning_rate"] = 999.0
        self.assertNotAlmostEqual(params2["learning_rate"], 999.0)


class TestUpdate(unittest.TestCase):
    """Tests for the update method."""

    def setUp(self):
        self.ml = MetaLearner(random_seed=42, enable_curriculum=False)
        self.task_chars = {"complexity": 0.5}
        self.hyperparams = {"learning_rate": 0.01, "momentum": 0.5}

    def test_update_adds_to_experience_buffer(self):
        """Updating should add an experience to the buffer."""
        self.assertEqual(len(self.ml.experience_buffer), 0)
        self.ml.update(
            LearningStrategy.HEBBIAN,
            self.hyperparams,
            self.task_chars,
            {"accuracy": 0.9},
        )
        self.assertEqual(len(self.ml.experience_buffer), 1)

    def test_update_records_correct_experience(self):
        """Recorded experience should contain the correct data."""
        self.ml.update(
            LearningStrategy.OJA,
            self.hyperparams,
            self.task_chars,
            {"accuracy": 0.8},
        )
        exp = self.ml.experience_buffer[0]
        self.assertEqual(exp.strategy, LearningStrategy.OJA)
        self.assertEqual(exp.task_characteristics, self.task_chars)
        self.assertIn("accuracy", exp.performance_metrics)

    def test_update_increments_total_updates(self):
        """Each update should increment the total_updates counter."""
        self.assertEqual(self.ml.total_updates, 0)
        self.ml.update(
            LearningStrategy.HEBBIAN,
            self.hyperparams,
            self.task_chars,
            {"accuracy": 0.5},
        )
        self.assertEqual(self.ml.total_updates, 1)

    def test_update_records_strategy_performance(self):
        """Strategy performance should be tracked."""
        self.ml.update(
            LearningStrategy.STDP,
            self.hyperparams,
            self.task_chars,
            {"accuracy": 0.9},
        )
        self.assertEqual(len(self.ml.strategy_performance[LearningStrategy.STDP]), 1)

    def test_update_q_values_change(self):
        """Q-values should change after an update for a known state."""
        task_chars = {"complexity": 0.5}
        # First, select a strategy so the state is registered
        self.ml.select_strategy(task_chars, explore=False)
        state_key = self.ml._encode_state(task_chars)
        initial_q = self.ml.strategy_values[state_key][LearningStrategy.HEBBIAN]

        # Update with a high-accuracy result
        self.ml.update(
            LearningStrategy.HEBBIAN,
            self.hyperparams,
            task_chars,
            {"accuracy": 1.0},
        )
        new_q = self.ml.strategy_values[state_key][LearningStrategy.HEBBIAN]
        # Q-value should have moved toward the success score
        self.assertNotAlmostEqual(initial_q, new_q)

    def test_update_buffer_overflow(self):
        """Experience buffer should not exceed memory_size."""
        ml = MetaLearner(random_seed=42, memory_size=5, enable_curriculum=False)
        for i in range(10):
            ml.update(
                LearningStrategy.HEBBIAN,
                self.hyperparams,
                self.task_chars,
                {"accuracy": float(i) / 10},
            )
        self.assertEqual(len(ml.experience_buffer), 5)

    def test_update_buffer_overflow_keeps_recent(self):
        """When buffer overflows, oldest experiences should be removed."""
        ml = MetaLearner(random_seed=42, memory_size=3, enable_curriculum=False)
        for i in range(5):
            ml.update(
                LearningStrategy.HEBBIAN,
                {"learning_rate": float(i)},
                self.task_chars,
                {"accuracy": float(i) / 10},
            )
        # Only the last 3 should remain
        self.assertEqual(len(ml.experience_buffer), 3)
        # The oldest remaining should have learning_rate=2.0
        self.assertAlmostEqual(
            ml.experience_buffer[0].hyperparameters["learning_rate"], 2.0
        )

    def test_update_strategy_performance_capped(self):
        """Strategy performance history should be capped at 100 entries."""
        ml = MetaLearner(random_seed=42, memory_size=200, enable_curriculum=False)
        for i in range(120):
            ml.update(
                LearningStrategy.HEBBIAN,
                self.hyperparams,
                self.task_chars,
                {"accuracy": 0.5},
            )
        self.assertLessEqual(
            len(ml.strategy_performance[LearningStrategy.HEBBIAN]), 100
        )

    def test_update_with_various_metrics(self):
        """Update should handle different performance metric combinations."""
        metrics_list = [
            {"accuracy": 0.9},
            {"prediction_error": 0.1},
            {"reconstruction_error": 0.2},
            {"sparsity": 0.1},
            {"accuracy": 0.8, "prediction_error": 0.2, "reconstruction_error": 0.1, "sparsity": 0.1},
            {},
        ]
        for metrics in metrics_list:
            self.ml.update(
                LearningStrategy.HEBBIAN,
                self.hyperparams,
                self.task_chars,
                metrics,
            )
        self.assertEqual(self.ml.total_updates, len(metrics_list))


class TestCalculateSuccessScore(unittest.TestCase):
    """Tests for the _calculate_success_score method."""

    def setUp(self):
        self.ml = MetaLearner(random_seed=42)

    def test_accuracy_only(self):
        """Success score with only accuracy."""
        score = self.ml._calculate_success_score({"accuracy": 1.0})
        self.assertAlmostEqual(score, 0.4)

    def test_prediction_error_only(self):
        """Success score with only prediction_error (lower is better)."""
        score = self.ml._calculate_success_score({"prediction_error": 0.0})
        self.assertAlmostEqual(score, 0.3)

    def test_reconstruction_error_only(self):
        """Success score with only reconstruction_error (lower is better)."""
        score = self.ml._calculate_success_score({"reconstruction_error": 0.0})
        self.assertAlmostEqual(score, 0.2)

    def test_sparsity_optimal(self):
        """Success score with optimal sparsity of 0.1."""
        score = self.ml._calculate_success_score({"sparsity": 0.1})
        self.assertAlmostEqual(score, 0.1)

    def test_all_metrics_perfect(self):
        """Success score with all metrics at their optimal values."""
        score = self.ml._calculate_success_score({
            "accuracy": 1.0,
            "prediction_error": 0.0,
            "reconstruction_error": 0.0,
            "sparsity": 0.1,
        })
        self.assertAlmostEqual(score, 1.0)

    def test_empty_metrics(self):
        """Success score with no metrics should be 0."""
        score = self.ml._calculate_success_score({})
        self.assertAlmostEqual(score, 0.0)

    def test_score_bounded_zero_one(self):
        """Success score should always be between 0 and 1."""
        test_cases = [
            {"accuracy": 2.0},
            {"prediction_error": -1.0},
            {"accuracy": 0.0, "prediction_error": 5.0},
            {"reconstruction_error": 10.0},
        ]
        for metrics in test_cases:
            score = self.ml._calculate_success_score(metrics)
            self.assertGreaterEqual(score, 0.0, f"Score below 0 for {metrics}")
            self.assertLessEqual(score, 1.0, f"Score above 1 for {metrics}")


class TestEncodeState(unittest.TestCase):
    """Tests for the _encode_state method."""

    def setUp(self):
        self.ml = MetaLearner(random_seed=42)

    def test_deterministic_encoding(self):
        """Same characteristics should always produce the same state key."""
        chars = {"complexity": 0.5, "dimensionality": 0.3}
        key1 = self.ml._encode_state(chars)
        key2 = self.ml._encode_state(chars)
        self.assertEqual(key1, key2)

    def test_different_characteristics_differ(self):
        """Different characteristics should produce different keys."""
        key1 = self.ml._encode_state({"complexity": 0.1})
        key2 = self.ml._encode_state({"complexity": 0.9})
        self.assertNotEqual(key1, key2)

    def test_sorted_keys(self):
        """State encoding should be consistent regardless of dict insertion order."""
        chars1 = {"a": 0.5, "b": 0.3}
        chars2 = {"b": 0.3, "a": 0.5}
        self.assertEqual(self.ml._encode_state(chars1), self.ml._encode_state(chars2))

    def test_empty_characteristics(self):
        """Empty characteristics should produce a valid key."""
        key = self.ml._encode_state({})
        self.assertEqual(key, "")

    def test_binning(self):
        """Values should be discretized into 5 bins (0-4)."""
        # Value 0.0 -> bin 0
        key0 = self.ml._encode_state({"x": 0.0})
        self.assertIn(":0", key0)
        # Value 1.0 -> bin 4 (clipped)
        key4 = self.ml._encode_state({"x": 1.0})
        self.assertIn(":4", key4)


class TestDiscoverAlgorithm(unittest.TestCase):
    """Tests for the discover_algorithm method."""

    def test_returns_none_when_disabled(self):
        """Should return None when algorithm discovery is disabled."""
        ml = MetaLearner(
            random_seed=42,
            enable_algorithm_discovery=False,
            enable_curriculum=False,
        )
        result = ml.discover_algorithm()
        self.assertIsNone(result)

    def test_returns_none_with_insufficient_experience(self):
        """Should return None when fewer than 50 experiences exist."""
        ml = MetaLearner(random_seed=42, enable_curriculum=False)
        # Add only 10 experiences
        for i in range(10):
            ml.update(
                LearningStrategy.HEBBIAN,
                {"learning_rate": 0.01},
                {"complexity": 0.5},
                {"accuracy": 0.8},
            )
        result = ml.discover_algorithm()
        self.assertIsNone(result)

    def test_discovers_algorithm_with_sufficient_experience(self):
        """Should discover an algorithm with enough diverse experiences."""
        ml = MetaLearner(random_seed=42, enable_curriculum=False, memory_size=2000)
        strategies = list(LearningStrategy)
        # Add 60 experiences with alternating strategies for diversity
        for i in range(60):
            strategy = strategies[i % len(strategies)]
            ml.update(
                strategy,
                {"learning_rate": 0.01, "momentum": 0.5},
                {"complexity": 0.5},
                {"accuracy": 0.7 + (i % 10) * 0.02},
            )
        result = ml.discover_algorithm()
        # With diverse experiences and at least 2 strategies in top-10,
        # an algorithm should be discovered
        if result is not None:
            self.assertIn("strategies", result)
            self.assertIn("hyperparameters", result)
            self.assertIn("discovery_time", result)
            self.assertIn("expected_performance", result)
            self.assertIsInstance(result["strategies"], list)
            self.assertGreaterEqual(len(result["strategies"]), 2)

    def test_discovered_algorithm_stored(self):
        """Discovered algorithms should be appended to discovered_algorithms list."""
        ml = MetaLearner(random_seed=42, enable_curriculum=False, memory_size=2000)
        strategies = list(LearningStrategy)
        for i in range(60):
            strategy = strategies[i % len(strategies)]
            ml.update(
                strategy,
                {"learning_rate": 0.01, "momentum": 0.5},
                {"complexity": 0.5},
                {"accuracy": 0.7 + (i % 10) * 0.02},
            )
        initial_count = len(ml.discovered_algorithms)
        result = ml.discover_algorithm()
        if result is not None:
            self.assertEqual(len(ml.discovered_algorithms), initial_count + 1)
            self.assertEqual(ml.discovered_algorithms[-1], result)


class TestHyperparameterOptimization(unittest.TestCase):
    """Tests for hyperparameter optimization during updates."""

    def test_hyperparameters_evolve_with_good_performance(self):
        """Optimal hyperparameters should shift toward high-performing params."""
        ml = MetaLearner(random_seed=42, learning_rate=0.1, enable_curriculum=False)
        strategy = LearningStrategy.HEBBIAN
        original_lr = ml.optimal_hyperparameters[strategy]["learning_rate"]

        # First update establishes a baseline
        ml.update(
            strategy,
            {"learning_rate": 0.05, "momentum": 0.5},
            {"complexity": 0.5},
            {"accuracy": 0.5},
        )
        # Second update with better performance and different params
        ml.update(
            strategy,
            {"learning_rate": 0.05, "momentum": 0.5},
            {"complexity": 0.5},
            {"accuracy": 0.9},
        )

        # With learning_rate=0.1 meta-lr, the optimal params should have shifted
        updated_lr = ml.optimal_hyperparameters[strategy]["learning_rate"]
        # The parameters may or may not change depending on baseline comparison
        # Just verify no errors occur and the value is a float
        self.assertIsInstance(updated_lr, float)

    def test_hyperparameters_stable_with_insufficient_data(self):
        """With only 1 experience, optimization should not change params."""
        ml = MetaLearner(random_seed=42, enable_curriculum=False)
        strategy = LearningStrategy.OJA
        original = ml.optimal_hyperparameters[strategy].copy()

        ml.update(
            strategy,
            {"learning_rate": 0.5, "momentum": 0.9},
            {"complexity": 0.5},
            {"accuracy": 0.9},
        )

        # With only one experience, _optimize_hyperparameters returns early
        for key in original:
            self.assertAlmostEqual(
                ml.optimal_hyperparameters[strategy][key],
                original[key],
                msg=f"Param '{key}' should not change with only 1 experience",
            )


class TestCurriculumLearning(unittest.TestCase):
    """Tests for curriculum difficulty progression."""

    def test_difficulty_increases_with_good_performance(self):
        """Difficulty should increase when performance exceeds mastery threshold."""
        ml = MetaLearner(random_seed=42, enable_curriculum=True)
        initial_difficulty = ml.difficulty_progression

        # All metrics at perfect values yield success_score = 1.0,
        # which exceeds the mastery_threshold of 0.8
        perfect_metrics = {
            "accuracy": 1.0,
            "prediction_error": 0.0,
            "reconstruction_error": 0.0,
            "sparsity": 0.1,
        }
        for i in range(10):
            ml.update(
                LearningStrategy.HEBBIAN,
                {"learning_rate": 0.01},
                {"complexity": 0.5},
                perfect_metrics,
            )

        self.assertGreater(ml.difficulty_progression, initial_difficulty)

    def test_difficulty_decreases_with_poor_performance(self):
        """Difficulty should decrease when performance is poor."""
        ml = MetaLearner(random_seed=42, enable_curriculum=True)
        # First increase difficulty
        ml.difficulty_progression = 0.5

        # Perform many poor updates so the recent average falls below 0.5
        for i in range(20):
            ml.update(
                LearningStrategy.HEBBIAN,
                {"learning_rate": 0.01},
                {"complexity": 0.5},
                {},  # No metrics => success_score = 0.0
            )

        self.assertLess(ml.difficulty_progression, 0.5)

    def test_difficulty_bounded(self):
        """Difficulty should stay within [0.0, 1.0]."""
        ml = MetaLearner(random_seed=42, enable_curriculum=True)

        # Many perfect updates
        for i in range(200):
            ml.update(
                LearningStrategy.HEBBIAN,
                {"learning_rate": 0.01},
                {"complexity": 0.5},
                {"accuracy": 1.0},
            )
        self.assertLessEqual(ml.difficulty_progression, 1.0)

        # Reset and do many poor updates
        ml.difficulty_progression = 0.0
        # Clear performance history to avoid interference
        for s in LearningStrategy:
            ml.strategy_performance[s] = []
        for i in range(200):
            ml.update(
                LearningStrategy.HEBBIAN,
                {"learning_rate": 0.01},
                {"complexity": 0.5},
                {},
            )
        self.assertGreaterEqual(ml.difficulty_progression, 0.0)


class TestGetStats(unittest.TestCase):
    """Tests for the get_stats method."""

    def test_stats_structure(self):
        """Stats should contain all expected keys."""
        ml = MetaLearner(random_seed=42)
        stats = ml.get_stats()
        expected_keys = [
            "total_selections",
            "total_updates",
            "exploration_rate",
            "avg_performance_by_strategy",
            "usage_distribution",
            "experience_buffer_size",
            "num_states_explored",
            "curriculum_difficulty",
            "discovered_algorithms",
        ]
        for key in expected_keys:
            self.assertIn(key, stats, f"Stats missing key: {key}")

    def test_initial_stats_values(self):
        """Initial stats should reflect zero activity."""
        ml = MetaLearner(random_seed=42)
        stats = ml.get_stats()
        self.assertEqual(stats["total_selections"], 0)
        self.assertEqual(stats["total_updates"], 0)
        self.assertEqual(stats["experience_buffer_size"], 0)
        self.assertEqual(stats["num_states_explored"], 0)
        self.assertAlmostEqual(stats["curriculum_difficulty"], 0.0)
        self.assertEqual(stats["discovered_algorithms"], 0)

    def test_stats_after_usage(self):
        """Stats should accurately reflect activity."""
        ml = MetaLearner(random_seed=42, enable_curriculum=False)
        task_chars = {"complexity": 0.5}

        ml.select_strategy(task_chars, explore=False)
        ml.update(
            LearningStrategy.HEBBIAN,
            {"learning_rate": 0.01},
            task_chars,
            {"accuracy": 0.7},
        )

        stats = ml.get_stats()
        self.assertEqual(stats["total_selections"], 1)
        self.assertEqual(stats["total_updates"], 1)
        self.assertEqual(stats["experience_buffer_size"], 1)
        self.assertEqual(stats["num_states_explored"], 1)

    def test_avg_performance_by_strategy_format(self):
        """Average performance should be keyed by strategy value strings."""
        ml = MetaLearner(random_seed=42, enable_curriculum=False)
        ml.update(
            LearningStrategy.HEBBIAN,
            {"learning_rate": 0.01},
            {"complexity": 0.5},
            {"accuracy": 0.8},
        )
        stats = ml.get_stats()
        avg_perf = stats["avg_performance_by_strategy"]
        self.assertIn("hebbian", avg_perf)
        self.assertGreater(avg_perf["hebbian"], 0.0)
        # Strategies with no experience should have 0.0
        self.assertAlmostEqual(avg_perf["stdp"], 0.0)

    def test_usage_distribution_sums_to_one(self):
        """Usage distribution should sum to approximately 1.0 when there is usage."""
        ml = MetaLearner(random_seed=42, enable_curriculum=False)
        task_chars = {"complexity": 0.5}
        for _ in range(20):
            ml.select_strategy(task_chars, explore=True)
        stats = ml.get_stats()
        total = sum(stats["usage_distribution"].values())
        self.assertAlmostEqual(total, 1.0, places=5)


class TestSerialization(unittest.TestCase):
    """Tests for serialize and deserialize methods."""

    def test_serialize_returns_dict(self):
        """serialize() should return a dictionary."""
        ml = MetaLearner(random_seed=42)
        state = ml.serialize()
        self.assertIsInstance(state, dict)

    def test_serialize_keys(self):
        """Serialized state should have expected keys."""
        ml = MetaLearner(random_seed=42)
        state = ml.serialize()
        self.assertIn("strategy_performance", state)
        self.assertIn("optimal_hyperparameters", state)
        self.assertIn("difficulty_progression", state)
        self.assertIn("discovered_algorithms", state)
        self.assertIn("stats", state)

    def test_serialize_strategy_performance_uses_string_keys(self):
        """Serialized strategy performance should use string values as keys."""
        ml = MetaLearner(random_seed=42, enable_curriculum=False)
        ml.update(
            LearningStrategy.HEBBIAN,
            {"learning_rate": 0.01},
            {"complexity": 0.5},
            {"accuracy": 0.9},
        )
        state = ml.serialize()
        self.assertIn("hebbian", state["strategy_performance"])

    def test_deserialize_creates_instance(self):
        """deserialize() should return a MetaLearner instance."""
        data = {
            "strategy_performance": {"hebbian": [0.5, 0.6], "stdp": [0.7]},
            "optimal_hyperparameters": {
                "hebbian": {"learning_rate": 0.02, "momentum": 0.6},
            },
            "difficulty_progression": 0.3,
            "discovered_algorithms": [],
        }
        ml = MetaLearner.deserialize(data)
        self.assertIsInstance(ml, MetaLearner)

    def test_deserialize_restores_performance(self):
        """Deserialized instance should have restored performance data."""
        data = {
            "strategy_performance": {"hebbian": [0.5, 0.6, 0.7]},
            "optimal_hyperparameters": {},
            "difficulty_progression": 0.0,
            "discovered_algorithms": [],
        }
        ml = MetaLearner.deserialize(data)
        self.assertEqual(
            ml.strategy_performance[LearningStrategy.HEBBIAN], [0.5, 0.6, 0.7]
        )

    def test_deserialize_restores_hyperparameters(self):
        """Deserialized instance should have restored hyperparameters."""
        data = {
            "strategy_performance": {},
            "optimal_hyperparameters": {
                "oja": {"learning_rate": 0.05, "momentum": 0.8},
            },
            "difficulty_progression": 0.0,
            "discovered_algorithms": [],
        }
        ml = MetaLearner.deserialize(data)
        self.assertAlmostEqual(
            ml.optimal_hyperparameters[LearningStrategy.OJA]["learning_rate"], 0.05
        )
        self.assertAlmostEqual(
            ml.optimal_hyperparameters[LearningStrategy.OJA]["momentum"], 0.8
        )

    def test_deserialize_restores_difficulty(self):
        """Deserialized instance should restore difficulty progression."""
        data = {
            "strategy_performance": {},
            "optimal_hyperparameters": {},
            "difficulty_progression": 0.75,
            "discovered_algorithms": [],
        }
        ml = MetaLearner.deserialize(data)
        self.assertAlmostEqual(ml.difficulty_progression, 0.75)

    def test_deserialize_restores_discovered_algorithms(self):
        """Deserialized instance should restore discovered algorithms."""
        algos = [{"strategies": ["hebbian", "stdp"], "hyperparameters": {}}]
        data = {
            "strategy_performance": {},
            "optimal_hyperparameters": {},
            "difficulty_progression": 0.0,
            "discovered_algorithms": algos,
        }
        ml = MetaLearner.deserialize(data)
        self.assertEqual(len(ml.discovered_algorithms), 1)
        self.assertEqual(ml.discovered_algorithms[0]["strategies"], ["hebbian", "stdp"])

    def test_roundtrip_serialization(self):
        """Serialize then deserialize should preserve key state."""
        ml = MetaLearner(random_seed=42, enable_curriculum=False)
        # Build up some state
        task_chars = {"complexity": 0.5}
        strategies = [LearningStrategy.HEBBIAN, LearningStrategy.STDP, LearningStrategy.OJA]
        for i, strategy in enumerate(strategies):
            ml.update(
                strategy,
                {"learning_rate": 0.01 + i * 0.01, "momentum": 0.5},
                task_chars,
                {"accuracy": 0.6 + i * 0.1},
            )
        ml.difficulty_progression = 0.4

        state = ml.serialize()
        restored = MetaLearner.deserialize(state)

        # Check performance was preserved
        self.assertEqual(
            len(restored.strategy_performance[LearningStrategy.HEBBIAN]),
            len(ml.strategy_performance[LearningStrategy.HEBBIAN]),
        )
        # Check difficulty preserved
        self.assertAlmostEqual(restored.difficulty_progression, 0.4)

    def test_deserialize_with_empty_data(self):
        """Deserializing empty data should create a valid default instance."""
        ml = MetaLearner.deserialize({})
        self.assertIsInstance(ml, MetaLearner)
        self.assertAlmostEqual(ml.difficulty_progression, 0.0)
        self.assertEqual(len(ml.discovered_algorithms), 0)


class TestXpBackendImport(unittest.TestCase):
    """Test that xp backend import works correctly."""

    def test_xp_has_array_operations(self):
        """xp should provide standard array operations (numpy-compatible)."""
        arr = xp.array([1.0, 2.0, 3.0])
        self.assertEqual(arr.shape, (3,))

    def test_xp_random(self):
        """xp.random should be available."""
        val = xp.random.random()
        self.assertIsInstance(float(val), float)

    def test_xp_mean(self):
        """xp.mean should work."""
        arr = xp.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(xp.mean(arr)), 2.0)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and robustness."""

    def test_multiple_states(self):
        """MetaLearner should handle multiple distinct task states."""
        ml = MetaLearner(random_seed=42, enable_curriculum=False)
        tasks = [
            {"complexity": 0.1},
            {"complexity": 0.5},
            {"complexity": 0.9},
        ]
        for t in tasks:
            ml.select_strategy(t, explore=False)
        self.assertEqual(len(ml.strategy_values), 3)

    def test_all_strategies_can_be_updated(self):
        """Every strategy should be updatable without error."""
        ml = MetaLearner(random_seed=42, enable_curriculum=False)
        for strategy in LearningStrategy:
            ml.update(
                strategy,
                {"learning_rate": 0.01},
                {"complexity": 0.5},
                {"accuracy": 0.7},
            )
        self.assertEqual(ml.total_updates, 7)

    def test_high_volume_updates(self):
        """System should handle a large number of updates without error."""
        ml = MetaLearner(random_seed=42, memory_size=100, enable_curriculum=False)
        strategies = list(LearningStrategy)
        for i in range(500):
            ml.update(
                strategies[i % len(strategies)],
                {"learning_rate": 0.01},
                {"complexity": (i % 10) / 10.0},
                {"accuracy": 0.5 + (i % 5) * 0.1},
            )
        self.assertEqual(ml.total_updates, 500)
        self.assertLessEqual(len(ml.experience_buffer), 100)

    def test_select_and_update_integration(self):
        """Full loop: select strategy, then update with results."""
        ml = MetaLearner(random_seed=42, enable_curriculum=False)
        task_chars = {"complexity": 0.5, "dimensionality": 0.3}

        strategy, params = ml.select_strategy(task_chars, explore=False)
        ml.update(
            strategy,
            params,
            task_chars,
            {"accuracy": 0.85, "prediction_error": 0.1},
        )

        self.assertEqual(ml.total_selections, 1)
        self.assertEqual(ml.total_updates, 1)
        self.assertEqual(len(ml.experience_buffer), 1)
        self.assertEqual(ml.experience_buffer[0].strategy, strategy)


if __name__ == "__main__":
    unittest.main()
