"""
Comprehensive tests for the RecursiveSelfImprovement system.

Tests cover initialization, performance metric tracking, modification proposals,
capability assessment, safety level checks, and serialization/deserialization.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.self_improvement import (
    RecursiveSelfImprovement,
    ImprovementType,
    SafetyLevel,
    ModificationStatus,
    PerformanceMetric,
    Modification,
    Capability,
)


class TestRecursiveSelfImprovementInitialization(unittest.TestCase):
    """Tests for RecursiveSelfImprovement initialization."""

    def setUp(self):
        self.rsi = RecursiveSelfImprovement(random_seed=42)

    def test_default_initialization(self):
        """Test that default construction sets expected attribute values."""
        self.assertEqual(self.rsi.num_hyperparameters, 20)
        self.assertEqual(self.rsi.max_modifications, 100)
        self.assertEqual(self.rsi.safety_level, SafetyLevel.MODERATE)
        self.assertAlmostEqual(self.rsi.improvement_threshold, 0.05)
        self.assertAlmostEqual(self.rsi.reversion_threshold, -0.1)

    def test_custom_initialization(self):
        """Test construction with non-default parameters."""
        rsi = RecursiveSelfImprovement(
            num_hyperparameters=10,
            max_modifications=50,
            safety_level=SafetyLevel.MINIMAL,
            improvement_threshold=0.1,
            reversion_threshold=-0.2,
            random_seed=99,
        )
        self.assertEqual(rsi.num_hyperparameters, 10)
        self.assertEqual(rsi.max_modifications, 50)
        self.assertEqual(rsi.safety_level, SafetyLevel.MINIMAL)
        self.assertAlmostEqual(rsi.improvement_threshold, 0.1)
        self.assertAlmostEqual(rsi.reversion_threshold, -0.2)

    def test_hyperparameters_initialized(self):
        """Test that default hyperparameters are populated."""
        expected_params = [
            'learning_rate', 'momentum', 'weight_decay', 'batch_size',
            'hidden_dim', 'num_layers', 'dropout', 'temperature',
            'exploration_rate', 'discount_factor', 'plasticity_rate',
            'hebbian_rate', 'stdp_window', 'homeostatic_rate',
            'growth_threshold', 'pruning_threshold', 'attention_heads',
            'memory_capacity', 'consolidation_rate', 'creativity_temperature',
        ]
        for param in expected_params:
            self.assertIn(param, self.rsi.hyperparameters)
            self.assertIn(param, self.rsi.hyperparameter_ranges)

    def test_hyperparameter_ranges_valid(self):
        """Test that every hyperparameter value falls within its declared range."""
        for name, value in self.rsi.hyperparameters.items():
            min_val, max_val = self.rsi.hyperparameter_ranges[name]
            self.assertGreaterEqual(value, min_val,
                                    f"{name} value {value} below min {min_val}")
            self.assertLessEqual(value, max_val,
                                 f"{name} value {value} above max {max_val}")

    def test_metrics_initialized(self):
        """Test that performance metrics are populated."""
        expected_metrics = [
            'prediction_accuracy', 'learning_speed', 'memory_retention',
            'generalization', 'creativity_score', 'reasoning_depth',
            'efficiency', 'robustness', 'adaptability', 'overall_capability',
        ]
        for metric_name in expected_metrics:
            self.assertIn(metric_name, self.rsi.metrics)
            m = self.rsi.metrics[metric_name]
            self.assertIsInstance(m, PerformanceMetric)
            self.assertEqual(m.name, metric_name)

    def test_capabilities_initialized(self):
        """Test that capabilities are populated."""
        expected_capabilities = [
            'perception', 'memory', 'reasoning', 'planning',
            'learning', 'creativity', 'language', 'social',
            'metacognition', 'self_improvement',
        ]
        for cap_name in expected_capabilities:
            self.assertIn(cap_name, self.rsi.capabilities)
            c = self.rsi.capabilities[cap_name]
            self.assertIsInstance(c, Capability)
            self.assertEqual(c.name, cap_name)
            self.assertGreaterEqual(c.level, 0.0)
            self.assertLessEqual(c.level, 1.0)

    def test_initial_counters_zero(self):
        """Test that modification counters start at zero."""
        self.assertEqual(self.rsi.modification_counter, 0)
        self.assertEqual(self.rsi.safety_violations, 0)
        self.assertEqual(self.rsi.reverted_modifications, 0)
        self.assertEqual(self.rsi.successful_improvements, 0)
        self.assertEqual(self.rsi.generation, 0)
        self.assertEqual(len(self.rsi.modifications), 0)
        self.assertEqual(len(self.rsi.pending_modifications), 0)

    def test_initial_checkpoint_created(self):
        """Test that an initial checkpoint is created during construction."""
        self.assertGreaterEqual(len(self.rsi.checkpoints), 1)
        initial_cp = self.rsi.checkpoints[0]
        self.assertEqual(initial_cp['name'], 'initial')
        self.assertIn('hyperparameters', initial_cp)
        self.assertIn('metrics', initial_cp)
        self.assertIn('capabilities', initial_cp)

    def test_search_state_initially_empty(self):
        """Test that evolutionary search state is empty at init."""
        self.assertEqual(len(self.rsi.search_population), 0)
        self.assertIsNone(self.rsi.best_configuration)


class TestPerformanceMetricTracking(unittest.TestCase):
    """Tests for the PerformanceMetric dataclass and its tracking logic."""

    def test_metric_update_changes_value(self):
        """Test that updating a metric stores the new current value."""
        metric = PerformanceMetric(
            name='test_metric',
            current_value=0.5,
            baseline_value=0.5,
            target_value=0.9,
        )
        metric.update(0.7)
        self.assertAlmostEqual(metric.current_value, 0.7)

    def test_metric_history_records_entries(self):
        """Test that each update adds an entry to the history deque."""
        metric = PerformanceMetric(
            name='test_metric',
            current_value=0.5,
            baseline_value=0.5,
            target_value=0.9,
        )
        metric.update(0.6)
        metric.update(0.7)
        metric.update(0.8)
        self.assertEqual(len(metric.history), 3)
        values = [h['value'] for h in metric.history]
        self.assertEqual(values, [0.6, 0.7, 0.8])

    def test_metric_history_contains_timestamps(self):
        """Test that history entries include a timestamp."""
        metric = PerformanceMetric(
            name='test_metric',
            current_value=0.5,
            baseline_value=0.5,
            target_value=0.9,
        )
        metric.update(0.6)
        entry = metric.history[0]
        self.assertIn('timestamp', entry)
        self.assertIsInstance(entry['timestamp'], float)

    def test_improvement_rate_computation(self):
        """Test that improvement_rate is computed after at least two updates."""
        metric = PerformanceMetric(
            name='test_metric',
            current_value=1.0,
            baseline_value=1.0,
            target_value=2.0,
        )
        # First update: only one entry, improvement_rate should stay 0
        metric.update(1.0)
        self.assertAlmostEqual(metric.improvement_rate, 0.0)

        # Second update: improvement_rate should reflect the change
        metric.update(1.5)
        # rate = (1.5 - 1.0) / (|1.0| + 1e-8) = 0.5 / 1.0 ~ 0.5
        self.assertAlmostEqual(metric.improvement_rate, 0.5, places=5)

    def test_improvement_rate_negative(self):
        """Test that a decrease yields a negative improvement rate."""
        metric = PerformanceMetric(
            name='test_metric',
            current_value=1.0,
            baseline_value=1.0,
            target_value=2.0,
        )
        metric.update(1.0)
        metric.update(0.5)
        # rate = (0.5 - 1.0) / (|1.0| + 1e-8) = -0.5
        self.assertLess(metric.improvement_rate, 0.0)

    def test_get_trend_with_no_history(self):
        """Test that trend returns 0.0 when there is no history."""
        metric = PerformanceMetric(
            name='test_metric',
            current_value=0.5,
            baseline_value=0.5,
            target_value=0.9,
        )
        self.assertAlmostEqual(metric.get_trend(), 0.0)

    def test_get_trend_with_single_entry(self):
        """Test that trend returns 0.0 with only one history entry."""
        metric = PerformanceMetric(
            name='test_metric',
            current_value=0.5,
            baseline_value=0.5,
            target_value=0.9,
        )
        metric.update(0.6)
        self.assertAlmostEqual(metric.get_trend(), 0.0)

    def test_get_trend_positive(self):
        """Test that an increasing sequence produces a positive trend."""
        metric = PerformanceMetric(
            name='test_metric',
            current_value=0.0,
            baseline_value=0.0,
            target_value=1.0,
        )
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            metric.update(v)
        trend = metric.get_trend()
        self.assertGreater(trend, 0.0)

    def test_get_trend_negative(self):
        """Test that a decreasing sequence produces a negative trend."""
        metric = PerformanceMetric(
            name='test_metric',
            current_value=1.0,
            baseline_value=1.0,
            target_value=0.0,
        )
        for v in [0.9, 0.8, 0.7, 0.6, 0.5]:
            metric.update(v)
        trend = metric.get_trend()
        self.assertLess(trend, 0.0)

    def test_get_trend_window_parameter(self):
        """Test that the window parameter limits the trend calculation."""
        metric = PerformanceMetric(
            name='test_metric',
            current_value=0.0,
            baseline_value=0.0,
            target_value=1.0,
        )
        # Insert 10 decreasing values, then 3 increasing values
        for v in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            metric.update(v)
        for v in [0.3, 0.5, 0.7]:
            metric.update(v)
        # With a window of 3, the trend should be positive (0.3 -> 0.5 -> 0.7)
        trend = metric.get_trend(window=3)
        self.assertGreater(trend, 0.0)

    def test_metric_history_maxlen(self):
        """Test that history deque respects its maxlen of 1000."""
        metric = PerformanceMetric(
            name='test_metric',
            current_value=0.0,
            baseline_value=0.0,
            target_value=1.0,
        )
        for i in range(1100):
            metric.update(float(i))
        self.assertEqual(len(metric.history), 1000)

    def test_rsi_metric_update_through_system(self):
        """Test updating a metric through the RSI system's metrics dict."""
        rsi = RecursiveSelfImprovement(random_seed=42)
        metric = rsi.metrics['prediction_accuracy']
        original = metric.current_value
        metric.update(original + 0.1)
        self.assertAlmostEqual(metric.current_value, original + 0.1)
        self.assertEqual(len(metric.history), 1)


class TestModificationProposals(unittest.TestCase):
    """Tests for proposing modifications."""

    def setUp(self):
        self.rsi = RecursiveSelfImprovement(random_seed=42)

    def test_propose_hyperparameter_change_returns_modification(self):
        """Test that proposing a hyperparameter change returns a Modification."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER,
            'learning_rate',
            'gradient_free',
        )
        self.assertIsNotNone(mod)
        self.assertIsInstance(mod, Modification)
        self.assertEqual(mod.improvement_type, ImprovementType.HYPERPARAMETER)
        self.assertEqual(mod.status, ModificationStatus.PENDING)
        self.assertEqual(mod.parameters['param_name'], 'learning_rate')

    def test_propose_hyperparameter_stores_original_and_new(self):
        """Test that original and new values are captured correctly."""
        original_lr = self.rsi.hyperparameters['learning_rate']
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER,
            'learning_rate',
            'gradient_free',
        )
        self.assertAlmostEqual(mod.original_values['learning_rate'], original_lr)
        self.assertIn('learning_rate', mod.new_values)
        # New value should differ from original (with overwhelming probability at seed=42)
        # But it must be within range
        min_val, max_val = self.rsi.hyperparameter_ranges['learning_rate']
        self.assertGreaterEqual(mod.new_values['learning_rate'], min_val)
        self.assertLessEqual(mod.new_values['learning_rate'], max_val)

    def test_propose_hyperparameter_increments_counter(self):
        """Test that each proposal increments the modification counter."""
        self.assertEqual(self.rsi.modification_counter, 0)
        self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        self.assertEqual(self.rsi.modification_counter, 1)
        self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'momentum',
        )
        self.assertEqual(self.rsi.modification_counter, 2)

    def test_propose_hyperparameter_adds_to_pending(self):
        """Test that proposal is added to the pending list."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'dropout',
        )
        self.assertIn(mod.modification_id, self.rsi.pending_modifications)
        self.assertIn(mod.modification_id, self.rsi.modifications)

    def test_propose_hyperparameter_invalid_name(self):
        """Test that proposing a change for a nonexistent hyperparameter returns None."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'nonexistent_param',
        )
        self.assertIsNone(mod)

    def test_propose_hyperparameter_evolutionary_strategy(self):
        """Test evolutionary strategy for hyperparameter change."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER,
            'momentum',
            'evolutionary',
        )
        self.assertIsNotNone(mod)
        min_val, max_val = self.rsi.hyperparameter_ranges['momentum']
        self.assertGreaterEqual(mod.new_values['momentum'], min_val)
        self.assertLessEqual(mod.new_values['momentum'], max_val)

    def test_propose_hyperparameter_default_strategy(self):
        """Test the fallback strategy (not gradient_free or evolutionary)."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER,
            'temperature',
            'other_strategy',
        )
        self.assertIsNotNone(mod)
        min_val, max_val = self.rsi.hyperparameter_ranges['temperature']
        self.assertGreaterEqual(mod.new_values['temperature'], min_val)
        self.assertLessEqual(mod.new_values['temperature'], max_val)

    def test_propose_capability_enhancement_returns_modification(self):
        """Test that a capability enhancement proposal returns a Modification."""
        mod = self.rsi.propose_improvement(
            ImprovementType.CAPABILITY, 'perception',
        )
        self.assertIsNotNone(mod)
        self.assertEqual(mod.improvement_type, ImprovementType.CAPABILITY)
        self.assertEqual(mod.status, ModificationStatus.PENDING)

    def test_propose_capability_invalid_name(self):
        """Test that proposing a capability enhancement for a nonexistent capability returns None."""
        mod = self.rsi.propose_improvement(
            ImprovementType.CAPABILITY, 'nonexistent_capability',
        )
        self.assertIsNone(mod)

    def test_propose_architecture_change_returns_modification(self):
        """Test that an architecture change proposal returns a Modification."""
        mod = self.rsi.propose_improvement(
            ImprovementType.ARCHITECTURE, 'network',
        )
        self.assertIsNotNone(mod)
        self.assertEqual(mod.improvement_type, ImprovementType.ARCHITECTURE)
        # Architecture changes set safety to MODERATE
        self.assertEqual(mod.safety_level, SafetyLevel.MODERATE)

    def test_propose_architecture_change_modifies_arch_params(self):
        """Test that architecture proposals touch hidden_dim, num_layers, attention_heads."""
        mod = self.rsi.propose_improvement(
            ImprovementType.ARCHITECTURE, 'network',
        )
        for param in ['hidden_dim', 'num_layers', 'attention_heads']:
            self.assertIn(param, mod.original_values)
            self.assertIn(param, mod.new_values)

    def test_propose_unsupported_type_returns_none(self):
        """Test that unsupported improvement types return None."""
        mod = self.rsi.propose_improvement(
            ImprovementType.EFFICIENCY, 'something',
        )
        self.assertIsNone(mod)

    def test_apply_modification_success(self):
        """Test applying a pending modification successfully."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        result = self.rsi.apply_modification(mod.modification_id)
        self.assertTrue(result['success'])
        self.assertEqual(result['status'], 'testing')
        self.assertEqual(mod.status, ModificationStatus.TESTING)
        # Hyperparameter should be updated
        self.assertAlmostEqual(
            self.rsi.hyperparameters['learning_rate'],
            mod.new_values['learning_rate'],
        )

    def test_apply_modification_not_found(self):
        """Test applying a non-existent modification returns failure."""
        result = self.rsi.apply_modification('nonexistent')
        self.assertFalse(result['success'])
        self.assertIn('not found', result['reason'].lower())

    def test_apply_modification_already_applied(self):
        """Test that re-applying an already applied modification fails."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        self.rsi.apply_modification(mod.modification_id)
        # Status is now TESTING, not PENDING
        result = self.rsi.apply_modification(mod.modification_id)
        self.assertFalse(result['success'])

    def test_apply_modification_creates_checkpoint(self):
        """Test that applying a modification creates a checkpoint."""
        initial_checkpoints = len(self.rsi.checkpoints)
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        self.rsi.apply_modification(mod.modification_id)
        self.assertGreater(len(self.rsi.checkpoints), initial_checkpoints)

    def test_apply_modification_removes_from_pending(self):
        """Test that applying removes the modification from pending list."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        self.assertIn(mod.modification_id, self.rsi.pending_modifications)
        self.rsi.apply_modification(mod.modification_id)
        self.assertNotIn(mod.modification_id, self.rsi.pending_modifications)

    def test_apply_capability_modification(self):
        """Test applying a capability enhancement modification."""
        mod = self.rsi.propose_improvement(
            ImprovementType.CAPABILITY, 'perception',
        )
        cap_name = mod.parameters['capability_name']
        expected_level = mod.new_values[cap_name]
        self.rsi.apply_modification(mod.modification_id)
        self.assertAlmostEqual(self.rsi.capabilities[cap_name].level, expected_level)

    def test_apply_architecture_modification(self):
        """Test applying an architecture modification updates hyperparameters."""
        mod = self.rsi.propose_improvement(
            ImprovementType.ARCHITECTURE, 'network',
        )
        self.rsi.apply_modification(mod.modification_id)
        for param, value in mod.new_values.items():
            self.assertAlmostEqual(self.rsi.hyperparameters[param], value)

    def test_evaluate_modification_keep(self):
        """Test that a sufficiently good modification is kept."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        self.rsi.apply_modification(mod.modification_id)
        result = self.rsi.evaluate_modification(
            mod.modification_id,
            {'overall_capability': 0.2},  # Well above threshold of 0.05
        )
        self.assertTrue(result['success'])
        self.assertEqual(result['decision'], 'keep')
        self.assertEqual(mod.status, ModificationStatus.APPLIED)

    def test_evaluate_modification_revert(self):
        """Test that a badly performing modification is reverted."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        original_lr = mod.original_values['learning_rate']
        self.rsi.apply_modification(mod.modification_id)
        result = self.rsi.evaluate_modification(
            mod.modification_id,
            {'overall_capability': -0.5},  # Below reversion threshold of -0.1
        )
        self.assertTrue(result['success'])
        self.assertEqual(result['decision'], 'revert')
        self.assertEqual(mod.status, ModificationStatus.REVERTED)
        # Value should be restored
        self.assertAlmostEqual(self.rsi.hyperparameters['learning_rate'], original_lr)

    def test_evaluate_modification_keep_marginal(self):
        """Test that marginal improvements are kept but marked appropriately."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        self.rsi.apply_modification(mod.modification_id)
        result = self.rsi.evaluate_modification(
            mod.modification_id,
            {'overall_capability': 0.0},  # Between thresholds
        )
        self.assertTrue(result['success'])
        self.assertEqual(result['decision'], 'keep_marginal')

    def test_evaluate_modification_not_found(self):
        """Test evaluating a non-existent modification returns failure."""
        result = self.rsi.evaluate_modification('nonexistent', {})
        self.assertFalse(result['success'])

    def test_revert_modification(self):
        """Test explicit reversion of a modification."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'momentum',
        )
        original_value = mod.original_values['momentum']
        self.rsi.apply_modification(mod.modification_id)
        result = self.rsi.revert_modification(mod.modification_id)
        self.assertTrue(result['success'])
        self.assertEqual(result['status'], 'reverted')
        self.assertEqual(mod.status, ModificationStatus.REVERTED)
        self.assertAlmostEqual(self.rsi.hyperparameters['momentum'], original_value)
        self.assertEqual(self.rsi.reverted_modifications, 1)

    def test_revert_modification_not_found(self):
        """Test reverting a non-existent modification returns failure."""
        result = self.rsi.revert_modification('nonexistent')
        self.assertFalse(result['success'])

    def test_revert_capability_modification(self):
        """Test reverting a capability enhancement restores original level."""
        mod = self.rsi.propose_improvement(
            ImprovementType.CAPABILITY, 'perception',
        )
        cap_name = mod.parameters['capability_name']
        original_level = mod.original_values[cap_name]
        self.rsi.apply_modification(mod.modification_id)
        self.rsi.revert_modification(mod.modification_id)
        self.assertAlmostEqual(self.rsi.capabilities[cap_name].level, original_level)

    def test_revert_architecture_modification(self):
        """Test reverting an architecture modification restores original params."""
        mod = self.rsi.propose_improvement(
            ImprovementType.ARCHITECTURE, 'network',
        )
        original_values = dict(mod.original_values)
        self.rsi.apply_modification(mod.modification_id)
        self.rsi.revert_modification(mod.modification_id)
        for param, value in original_values.items():
            self.assertAlmostEqual(self.rsi.hyperparameters[param], value)

    def test_modification_id_format(self):
        """Test that modification IDs follow the expected format."""
        mod1 = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        mod2 = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'momentum',
        )
        self.assertEqual(mod1.modification_id, 'mod_0')
        self.assertEqual(mod2.modification_id, 'mod_1')


class TestCapabilityAssessment(unittest.TestCase):
    """Tests for capability tracking and assessment."""

    def setUp(self):
        self.rsi = RecursiveSelfImprovement(random_seed=42)

    def test_capabilities_have_correct_dependencies(self):
        """Test that capabilities have the expected dependency structure."""
        self.assertEqual(self.rsi.capabilities['perception'].dependencies, [])
        self.assertEqual(self.rsi.capabilities['memory'].dependencies, [])
        self.assertEqual(self.rsi.capabilities['reasoning'].dependencies, ['memory'])
        self.assertEqual(
            self.rsi.capabilities['planning'].dependencies,
            ['reasoning', 'memory'],
        )
        self.assertEqual(
            self.rsi.capabilities['creativity'].dependencies,
            ['memory', 'reasoning'],
        )
        self.assertEqual(
            self.rsi.capabilities['self_improvement'].dependencies,
            ['metacognition', 'learning'],
        )

    def test_capability_enhancement_increases_level(self):
        """Test that enhancing a capability increases its level."""
        original_level = self.rsi.capabilities['perception'].level
        mod = self.rsi.propose_improvement(
            ImprovementType.CAPABILITY, 'perception',
        )
        self.rsi.apply_modification(mod.modification_id)
        self.assertGreater(self.rsi.capabilities['perception'].level, original_level)

    def test_capability_level_capped_at_one(self):
        """Test that capability level does not exceed 1.0."""
        # Set level close to max
        self.rsi.capabilities['perception'].level = 0.98
        mod = self.rsi.propose_improvement(
            ImprovementType.CAPABILITY, 'perception',
        )
        self.rsi.apply_modification(mod.modification_id)
        self.assertLessEqual(self.rsi.capabilities['perception'].level, 1.0)

    def test_capability_dependency_redirect(self):
        """Test that enhancing a capability with a weak dependency redirects to the dependency."""
        # Set 'reasoning' (depends on 'memory') and make 'memory' weak
        self.rsi.capabilities['memory'].level = 0.3  # Below 0.5 threshold
        mod = self.rsi.propose_improvement(
            ImprovementType.CAPABILITY, 'reasoning',
        )
        # Should redirect to enhance 'memory' first
        self.assertIsNotNone(mod)
        self.assertEqual(mod.parameters['capability_name'], 'memory')

    def test_capability_no_redirect_when_deps_strong(self):
        """Test that capability enhancement does not redirect when dependencies are strong."""
        # Ensure dependencies of 'reasoning' are strong
        self.rsi.capabilities['memory'].level = 0.8
        mod = self.rsi.propose_improvement(
            ImprovementType.CAPABILITY, 'reasoning',
        )
        self.assertIsNotNone(mod)
        self.assertEqual(mod.parameters['capability_name'], 'reasoning')

    def test_get_improvement_recommendations_returns_list(self):
        """Test that recommendations are returned as a list."""
        recs = self.rsi.get_improvement_recommendations()
        self.assertIsInstance(recs, list)
        self.assertLessEqual(len(recs), 5)

    def test_recommendations_sorted_by_priority(self):
        """Test that recommendations are sorted by priority (descending)."""
        recs = self.rsi.get_improvement_recommendations()
        if len(recs) > 1:
            priorities = [r['priority'] for r in recs]
            for i in range(len(priorities) - 1):
                self.assertGreaterEqual(priorities[i], priorities[i + 1])

    def test_recommendations_contain_required_keys(self):
        """Test that each recommendation has the expected keys."""
        recs = self.rsi.get_improvement_recommendations()
        for rec in recs:
            self.assertIn('type', rec)
            self.assertIn('target', rec)
            self.assertIn('priority', rec)

    def test_capability_enhancement_amount_by_safety_level(self):
        """Test that the enhancement amount varies with safety level."""
        # MINIMAL: 0.05 increase
        rsi_min = RecursiveSelfImprovement(
            safety_level=SafetyLevel.MINIMAL, random_seed=42,
        )
        mod_min = rsi_min.propose_improvement(
            ImprovementType.CAPABILITY, 'perception',
        )
        original = rsi_min.capabilities['perception'].level
        expected_min = min(1.0, original + 0.05)
        self.assertAlmostEqual(mod_min.new_values['perception'], expected_min)

        # MODERATE: 0.1 increase
        rsi_mod = RecursiveSelfImprovement(
            safety_level=SafetyLevel.MODERATE, random_seed=42,
        )
        mod_mod = rsi_mod.propose_improvement(
            ImprovementType.CAPABILITY, 'perception',
        )
        original = rsi_mod.capabilities['perception'].level
        expected_mod = min(1.0, original + 0.1)
        self.assertAlmostEqual(mod_mod.new_values['perception'], expected_mod)

        # AGGRESSIVE: 0.15 increase
        rsi_agg = RecursiveSelfImprovement(
            safety_level=SafetyLevel.AGGRESSIVE, random_seed=42,
        )
        mod_agg = rsi_agg.propose_improvement(
            ImprovementType.CAPABILITY, 'perception',
        )
        original = rsi_agg.capabilities['perception'].level
        expected_agg = min(1.0, original + 0.15)
        self.assertAlmostEqual(mod_agg.new_values['perception'], expected_agg)


class TestSafetyLevelChecks(unittest.TestCase):
    """Tests for safety levels and their impact on modifications."""

    def test_safety_level_enum_values(self):
        """Test that SafetyLevel enum has the expected values."""
        self.assertEqual(SafetyLevel.MINIMAL.value, 'minimal')
        self.assertEqual(SafetyLevel.MODERATE.value, 'moderate')
        self.assertEqual(SafetyLevel.AGGRESSIVE.value, 'aggressive')

    def test_modification_status_enum_values(self):
        """Test that ModificationStatus enum has the expected values."""
        self.assertEqual(ModificationStatus.PENDING.value, 'pending')
        self.assertEqual(ModificationStatus.TESTING.value, 'testing')
        self.assertEqual(ModificationStatus.APPLIED.value, 'applied')
        self.assertEqual(ModificationStatus.REVERTED.value, 'reverted')
        self.assertEqual(ModificationStatus.FAILED.value, 'failed')

    def test_improvement_type_enum_values(self):
        """Test that ImprovementType enum has the expected values."""
        self.assertEqual(ImprovementType.HYPERPARAMETER.value, 'hyperparameter')
        self.assertEqual(ImprovementType.ARCHITECTURE.value, 'architecture')
        self.assertEqual(ImprovementType.ALGORITHM.value, 'algorithm')
        self.assertEqual(ImprovementType.CAPABILITY.value, 'capability')
        self.assertEqual(ImprovementType.EFFICIENCY.value, 'efficiency')
        self.assertEqual(ImprovementType.ROBUSTNESS.value, 'robustness')

    def test_safety_level_stored_on_rsi(self):
        """Test that the safety level is properly stored on the RSI object."""
        for level in SafetyLevel:
            rsi = RecursiveSelfImprovement(safety_level=level, random_seed=42)
            self.assertEqual(rsi.safety_level, level)

    def test_safety_level_affects_hyperparameter_scale(self):
        """Test that different safety levels produce different perturbation scales."""
        # Use the same seed for each to get comparable random draws
        results = {}
        for level in SafetyLevel:
            rsi = RecursiveSelfImprovement(safety_level=level, random_seed=42)
            original = rsi.hyperparameters['learning_rate']
            mod = rsi.propose_improvement(
                ImprovementType.HYPERPARAMETER,
                'learning_rate',
                'gradient_free',
            )
            delta = abs(mod.new_values['learning_rate'] - original)
            results[level] = delta

        # With the same random draw, AGGRESSIVE should produce larger deltas
        # than MINIMAL (since the scale multiplier is larger).
        # Note: due to clipping this is not guaranteed in all cases, but with
        # seed=42 the relationship holds.
        self.assertLessEqual(results[SafetyLevel.MINIMAL], results[SafetyLevel.AGGRESSIVE] + 1e-10)

    def test_modification_records_safety_level(self):
        """Test that a modification records its safety level."""
        rsi = RecursiveSelfImprovement(
            safety_level=SafetyLevel.MINIMAL, random_seed=42,
        )
        mod = rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        self.assertEqual(mod.safety_level, SafetyLevel.MINIMAL)

    def test_safety_violations_counter(self):
        """Test that safety_violations starts at zero and is accessible."""
        rsi = RecursiveSelfImprovement(random_seed=42)
        self.assertEqual(rsi.safety_violations, 0)

    def test_reversion_increments_counter(self):
        """Test that reverting a modification increments the reversion counter."""
        rsi = RecursiveSelfImprovement(random_seed=42)
        mod = rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        rsi.apply_modification(mod.modification_id)
        self.assertEqual(rsi.reverted_modifications, 0)
        rsi.revert_modification(mod.modification_id)
        self.assertEqual(rsi.reverted_modifications, 1)

    def test_successful_improvement_increments_counter(self):
        """Test that a kept modification increments the success counter."""
        rsi = RecursiveSelfImprovement(random_seed=42)
        mod = rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        rsi.apply_modification(mod.modification_id)
        self.assertEqual(rsi.successful_improvements, 0)
        rsi.evaluate_modification(
            mod.modification_id,
            {'overall_capability': 0.2},
        )
        self.assertEqual(rsi.successful_improvements, 1)

    def test_evaluate_revert_does_not_increment_success(self):
        """Test that a reverted modification does not count as successful."""
        rsi = RecursiveSelfImprovement(random_seed=42)
        mod = rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        rsi.apply_modification(mod.modification_id)
        rsi.evaluate_modification(
            mod.modification_id,
            {'overall_capability': -0.5},
        )
        self.assertEqual(rsi.successful_improvements, 0)
        self.assertEqual(rsi.reverted_modifications, 1)


class TestGetStateAndSerialization(unittest.TestCase):
    """Tests for get_stats() and serialize()/deserialize()."""

    def setUp(self):
        self.rsi = RecursiveSelfImprovement(random_seed=42)

    def test_get_stats_returns_dict(self):
        """Test that get_stats() returns a dictionary."""
        stats = self.rsi.get_stats()
        self.assertIsInstance(stats, dict)

    def test_get_stats_contains_expected_keys(self):
        """Test that get_stats() has all expected keys."""
        stats = self.rsi.get_stats()
        expected_keys = [
            'generation', 'total_modifications', 'pending_modifications',
            'successful_improvements', 'reverted_modifications',
            'safety_violations', 'safety_level', 'checkpoints',
            'current_metrics', 'current_capabilities', 'best_configuration',
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_get_stats_initial_values(self):
        """Test that initial stats reflect a freshly created system."""
        stats = self.rsi.get_stats()
        self.assertEqual(stats['generation'], 0)
        self.assertEqual(stats['total_modifications'], 0)
        self.assertEqual(stats['pending_modifications'], 0)
        self.assertEqual(stats['successful_improvements'], 0)
        self.assertEqual(stats['reverted_modifications'], 0)
        self.assertEqual(stats['safety_violations'], 0)
        self.assertEqual(stats['safety_level'], 'moderate')
        self.assertIsNone(stats['best_configuration'])

    def test_get_stats_metrics_match(self):
        """Test that stats metrics match actual metric values."""
        stats = self.rsi.get_stats()
        for name, value in stats['current_metrics'].items():
            self.assertAlmostEqual(value, self.rsi.metrics[name].current_value)

    def test_get_stats_capabilities_match(self):
        """Test that stats capabilities match actual capability levels."""
        stats = self.rsi.get_stats()
        for name, level in stats['current_capabilities'].items():
            self.assertAlmostEqual(level, self.rsi.capabilities[name].level)

    def test_get_stats_updates_after_modifications(self):
        """Test that stats reflect changes after modifications are applied."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        self.rsi.apply_modification(mod.modification_id)
        self.rsi.evaluate_modification(
            mod.modification_id,
            {'overall_capability': 0.2},
        )
        stats = self.rsi.get_stats()
        self.assertEqual(stats['total_modifications'], 1)
        self.assertEqual(stats['successful_improvements'], 1)

    def test_serialize_returns_dict(self):
        """Test that serialize() returns a dictionary."""
        data = self.rsi.serialize()
        self.assertIsInstance(data, dict)

    def test_serialize_contains_expected_keys(self):
        """Test that serialize() output has all expected top-level keys."""
        data = self.rsi.serialize()
        expected_keys = [
            'num_hyperparameters', 'max_modifications', 'safety_level',
            'improvement_threshold', 'reversion_threshold',
            'hyperparameters', 'hyperparameter_ranges', 'metrics',
            'capabilities', 'modifications', 'generation',
            'best_configuration', 'stats',
        ]
        for key in expected_keys:
            self.assertIn(key, data)

    def test_serialize_preserves_hyperparameters(self):
        """Test that serialized hyperparameters match the current values."""
        data = self.rsi.serialize()
        for name, value in self.rsi.hyperparameters.items():
            self.assertAlmostEqual(data['hyperparameters'][name], value)

    def test_serialize_preserves_metrics(self):
        """Test that serialized metrics contain the right structure."""
        data = self.rsi.serialize()
        for name, m_data in data['metrics'].items():
            self.assertIn('current_value', m_data)
            self.assertIn('baseline_value', m_data)
            self.assertIn('target_value', m_data)
            self.assertIn('improvement_rate', m_data)
            self.assertAlmostEqual(
                m_data['current_value'],
                self.rsi.metrics[name].current_value,
            )

    def test_serialize_preserves_capabilities(self):
        """Test that serialized capabilities contain the right structure."""
        data = self.rsi.serialize()
        for name, c_data in data['capabilities'].items():
            self.assertIn('level', c_data)
            self.assertIn('dependencies', c_data)
            self.assertAlmostEqual(
                c_data['level'],
                self.rsi.capabilities[name].level,
            )

    def test_serialize_includes_modifications(self):
        """Test that serialized data includes modifications after proposals."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        data = self.rsi.serialize()
        self.assertIn(mod.modification_id, data['modifications'])
        mod_data = data['modifications'][mod.modification_id]
        self.assertEqual(mod_data['modification_id'], mod.modification_id)
        self.assertEqual(mod_data['improvement_type'], 'hyperparameter')
        self.assertEqual(mod_data['status'], 'pending')

    def test_deserialize_roundtrip(self):
        """Test that serialize -> deserialize produces an equivalent system."""
        # Perform some operations first
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'learning_rate',
        )
        self.rsi.apply_modification(mod.modification_id)
        self.rsi.evaluate_modification(
            mod.modification_id,
            {'overall_capability': 0.2},
        )

        data = self.rsi.serialize()
        restored = RecursiveSelfImprovement.deserialize(data)

        # Check top-level attributes
        self.assertEqual(restored.num_hyperparameters, self.rsi.num_hyperparameters)
        self.assertEqual(restored.max_modifications, self.rsi.max_modifications)
        self.assertEqual(restored.safety_level, self.rsi.safety_level)
        self.assertAlmostEqual(
            restored.improvement_threshold, self.rsi.improvement_threshold,
        )
        self.assertAlmostEqual(
            restored.reversion_threshold, self.rsi.reversion_threshold,
        )
        self.assertEqual(restored.generation, self.rsi.generation)
        self.assertEqual(restored.best_configuration, self.rsi.best_configuration)

    def test_deserialize_restores_hyperparameters(self):
        """Test that deserialization restores hyperparameter values."""
        # Modify a hyperparameter
        self.rsi.hyperparameters['learning_rate'] = 0.05
        data = self.rsi.serialize()
        restored = RecursiveSelfImprovement.deserialize(data)
        self.assertAlmostEqual(
            restored.hyperparameters['learning_rate'], 0.05,
        )

    def test_deserialize_restores_metrics(self):
        """Test that deserialization restores metric current values."""
        self.rsi.metrics['prediction_accuracy'].update(0.8)
        data = self.rsi.serialize()
        restored = RecursiveSelfImprovement.deserialize(data)
        self.assertAlmostEqual(
            restored.metrics['prediction_accuracy'].current_value, 0.8,
        )

    def test_deserialize_restores_capabilities(self):
        """Test that deserialization restores capability levels."""
        self.rsi.capabilities['perception'].level = 0.9
        data = self.rsi.serialize()
        restored = RecursiveSelfImprovement.deserialize(data)
        self.assertAlmostEqual(
            restored.capabilities['perception'].level, 0.9,
        )

    def test_deserialize_restores_safety_level(self):
        """Test that deserialization restores the safety level enum."""
        for level in SafetyLevel:
            rsi = RecursiveSelfImprovement(safety_level=level, random_seed=42)
            data = rsi.serialize()
            restored = RecursiveSelfImprovement.deserialize(data)
            self.assertEqual(restored.safety_level, level)

    def test_serialize_safety_level_is_string(self):
        """Test that safety_level is serialized as a string."""
        data = self.rsi.serialize()
        self.assertIsInstance(data['safety_level'], str)
        self.assertEqual(data['safety_level'], 'moderate')


class TestRunOptimizationCycle(unittest.TestCase):
    """Tests for the run_optimization_cycle method."""

    def setUp(self):
        self.rsi = RecursiveSelfImprovement(random_seed=42)

    def test_optimization_cycle_returns_results(self):
        """Test that an optimization cycle returns the expected structure."""
        results = self.rsi.run_optimization_cycle(n_proposals=3)
        self.assertIn('proposals', results)
        self.assertIn('applied', results)
        self.assertIn('improved', results)
        self.assertIn('reverted', results)

    def test_optimization_cycle_creates_proposals(self):
        """Test that proposals are generated in the cycle."""
        results = self.rsi.run_optimization_cycle(n_proposals=3)
        self.assertGreater(len(results['proposals']), 0)
        self.assertLessEqual(len(results['proposals']), 3)

    def test_optimization_cycle_increments_generation(self):
        """Test that the generation counter increments after a cycle."""
        self.assertEqual(self.rsi.generation, 0)
        self.rsi.run_optimization_cycle(n_proposals=2)
        self.assertEqual(self.rsi.generation, 1)

    def test_optimization_cycle_with_custom_eval(self):
        """Test that a custom evaluation function is used if provided."""
        def always_good(config):
            return {'overall_capability': 0.5}

        results = self.rsi.run_optimization_cycle(
            n_proposals=2, evaluation_fn=always_good,
        )
        # All should be improved since 0.5 > improvement_threshold
        self.assertEqual(len(results['improved']), len(results['applied']))


class TestEvolutionarySearch(unittest.TestCase):
    """Tests for the evolutionary_search method."""

    def setUp(self):
        self.rsi = RecursiveSelfImprovement(random_seed=42)

    def test_evolutionary_search_returns_results(self):
        """Test that evolutionary search returns the expected structure."""
        results = self.rsi.evolutionary_search(
            population_size=6, generations=2,
        )
        self.assertIn('generations', results)
        self.assertIn('best_fitness', results)
        self.assertIn('best_configuration', results)
        self.assertIn('final_population_size', results)

    def test_evolutionary_search_sets_best_configuration(self):
        """Test that best_configuration is set after search."""
        self.rsi.evolutionary_search(population_size=6, generations=2)
        self.assertIsNotNone(self.rsi.best_configuration)

    def test_evolutionary_search_population_size(self):
        """Test that final population matches the requested size."""
        results = self.rsi.evolutionary_search(
            population_size=8, generations=2,
        )
        self.assertEqual(results['final_population_size'], 8)

    def test_apply_best_configuration_without_search(self):
        """Test that apply_best_configuration fails when no search has been run."""
        result = self.rsi.apply_best_configuration()
        self.assertFalse(result['success'])

    def test_apply_best_configuration_after_search(self):
        """Test that apply_best_configuration succeeds after evolutionary search."""
        self.rsi.evolutionary_search(population_size=6, generations=2)
        result = self.rsi.apply_best_configuration()
        self.assertTrue(result['success'])
        self.assertIn('applied_config', result)

    def test_evolutionary_search_with_custom_eval(self):
        """Test evolutionary search with a custom fitness function."""
        def custom_fitness(individual):
            # Prefer higher learning rate
            return individual.get('learning_rate', 0.0)

        results = self.rsi.evolutionary_search(
            population_size=6,
            generations=3,
            evaluation_fn=custom_fitness,
        )
        self.assertIsNotNone(results['best_configuration'])
        self.assertGreater(results['best_fitness'], 0.0)


class TestDataclasses(unittest.TestCase):
    """Tests for standalone dataclass construction."""

    def test_modification_defaults(self):
        """Test Modification dataclass default values."""
        mod = Modification(
            modification_id='test_mod',
            improvement_type=ImprovementType.HYPERPARAMETER,
            description='Test modification',
            parameters={'param_name': 'lr'},
            original_values={'lr': 0.01},
            new_values={'lr': 0.02},
        )
        self.assertEqual(mod.modification_id, 'test_mod')
        self.assertEqual(mod.status, ModificationStatus.PENDING)
        self.assertAlmostEqual(mod.expected_improvement, 0.0)
        self.assertAlmostEqual(mod.actual_improvement, 0.0)
        self.assertEqual(mod.safety_level, SafetyLevel.MINIMAL)
        self.assertIsNone(mod.application_time)
        self.assertIsNone(mod.reversion_time)

    def test_capability_defaults(self):
        """Test Capability dataclass default values."""
        cap = Capability(name='test_cap', level=0.5, dependencies=['dep1'])
        self.assertEqual(cap.name, 'test_cap')
        self.assertAlmostEqual(cap.level, 0.5)
        self.assertEqual(cap.dependencies, ['dep1'])
        self.assertEqual(cap.enhancement_history, [])

    def test_performance_metric_defaults(self):
        """Test PerformanceMetric dataclass default values."""
        pm = PerformanceMetric(
            name='test_pm',
            current_value=0.5,
            baseline_value=0.5,
            target_value=0.9,
        )
        self.assertEqual(pm.name, 'test_pm')
        self.assertAlmostEqual(pm.improvement_rate, 0.0)
        self.assertEqual(len(pm.history), 0)


if __name__ == '__main__':
    unittest.main()
