"""
Comprehensive tests for the RecursiveSelfImprovement system in core/.

Tests cover initialization, code analysis, bottleneck detection, modification
proposals, capability assessment, safety levels, and serialization.
"""

import os
import sys
import unittest
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.self_improvement import (
    RecursiveSelfImprovement,
    ImprovementType,
    SafetyLevel,
    ModificationStatus,
    PerformanceMetric,
    CodeModification,
    CodeMetrics,
    PerformanceBottleneck,
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
        self.assertTrue(self.rsi.code_analysis_enabled)
        self.assertFalse(self.rsi.auto_improvement_enabled)

    def test_custom_initialization(self):
        """Test construction with non-default parameters."""
        rsi = RecursiveSelfImprovement(
            num_hyperparameters=10,
            max_modifications=50,
            safety_level=SafetyLevel.MINIMAL,
            improvement_threshold=0.1,
            reversion_threshold=-0.2,
            code_analysis_enabled=False,
            auto_improvement_enabled=True,
            random_seed=99,
        )
        self.assertEqual(rsi.num_hyperparameters, 10)
        self.assertEqual(rsi.max_modifications, 50)
        self.assertEqual(rsi.safety_level, SafetyLevel.MINIMAL)
        self.assertAlmostEqual(rsi.improvement_threshold, 0.1)
        self.assertAlmostEqual(rsi.reversion_threshold, -0.2)
        self.assertFalse(rsi.code_analysis_enabled)
        self.assertTrue(rsi.auto_improvement_enabled)

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
            'code_quality', 'execution_speed'
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
            'metacognition', 'self_improvement', 'code_analysis', 'optimization'
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
        self.assertEqual(len(self.rsi.code_modifications), 0)
        self.assertEqual(len(self.rsi.pending_modifications), 0)

    def test_initial_checkpoint_created(self):
        """Test that an initial checkpoint is created during construction."""
        self.assertGreaterEqual(len(self.rsi.checkpoints), 1)
        initial_cp = self.rsi.checkpoints[0]
        self.assertEqual(initial_cp['name'], 'initial')
        self.assertIn('hyperparameters', initial_cp)
        self.assertIn('metrics', initial_cp)
        self.assertIn('capabilities', initial_cp)


class TestCodeAnalysis(unittest.TestCase):
    """Tests for code analysis functionality."""

    def setUp(self):
        self.rsi = RecursiveSelfImprovement(random_seed=42)

    def test_analyze_module_with_valid_python(self):
        """Test analyzing a valid Python module."""
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def hello():
    """Say hello."""
    return "Hello"

class MyClass:
    """A class."""
    def method(self):
        pass
''')
            temp_path = f.name
        
        try:
            metrics = self.rsi.analyze_module(temp_path)
            
            self.assertIsInstance(metrics, CodeMetrics)
            self.assertGreater(metrics.lines_of_code, 0)
            self.assertEqual(metrics.function_count, 2)  # hello and method
            self.assertEqual(metrics.class_count, 1)
            self.assertGreater(metrics.docstring_coverage, 0)
            
            # Check stored
            self.assertIn(temp_path, self.rsi.code_metrics)
        finally:
            os.unlink(temp_path)

    def test_analyze_module_disabled(self):
        """Test that analysis is skipped when disabled."""
        self.rsi.code_analysis_enabled = False
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('x = 1')
            temp_path = f.name
        
        try:
            metrics = self.rsi.analyze_module(temp_path)
            self.assertEqual(metrics.lines_of_code, 0)
        finally:
            os.unlink(temp_path)

    def test_detect_bottlenecks_nested_loops(self):
        """Test detection of nested loop bottlenecks."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def slow_function():
    """A function with nested loops."""
    for i in range(10):
        for j in range(10):
            print(i, j)
''')
            temp_path = f.name
        
        try:
            bottlenecks = self.rsi.detect_bottlenecks(temp_path)
            
            # Should detect the nested loop
            self.assertTrue(any(
                b.function_name == 'slow_function' and b.bottleneck_type == 'algorithm'
                for b in bottlenecks
            ))
        finally:
            os.unlink(temp_path)

    def test_detect_bottlenecks_long_function(self):
        """Test detection of long function bottlenecks."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('def long_func():\n')
            for i in range(60):
                f.write(f'    x{i} = {i}\n')
            temp_path = f.name
        
        try:
            bottlenecks = self.rsi.detect_bottlenecks(temp_path)
            
            # Should detect the long function
            long_func_bottlenecks = [b for b in bottlenecks if b.function_name == 'long_func']
            self.assertTrue(len(long_func_bottlenecks) > 0)
        finally:
            os.unlink(temp_path)


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

    def test_improvement_rate_computation(self):
        """Test that improvement_rate is computed after at least two updates."""
        metric = PerformanceMetric(
            name='test_metric',
            current_value=1.0,
            baseline_value=1.0,
            target_value=2.0,
        )
        metric.update(1.0)
        self.assertAlmostEqual(metric.improvement_rate, 0.0)
        
        metric.update(1.5)
        self.assertAlmostEqual(metric.improvement_rate, 0.5, places=5)

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


class TestModificationProposals(unittest.TestCase):
    """Tests for proposing modifications."""

    def setUp(self):
        self.rsi = RecursiveSelfImprovement(random_seed=42)

    def test_propose_hyperparameter_change_returns_modification(self):
        """Test that proposing a hyperparameter change returns a modification."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER,
            'learning_rate',
            'gradient_free',
        )
        self.assertIsNotNone(mod)
        self.assertEqual(mod['improvement_type'], ImprovementType.HYPERPARAMETER)
        self.assertEqual(mod['parameters']['param_name'], 'learning_rate')

    def test_propose_hyperparameter_invalid_name(self):
        """Test that proposing a change for a nonexistent hyperparameter returns None."""
        mod = self.rsi.propose_improvement(
            ImprovementType.HYPERPARAMETER, 'nonexistent_param',
        )
        self.assertIsNone(mod)

    def test_propose_capability_enhancement(self):
        """Test that a capability enhancement proposal returns a modification."""
        mod = self.rsi.propose_improvement(
            ImprovementType.CAPABILITY, 'perception',
        )
        self.assertIsNotNone(mod)
        self.assertEqual(mod['improvement_type'], ImprovementType.CAPABILITY)

    def test_propose_capability_invalid_name(self):
        """Test that proposing a capability enhancement for a nonexistent capability returns None."""
        mod = self.rsi.propose_improvement(
            ImprovementType.CAPABILITY, 'nonexistent_capability',
        )
        self.assertIsNone(mod)

    def test_propose_architecture_change(self):
        """Test that an architecture change proposal returns a modification."""
        mod = self.rsi.propose_improvement(
            ImprovementType.ARCHITECTURE, 'network',
        )
        self.assertIsNotNone(mod)
        self.assertEqual(mod['improvement_type'], ImprovementType.ARCHITECTURE)

    def test_propose_unsupported_type_returns_none(self):
        """Test that unsupported improvement types return None."""
        mod = self.rsi.propose_improvement(
            ImprovementType.EFFICIENCY, 'something',
        )
        self.assertIsNone(mod)


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
            self.rsi.capabilities['self_improvement'].dependencies,
            ['metacognition', 'learning'],
        )

    def test_get_improvement_recommendations(self):
        """Test that recommendations are returned."""
        recs = self.rsi.get_improvement_recommendations()
        self.assertIsInstance(recs, list)

    def test_get_stats(self):
        """Test that get_stats returns expected structure."""
        stats = self.rsi.get_stats()
        self.assertIn('generation', stats)
        self.assertIn('total_modifications', stats)
        self.assertIn('current_metrics', stats)
        self.assertIn('current_capabilities', stats)
        self.assertIn('bottlenecks_found', stats)


class TestSafetyLevels(unittest.TestCase):
    """Tests for safety levels."""

    def test_safety_level_enum_values(self):
        """Test that SafetyLevel enum has the expected values."""
        self.assertEqual(SafetyLevel.MINIMAL.value, 'minimal')
        self.assertEqual(SafetyLevel.MODERATE.value, 'moderate')
        self.assertEqual(SafetyLevel.AGGRESSIVE.value, 'aggressive')
        self.assertEqual(SafetyLevel.EXPERIMENTAL.value, 'experimental')

    def test_improvement_type_enum_values(self):
        """Test that ImprovementType enum has the expected values."""
        self.assertEqual(ImprovementType.HYPERPARAMETER.value, 'hyperparameter')
        self.assertEqual(ImprovementType.ARCHITECTURE.value, 'architecture')
        self.assertEqual(ImprovementType.CODE_OPTIMIZATION.value, 'code_optimization')


class TestOptimizationCycles(unittest.TestCase):
    """Tests for optimization cycles."""

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

    def test_evolutionary_search(self):
        """Test evolutionary search."""
        results = self.rsi.evolutionary_search(
            population_size=6, generations=2,
        )
        self.assertIn('generations', results)
        self.assertIn('best_fitness', results)
        self.assertIn('best_configuration', results)


class TestSerialization(unittest.TestCase):
    """Tests for serialization/deserialization."""

    def setUp(self):
        self.rsi = RecursiveSelfImprovement(random_seed=42)

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
            'code_analysis_enabled', 'auto_improvement_enabled',
            'hyperparameters', 'hyperparameter_ranges', 'metrics',
            'capabilities', 'generation', 'stats'
        ]
        for key in expected_keys:
            self.assertIn(key, data)

    def test_deserialize_roundtrip(self):
        """Test that serialize -> deserialize produces an equivalent system."""
        data = self.rsi.serialize()
        restored = RecursiveSelfImprovement.deserialize(data)
        
        self.assertEqual(restored.num_hyperparameters, self.rsi.num_hyperparameters)
        self.assertEqual(restored.safety_level, self.rsi.safety_level)
        self.assertEqual(restored.generation, self.rsi.generation)


if __name__ == '__main__':
    unittest.main()
