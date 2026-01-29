"""
Comprehensive tests for the UnifiedSuperIntelligence system.

Tests cover initialization, cognitive processing cycles, multi-phase processing,
intelligence metrics, IntelligenceMode and CognitivePhase enums, thinking,
imagination, serialization, and get_state/get_stats methods.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.unified_intelligence import (
    UnifiedSuperIntelligence,
    IntelligenceMode,
    CognitivePhase,
    CognitiveState,
)


# ---------------------------------------------------------------------------
# Small, shared constants to keep tests fast
# ---------------------------------------------------------------------------
SMALL_DIM = 16
SEED = 42


class TestIntelligenceModeEnum(unittest.TestCase):
    """Tests for the IntelligenceMode enum."""

    def test_all_modes_exist(self):
        """Test that every expected mode is defined."""
        expected = [
            'PERCEPTION', 'REASONING', 'CREATIVE', 'SOCIAL',
            'LEARNING', 'PLANNING', 'METACOGNITIVE', 'UNIFIED',
        ]
        for name in expected:
            self.assertTrue(
                hasattr(IntelligenceMode, name),
                f"IntelligenceMode missing member {name}",
            )

    def test_mode_count(self):
        """Test that IntelligenceMode has exactly 8 members."""
        self.assertEqual(len(IntelligenceMode), 8)

    def test_modes_are_unique(self):
        """Test that all mode values are distinct."""
        values = [m.value for m in IntelligenceMode]
        self.assertEqual(len(values), len(set(values)))

    def test_mode_name_round_trip(self):
        """Test that accessing a mode by name returns the same member."""
        for mode in IntelligenceMode:
            self.assertIs(IntelligenceMode[mode.name], mode)


class TestCognitivePhaseEnum(unittest.TestCase):
    """Tests for the CognitivePhase enum."""

    def test_all_phases_exist(self):
        """Test that every expected phase is defined."""
        expected = [
            'SENSE', 'PERCEIVE', 'ATTEND', 'REMEMBER',
            'REASON', 'DECIDE', 'ACT', 'LEARN', 'REFLECT',
        ]
        for name in expected:
            self.assertTrue(
                hasattr(CognitivePhase, name),
                f"CognitivePhase missing member {name}",
            )

    def test_phase_count(self):
        """Test that CognitivePhase has exactly 9 members."""
        self.assertEqual(len(CognitivePhase), 9)

    def test_phases_are_unique(self):
        """Test that all phase values are distinct."""
        values = [p.value for p in CognitivePhase]
        self.assertEqual(len(values), len(set(values)))

    def test_phase_name_round_trip(self):
        """Test that accessing a phase by name returns the same member."""
        for phase in CognitivePhase:
            self.assertIs(CognitivePhase[phase.name], phase)


class TestCognitiveStateDefaults(unittest.TestCase):
    """Tests for CognitiveState dataclass defaults."""

    def setUp(self):
        self.state = CognitiveState()

    def test_default_timestamp(self):
        """Test that the default timestamp is 0.0."""
        self.assertAlmostEqual(self.state.timestamp, 0.0)

    def test_default_phase(self):
        """Test that the default phase is SENSE."""
        self.assertEqual(self.state.current_phase, CognitivePhase.SENSE)

    def test_default_mode(self):
        """Test that the default mode is UNIFIED."""
        self.assertEqual(self.state.current_mode, IntelligenceMode.UNIFIED)

    def test_default_sensory_none(self):
        """Test that sensory inputs default to None."""
        self.assertIsNone(self.state.visual_input)
        self.assertIsNone(self.state.audio_input)
        self.assertIsNone(self.state.multimodal_state)

    def test_default_lists_empty(self):
        """Test that list fields default to empty lists."""
        self.assertEqual(self.state.working_memory_contents, [])
        self.assertEqual(self.state.episodic_retrievals, [])
        self.assertEqual(self.state.current_goals, [])
        self.assertEqual(self.state.active_hypotheses, [])
        self.assertEqual(self.state.causal_inferences, [])
        self.assertEqual(self.state.creative_outputs, [])
        self.assertEqual(self.state.imagination_trajectory, [])

    def test_default_dicts_empty(self):
        """Test that dict fields default to empty dicts."""
        self.assertEqual(self.state.semantic_activations, {})
        self.assertEqual(self.state.agent_models, {})
        self.assertEqual(self.state.social_context, {})
        self.assertEqual(self.state.hyperparameters, {})
        self.assertEqual(self.state.performance_metrics, {})

    def test_default_scalars(self):
        """Test that scalar defaults are correct."""
        self.assertAlmostEqual(self.state.confidence_level, 0.5)
        self.assertAlmostEqual(self.state.uncertainty_level, 0.5)
        self.assertIsNone(self.state.current_task)
        self.assertIsNone(self.state.attention_focus)
        self.assertIsNone(self.state.learning_strategy)


class TestUnifiedSuperIntelligenceInitialization(unittest.TestCase):
    """Tests for UnifiedSuperIntelligence initialization."""

    def setUp(self):
        self.usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            learning_rate=0.01,
            random_seed=SEED,
        )

    def test_state_dim_stored(self):
        """Test that state_dim is stored correctly."""
        self.assertEqual(self.usi.state_dim, SMALL_DIM)

    def test_max_memory_capacity_stored(self):
        """Test that max_memory_capacity is stored correctly."""
        self.assertEqual(self.usi.max_memory_capacity, 100)

    def test_learning_rate_stored(self):
        """Test that learning_rate is stored correctly."""
        self.assertAlmostEqual(self.usi.learning_rate, 0.01)

    def test_random_seed_stored(self):
        """Test that random_seed is stored correctly."""
        self.assertEqual(self.usi.random_seed, SEED)

    def test_systems_initialized_flag(self):
        """Test that systems initialization completes successfully."""
        self.assertTrue(self.usi._systems_initialized)

    def test_feature_flags_default_enabled(self):
        """Test that self-improvement, creativity, and social cognition are enabled by default."""
        self.assertTrue(self.usi.enable_self_improvement)
        self.assertTrue(self.usi.enable_creativity)
        self.assertTrue(self.usi.enable_social_cognition)

    def test_initial_counters_zero(self):
        """Test that all processing counters start at zero."""
        self.assertEqual(self.usi.total_cycles, 0)
        self.assertEqual(self.usi.total_experiences, 0)
        self.assertEqual(self.usi.total_inferences, 0)
        self.assertEqual(self.usi.total_decisions, 0)
        self.assertEqual(self.usi.total_creative_outputs, 0)
        self.assertEqual(self.usi.total_learning_updates, 0)

    def test_cognitive_state_exists(self):
        """Test that a CognitiveState is created."""
        self.assertIsInstance(self.usi.state, CognitiveState)

    def test_initial_state_timestamp_positive(self):
        """Test that the initial state has a positive timestamp."""
        self.assertGreater(self.usi.state.timestamp, 0.0)

    def test_integration_weights_created(self):
        """Test that integration weight matrices exist with correct shapes."""
        expected_keys = [
            'memory_to_reasoning', 'reasoning_to_executive',
            'executive_to_output', 'creativity_blend', 'social_context',
        ]
        for key in expected_keys:
            self.assertIn(key, self.usi.integration_weights)
            weight = self.usi.integration_weights[key]
            self.assertEqual(weight.shape, (SMALL_DIM, SMALL_DIM))

    def test_subsystem_refs_populated(self):
        """Test that _system_refs contains all expected subsystems."""
        expected_subsystems = [
            'episodic_memory', 'semantic_memory', 'working_memory',
            'meta_learner', 'goal_planner', 'causal_reasoner',
            'knowledge_base', 'language_grounding', 'world_model',
            'executive_control', 'creativity_engine', 'theory_of_mind',
            'self_improvement',
        ]
        for name in expected_subsystems:
            self.assertIn(name, self.usi._system_refs)
            self.assertIsNotNone(self.usi._system_refs[name])

    def test_subsystems_as_attributes(self):
        """Test that subsystems are accessible as direct attributes."""
        self.assertIsNotNone(self.usi.episodic_memory)
        self.assertIsNotNone(self.usi.semantic_memory)
        self.assertIsNotNone(self.usi.working_memory)
        self.assertIsNotNone(self.usi.meta_learner)
        self.assertIsNotNone(self.usi.goal_planner)
        self.assertIsNotNone(self.usi.causal_reasoner)
        self.assertIsNotNone(self.usi.knowledge_base)
        self.assertIsNotNone(self.usi.language_grounding)
        self.assertIsNotNone(self.usi.world_model)
        self.assertIsNotNone(self.usi.executive_control)
        self.assertIsNotNone(self.usi.creativity_engine)
        self.assertIsNotNone(self.usi.theory_of_mind)
        self.assertIsNotNone(self.usi.self_improvement)


class TestInitializationWithDisabledSystems(unittest.TestCase):
    """Tests for initialization with optional systems disabled."""

    def test_creativity_disabled(self):
        """Test that disabling creativity sets creativity_engine to None."""
        usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            enable_creativity=False,
            random_seed=SEED,
        )
        self.assertIsNone(usi.creativity_engine)
        self.assertFalse(usi.enable_creativity)

    def test_social_cognition_disabled(self):
        """Test that disabling social cognition sets theory_of_mind to None."""
        usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            enable_social_cognition=False,
            random_seed=SEED,
        )
        self.assertIsNone(usi.theory_of_mind)
        self.assertFalse(usi.enable_social_cognition)

    def test_self_improvement_disabled(self):
        """Test that disabling self-improvement sets self_improvement to None."""
        usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            enable_self_improvement=False,
            random_seed=SEED,
        )
        self.assertIsNone(usi.self_improvement)
        self.assertFalse(usi.enable_self_improvement)

    def test_all_optional_disabled(self):
        """Test initialization with all optional systems disabled."""
        usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            enable_creativity=False,
            enable_social_cognition=False,
            enable_self_improvement=False,
            random_seed=SEED,
        )
        self.assertTrue(usi._systems_initialized)
        self.assertIsNone(usi.creativity_engine)
        self.assertIsNone(usi.theory_of_mind)
        self.assertIsNone(usi.self_improvement)


class TestCognitiveProcessingCycle(unittest.TestCase):
    """Tests for the main process() method."""

    def setUp(self):
        self.usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            learning_rate=0.01,
            random_seed=SEED,
        )

    def test_process_returns_dict(self):
        """Test that process() returns a dictionary."""
        result = self.usi.process()
        self.assertIsInstance(result, dict)

    def test_process_result_structure(self):
        """Test that the result has all expected top-level keys."""
        result = self.usi.process()
        expected_keys = [
            'timestamp', 'mode', 'phases_completed',
            'outputs', 'metrics', 'processing_time',
        ]
        for key in expected_keys:
            self.assertIn(key, result)

    def test_process_no_error(self):
        """Test that process() completes without error key."""
        result = self.usi.process()
        self.assertNotIn('error', result)

    def test_process_increments_total_cycles(self):
        """Test that each process call increments the cycle counter."""
        self.assertEqual(self.usi.total_cycles, 0)
        self.usi.process()
        self.assertEqual(self.usi.total_cycles, 1)
        self.usi.process()
        self.assertEqual(self.usi.total_cycles, 2)

    def test_process_returns_mode_name(self):
        """Test that result includes the mode name string."""
        result = self.usi.process(mode=IntelligenceMode.UNIFIED)
        self.assertEqual(result['mode'], 'UNIFIED')

    def test_process_timestamp_positive(self):
        """Test that the result timestamp is a positive float."""
        result = self.usi.process()
        self.assertIsInstance(result['timestamp'], float)
        self.assertGreater(result['timestamp'], 0.0)

    def test_process_processing_time_positive(self):
        """Test that processing_time is non-negative."""
        result = self.usi.process()
        self.assertGreaterEqual(result['processing_time'], 0.0)

    def test_process_with_visual_input(self):
        """Test processing with visual sensory input."""
        np.random.seed(SEED)
        visual = np.random.rand(SMALL_DIM).astype(np.float64)
        result = self.usi.process(
            sensory_input={'visual': visual},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('SENSE', result['phases_completed'])
        self.assertIn('sensory', result['outputs'])
        self.assertIsNotNone(self.usi.state.visual_input)

    def test_process_with_audio_input(self):
        """Test processing with audio sensory input."""
        np.random.seed(SEED)
        audio = np.random.rand(SMALL_DIM).astype(np.float64)
        result = self.usi.process(
            sensory_input={'audio': audio},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('SENSE', result['phases_completed'])
        self.assertIsNotNone(self.usi.state.audio_input)

    def test_process_with_both_visual_and_audio(self):
        """Test processing with both visual and audio produces multimodal state."""
        np.random.seed(SEED)
        visual = np.random.rand(SMALL_DIM).astype(np.float64)
        audio = np.random.rand(SMALL_DIM).astype(np.float64)
        self.usi.process(
            sensory_input={'visual': visual, 'audio': audio},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIsNotNone(self.usi.state.multimodal_state)
        expected = (visual + audio) / 2
        np.testing.assert_array_almost_equal(
            self.usi.state.multimodal_state, expected,
        )

    def test_process_increments_experiences_with_sensory(self):
        """Test that sensory input increments total_experiences."""
        np.random.seed(SEED)
        visual = np.random.rand(SMALL_DIM).astype(np.float64)
        self.assertEqual(self.usi.total_experiences, 0)
        self.usi.process(sensory_input={'visual': visual})
        self.assertEqual(self.usi.total_experiences, 1)

    def test_process_updates_state_timestamp(self):
        """Test that the cognitive state timestamp is updated after processing."""
        old_ts = self.usi.state.timestamp
        self.usi.process()
        self.assertGreaterEqual(self.usi.state.timestamp, old_ts)


class TestMultiPhaseProcessing(unittest.TestCase):
    """Tests for multi-phase cognitive processing in UNIFIED mode."""

    def setUp(self):
        np.random.seed(SEED)
        self.usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            learning_rate=0.01,
            random_seed=SEED,
        )
        self.visual = np.random.rand(SMALL_DIM).astype(np.float64)

    def test_unified_mode_completes_all_core_phases(self):
        """Test that UNIFIED mode completes the core processing phases."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.UNIFIED,
        )
        for phase in ['PERCEIVE', 'ATTEND', 'REMEMBER', 'ACT', 'LEARN']:
            self.assertIn(phase, result['phases_completed'],
                          f"Phase {phase} not completed in UNIFIED mode")

    def test_unified_mode_includes_reason_phase(self):
        """Test that UNIFIED mode includes the REASON phase."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('REASON', result['phases_completed'])

    def test_unified_mode_includes_decide_phase(self):
        """Test that UNIFIED mode includes the DECIDE phase."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('DECIDE', result['phases_completed'])

    def test_unified_mode_includes_reflect_phase(self):
        """Test that UNIFIED mode includes the REFLECT phase."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('REFLECT', result['phases_completed'])

    def test_perception_mode_skips_reason_phase(self):
        """Test that PERCEPTION mode does not execute REASON."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.PERCEPTION,
        )
        self.assertNotIn('REASON', result['phases_completed'])

    def test_perception_mode_skips_reflect_phase(self):
        """Test that PERCEPTION mode does not execute REFLECT."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.PERCEPTION,
        )
        self.assertNotIn('REFLECT', result['phases_completed'])

    def test_reasoning_mode_includes_reason(self):
        """Test that REASONING mode includes the REASON phase."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.REASONING,
        )
        self.assertIn('REASON', result['phases_completed'])

    def test_planning_mode_includes_decide(self):
        """Test that PLANNING mode includes the DECIDE phase."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.PLANNING,
        )
        self.assertIn('DECIDE', result['phases_completed'])

    def test_metacognitive_mode_includes_reflect(self):
        """Test that METACOGNITIVE mode includes the REFLECT phase."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.METACOGNITIVE,
        )
        self.assertIn('REFLECT', result['phases_completed'])

    def test_process_without_sensory_skips_sense(self):
        """Test that processing without sensory input skips SENSE phase."""
        result = self.usi.process(mode=IntelligenceMode.UNIFIED)
        self.assertNotIn('SENSE', result['phases_completed'])

    def test_process_with_goals(self):
        """Test that providing goals triggers DECIDE phase."""
        result = self.usi.process(
            goals=['learn new skill'],
            mode=IntelligenceMode.LEARNING,
        )
        self.assertIn('DECIDE', result['phases_completed'])

    def test_multiple_cycles_accumulate(self):
        """Test that running multiple cycles accumulates statistics correctly."""
        for _ in range(3):
            self.usi.process(
                sensory_input={'visual': self.visual},
                mode=IntelligenceMode.UNIFIED,
            )
        self.assertEqual(self.usi.total_cycles, 3)
        self.assertEqual(self.usi.total_experiences, 3)


class TestProcessOutputStructure(unittest.TestCase):
    """Tests for the structure of individual phase outputs."""

    def setUp(self):
        np.random.seed(SEED)
        self.usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            learning_rate=0.01,
            random_seed=SEED,
        )
        self.visual = np.random.rand(SMALL_DIM).astype(np.float64)

    def test_perception_output_has_patterns(self):
        """Test that perception output has patterns_detected list."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('perception', result['outputs'])
        self.assertIn('patterns_detected', result['outputs']['perception'])

    def test_attention_output_has_allocated_flag(self):
        """Test that attention output has attention_allocated flag."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('attention', result['outputs'])
        self.assertTrue(result['outputs']['attention']['attention_allocated'])

    def test_memory_output_has_retrieval_counts(self):
        """Test that memory output has retrieval count fields."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('memory', result['outputs'])
        self.assertIn('episodic_retrieved', result['outputs']['memory'])
        self.assertIn('semantic_retrieved', result['outputs']['memory'])

    def test_reasoning_output_has_inference_counts(self):
        """Test that reasoning output has expected fields."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('reasoning', result['outputs'])
        self.assertIn('causal_inferences', result['outputs']['reasoning'])
        self.assertIn('goals_active', result['outputs']['reasoning'])

    def test_action_output_has_response_fields(self):
        """Test that action output has response_generated flag."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('action', result['outputs'])
        self.assertIn('response_generated', result['outputs']['action'])
        self.assertIn('response_type', result['outputs']['action'])

    def test_learning_output_has_update_counts(self):
        """Test that learning output reports episodes stored and meta updates."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('learning', result['outputs'])
        self.assertIn('episodes_stored', result['outputs']['learning'])
        self.assertIn('meta_updates', result['outputs']['learning'])

    def test_reflection_output_has_completion_flag(self):
        """Test that reflection output has reflection_complete flag."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('reflection', result['outputs'])
        self.assertTrue(result['outputs']['reflection']['reflection_complete'])

    def test_decision_output_has_confidence(self):
        """Test that decision output includes confidence field."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('decision', result['outputs'])
        self.assertIn('confidence', result['outputs']['decision'])


class TestIntelligenceMetrics(unittest.TestCase):
    """Tests for intelligence metrics reporting."""

    def setUp(self):
        np.random.seed(SEED)
        self.usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            learning_rate=0.01,
            random_seed=SEED,
        )

    def test_get_metrics_returns_dict(self):
        """Test that get_metrics() returns a dictionary."""
        metrics = self.usi.get_metrics()
        self.assertIsInstance(metrics, dict)

    def test_get_metrics_has_all_expected_keys(self):
        """Test that metrics dictionary has all expected keys."""
        metrics = self.usi.get_metrics()
        expected_keys = [
            'perception_accuracy', 'reasoning_depth', 'memory_capacity',
            'learning_speed', 'creativity_score', 'social_understanding',
            'planning_horizon', 'metacognitive_awareness',
            'cross_modal_coherence', 'temporal_consistency',
            'goal_achievement_rate', 'unified_intelligence_quotient',
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)

    def test_initial_metrics_are_numeric(self):
        """Test that all initial metric values are floats."""
        metrics = self.usi.get_metrics()
        for key, value in metrics.items():
            self.assertIsInstance(value, (int, float, np.floating),
                                 f"Metric {key} is not numeric: {type(value)}")

    def test_initial_metrics_within_bounds(self):
        """Test that initial metric values are in [0, 1]."""
        metrics = self.usi.get_metrics()
        for key, value in metrics.items():
            self.assertGreaterEqual(float(value), 0.0,
                                    f"Metric {key} is below 0.0")
            self.assertLessEqual(float(value), 1.0,
                                 f"Metric {key} is above 1.0")

    def test_metrics_update_after_processing(self):
        """Test that metrics are updated after a processing cycle."""
        visual = np.random.rand(SMALL_DIM).astype(np.float64)
        result = self.usi.process(
            sensory_input={'visual': visual},
            mode=IntelligenceMode.UNIFIED,
        )
        self.assertIn('metrics', result)
        metrics = result['metrics']
        # After one cycle with sensory input, some metrics should be non-zero
        self.assertGreater(metrics['perception_accuracy'], 0.0)
        self.assertGreater(metrics['learning_speed'], 0.0)

    def test_perception_metric_increases_with_experience(self):
        """Test that perception_accuracy increases with more experiences."""
        visual = np.random.rand(SMALL_DIM).astype(np.float64)
        for _ in range(5):
            self.usi.process(sensory_input={'visual': visual},
                             mode=IntelligenceMode.UNIFIED)
        metrics = self.usi.get_metrics()
        self.assertGreater(metrics['perception_accuracy'], 0.0)

    def test_reasoning_metric_increases_with_inferences(self):
        """Test that reasoning_depth increases after reasoning cycles."""
        visual = np.random.rand(SMALL_DIM).astype(np.float64)
        for _ in range(3):
            self.usi.process(sensory_input={'visual': visual},
                             mode=IntelligenceMode.UNIFIED)
        metrics = self.usi.get_metrics()
        self.assertGreater(metrics['reasoning_depth'], 0.0)

    def test_temporal_consistency_grows_with_cycles(self):
        """Test that temporal_consistency increases with more cycles."""
        for _ in range(5):
            self.usi.process(mode=IntelligenceMode.UNIFIED)
        metrics = self.usi.get_metrics()
        self.assertGreater(metrics['temporal_consistency'], 0.0)

    def test_cross_modal_coherence_with_sensory(self):
        """Test that cross_modal_coherence is set when multimodal state exists."""
        visual = np.random.rand(SMALL_DIM).astype(np.float64)
        self.usi.process(sensory_input={'visual': visual},
                         mode=IntelligenceMode.UNIFIED)
        metrics = self.usi.get_metrics()
        self.assertAlmostEqual(metrics['cross_modal_coherence'], 0.7)

    def test_cross_modal_coherence_zero_without_sensory(self):
        """Test that cross_modal_coherence is zero without sensory input."""
        self.usi.process(mode=IntelligenceMode.UNIFIED)
        metrics = self.usi.get_metrics()
        self.assertAlmostEqual(metrics['cross_modal_coherence'], 0.0)

    def test_social_understanding_with_tom_enabled(self):
        """Test social_understanding is 0.5 when theory of mind is enabled."""
        visual = np.random.rand(SMALL_DIM).astype(np.float64)
        self.usi.process(sensory_input={'visual': visual},
                         mode=IntelligenceMode.UNIFIED)
        metrics = self.usi.get_metrics()
        self.assertAlmostEqual(metrics['social_understanding'], 0.5)

    def test_social_understanding_zero_without_tom(self):
        """Test social_understanding is zero when theory of mind is disabled."""
        usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            enable_social_cognition=False,
            random_seed=SEED,
        )
        usi.process(mode=IntelligenceMode.UNIFIED)
        metrics = usi.get_metrics()
        self.assertAlmostEqual(metrics['social_understanding'], 0.0)

    def test_unified_intelligence_quotient_is_mean(self):
        """Test that UIQ is the mean of component scores."""
        visual = np.random.rand(SMALL_DIM).astype(np.float64)
        self.usi.process(sensory_input={'visual': visual},
                         mode=IntelligenceMode.UNIFIED)
        metrics = self.usi.get_metrics()
        component_keys = [
            'perception_accuracy', 'reasoning_depth', 'memory_capacity',
            'learning_speed', 'creativity_score', 'social_understanding',
            'planning_horizon', 'metacognitive_awareness',
        ]
        expected_mean = np.mean([metrics[k] for k in component_keys])
        self.assertAlmostEqual(
            metrics['unified_intelligence_quotient'],
            float(expected_mean),
            places=5,
        )


class TestThinkMethod(unittest.TestCase):
    """Tests for the think() high-level reasoning interface."""

    def setUp(self):
        self.usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            learning_rate=0.01,
            random_seed=SEED,
        )

    def test_think_returns_dict(self):
        """Test that think() returns a dictionary."""
        result = self.usi.think("What is the meaning of life?")
        self.assertIsInstance(result, dict)

    def test_think_result_structure(self):
        """Test that think result has all expected keys."""
        result = self.usi.think("test query")
        expected_keys = ['query', 'reasoning_chain', 'conclusion', 'confidence']
        for key in expected_keys:
            self.assertIn(key, result)

    def test_think_query_echoed(self):
        """Test that the query is echoed back in the result."""
        query = "What is the meaning of life?"
        result = self.usi.think(query)
        self.assertEqual(result['query'], query)

    def test_think_reasoning_chain_length_matches_depth(self):
        """Test that the reasoning chain has depth steps."""
        for depth in [1, 2, 4]:
            result = self.usi.think("test", depth=depth)
            self.assertEqual(len(result['reasoning_chain']), depth)

    def test_think_each_step_has_operation(self):
        """Test that each reasoning step has an operation field."""
        result = self.usi.think("test query", depth=3)
        for step in result['reasoning_chain']:
            self.assertIn('step', step)
            self.assertIn('operation', step)
            self.assertIn('result', step)

    def test_think_step_operations_vary(self):
        """Test that different depth steps use different reasoning operations."""
        result = self.usi.think("test query", depth=3)
        steps = result['reasoning_chain']
        self.assertEqual(steps[0]['operation'], 'pattern_matching')
        self.assertEqual(steps[1]['operation'], 'causal_inference')
        self.assertEqual(steps[2]['operation'], 'creative_synthesis')

    def test_think_conclusion_not_none(self):
        """Test that the conclusion is not None."""
        result = self.usi.think("test query")
        self.assertIsNotNone(result['conclusion'])

    def test_think_confidence_increases_with_depth(self):
        """Test that confidence increases with deeper reasoning."""
        result_shallow = self.usi.think("test", depth=1)
        result_deep = self.usi.think("test", depth=5)
        self.assertGreater(result_deep['confidence'], result_shallow['confidence'])

    def test_think_confidence_formula(self):
        """Test the exact confidence formula: 0.7 + 0.1 * depth."""
        for depth in [1, 3, 5]:
            result = self.usi.think("test", depth=depth)
            expected = 0.7 + 0.1 * depth
            self.assertAlmostEqual(result['confidence'], expected)

    def test_think_processes_language_words(self):
        """Test that think grounds the query words in the language system."""
        query = "hello world test"
        self.usi.think(query)
        # After thinking, the language system should have encountered these words
        for word in query.split():
            self.assertIn(
                word,
                self.usi.language_grounding.vocabulary,
            )


class TestImagineMethod(unittest.TestCase):
    """Tests for the imagine() creative scenario generation."""

    def setUp(self):
        self.usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            learning_rate=0.01,
            random_seed=SEED,
        )

    def test_imagine_returns_dict(self):
        """Test that imagine() returns a dictionary."""
        result = self.usi.imagine("A flying car in the city")
        self.assertIsInstance(result, dict)

    def test_imagine_result_structure(self):
        """Test that imagine result has all expected keys."""
        result = self.usi.imagine("A flying car in the city")
        expected_keys = [
            'scenario', 'trajectory_length', 'plausibility',
            'novelty', 'insights_generated',
        ]
        for key in expected_keys:
            self.assertIn(key, result)

    def test_imagine_scenario_echoed(self):
        """Test that the scenario is echoed back in the result."""
        scenario = "Exploring deep space"
        result = self.usi.imagine(scenario)
        self.assertEqual(result['scenario'], scenario)

    def test_imagine_trajectory_length_positive(self):
        """Test that the trajectory length is positive."""
        result = self.usi.imagine("test scenario", steps=5)
        self.assertGreater(result['trajectory_length'], 0)

    def test_imagine_plausibility_numeric(self):
        """Test that plausibility is a numeric value."""
        result = self.usi.imagine("test scenario")
        self.assertIsInstance(result['plausibility'], float)

    def test_imagine_novelty_numeric(self):
        """Test that novelty is a numeric value."""
        result = self.usi.imagine("test scenario")
        self.assertIsInstance(result['novelty'], float)

    def test_imagine_disabled_creativity_returns_error(self):
        """Test that imagination fails when creativity is disabled."""
        usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            enable_creativity=False,
            random_seed=SEED,
        )
        result = usi.imagine("test scenario")
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Creativity not enabled')

    def test_imagine_insights_non_negative(self):
        """Test that insights_generated is non-negative."""
        result = self.usi.imagine("test scenario", steps=10)
        self.assertGreaterEqual(result['insights_generated'], 0)


class TestGetStatsMethod(unittest.TestCase):
    """Tests for the get_stats() method."""

    def setUp(self):
        self.usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            learning_rate=0.01,
            random_seed=SEED,
        )

    def test_get_stats_returns_dict(self):
        """Test that get_stats() returns a dictionary."""
        stats = self.usi.get_stats()
        self.assertIsInstance(stats, dict)

    def test_get_stats_has_expected_keys(self):
        """Test that stats dict has all top-level keys."""
        stats = self.usi.get_stats()
        expected_keys = [
            'state_dim', 'systems_initialized', 'total_cycles',
            'total_experiences', 'total_inferences', 'total_decisions',
            'total_creative_outputs', 'total_learning_updates',
            'current_phase', 'current_mode', 'metrics', 'subsystem_stats',
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_get_stats_initial_values(self):
        """Test that initial stats reflect a fresh system."""
        stats = self.usi.get_stats()
        self.assertEqual(stats['state_dim'], SMALL_DIM)
        self.assertTrue(stats['systems_initialized'])
        self.assertEqual(stats['total_cycles'], 0)
        self.assertEqual(stats['total_experiences'], 0)
        self.assertEqual(stats['total_inferences'], 0)
        self.assertEqual(stats['total_decisions'], 0)
        self.assertEqual(stats['total_creative_outputs'], 0)
        self.assertEqual(stats['total_learning_updates'], 0)

    def test_get_stats_current_phase_is_string(self):
        """Test that current_phase is reported as a string."""
        stats = self.usi.get_stats()
        self.assertIsInstance(stats['current_phase'], str)

    def test_get_stats_current_mode_is_string(self):
        """Test that current_mode is reported as a string."""
        stats = self.usi.get_stats()
        self.assertIsInstance(stats['current_mode'], str)

    def test_get_stats_includes_metrics(self):
        """Test that stats include the full metrics dictionary."""
        stats = self.usi.get_stats()
        self.assertIsInstance(stats['metrics'], dict)
        self.assertIn('unified_intelligence_quotient', stats['metrics'])

    def test_get_stats_includes_subsystem_stats(self):
        """Test that stats include subsystem statistics."""
        stats = self.usi.get_stats()
        self.assertIsInstance(stats['subsystem_stats'], dict)

    def test_get_stats_updates_after_processing(self):
        """Test that stats reflect changes after processing cycles."""
        np.random.seed(SEED)
        visual = np.random.rand(SMALL_DIM).astype(np.float64)
        self.usi.process(
            sensory_input={'visual': visual},
            mode=IntelligenceMode.UNIFIED,
        )
        stats = self.usi.get_stats()
        self.assertEqual(stats['total_cycles'], 1)
        self.assertEqual(stats['total_experiences'], 1)
        self.assertGreater(stats['total_inferences'], 0)
        self.assertGreater(stats['total_decisions'], 0)
        self.assertGreater(stats['total_learning_updates'], 0)


class TestSerialization(unittest.TestCase):
    """Tests for serialize() and deserialize() methods."""

    def setUp(self):
        self.usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            learning_rate=0.01,
            random_seed=SEED,
        )

    def test_serialize_returns_dict(self):
        """Test that serialize() returns a dictionary."""
        data = self.usi.serialize()
        self.assertIsInstance(data, dict)

    def test_serialize_has_expected_keys(self):
        """Test that serialized data has all expected keys."""
        data = self.usi.serialize()
        expected_keys = [
            'state_dim', 'max_memory_capacity', 'learning_rate',
            'enable_self_improvement', 'enable_creativity',
            'enable_social_cognition', 'random_seed',
            'total_cycles', 'total_experiences', 'total_inferences',
            'total_decisions', 'total_creative_outputs',
            'total_learning_updates', 'metrics', 'integration_weights',
        ]
        for key in expected_keys:
            self.assertIn(key, data)

    def test_serialize_preserves_state_dim(self):
        """Test that serialized state_dim matches the original."""
        data = self.usi.serialize()
        self.assertEqual(data['state_dim'], SMALL_DIM)

    def test_serialize_preserves_learning_rate(self):
        """Test that serialized learning_rate matches the original."""
        data = self.usi.serialize()
        self.assertAlmostEqual(data['learning_rate'], 0.01)

    def test_serialize_preserves_feature_flags(self):
        """Test that serialized feature flags match the originals."""
        data = self.usi.serialize()
        self.assertTrue(data['enable_self_improvement'])
        self.assertTrue(data['enable_creativity'])
        self.assertTrue(data['enable_social_cognition'])

    def test_serialize_preserves_counters(self):
        """Test that serialized counters match the current state."""
        np.random.seed(SEED)
        visual = np.random.rand(SMALL_DIM).astype(np.float64)
        self.usi.process(sensory_input={'visual': visual},
                         mode=IntelligenceMode.UNIFIED)
        data = self.usi.serialize()
        self.assertEqual(data['total_cycles'], 1)
        self.assertEqual(data['total_experiences'], 1)

    def test_serialize_integration_weights_are_lists(self):
        """Test that integration weights are serialized as lists (not ndarrays)."""
        data = self.usi.serialize()
        for key, value in data['integration_weights'].items():
            self.assertIsInstance(value, list)

    def test_serialize_metrics_is_dict(self):
        """Test that serialized metrics is a dictionary."""
        data = self.usi.serialize()
        self.assertIsInstance(data['metrics'], dict)

    def test_deserialize_returns_instance(self):
        """Test that deserialize() returns a UnifiedSuperIntelligence instance."""
        data = self.usi.serialize()
        restored = UnifiedSuperIntelligence.deserialize(data)
        self.assertIsInstance(restored, UnifiedSuperIntelligence)

    def test_deserialize_preserves_state_dim(self):
        """Test that deserialized state_dim matches the original."""
        data = self.usi.serialize()
        restored = UnifiedSuperIntelligence.deserialize(data)
        self.assertEqual(restored.state_dim, SMALL_DIM)

    def test_deserialize_preserves_learning_rate(self):
        """Test that deserialized learning_rate matches the original."""
        data = self.usi.serialize()
        restored = UnifiedSuperIntelligence.deserialize(data)
        self.assertAlmostEqual(restored.learning_rate, 0.01)

    def test_deserialize_preserves_feature_flags(self):
        """Test that deserialized feature flags match the originals."""
        data = self.usi.serialize()
        restored = UnifiedSuperIntelligence.deserialize(data)
        self.assertEqual(restored.enable_self_improvement, self.usi.enable_self_improvement)
        self.assertEqual(restored.enable_creativity, self.usi.enable_creativity)
        self.assertEqual(restored.enable_social_cognition, self.usi.enable_social_cognition)

    def test_deserialize_preserves_random_seed(self):
        """Test that deserialized random_seed matches the original."""
        data = self.usi.serialize()
        restored = UnifiedSuperIntelligence.deserialize(data)
        self.assertEqual(restored.random_seed, SEED)

    def test_deserialize_preserves_counters(self):
        """Test that deserialized counters match after processing."""
        np.random.seed(SEED)
        visual = np.random.rand(SMALL_DIM).astype(np.float64)
        self.usi.process(sensory_input={'visual': visual},
                         mode=IntelligenceMode.UNIFIED)
        data = self.usi.serialize()
        restored = UnifiedSuperIntelligence.deserialize(data)
        self.assertEqual(restored.total_cycles, self.usi.total_cycles)
        self.assertEqual(restored.total_experiences, self.usi.total_experiences)
        self.assertEqual(restored.total_inferences, self.usi.total_inferences)
        self.assertEqual(restored.total_decisions, self.usi.total_decisions)
        self.assertEqual(restored.total_creative_outputs, self.usi.total_creative_outputs)
        self.assertEqual(restored.total_learning_updates, self.usi.total_learning_updates)

    def test_deserialize_restores_integration_weights(self):
        """Test that deserialized integration weights are numpy arrays with correct shape."""
        data = self.usi.serialize()
        restored = UnifiedSuperIntelligence.deserialize(data)
        for key in self.usi.integration_weights:
            self.assertIn(key, restored.integration_weights)
            np.testing.assert_array_almost_equal(
                restored.integration_weights[key],
                self.usi.integration_weights[key],
            )

    def test_deserialize_systems_initialized(self):
        """Test that deserialized system has all subsystems initialized."""
        data = self.usi.serialize()
        restored = UnifiedSuperIntelligence.deserialize(data)
        self.assertTrue(restored._systems_initialized)

    def test_deserialize_can_process(self):
        """Test that a deserialized system can run process() successfully."""
        data = self.usi.serialize()
        restored = UnifiedSuperIntelligence.deserialize(data)
        result = restored.process(mode=IntelligenceMode.UNIFIED)
        self.assertNotIn('error', result)
        self.assertIn('phases_completed', result)

    def test_serialize_roundtrip_with_disabled_features(self):
        """Test serialize/deserialize roundtrip with disabled optional features."""
        usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            enable_creativity=False,
            enable_social_cognition=False,
            enable_self_improvement=False,
            random_seed=SEED,
        )
        data = usi.serialize()
        restored = UnifiedSuperIntelligence.deserialize(data)
        self.assertFalse(restored.enable_creativity)
        self.assertFalse(restored.enable_social_cognition)
        self.assertFalse(restored.enable_self_improvement)
        self.assertIsNone(restored.creativity_engine)
        self.assertIsNone(restored.theory_of_mind)
        self.assertIsNone(restored.self_improvement)


class TestModelAgent(unittest.TestCase):
    """Tests for the model_agent() social cognition interface."""

    def setUp(self):
        self.usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            learning_rate=0.01,
            random_seed=SEED,
        )

    def test_model_agent_returns_dict(self):
        """Test that model_agent() returns a dictionary."""
        np.random.seed(SEED)
        obs = [(np.random.rand(SMALL_DIM), np.random.rand(SMALL_DIM))]
        result = self.usi.model_agent("agent_1", obs)
        self.assertIsInstance(result, dict)

    def test_model_agent_result_structure(self):
        """Test that model_agent result has expected keys."""
        np.random.seed(SEED)
        obs = [(np.random.rand(SMALL_DIM), np.random.rand(SMALL_DIM))]
        result = self.usi.model_agent("agent_1", obs)
        self.assertIn('agent_id', result)
        self.assertIn('observations_processed', result)

    def test_model_agent_observations_count(self):
        """Test that observations_processed matches the number of observations."""
        np.random.seed(SEED)
        obs = [
            (np.random.rand(SMALL_DIM), np.random.rand(SMALL_DIM))
            for _ in range(3)
        ]
        result = self.usi.model_agent("agent_1", obs)
        self.assertEqual(result['observations_processed'], 3)

    def test_model_agent_disabled_returns_error(self):
        """Test that model_agent returns error when social cognition is disabled."""
        usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            enable_social_cognition=False,
            random_seed=SEED,
        )
        np.random.seed(SEED)
        obs = [(np.random.rand(SMALL_DIM), np.random.rand(SMALL_DIM))]
        result = usi.model_agent("agent_1", obs)
        self.assertIn('error', result)


class TestImproveSelf(unittest.TestCase):
    """Tests for the improve_self() method."""

    def setUp(self):
        self.usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            learning_rate=0.01,
            random_seed=SEED,
        )

    def test_improve_self_returns_dict(self):
        """Test that improve_self() returns a dictionary."""
        result = self.usi.improve_self(n_cycles=1)
        self.assertIsInstance(result, dict)

    def test_improve_self_result_structure(self):
        """Test that improve_self result has expected keys."""
        result = self.usi.improve_self(n_cycles=1)
        self.assertIn('cycles_run', result)
        self.assertIn('total_proposals', result)
        self.assertIn('improvements_applied', result)

    def test_improve_self_cycles_reported(self):
        """Test that the number of cycles run matches the request."""
        result = self.usi.improve_self(n_cycles=2)
        self.assertEqual(result['cycles_run'], 2)

    def test_improve_self_disabled_returns_error(self):
        """Test that improve_self returns error when self-improvement is disabled."""
        usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            enable_self_improvement=False,
            random_seed=SEED,
        )
        result = usi.improve_self(n_cycles=1)
        self.assertIn('error', result)


class TestDifferentModesProcessing(unittest.TestCase):
    """Tests for processing under different IntelligenceMode values."""

    def setUp(self):
        np.random.seed(SEED)
        self.usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            learning_rate=0.01,
            random_seed=SEED,
        )
        self.visual = np.random.rand(SMALL_DIM).astype(np.float64)

    def test_all_modes_complete_without_error(self):
        """Test that processing completes without error for every mode."""
        for mode in IntelligenceMode:
            result = self.usi.process(
                sensory_input={'visual': self.visual},
                mode=mode,
            )
            self.assertNotIn('error', result,
                             f"Error in mode {mode.name}: {result.get('error')}")

    def test_all_modes_return_phases_completed(self):
        """Test that every mode returns at least one completed phase."""
        for mode in IntelligenceMode:
            result = self.usi.process(
                sensory_input={'visual': self.visual},
                mode=mode,
            )
            self.assertGreater(
                len(result['phases_completed']), 0,
                f"Mode {mode.name} completed no phases",
            )

    def test_all_modes_report_correct_mode_name(self):
        """Test that the result mode name matches the requested mode."""
        for mode in IntelligenceMode:
            result = self.usi.process(
                sensory_input={'visual': self.visual},
                mode=mode,
            )
            self.assertEqual(result['mode'], mode.name)

    def test_creative_mode_processes(self):
        """Test that CREATIVE mode completes successfully."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.CREATIVE,
        )
        self.assertNotIn('error', result)
        self.assertIn('PERCEIVE', result['phases_completed'])

    def test_social_mode_processes(self):
        """Test that SOCIAL mode completes successfully."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.SOCIAL,
        )
        self.assertNotIn('error', result)
        self.assertIn('PERCEIVE', result['phases_completed'])

    def test_learning_mode_processes(self):
        """Test that LEARNING mode completes successfully."""
        result = self.usi.process(
            sensory_input={'visual': self.visual},
            mode=IntelligenceMode.LEARNING,
        )
        self.assertNotIn('error', result)
        self.assertIn('LEARN', result['phases_completed'])


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def setUp(self):
        self.usi = UnifiedSuperIntelligence(
            state_dim=SMALL_DIM,
            max_memory_capacity=100,
            learning_rate=0.01,
            random_seed=SEED,
        )

    def test_process_with_empty_sensory_dict(self):
        """Test that an empty sensory dict does not crash."""
        result = self.usi.process(sensory_input={})
        self.assertNotIn('error', result)
        self.assertNotIn('SENSE', result['phases_completed'])

    def test_process_with_none_sensory(self):
        """Test that None sensory_input processes normally."""
        result = self.usi.process(sensory_input=None)
        self.assertNotIn('error', result)

    def test_process_with_empty_goals_list(self):
        """Test that an empty goals list is handled."""
        result = self.usi.process(goals=[], mode=IntelligenceMode.UNIFIED)
        self.assertNotIn('error', result)

    def test_think_with_single_word(self):
        """Test thinking with a single word query."""
        result = self.usi.think("hello")
        self.assertIsNotNone(result['conclusion'])

    def test_think_with_depth_one(self):
        """Test thinking at minimum depth."""
        result = self.usi.think("test", depth=1)
        self.assertEqual(len(result['reasoning_chain']), 1)
        self.assertEqual(result['reasoning_chain'][0]['operation'], 'pattern_matching')

    def test_think_with_large_depth(self):
        """Test thinking at a larger depth (beyond special-cased steps)."""
        result = self.usi.think("test", depth=5)
        self.assertEqual(len(result['reasoning_chain']), 5)
        # Steps beyond index 1 should be creative_synthesis
        for i in range(2, 5):
            self.assertEqual(
                result['reasoning_chain'][i]['operation'],
                'creative_synthesis',
            )

    def test_multiple_serialization_roundtrips(self):
        """Test that multiple serialize/deserialize roundtrips are stable."""
        np.random.seed(SEED)
        visual = np.random.rand(SMALL_DIM).astype(np.float64)
        self.usi.process(sensory_input={'visual': visual},
                         mode=IntelligenceMode.UNIFIED)

        current = self.usi
        for _ in range(3):
            data = current.serialize()
            current = UnifiedSuperIntelligence.deserialize(data)

        self.assertEqual(current.state_dim, SMALL_DIM)
        self.assertEqual(current.total_cycles, 1)
        self.assertTrue(current._systems_initialized)


if __name__ == '__main__':
    unittest.main()
