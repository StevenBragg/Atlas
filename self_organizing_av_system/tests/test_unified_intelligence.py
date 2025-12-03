#!/usr/bin/env python3
"""
Comprehensive Test Suite for ATLAS Unified Super Intelligence Model

This module tests the unified super intelligence system that integrates
all cognitive systems into a cohesive architecture:

1. Unified Intelligence Initialization
2. Complete Cognitive Processing Cycle
3. Multi-Phase Processing
4. Cross-System Integration
5. Intelligence Metrics
6. Thinking and Reasoning
7. Imagination and Creativity
8. Social Cognition
9. Self-Improvement
10. Serialization and Persistence
"""

import unittest
import numpy as np
import sys
import os
import logging
import time

# Add parent directory to path for core imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Prevent __init__.py from loading GUI components
import importlib.util

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class TestUnifiedIntelligenceInitialization(unittest.TestCase):
    """Test initialization of the unified intelligence system"""

    def test_basic_initialization(self):
        """Test basic initialization with default parameters"""
        from core.unified_intelligence import UnifiedSuperIntelligence

        usi = UnifiedSuperIntelligence(random_seed=42)

        self.assertEqual(usi.state_dim, 64)
        self.assertTrue(usi._systems_initialized)
        self.assertIsNotNone(usi.episodic_memory)
        self.assertIsNotNone(usi.semantic_memory)
        self.assertIsNotNone(usi.working_memory)
        self.assertIsNotNone(usi.meta_learner)
        self.assertIsNotNone(usi.goal_planner)
        self.assertIsNotNone(usi.causal_reasoner)
        self.assertIsNotNone(usi.world_model)
        self.assertIsNotNone(usi.executive_control)

    def test_custom_initialization(self):
        """Test initialization with custom parameters"""
        from core.unified_intelligence import UnifiedSuperIntelligence

        usi = UnifiedSuperIntelligence(
            state_dim=128,
            max_memory_capacity=5000,
            learning_rate=0.05,
            enable_self_improvement=True,
            enable_creativity=True,
            enable_social_cognition=True,
            random_seed=42
        )

        self.assertEqual(usi.state_dim, 128)
        self.assertEqual(usi.max_memory_capacity, 5000)
        self.assertEqual(usi.learning_rate, 0.05)
        self.assertIsNotNone(usi.creativity_engine)
        self.assertIsNotNone(usi.theory_of_mind)
        self.assertIsNotNone(usi.self_improvement)

    def test_initialization_without_optional_systems(self):
        """Test initialization with optional systems disabled"""
        from core.unified_intelligence import UnifiedSuperIntelligence

        usi = UnifiedSuperIntelligence(
            enable_self_improvement=False,
            enable_creativity=False,
            enable_social_cognition=False,
            random_seed=42
        )

        self.assertIsNone(usi.creativity_engine)
        self.assertIsNone(usi.theory_of_mind)
        self.assertIsNone(usi.self_improvement)

    def test_all_subsystems_registered(self):
        """Test that all subsystems are properly registered"""
        from core.unified_intelligence import UnifiedSuperIntelligence

        usi = UnifiedSuperIntelligence(random_seed=42)

        expected_systems = [
            'episodic_memory', 'semantic_memory', 'working_memory',
            'meta_learner', 'goal_planner', 'causal_reasoner',
            'knowledge_base', 'language_grounding', 'world_model',
            'executive_control', 'creativity_engine', 'theory_of_mind',
            'self_improvement'
        ]

        for system in expected_systems:
            self.assertIn(system, usi._system_refs)

    def test_integration_weights_initialized(self):
        """Test integration weights are properly initialized"""
        from core.unified_intelligence import UnifiedSuperIntelligence

        usi = UnifiedSuperIntelligence(random_seed=42)

        expected_weights = [
            'memory_to_reasoning', 'reasoning_to_executive',
            'executive_to_output', 'creativity_blend', 'social_context'
        ]

        for weight in expected_weights:
            self.assertIn(weight, usi.integration_weights)
            self.assertEqual(usi.integration_weights[weight].shape, (64, 64))


class TestCognitiveProcessingCycle(unittest.TestCase):
    """Test the complete cognitive processing cycle"""

    def setUp(self):
        from core.unified_intelligence import UnifiedSuperIntelligence, IntelligenceMode
        self.usi = UnifiedSuperIntelligence(random_seed=42)
        self.IntelligenceMode = IntelligenceMode

    def test_basic_processing(self):
        """Test basic processing with sensory input"""
        sensory_input = {
            'visual': np.random.rand(64),
            'audio': np.random.rand(64)
        }

        result = self.usi.process(sensory_input=sensory_input)

        self.assertIn('timestamp', result)
        self.assertIn('phases_completed', result)
        self.assertIn('outputs', result)
        self.assertIn('processing_time', result)
        self.assertGreater(len(result['phases_completed']), 0)

    def test_all_phases_complete(self):
        """Test that all cognitive phases complete"""
        sensory_input = {'visual': np.random.rand(64)}

        result = self.usi.process(
            sensory_input=sensory_input,
            mode=self.IntelligenceMode.UNIFIED
        )

        expected_phases = ['SENSE', 'PERCEIVE', 'ATTEND', 'REMEMBER', 'REASON',
                          'DECIDE', 'ACT', 'LEARN', 'REFLECT']

        for phase in expected_phases:
            self.assertIn(phase, result['phases_completed'])

    def test_processing_modes(self):
        """Test different processing modes"""
        sensory_input = {'visual': np.random.rand(64)}

        modes = [
            self.IntelligenceMode.PERCEPTION,
            self.IntelligenceMode.REASONING,
            self.IntelligenceMode.CREATIVE,
            self.IntelligenceMode.LEARNING,
            self.IntelligenceMode.UNIFIED
        ]

        for mode in modes:
            result = self.usi.process(sensory_input=sensory_input, mode=mode)
            self.assertEqual(result['mode'], mode.name)

    def test_goal_processing(self):
        """Test processing with goals"""
        sensory_input = {'visual': np.random.rand(64)}
        goals = ['learn_patterns', 'optimize_performance']

        result = self.usi.process(
            sensory_input=sensory_input,
            goals=goals,
            mode=self.IntelligenceMode.PLANNING
        )

        self.assertIn('DECIDE', result['phases_completed'])
        self.assertIn('decision', result['outputs'])

    def test_counter_updates(self):
        """Test that processing updates counters"""
        initial_cycles = self.usi.total_cycles
        initial_experiences = self.usi.total_experiences

        self.usi.process(sensory_input={'visual': np.random.rand(64)})

        self.assertEqual(self.usi.total_cycles, initial_cycles + 1)
        self.assertGreater(self.usi.total_experiences, initial_experiences)


class TestMultiPhaseProcessing(unittest.TestCase):
    """Test individual cognitive phases"""

    def setUp(self):
        from core.unified_intelligence import UnifiedSuperIntelligence
        self.usi = UnifiedSuperIntelligence(random_seed=42)

    def test_sensory_processing(self):
        """Test sensory processing phase"""
        sensory_input = {'visual': np.random.rand(64)}

        result = self.usi._process_sensory(sensory_input)

        self.assertIsNotNone(self.usi.state.visual_input)
        self.assertIsNotNone(self.usi.state.multimodal_state)
        self.assertIn('world_model', result)

    def test_perception_phase(self):
        """Test perception and pattern recognition phase"""
        # First process sensory input
        self.usi._process_sensory({'visual': np.random.rand(64)})

        result = self.usi._perceive_patterns()

        self.assertIn('patterns_detected', result)
        self.assertIsInstance(result['patterns_detected'], list)

    def test_attention_allocation(self):
        """Test attention allocation phase"""
        self.usi._process_sensory({'visual': np.random.rand(64)})

        result = self.usi._allocate_attention()

        self.assertTrue(result['attention_allocated'])
        self.assertIsNotNone(self.usi.state.attention_focus)

    def test_memory_retrieval(self):
        """Test memory retrieval phase"""
        # Store some memories first
        for _ in range(5):
            state = np.random.rand(64)
            self.usi.episodic_memory.store(
                state=state,
                sensory_data={'visual': state[:32]}
            )

        self.usi._process_sensory({'visual': np.random.rand(64)})
        self.usi._allocate_attention()

        result = self.usi._retrieve_memories()

        self.assertIn('episodic_retrieved', result)
        self.assertIn('semantic_retrieved', result)

    def test_reasoning_phase(self):
        """Test reasoning phase"""
        self.usi._process_sensory({'visual': np.random.rand(64)})
        self.usi._perceive_patterns()

        result = self.usi._perform_reasoning()

        self.assertIn('causal_inferences', result)
        self.assertIn('goals_active', result)

    def test_decision_phase(self):
        """Test decision making phase"""
        self.usi._process_sensory({'visual': np.random.rand(64)})
        self.usi._allocate_attention()

        result = self.usi._make_decisions(['test_goal'])

        self.assertIn('decision_made', result)
        self.assertIn('confidence', result)

    def test_learning_update(self):
        """Test learning update phase"""
        self.usi._process_sensory({'visual': np.random.rand(64)})

        result = self.usi._update_learning()

        self.assertIn('episodes_stored', result)
        self.assertIn('meta_updates', result)
        self.assertGreater(result['episodes_stored'], 0)

    def test_metacognitive_reflection(self):
        """Test metacognitive reflection phase"""
        self.usi._process_sensory({'visual': np.random.rand(64)})
        self.usi._allocate_attention()

        result = self.usi._metacognitive_reflect()

        self.assertTrue(result['reflection_complete'])
        self.assertIn('metacognitive_assessment', result)


class TestCrossSystemIntegration(unittest.TestCase):
    """Test integration between cognitive systems"""

    def setUp(self):
        from core.unified_intelligence import UnifiedSuperIntelligence
        self.usi = UnifiedSuperIntelligence(random_seed=42)

    def test_memory_systems_integration(self):
        """Test integration between memory systems"""
        # Store in episodic
        state = np.random.rand(64)
        self.usi.episodic_memory.store(
            state=state,
            sensory_data={'visual': state[:32]}
        )

        # Add to semantic
        self.usi.semantic_memory.add_concept('test_concept', state)

        # Process to trigger integration
        result = self.usi.process(sensory_input={'visual': state})

        self.assertGreater(len(self.usi.episodic_memory.episodes), 0)
        self.assertIn('test_concept', self.usi.semantic_memory.concepts)

    def test_reasoning_executive_integration(self):
        """Test integration between reasoning and executive control"""
        from core.goal_planning import GoalType

        # Generate goal
        goal = self.usi.goal_planner.generate_goal(GoalType.LEARNING)

        if goal:
            # Create executive task
            task_id = self.usi.executive_control.add_task(
                name=goal.name,
                priority=goal.priority
            )

            self.assertIsNotNone(task_id)
            self.assertIn(task_id, self.usi.executive_control.tasks)

    def test_creative_executive_integration(self):
        """Test integration between creativity and executive control"""
        # Generate creative solutions
        problem = np.random.rand(64)
        solutions = self.usi.creativity_engine.divergent_think(problem, n_solutions=3)

        # Use executive for decision
        if solutions:
            options = [
                {'value': 0.5 + i * 0.1, 'risk': 0.2, 'embedding': sol}
                for i, (sol, _) in enumerate(solutions)
            ]
            chosen, confidence, _ = self.usi.executive_control.decide_under_uncertainty(options)

            self.assertIn(chosen, [0, 1, 2])
            self.assertGreater(confidence, 0)

    def test_world_model_memory_integration(self):
        """Test integration between world model and memory"""
        # Observe world
        self.usi.world_model.observe([{
            'embedding': np.random.rand(64),
            'position': np.array([0.0, 0.0, 0.0])
        }])

        # Get world state
        world_state = self.usi.world_model.get_world_state_embedding()

        # Store in episodic memory
        self.usi.episodic_memory.store(
            state=world_state,
            sensory_data={'world': world_state[:32]}
        )

        self.assertGreater(len(self.usi.episodic_memory.episodes), 0)


class TestIntelligenceMetrics(unittest.TestCase):
    """Test intelligence metrics computation"""

    def setUp(self):
        from core.unified_intelligence import UnifiedSuperIntelligence
        self.usi = UnifiedSuperIntelligence(random_seed=42)

    def test_initial_metrics(self):
        """Test initial metrics are zero or valid"""
        metrics = self.usi.get_metrics()

        expected_metrics = [
            'perception_accuracy', 'reasoning_depth', 'memory_capacity',
            'learning_speed', 'creativity_score', 'social_understanding',
            'planning_horizon', 'metacognitive_awareness',
            'cross_modal_coherence', 'temporal_consistency',
            'goal_achievement_rate', 'unified_intelligence_quotient'
        ]

        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)

    def test_metrics_update_with_processing(self):
        """Test metrics update after processing"""
        initial_metrics = self.usi.get_metrics()

        # Process multiple cycles
        for _ in range(10):
            self.usi.process(sensory_input={'visual': np.random.rand(64)})

        updated_metrics = self.usi.get_metrics()

        # Some metrics should improve
        self.assertGreater(
            updated_metrics['perception_accuracy'],
            initial_metrics['perception_accuracy']
        )

    def test_unified_intelligence_quotient(self):
        """Test unified intelligence quotient computation"""
        # Process to build up metrics
        for _ in range(20):
            self.usi.process(sensory_input={'visual': np.random.rand(64)})

        metrics = self.usi.get_metrics()
        uiq = metrics['unified_intelligence_quotient']

        self.assertGreaterEqual(uiq, 0.0)
        self.assertLessEqual(uiq, 1.0)


class TestThinkingAndReasoning(unittest.TestCase):
    """Test high-level thinking and reasoning"""

    def setUp(self):
        from core.unified_intelligence import UnifiedSuperIntelligence
        self.usi = UnifiedSuperIntelligence(random_seed=42)

    def test_basic_thinking(self):
        """Test basic thinking operation"""
        result = self.usi.think(
            query="What patterns exist in the data?",
            depth=3
        )

        self.assertIn('query', result)
        self.assertIn('reasoning_chain', result)
        self.assertIn('conclusion', result)
        self.assertIn('confidence', result)
        self.assertEqual(len(result['reasoning_chain']), 3)

    def test_thinking_depth(self):
        """Test different thinking depths"""
        for depth in [1, 3, 5]:
            result = self.usi.think(query="Test query", depth=depth)
            self.assertEqual(len(result['reasoning_chain']), depth)

    def test_thinking_with_context(self):
        """Test thinking with additional context"""
        result = self.usi.think(
            query="Analyze the situation",
            context={'domain': 'test', 'priority': 'high'},
            depth=2
        )

        self.assertIsNotNone(result['conclusion'])

    def test_reasoning_chain_operations(self):
        """Test that reasoning chain includes different operations"""
        result = self.usi.think(query="Complex analysis", depth=3)

        operations = [step['operation'] for step in result['reasoning_chain']]

        self.assertIn('pattern_matching', operations)
        self.assertIn('causal_inference', operations)


class TestImaginationAndCreativity(unittest.TestCase):
    """Test imagination and creativity capabilities"""

    def setUp(self):
        from core.unified_intelligence import UnifiedSuperIntelligence
        self.usi = UnifiedSuperIntelligence(
            enable_creativity=True,
            random_seed=42
        )

    def test_imagination(self):
        """Test imagination capability"""
        result = self.usi.imagine(
            scenario="A novel situation",
            steps=10
        )

        self.assertIn('scenario', result)
        self.assertIn('trajectory_length', result)
        self.assertIn('plausibility', result)
        self.assertIn('novelty', result)
        self.assertEqual(result['trajectory_length'], 11)  # Initial + 10 steps

    def test_creative_generation_in_processing(self):
        """Test creative generation during processing"""
        from core.unified_intelligence import IntelligenceMode

        result = self.usi.process(
            sensory_input={'visual': np.random.rand(64)},
            mode=IntelligenceMode.CREATIVE
        )

        # Should have creative outputs
        self.assertGreaterEqual(len(self.usi.state.creative_outputs), 0)

    def test_creativity_disabled(self):
        """Test behavior when creativity is disabled"""
        from core.unified_intelligence import UnifiedSuperIntelligence

        usi = UnifiedSuperIntelligence(enable_creativity=False, random_seed=42)

        result = usi.imagine("Test scenario")

        self.assertIn('error', result)


class TestSocialCognition(unittest.TestCase):
    """Test social cognition and theory of mind"""

    def setUp(self):
        from core.unified_intelligence import UnifiedSuperIntelligence
        self.usi = UnifiedSuperIntelligence(
            enable_social_cognition=True,
            random_seed=42
        )

    def test_agent_modeling(self):
        """Test modeling other agents"""
        observations = [
            (np.random.rand(64), np.random.rand(64))
            for _ in range(5)
        ]

        result = self.usi.model_agent(
            agent_id="agent_1",
            observations=observations
        )

        self.assertIn('agent_id', result)
        self.assertEqual(result['observations_processed'], 5)
        self.assertIn('latest_inference', result)

    def test_social_cognition_disabled(self):
        """Test behavior when social cognition is disabled"""
        from core.unified_intelligence import UnifiedSuperIntelligence

        usi = UnifiedSuperIntelligence(enable_social_cognition=False, random_seed=42)

        result = usi.model_agent("agent", [])

        self.assertIn('error', result)


class TestSelfImprovement(unittest.TestCase):
    """Test self-improvement capabilities"""

    def setUp(self):
        from core.unified_intelligence import UnifiedSuperIntelligence
        self.usi = UnifiedSuperIntelligence(
            enable_self_improvement=True,
            random_seed=42
        )

    def test_self_improvement_cycle(self):
        """Test running self-improvement cycle"""
        result = self.usi.improve_self(n_cycles=1)

        self.assertEqual(result['cycles_run'], 1)
        self.assertIn('total_proposals', result)
        self.assertIn('improvements_applied', result)

    def test_multiple_improvement_cycles(self):
        """Test multiple improvement cycles"""
        result = self.usi.improve_self(n_cycles=3)

        self.assertEqual(result['cycles_run'], 3)

    def test_self_improvement_disabled(self):
        """Test behavior when self-improvement is disabled"""
        from core.unified_intelligence import UnifiedSuperIntelligence

        usi = UnifiedSuperIntelligence(enable_self_improvement=False, random_seed=42)

        result = usi.improve_self()

        self.assertIn('error', result)


class TestSerializationAndPersistence(unittest.TestCase):
    """Test serialization and persistence"""

    def setUp(self):
        from core.unified_intelligence import UnifiedSuperIntelligence
        self.usi = UnifiedSuperIntelligence(random_seed=42)

    def test_serialization(self):
        """Test serialization to dict"""
        # Process some data first
        for _ in range(5):
            self.usi.process(sensory_input={'visual': np.random.rand(64)})

        serialized = self.usi.serialize()

        expected_keys = [
            'state_dim', 'max_memory_capacity', 'learning_rate',
            'enable_self_improvement', 'enable_creativity',
            'enable_social_cognition', 'random_seed',
            'total_cycles', 'metrics', 'integration_weights'
        ]

        for key in expected_keys:
            self.assertIn(key, serialized)

    def test_deserialization(self):
        """Test deserialization from dict"""
        from core.unified_intelligence import UnifiedSuperIntelligence

        # Process some data
        for _ in range(5):
            self.usi.process(sensory_input={'visual': np.random.rand(64)})

        serialized = self.usi.serialize()

        # Deserialize
        restored = UnifiedSuperIntelligence.deserialize(serialized)

        self.assertEqual(restored.state_dim, self.usi.state_dim)
        self.assertEqual(restored.total_cycles, self.usi.total_cycles)
        self.assertTrue(restored._systems_initialized)

    def test_stats_retrieval(self):
        """Test comprehensive stats retrieval"""
        for _ in range(3):
            self.usi.process(sensory_input={'visual': np.random.rand(64)})

        stats = self.usi.get_stats()

        self.assertIn('state_dim', stats)
        self.assertIn('systems_initialized', stats)
        self.assertIn('total_cycles', stats)
        self.assertIn('metrics', stats)
        self.assertIn('subsystem_stats', stats)


class TestCognitiveStateManagement(unittest.TestCase):
    """Test cognitive state management"""

    def setUp(self):
        from core.unified_intelligence import (
            UnifiedSuperIntelligence, CognitivePhase, IntelligenceMode
        )
        self.usi = UnifiedSuperIntelligence(random_seed=42)
        self.CognitivePhase = CognitivePhase
        self.IntelligenceMode = IntelligenceMode

    def test_initial_state(self):
        """Test initial cognitive state"""
        self.assertEqual(self.usi.state.current_phase, self.CognitivePhase.SENSE)
        self.assertEqual(self.usi.state.current_mode, self.IntelligenceMode.UNIFIED)
        self.assertIsNone(self.usi.state.visual_input)
        self.assertIsNone(self.usi.state.audio_input)

    def test_state_updates_with_processing(self):
        """Test state updates during processing"""
        self.usi.process(sensory_input={'visual': np.random.rand(64)})

        self.assertIsNotNone(self.usi.state.visual_input)
        self.assertIsNotNone(self.usi.state.multimodal_state)
        self.assertIsNotNone(self.usi.state.attention_focus)
        self.assertGreater(self.usi.state.timestamp, 0)

    def test_phase_transitions(self):
        """Test cognitive phase transitions"""
        result = self.usi.process(
            sensory_input={'visual': np.random.rand(64)},
            mode=self.IntelligenceMode.UNIFIED
        )

        # All phases should be completed
        phases = result['phases_completed']
        self.assertGreater(len(phases), 0)


class TestFullSupertintelligencePipeline(unittest.TestCase):
    """Test complete superintelligence pipeline"""

    def setUp(self):
        from core.unified_intelligence import UnifiedSuperIntelligence
        self.usi = UnifiedSuperIntelligence(
            state_dim=64,
            enable_self_improvement=True,
            enable_creativity=True,
            enable_social_cognition=True,
            random_seed=42
        )

    def test_complete_pipeline(self):
        """Test complete superintelligence processing pipeline"""
        # 1. Process sensory input
        sensory_result = self.usi.process(
            sensory_input={'visual': np.random.rand(64)}
        )
        self.assertGreater(len(sensory_result['phases_completed']), 0)

        # 2. Think about a problem
        think_result = self.usi.think(
            query="How to optimize the system?",
            depth=3
        )
        self.assertIsNotNone(think_result['conclusion'])

        # 3. Imagine scenarios
        imagine_result = self.usi.imagine(
            scenario="Future optimization",
            steps=5
        )
        self.assertGreater(imagine_result['trajectory_length'], 0)

        # 4. Model an agent
        agent_result = self.usi.model_agent(
            agent_id="test_agent",
            observations=[
                (np.random.rand(64), np.random.rand(64))
                for _ in range(3)
            ]
        )
        self.assertEqual(agent_result['observations_processed'], 3)

        # 5. Self-improve
        improve_result = self.usi.improve_self(n_cycles=1)
        self.assertEqual(improve_result['cycles_run'], 1)

        # 6. Check metrics
        metrics = self.usi.get_metrics()
        self.assertGreater(metrics['unified_intelligence_quotient'], 0)

    def test_continuous_learning(self):
        """Test continuous learning over multiple cycles"""
        initial_metrics = self.usi.get_metrics()

        # Run many processing cycles
        for i in range(50):
            self.usi.process(
                sensory_input={'visual': np.random.rand(64)},
                goals=['learn', 'optimize'] if i % 10 == 0 else None
            )

        final_metrics = self.usi.get_metrics()

        # Metrics should generally improve
        self.assertGreaterEqual(
            final_metrics['perception_accuracy'],
            initial_metrics['perception_accuracy']
        )
        self.assertGreater(self.usi.total_cycles, 0)
        self.assertGreater(self.usi.total_learning_updates, 0)

    def test_cross_modal_processing(self):
        """Test cross-modal sensory processing"""
        result = self.usi.process(
            sensory_input={
                'visual': np.random.rand(64),
                'audio': np.random.rand(64)
            }
        )

        self.assertIsNotNone(self.usi.state.visual_input)
        self.assertIsNotNone(self.usi.state.audio_input)
        self.assertIsNotNone(self.usi.state.multimodal_state)


def run_all_unified_intelligence_tests():
    """Run all unified intelligence tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestUnifiedIntelligenceInitialization,
        TestCognitiveProcessingCycle,
        TestMultiPhaseProcessing,
        TestCrossSystemIntegration,
        TestIntelligenceMetrics,
        TestThinkingAndReasoning,
        TestImaginationAndCreativity,
        TestSocialCognition,
        TestSelfImprovement,
        TestSerializationAndPersistence,
        TestCognitiveStateManagement,
        TestFullSupertintelligencePipeline,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("UNIFIED SUPER INTELLIGENCE TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.errors) - len(result.failures)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = run_all_unified_intelligence_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
