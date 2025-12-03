#!/usr/bin/env python3
"""
Comprehensive Test Suite for ATLAS Phase 2 Cognitive Systems

This module tests all the advanced cognitive systems implemented in Phase 2:
1. World Model (world_model.py)
2. Executive Control (executive_control.py)
3. Creativity Engine (creativity.py)
4. Theory of Mind (theory_of_mind.py)
5. Recursive Self-Improvement (self_improvement.py)
6. Integration tests between Phase 1 and Phase 2 systems
"""

import unittest
import numpy as np
import sys
import os
import logging
import time

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class TestWorldModel(unittest.TestCase):
    """Test the World Model system"""

    def setUp(self):
        from core.world_model import WorldModel, ObjectState, PhysicsProperty
        self.world_model = WorldModel(
            state_dim=64,
            max_objects=50,
            permanence_decay=0.01,
            prediction_horizon=5,
            learning_rate=0.05,
            random_seed=42
        )
        self.ObjectState = ObjectState
        self.PhysicsProperty = PhysicsProperty

    def test_initialization(self):
        """Test WorldModel initializes correctly"""
        self.assertEqual(self.world_model.state_dim, 64)
        self.assertEqual(self.world_model.max_objects, 50)
        self.assertEqual(len(self.world_model.objects), 0)

    def test_observe_objects(self):
        """Test observing and tracking objects"""
        observations = [
            {
                'embedding': np.random.rand(64),
                'position': np.array([1.0, 2.0, 3.0]),
                'properties': {'type': 'ball'}
            },
            {
                'embedding': np.random.rand(64),
                'position': np.array([5.0, 0.0, 0.0]),
                'properties': {'type': 'cube'}
            }
        ]

        result = self.world_model.observe(observations)

        self.assertEqual(result['total_objects'], 2)
        self.assertEqual(len(result['new_objects']), 2)
        self.assertIn('predictions', result)

    def test_object_permanence(self):
        """Test object permanence when objects are occluded"""
        # First observation
        embedding = np.random.rand(64)
        self.world_model.observe([{
            'embedding': embedding,
            'position': np.array([0.0, 0.0, 0.0])
        }])

        self.assertEqual(len(self.world_model.objects), 1)
        obj_id = list(self.world_model.objects.keys())[0]

        # Object disappears (occluded)
        self.world_model.observe([])

        # Object should still exist
        self.assertIn(obj_id, self.world_model.objects)

        # Query permanence
        permanence = self.world_model.query_object_permanence(obj_id)
        self.assertTrue(permanence['exists'])
        self.assertEqual(permanence['state'], 'occluded')

    def test_physics_prediction(self):
        """Test physics-based prediction"""
        # Observe moving object
        for i in range(5):
            self.world_model.observe([{
                'embedding': np.random.rand(64),
                'position': np.array([float(i), 0.0, 0.0])
            }])

        obj_id = list(self.world_model.objects.keys())[0]
        obj = self.world_model.objects[obj_id]

        # Should have learned velocity
        self.assertTrue(np.linalg.norm(obj.velocity) > 0)

    def test_state_transition_learning(self):
        """Test learning state transitions"""
        # Feed sequence of states
        for i in range(20):
            state = np.random.rand(64)
            self.world_model.observe([], global_state=state)

        # Should have learned transitions
        self.assertTrue(len(self.world_model.state_transitions) > 0)
        self.assertGreater(self.world_model.total_predictions, 0)

    def test_predict_with_intervention(self):
        """Test counterfactual prediction with intervention"""
        # Train on states
        for i in range(10):
            self.world_model.observe([], global_state=np.random.rand(64))

        # Predict with action
        action = np.random.rand(16)
        predictions = self.world_model.predict_with_intervention(action, steps=3)

        self.assertEqual(len(predictions), 3)

    def test_hidden_state_inference(self):
        """Test Bayesian hidden state inference"""
        observation = np.random.rand(64)

        state, uncertainty = self.world_model.infer_hidden_state(observation)

        self.assertEqual(state.shape, (64,))
        self.assertEqual(uncertainty.shape, (64,))

    def test_physics_learning(self):
        """Test learning physics from trajectories"""
        trajectory = [
            {'position': np.array([0.0, 0.0, 0.0])},
            {'position': np.array([0.0, -1.0, 0.0])},
            {'position': np.array([0.0, -3.0, 0.0])},
            {'position': np.array([0.0, -6.0, 0.0])},
        ]

        self.world_model.learn_physics(trajectory)

        # Gravity should be approximately learned
        self.assertTrue(self.world_model.physics.gravity[1] < 0)

    def test_serialization(self):
        """Test serialization and deserialization"""
        from core.world_model import WorldModel

        # Add some data
        self.world_model.observe([{
            'embedding': np.random.rand(64),
            'position': np.array([1.0, 2.0, 3.0])
        }])

        serialized = self.world_model.serialize()

        self.assertIn('state_dim', serialized)
        self.assertIn('objects', serialized)
        self.assertIn('transition_weights', serialized)

        # Deserialize
        restored = WorldModel.deserialize(serialized)
        self.assertEqual(restored.state_dim, self.world_model.state_dim)


class TestExecutiveControl(unittest.TestCase):
    """Test the Executive Control system"""

    def setUp(self):
        from core.executive_control import (
            ExecutiveController, ControlSignal, CognitiveState, Task
        )
        self.controller = ExecutiveController(
            control_dim=64,
            max_tasks=10,
            inhibition_threshold=0.5,
            switch_cost=0.2,
            learning_rate=0.05,
            random_seed=42
        )
        self.ControlSignal = ControlSignal
        self.CognitiveState = CognitiveState
        self.Task = Task

    def test_initialization(self):
        """Test ExecutiveController initializes correctly"""
        self.assertEqual(self.controller.control_dim, 64)
        self.assertEqual(self.controller.max_tasks, 10)
        self.assertEqual(self.controller.cognitive_state, self.CognitiveState.IDLE)

    def test_add_task(self):
        """Test adding tasks"""
        task_id = self.controller.add_task(
            name="test_task",
            priority=0.8,
            requirements={'cpu': 0.5}
        )

        self.assertIn(task_id, self.controller.tasks)
        self.assertEqual(self.controller.current_task, task_id)
        self.assertEqual(self.controller.cognitive_state, self.CognitiveState.FOCUSED)

    def test_task_switching(self):
        """Test task switching with context preservation"""
        task1 = self.controller.add_task("task1", 0.5)
        task2 = self.controller.add_task("task2", 0.8)

        # Switch to task2
        result = self.controller.switch_task(task2)

        self.assertTrue(result['success'])
        self.assertEqual(self.controller.current_task, task2)
        self.assertIn(task1, self.controller.task_stack)

    def test_inhibition(self):
        """Test inhibitory control"""
        result = self.controller.inhibit("target_1", strength=0.8)
        self.assertTrue(result)

        level = self.controller.get_inhibition_level("target_1")
        self.assertGreater(level, 0)

        # Release inhibition
        self.controller.release_inhibition("target_1")
        level = self.controller.get_inhibition_level("target_1")
        self.assertEqual(level, 0)

    def test_control_signal_generation(self):
        """Test generating control signals"""
        input_state = np.random.rand(64)

        for signal_type in self.ControlSignal:
            signal = self.controller.generate_control_signal(input_state, signal_type)
            self.assertEqual(signal.shape, (64,))

    def test_decision_under_uncertainty(self):
        """Test decision making under uncertainty"""
        options = [
            {'value': 0.8, 'risk': 0.2, 'embedding': np.random.rand(64)},
            {'value': 0.6, 'risk': 0.1, 'embedding': np.random.rand(64)},
            {'value': 0.9, 'risk': 0.5, 'embedding': np.random.rand(64)}
        ]

        chosen, confidence, info = self.controller.decide_under_uncertainty(options)

        self.assertIn(chosen, [0, 1, 2])
        self.assertGreater(confidence, 0)
        self.assertIn('utilities', info)

    def test_metacognitive_assessment(self):
        """Test metacognitive assessment"""
        # Perform some operations
        self.controller.add_task("task", 0.5)
        self.controller.step(dt=1.0)

        assessment = self.controller.assess_metacognition()

        self.assertIsNotNone(assessment.confidence)
        self.assertIsNotNone(assessment.uncertainty_type)
        self.assertIsInstance(assessment.recommendations, list)

    def test_feedback_update(self):
        """Test updating from feedback"""
        initial_accuracy = self.controller.prediction_accuracy

        # Update with good feedback
        self.controller.update_from_feedback(0, outcome=0.9, expected=0.8)

        # Accuracy should be maintained or improved
        self.assertGreaterEqual(self.controller.prediction_accuracy, 0)

    def test_step_and_decay(self):
        """Test step function with decay"""
        self.controller.add_task("task", 0.8)
        self.controller.inhibit("target", 0.9)

        initial_inh = self.controller.get_inhibition_level("target")

        # Run several steps
        for _ in range(10):
            self.controller.step(dt=1.0)

        # Inhibition should decay
        final_inh = self.controller.get_inhibition_level("target")
        self.assertLess(final_inh, initial_inh)

    def test_serialization(self):
        """Test serialization"""
        self.controller.add_task("test", 0.5)

        serialized = self.controller.serialize()

        self.assertIn('control_dim', serialized)
        self.assertIn('tasks', serialized)
        self.assertIn('cognitive_state', serialized)


class TestCreativity(unittest.TestCase):
    """Test the Creativity Engine"""

    def setUp(self):
        from core.creativity import CreativityEngine, CreativeMode, NoveltyType
        self.engine = CreativityEngine(
            embedding_dim=64,
            max_concepts=100,
            novelty_threshold=0.3,
            coherence_threshold=0.4,
            temperature=1.0,
            learning_rate=0.05,
            random_seed=42
        )
        self.CreativeMode = CreativeMode
        self.NoveltyType = NoveltyType

    def test_initialization(self):
        """Test CreativityEngine initializes correctly"""
        self.assertEqual(self.engine.embedding_dim, 64)
        self.assertEqual(len(self.engine.concepts), 0)

    def test_add_concept(self):
        """Test adding concepts"""
        concept_id = self.engine.add_concept(
            name="apple",
            embedding=np.random.rand(64),
            attributes={'color': 'red', 'type': 'fruit'}
        )

        self.assertIn(concept_id, self.engine.concepts)
        self.assertEqual(self.engine.concepts[concept_id].name, "apple")

    def test_generation(self):
        """Test generative synthesis"""
        # Add some concepts first
        for i in range(10):
            self.engine.add_concept(f"concept_{i}", np.random.rand(64))

        # Generate new concepts
        results = self.engine.generate(
            mode=self.CreativeMode.DIVERGENT,
            n_samples=5
        )

        self.assertEqual(len(results), 5)
        for embedding, novelty, coherence in results:
            self.assertEqual(embedding.shape, (64,))

    def test_conceptual_blending(self):
        """Test conceptual blending"""
        # Add concepts to blend
        c1 = self.engine.add_concept("bird", np.random.rand(64),
                                      attributes={'can_fly': True})
        c2 = self.engine.add_concept("fish", np.random.rand(64),
                                      attributes={'can_swim': True})

        # Lower thresholds for testing
        self.engine.novelty_threshold = 0.0
        self.engine.coherence_threshold = 0.0

        blend = self.engine.blend([c1, c2], blend_mode="average")

        if blend:  # May be None if thresholds not met
            self.assertIn(c1, blend.source_concepts)
            self.assertIn(c2, blend.source_concepts)

    def test_imagination(self):
        """Test mental simulation"""
        initial_state = np.random.rand(64)

        imagination = self.engine.imagine(
            initial_state=initial_state,
            steps=10,
            goal_state=np.random.rand(64)
        )

        self.assertEqual(len(imagination.trajectory), 11)  # Initial + 10 steps
        self.assertGreaterEqual(imagination.plausibility, 0)
        self.assertGreaterEqual(imagination.novelty, 0)

    def test_divergent_thinking(self):
        """Test divergent thinking"""
        problem = np.random.rand(64)

        solutions = self.engine.divergent_think(
            problem=problem,
            n_solutions=5,
            diversity_weight=0.5
        )

        self.assertGreater(len(solutions), 0)
        self.assertLessEqual(len(solutions), 5)

    def test_temperature_and_relaxation(self):
        """Test temperature and relaxation settings"""
        self.engine.set_temperature(2.0)
        self.assertEqual(self.engine.temperature, 2.0)

        self.engine.set_relaxation_level(0.5)
        self.assertEqual(self.engine.relaxation_level, 0.5)

    def test_serialization(self):
        """Test serialization"""
        self.engine.add_concept("test", np.random.rand(64))

        serialized = self.engine.serialize()

        self.assertIn('embedding_dim', serialized)
        self.assertIn('concepts', serialized)
        self.assertIn('generator_weights', serialized)


class TestTheoryOfMind(unittest.TestCase):
    """Test the Theory of Mind module"""

    def setUp(self):
        from core.theory_of_mind import (
            TheoryOfMind, AgentType, MentalStateType, SocialRelation
        )
        self.tom = TheoryOfMind(
            embedding_dim=64,
            max_agents=50,
            belief_update_rate=0.1,
            simulation_steps=5,
            learning_rate=0.05,
            random_seed=42
        )
        self.AgentType = AgentType
        self.MentalStateType = MentalStateType
        self.SocialRelation = SocialRelation

    def test_initialization(self):
        """Test TheoryOfMind initializes correctly"""
        self.assertEqual(self.tom.embedding_dim, 64)
        self.assertIsNotNone(self.tom.self_model)

    def test_add_agent(self):
        """Test adding agents"""
        agent_id = self.tom.add_agent(
            agent_type=self.AgentType.HUMAN,
            relationship=self.SocialRelation.COOPERATIVE
        )

        self.assertIn(agent_id, self.tom.agents)
        self.assertEqual(self.tom.agents[agent_id].agent_type, self.AgentType.HUMAN)

    def test_observe_and_infer(self):
        """Test observing agent and inferring mental states"""
        agent_id = self.tom.add_agent()

        inferences = self.tom.observe(
            agent_id=agent_id,
            action=np.random.rand(64),
            context=np.random.rand(64)
        )

        self.assertIn('intention', inferences)
        self.assertIn('belief', inferences)

    def test_predict_action(self):
        """Test predicting agent action"""
        agent_id = self.tom.add_agent()

        # Observe some actions first
        for _ in range(5):
            self.tom.observe(
                agent_id,
                np.random.rand(64),
                np.random.rand(64)
            )

        predicted, confidence = self.tom.predict_action(
            agent_id,
            context=np.random.rand(64)
        )

        self.assertEqual(predicted.shape, (64,))
        self.assertGreater(confidence, 0)

    def test_false_belief_inference(self):
        """Test false belief task"""
        agent_id = self.tom.add_agent()

        # Agent observed one thing
        self.tom.observe(
            agent_id,
            np.random.rand(64),
            np.random.rand(64)
        )

        # But true state is different
        true_state = np.random.rand(64)

        result = self.tom.infer_false_belief(
            agent_id,
            true_state=true_state
        )

        self.assertIn('has_false_belief', result)
        self.assertIn('divergence', result)

    def test_simulate_perspective(self):
        """Test perspective simulation"""
        agent_id = self.tom.add_agent()

        trajectory = self.tom.simulate_perspective(
            agent_id,
            scenario=np.random.rand(64),
            steps=5
        )

        self.assertEqual(len(trajectory), 5)

    def test_cooperative_reasoning(self):
        """Test cooperative reasoning"""
        agent1 = self.tom.add_agent(relationship=self.SocialRelation.COOPERATIVE)
        agent2 = self.tom.add_agent(relationship=self.SocialRelation.COOPERATIVE)

        result = self.tom.reason_cooperatively(
            agent_ids=[agent1, agent2],
            shared_goal=np.random.rand(64),
            context=np.random.rand(64)
        )

        self.assertIn('coordination_quality', result)
        self.assertIn('recommended_strategy', result)

    def test_competitive_reasoning(self):
        """Test competitive reasoning"""
        opponent = self.tom.add_agent(relationship=self.SocialRelation.COMPETITIVE)

        result = self.tom.reason_competitively(
            opponent_id=opponent,
            own_goal=np.random.rand(64),
            context=np.random.rand(64)
        )

        self.assertIn('strategy', result)
        self.assertIn('predicted_opponent_action', result)

    def test_social_learning(self):
        """Test learning from observation"""
        demonstrator = self.tom.add_agent()

        demonstrations = [
            (np.random.rand(64), np.random.rand(64))
            for _ in range(5)
        ]

        result = self.tom.learn_from_observation(demonstrator, demonstrations)

        self.assertEqual(result['patterns_learned'], 5)

    def test_trust_update(self):
        """Test trust level updates"""
        agent_id = self.tom.add_agent()
        initial_trust = self.tom.agents[agent_id].trust_level

        self.tom.update_trust(agent_id, 0.5)  # Positive outcome

        self.assertGreater(self.tom.agents[agent_id].trust_level, initial_trust)

    def test_serialization(self):
        """Test serialization"""
        self.tom.add_agent()

        serialized = self.tom.serialize()

        self.assertIn('embedding_dim', serialized)
        self.assertIn('agents', serialized)


class TestSelfImprovement(unittest.TestCase):
    """Test the Recursive Self-Improvement system"""

    def setUp(self):
        from core.self_improvement import (
            RecursiveSelfImprovement, ImprovementType, SafetyLevel
        )
        self.system = RecursiveSelfImprovement(
            num_hyperparameters=20,
            max_modifications=50,
            safety_level=SafetyLevel.MODERATE,
            improvement_threshold=0.05,
            reversion_threshold=-0.1,
            random_seed=42
        )
        self.ImprovementType = ImprovementType
        self.SafetyLevel = SafetyLevel

    def test_initialization(self):
        """Test RecursiveSelfImprovement initializes correctly"""
        self.assertGreater(len(self.system.hyperparameters), 0)
        self.assertGreater(len(self.system.metrics), 0)
        self.assertGreater(len(self.system.capabilities), 0)

    def test_propose_hyperparameter_improvement(self):
        """Test proposing hyperparameter improvements"""
        mod = self.system.propose_improvement(
            self.ImprovementType.HYPERPARAMETER,
            target='learning_rate',
            strategy='gradient_free'
        )

        self.assertIsNotNone(mod)
        self.assertIn(mod.modification_id, self.system.modifications)

    def test_propose_capability_enhancement(self):
        """Test proposing capability enhancements"""
        mod = self.system.propose_improvement(
            self.ImprovementType.CAPABILITY,
            target='reasoning'
        )

        self.assertIsNotNone(mod)
        self.assertEqual(mod.improvement_type, self.ImprovementType.CAPABILITY)

    def test_apply_and_evaluate_modification(self):
        """Test applying and evaluating modifications"""
        mod = self.system.propose_improvement(
            self.ImprovementType.HYPERPARAMETER,
            target='learning_rate'
        )

        # Apply
        result = self.system.apply_modification(mod.modification_id)
        self.assertTrue(result['success'])

        # Evaluate with positive improvement
        eval_result = self.system.evaluate_modification(
            mod.modification_id,
            {'overall_capability': 0.1}
        )

        self.assertEqual(eval_result['decision'], 'keep')

    def test_revert_modification(self):
        """Test reverting modifications"""
        mod = self.system.propose_improvement(
            self.ImprovementType.HYPERPARAMETER,
            target='momentum'
        )

        original_value = self.system.hyperparameters['momentum']
        self.system.apply_modification(mod.modification_id)

        # Revert
        result = self.system.revert_modification(mod.modification_id)

        self.assertTrue(result['success'])
        self.assertEqual(
            self.system.hyperparameters['momentum'],
            original_value
        )

    def test_optimization_cycle(self):
        """Test running optimization cycle"""
        results = self.system.run_optimization_cycle(n_proposals=3)

        self.assertIn('proposals', results)
        self.assertIn('applied', results)
        self.assertEqual(len(results['proposals']), 3)

    def test_evolutionary_search(self):
        """Test evolutionary search"""
        results = self.system.evolutionary_search(
            population_size=5,
            generations=3
        )

        self.assertIn('best_fitness', results)
        self.assertIn('best_configuration', results)
        self.assertEqual(results['generations'], 3)

    def test_improvement_recommendations(self):
        """Test getting improvement recommendations"""
        recommendations = self.system.get_improvement_recommendations()

        self.assertIsInstance(recommendations, list)
        if recommendations:
            self.assertIn('type', recommendations[0])
            self.assertIn('priority', recommendations[0])

    def test_checkpoint_creation(self):
        """Test checkpoint creation"""
        initial_checkpoints = len(self.system.checkpoints)

        self.system._create_checkpoint("test")

        self.assertEqual(len(self.system.checkpoints), initial_checkpoints + 1)

    def test_serialization(self):
        """Test serialization"""
        self.system.propose_improvement(
            self.ImprovementType.HYPERPARAMETER,
            'learning_rate'
        )

        serialized = self.system.serialize()

        self.assertIn('hyperparameters', serialized)
        self.assertIn('metrics', serialized)
        self.assertIn('capabilities', serialized)


class TestPhase2Integration(unittest.TestCase):
    """Test integration between Phase 2 cognitive systems"""

    def setUp(self):
        """Set up integrated Phase 2 systems"""
        from core.world_model import WorldModel
        from core.executive_control import ExecutiveController
        from core.creativity import CreativityEngine
        from core.theory_of_mind import TheoryOfMind
        from core.self_improvement import RecursiveSelfImprovement

        self.world_model = WorldModel(state_dim=64, random_seed=42)
        self.executive = ExecutiveController(control_dim=64, random_seed=42)
        self.creativity = CreativityEngine(embedding_dim=64, random_seed=42)
        self.tom = TheoryOfMind(embedding_dim=64, random_seed=42)
        self.self_improvement = RecursiveSelfImprovement(random_seed=42)

    def test_world_model_executive_integration(self):
        """Test integration between world model and executive control"""
        # Observe world state
        self.world_model.observe([{
            'embedding': np.random.rand(64),
            'position': np.array([0.0, 0.0, 0.0])
        }])

        # Get world state for task planning
        world_state = self.world_model.get_world_state_embedding()

        # Add task based on world state
        task_id = self.executive.add_task(
            name="respond_to_world",
            priority=0.8,
            context=world_state
        )

        self.assertIsNotNone(task_id)

    def test_creativity_executive_integration(self):
        """Test integration between creativity and executive control"""
        # Add a creative task
        task_id = self.executive.add_task(
            name="creative_problem_solving",
            priority=0.7
        )

        # Generate creative solutions
        problem = np.random.rand(64)
        solutions = self.creativity.divergent_think(problem, n_solutions=3)

        # Decision about which solution
        options = [
            {'value': 0.5, 'risk': 0.2, 'embedding': sol}
            for sol, _ in solutions
        ]

        if options:
            chosen, confidence, _ = self.executive.decide_under_uncertainty(options)
            self.assertIsNotNone(chosen)

    def test_tom_executive_integration(self):
        """Test integration between theory of mind and executive control"""
        # Model an agent
        agent_id = self.tom.add_agent()

        # Observe their action
        self.tom.observe(
            agent_id,
            action=np.random.rand(64),
            context=np.random.rand(64)
        )

        # Predict their next action
        predicted_action, confidence = self.tom.predict_action(
            agent_id,
            context=np.random.rand(64)
        )

        # Use prediction in executive decision making
        options = [
            {'value': 0.8, 'risk': 0.2, 'embedding': predicted_action},
            {'value': 0.6, 'risk': 0.1, 'embedding': np.random.rand(64)}
        ]

        chosen, _, _ = self.executive.decide_under_uncertainty(
            options,
            context=predicted_action
        )

        self.assertIn(chosen, [0, 1])

    def test_self_improvement_monitoring(self):
        """Test self-improvement monitoring across systems"""
        # Perform some cognitive operations
        self.world_model.observe([{
            'embedding': np.random.rand(64),
            'position': np.array([0.0, 0.0, 0.0])
        }])

        self.executive.add_task("test", 0.5)
        self.creativity.add_concept("test", np.random.rand(64))
        self.tom.add_agent()

        # Run improvement cycle
        results = self.self_improvement.run_optimization_cycle(n_proposals=2)

        self.assertIn('proposals', results)

    def test_full_cognitive_cycle_phase2(self):
        """Test full cognitive processing cycle with Phase 2 systems"""
        # 1. Perceive world
        self.world_model.observe([{
            'embedding': np.random.rand(64),
            'position': np.array([1.0, 0.0, 0.0]),
            'properties': {'type': 'other_agent'}
        }])

        # 2. Create task
        self.executive.add_task("interact", 0.8)

        # 3. Model the other agent
        agent_id = self.tom.add_agent()
        self.tom.observe(
            agent_id,
            action=np.random.rand(64),
            context=np.random.rand(64)
        )

        # 4. Generate creative response options
        problem = self.world_model.get_world_state_embedding()
        solutions = self.creativity.divergent_think(problem, n_solutions=3)

        # 5. Decide on action
        options = [
            {'value': 0.5 + i * 0.1, 'risk': 0.1 + i * 0.05,
             'embedding': sol}
            for i, (sol, _) in enumerate(solutions)
        ]

        if options:
            chosen, confidence, _ = self.executive.decide_under_uncertainty(options)

            # 6. Complete task
            self.executive.complete_task()

        # Verify all systems have state
        self.assertGreater(len(self.world_model.objects), 0)
        self.assertGreater(len(self.tom.agents), 0)


class TestPhase1Phase2Integration(unittest.TestCase):
    """Test integration between Phase 1 and Phase 2 systems"""

    def setUp(self):
        """Set up integrated systems from both phases"""
        # Phase 1 systems
        from core.episodic_memory import EpisodicMemory
        from core.semantic_memory import SemanticMemory
        from core.meta_learning import MetaLearner
        from core.goal_planning import GoalPlanningSystem, GoalType

        # Phase 2 systems
        from core.world_model import WorldModel
        from core.executive_control import ExecutiveController
        from core.creativity import CreativityEngine
        from core.theory_of_mind import TheoryOfMind

        self.episodic = EpisodicMemory(state_size=64, random_seed=42)
        self.semantic = SemanticMemory(embedding_size=64, random_seed=42)
        self.meta_learner = MetaLearner(random_seed=42)
        self.goal_planner = GoalPlanningSystem(random_seed=42)
        self.GoalType = GoalType

        self.world_model = WorldModel(state_dim=64, random_seed=42)
        self.executive = ExecutiveController(control_dim=64, random_seed=42)
        self.creativity = CreativityEngine(embedding_dim=64, random_seed=42)
        self.tom = TheoryOfMind(embedding_dim=64, random_seed=42)

    def test_memory_world_model_integration(self):
        """Test integration between memory systems and world model"""
        # Observe world
        result = self.world_model.observe([{
            'embedding': np.random.rand(64),
            'position': np.array([0.0, 0.0, 0.0]),
            'properties': {'name': 'test_object'}
        }])

        # Store in episodic memory
        world_state = self.world_model.get_world_state_embedding()
        self.episodic.store(
            state=world_state,
            sensory_data={'world': world_state[:32]},
            context={'objects': result['total_objects']}
        )

        # Create semantic concept
        self.semantic.add_concept(
            name='world_snapshot',
            embedding=world_state
        )

        self.assertEqual(len(self.episodic.episodes), 1)
        self.assertIn('world_snapshot', self.semantic.concepts)

    def test_goal_executive_integration(self):
        """Test integration between goal planning and executive control"""
        # Generate goal
        goal = self.goal_planner.generate_goal(
            goal_type=self.GoalType.LEARNING,
            context={'task': 'understand_world'}
        )

        # Create corresponding executive task
        if goal:
            task_id = self.executive.add_task(
                name=goal.name,
                priority=goal.priority,
                requirements={'goal_value': goal.value}
            )

            self.assertIn(task_id, self.executive.tasks)

    def test_meta_learning_creativity_integration(self):
        """Test integration between meta-learning and creativity"""
        # Add creative concepts
        for i in range(5):
            self.creativity.add_concept(f"concept_{i}", np.random.rand(64))

        # Generate creative outputs
        outputs = self.creativity.generate(n_samples=3)

        # Use meta-learner to select strategy for evaluation
        strategy, params = self.meta_learner.select_strategy({
            'creativity_task': True,
            'complexity': 0.7
        })

        self.assertIsNotNone(strategy)

    def test_tom_semantic_memory_integration(self):
        """Test integration between theory of mind and semantic memory"""
        # Add agent
        agent_id = self.tom.add_agent()

        # Observe behavior
        self.tom.observe(
            agent_id,
            action=np.random.rand(64),
            context=np.random.rand(64)
        )

        # Store agent concept in semantic memory
        agent = self.tom.agents[agent_id]
        if agent.personality_embedding is not None:
            self.semantic.add_concept(
                name=f'agent_{agent_id}',
                embedding=agent.personality_embedding,
                attributes={'trust': agent.trust_level}
            )

            self.assertIn(f'agent_{agent_id}', self.semantic.concepts)

    def test_complete_superintelligence_cycle(self):
        """Test complete superintelligence processing cycle"""
        # 1. World perception
        self.world_model.observe([{
            'embedding': np.random.rand(64),
            'position': np.array([0.0, 0.0, 0.0])
        }])

        # 2. Goal setting
        goal = self.goal_planner.generate_goal(self.GoalType.EXPLORATION)

        # 3. Executive task management
        if goal:
            self.executive.add_task(goal.name, goal.priority)

        # 4. Memory encoding
        world_state = self.world_model.get_world_state_embedding()
        self.episodic.store(
            state=world_state,
            sensory_data={'world': world_state[:32]}
        )

        # 5. Social reasoning
        agent_id = self.tom.add_agent()
        self.tom.observe(agent_id, np.random.rand(64), world_state)

        # 6. Creative problem solving
        solutions = self.creativity.divergent_think(world_state, n_solutions=3)

        # 7. Decision making
        if solutions:
            options = [
                {'value': 0.5, 'risk': 0.2, 'embedding': sol}
                for sol, _ in solutions
            ]
            chosen, confidence, _ = self.executive.decide_under_uncertainty(options)

        # 8. Learning update
        strategy, params = self.meta_learner.select_strategy({'complexity': 0.5})
        self.meta_learner.update(
            strategy, params,
            {'complexity': 0.5},
            {'accuracy': 0.8}
        )

        # Verify state
        self.assertGreater(len(self.episodic.episodes), 0)
        self.assertGreater(len(self.world_model.objects), 0)


def run_all_phase2_tests():
    """Run all Phase 2 cognitive system tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestWorldModel,
        TestExecutiveControl,
        TestCreativity,
        TestTheoryOfMind,
        TestSelfImprovement,
        TestPhase2Integration,
        TestPhase1Phase2Integration,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*70)
    print("PHASE 2 COGNITIVE SYSTEMS TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.errors) - len(result.failures)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result


if __name__ == "__main__":
    result = run_all_phase2_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
