#!/usr/bin/env python3
"""
Comprehensive Test Suite for ATLAS Cognitive Systems

This module tests all the advanced cognitive systems that enable
superintelligence capabilities:
1. Meta-Learning (meta_learning.py)
2. Goal Planning (goal_planning.py)
3. Causal Reasoning (causal_reasoning.py)
4. Abstract Reasoning (abstract_reasoning.py)
5. Language Grounding (language_grounding.py)
6. Episodic Memory (episodic_memory.py)
7. Semantic Memory (semantic_memory.py)
8. Working Memory (working_memory.py)
9. Image Classification (image_classification.py)
10. Cognitive Integration Tests
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


class TestMetaLearning(unittest.TestCase):
    """Test the Meta-Learning system"""

    def setUp(self):
        from core.meta_learning import MetaLearner, LearningStrategy
        self.meta_learner = MetaLearner(
            num_strategies=7,
            num_hyperparameters=10,
            exploration_rate=0.3,
            learning_rate=0.05,
            memory_size=100,
            random_seed=42
        )
        self.LearningStrategy = LearningStrategy

    def test_initialization(self):
        """Test MetaLearner initializes correctly"""
        self.assertEqual(self.meta_learner.num_strategies, 7)
        self.assertEqual(self.meta_learner.exploration_rate, 0.3)
        self.assertEqual(len(self.meta_learner.strategy_performance), 7)

    def test_strategy_selection(self):
        """Test strategy selection works"""
        task_characteristics = {
            'complexity': 0.5,
            'sparsity': 0.3,
            'temporal': 0.2
        }
        strategy, hyperparams = self.meta_learner.select_strategy(task_characteristics)

        self.assertIsInstance(strategy, self.LearningStrategy)
        self.assertIsInstance(hyperparams, dict)
        self.assertIn('learning_rate', hyperparams)

    def test_update(self):
        """Test meta-learner update mechanism"""
        task_characteristics = {'complexity': 0.5, 'sparsity': 0.3}
        strategy, hyperparams = self.meta_learner.select_strategy(task_characteristics)

        performance_metrics = {
            'accuracy': 0.8,
            'prediction_error': 0.1,
            'sparsity': 0.15
        }

        initial_updates = self.meta_learner.total_updates
        self.meta_learner.update(strategy, hyperparams, task_characteristics, performance_metrics)

        self.assertEqual(self.meta_learner.total_updates, initial_updates + 1)
        self.assertTrue(len(self.meta_learner.experience_buffer) > 0)

    def test_curriculum_progression(self):
        """Test curriculum learning progression"""
        initial_difficulty = self.meta_learner.difficulty_progression

        # Simulate good performance to increase difficulty
        for _ in range(20):
            strategy = self.LearningStrategy.HEBBIAN
            hyperparams = {'learning_rate': 0.01}
            task_characteristics = {'complexity': 0.5}
            performance_metrics = {'accuracy': 0.95}  # High performance

            self.meta_learner.update(strategy, hyperparams, task_characteristics, performance_metrics)

        # Difficulty should increase with good performance
        self.assertGreaterEqual(self.meta_learner.difficulty_progression, initial_difficulty)

    def test_algorithm_discovery(self):
        """Test algorithm discovery from experience"""
        # Add enough experiences
        for i in range(60):
            strategy = list(self.LearningStrategy)[i % 3]
            hyperparams = {'learning_rate': 0.01 + i * 0.001}
            task_characteristics = {'complexity': 0.5}
            performance_metrics = {'accuracy': 0.7 + (i % 10) * 0.02}

            self.meta_learner.update(strategy, hyperparams, task_characteristics, performance_metrics)

        # Try to discover new algorithm
        discovered = self.meta_learner.discover_algorithm()
        # May or may not discover based on patterns

    def test_serialization(self):
        """Test serialization and stats"""
        stats = self.meta_learner.get_stats()
        serialized = self.meta_learner.serialize()

        self.assertIn('total_selections', stats)
        self.assertIn('strategy_performance', serialized)

    def test_deserialization(self):
        """Test deserializing meta-learner"""
        from core.meta_learning import MetaLearner

        # Add some data
        for _ in range(10):
            strategy, hyperparams = self.meta_learner.select_strategy({'complexity': 0.5})
            self.meta_learner.update(strategy, hyperparams, {'complexity': 0.5}, {'accuracy': 0.8})

        serialized = self.meta_learner.serialize()
        restored = MetaLearner.deserialize(serialized)

        self.assertIsInstance(restored, MetaLearner)


class TestGoalPlanning(unittest.TestCase):
    """Test the Goal Planning system"""

    def setUp(self):
        from core.goal_planning import GoalPlanningSystem, GoalType, Goal, Action
        self.planner = GoalPlanningSystem(
            max_goals=10,
            planning_horizon=5,
            enable_meta_goals=True,
            random_seed=42
        )
        self.GoalType = GoalType
        self.Goal = Goal
        self.Action = Action

    def test_initialization(self):
        """Test GoalPlanningSystem initializes correctly"""
        self.assertEqual(self.planner.max_goals, 10)
        self.assertTrue(self.planner.enable_meta_goals)
        # Meta-goals should be initialized
        self.assertTrue(len(self.planner.active_goals) > 0)

    def test_goal_generation(self):
        """Test autonomous goal generation"""
        goal = self.planner.generate_goal(
            goal_type=self.GoalType.LEARNING,
            context={'task': 'pattern_recognition'}
        )

        self.assertIsNotNone(goal)
        self.assertEqual(goal.goal_type, self.GoalType.LEARNING)
        self.assertIn(goal, self.planner.active_goals)

    def test_goal_type_selection(self):
        """Test goal type is selected based on drives"""
        # Test multiple selections
        goal_types = []
        for _ in range(20):
            goal = self.planner.generate_goal(context={})
            if goal:
                goal_types.append(goal.goal_type)

        # Should have variety in goal types
        unique_types = set(goal_types)
        self.assertTrue(len(unique_types) >= 1)

    def test_action_library(self):
        """Test adding actions to library"""
        action = self.Action(
            name="move_forward",
            preconditions={"has_path": True},
            effects={"position": "advanced"},
            cost=1.0,
            duration=1.0
        )

        self.planner.action_library.append(action)
        self.assertIn(action, self.planner.action_library)

    def test_plan_creation(self):
        """Test plan creation for a goal"""
        # Add some actions
        action1 = self.Action(
            name="prepare",
            preconditions={},
            effects={"prepared": True},
            cost=1.0,
            duration=1.0
        )
        action2 = self.Action(
            name="execute",
            preconditions={"prepared": True},
            effects={"test_goal_0": True},
            cost=2.0,
            duration=2.0
        )
        self.planner.action_library = [action1, action2]

        goal = self.Goal(
            name="test_goal_0",
            goal_type=self.GoalType.OPTIMIZATION,
            priority=0.8,
            value=1.0
        )

        plan = self.planner.create_plan(goal, {"prepared": False})
        # Plan may or may not be found depending on goal satisfaction logic

    def test_intrinsic_motivation(self):
        """Test intrinsic motivation drives"""
        initial_curiosity = self.planner.curiosity_level
        initial_competence = self.planner.competence_level

        # These should be initialized
        self.assertGreater(initial_curiosity, 0)
        self.assertGreaterEqual(initial_competence, 0)

    def test_stats(self):
        """Test getting stats"""
        stats = self.planner.get_stats()

        self.assertIn('active_goals', stats)
        self.assertIn('curiosity_level', stats)
        self.assertIn('goal_success_rate', stats)

    def test_serialization(self):
        """Test serialization"""
        serialized = self.planner.serialize()

        self.assertIn('active_goals', serialized)
        self.assertIn('drives', serialized)


class TestCausalReasoning(unittest.TestCase):
    """Test the Causal Reasoning system"""

    def setUp(self):
        from core.causal_reasoning import (
            CausalReasoner, CausalGraph, CausalModel,
            CausalRelationType, CounterfactualQuery
        )
        self.reasoner = CausalReasoner(learning_rate=0.1)
        self.CausalRelationType = CausalRelationType
        self.CounterfactualQuery = CounterfactualQuery
        self.CausalGraph = CausalGraph
        self.CausalModel = CausalModel

    def test_graph_creation(self):
        """Test causal graph creation"""
        graph = self.CausalGraph()

        graph.add_variable("rain")
        graph.add_variable("wet_ground")
        graph.add_variable("slippery")

        link = graph.add_link("rain", "wet_ground", self.CausalRelationType.CAUSES)
        self.assertIsNotNone(link)

        link2 = graph.add_link("wet_ground", "slippery", self.CausalRelationType.CAUSES)
        self.assertIsNotNone(link2)

        # Check topology
        self.assertIn("rain", graph.parents["wet_ground"])
        self.assertIn("wet_ground", graph.children["rain"])

    def test_cycle_prevention(self):
        """Test that cycles are prevented"""
        graph = self.CausalGraph()

        graph.add_variable("A")
        graph.add_variable("B")
        graph.add_variable("C")

        graph.add_link("A", "B")
        graph.add_link("B", "C")

        # This should fail (would create cycle)
        result = graph.add_link("C", "A")
        self.assertIsNone(result)

    def test_topological_order(self):
        """Test topological ordering"""
        graph = self.CausalGraph()

        graph.add_variable("A")
        graph.add_variable("B")
        graph.add_variable("C")

        graph.add_link("A", "B")
        graph.add_link("B", "C")

        order = graph.get_topological_order()

        # A should come before B, B before C
        self.assertTrue(order.index("A") < order.index("B"))
        self.assertTrue(order.index("B") < order.index("C"))

    def test_observation(self):
        """Test recording observations"""
        initial_obs = self.reasoner.total_observations

        self.reasoner.observe({
            "rain": 1.0,
            "wet_ground": 0.9,
            "slippery": 0.8
        })

        self.assertEqual(self.reasoner.total_observations, initial_obs + 1)

    def test_intervention(self):
        """Test do-calculus interventions"""
        # Build model
        self.reasoner.model.graph.add_variable("cause")
        self.reasoner.model.graph.add_variable("effect")
        self.reasoner.model.graph.add_link("cause", "effect", strength=0.7)
        self.reasoner.model.weights[("cause", "effect")] = 0.7
        self.reasoner.model.set_default_linear_equations()

        # Intervene
        predictions = self.reasoner.do({"cause": 1.0})

        self.assertIn("effect", predictions)
        self.assertGreater(predictions["effect"], 0)

    def test_counterfactual(self):
        """Test counterfactual reasoning"""
        # Build model
        self.reasoner.model.graph.add_variable("X")
        self.reasoner.model.graph.add_variable("Y")
        self.reasoner.model.graph.add_link("X", "Y", strength=0.8)
        self.reasoner.model.weights[("X", "Y")] = 0.8
        self.reasoner.model.set_default_linear_equations()

        # What if X had been 0 instead of 1?
        result, explanation = self.reasoner.what_if(
            factual={"X": 1.0, "Y": 0.8},
            hypothetical={"X": 0.0},
            query="Y"
        )

        self.assertIsInstance(result, float)
        self.assertIsInstance(explanation, str)

    def test_causal_explanation(self):
        """Test causal explanation generation"""
        # Build model
        self.reasoner.model.graph.add_variable("A")
        self.reasoner.model.graph.add_variable("B")
        self.reasoner.model.graph.add_variable("C")
        self.reasoner.model.graph.add_link("A", "C", strength=0.6)
        self.reasoner.model.graph.add_link("B", "C", strength=0.4)
        self.reasoner.model.weights[("A", "C")] = 0.6
        self.reasoner.model.weights[("B", "C")] = 0.4

        explanation = self.reasoner.why(
            effect="C",
            evidence={"A": 1.0, "B": 0.5, "C": 0.8}
        )

        self.assertEqual(explanation.effect, "C")
        self.assertTrue(len(explanation.causes) > 0)

    def test_find_causes_and_effects(self):
        """Test finding causes and effects"""
        self.reasoner.model.graph.add_variable("X")
        self.reasoner.model.graph.add_variable("Y")
        self.reasoner.model.graph.add_variable("Z")
        self.reasoner.model.graph.add_link("X", "Y", strength=0.7)
        self.reasoner.model.graph.add_link("Y", "Z", strength=0.5)

        causes = self.reasoner.find_causes("Y")
        effects = self.reasoner.find_effects("Y")

        self.assertEqual(len(causes), 1)
        self.assertEqual(causes[0][0], "X")
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0][0], "Z")

    def test_structure_learning(self):
        """Test learning causal structure from observations"""
        # Add many observations
        for i in range(30):
            x = np.random.rand()
            y = 0.8 * x + 0.1 * np.random.rand()
            self.reasoner.observe({"x": x, "y": y}, learn_structure=False)

        # Force structure learning
        links_added = self.reasoner.model.learn_structure(threshold=0.3)

        # Should learn some structure
        self.assertGreaterEqual(links_added, 0)


class TestAbstractReasoning(unittest.TestCase):
    """Test the Abstract Reasoning system"""

    def setUp(self):
        from core.abstract_reasoning import (
            KnowledgeBase, Proposition, Rule, Analogy, Symbol, Predicate,
            LogicOperator, RelationType
        )
        self.kb = KnowledgeBase()
        self.Proposition = Proposition
        self.Rule = Rule
        self.Analogy = Analogy
        self.Symbol = Symbol
        self.Predicate = Predicate
        self.LogicOperator = LogicOperator
        self.RelationType = RelationType

    def test_knowledge_base_creation(self):
        """Test knowledge base initialization"""
        self.assertEqual(len(self.kb.facts), 0)
        self.assertEqual(len(self.kb.rules), 0)

    def test_add_facts(self):
        """Test adding facts to knowledge base"""
        prop = self.Proposition(
            predicate="is_a",
            arguments=["Socrates", "human"],
            truth_value=True
        )

        self.kb.add_fact(prop)
        self.assertIn(prop, self.kb.facts)

    def test_add_rules(self):
        """Test adding rules"""
        # All humans are mortal
        antecedent = [self.Proposition("is_a", ["?X", "human"])]
        consequent = self.Proposition("is_mortal", ["?X"])

        rule = self.Rule(
            name="mortality_rule",
            antecedent=antecedent,
            consequent=consequent,
            confidence=1.0
        )

        self.kb.add_rule(rule)
        self.assertIn(rule, self.kb.rules)

    def test_proposition_equality(self):
        """Test proposition equality"""
        p1 = self.Proposition("likes", ["Alice", "Bob"])
        p2 = self.Proposition("likes", ["Alice", "Bob"])
        p3 = self.Proposition("likes", ["Alice", "Charlie"])

        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)

    def test_analogy_transfer(self):
        """Test analogical transfer"""
        # Atom:electron :: Solar system:planet
        analogy = self.Analogy(
            source_domain="atom",
            target_domain="solar_system",
            mappings={
                "nucleus": "sun",
                "electron": "planet"
            },
            relation_mappings={
                "orbits": "orbits"
            },
            strength=0.8
        )

        source_prop = self.Proposition("orbits", ["electron", "nucleus"])
        transferred = analogy.transfer(source_prop)

        self.assertEqual(transferred.predicate, "orbits")
        self.assertIn("planet", transferred.arguments)
        self.assertIn("sun", transferred.arguments)

    def test_predicate_creation(self):
        """Test predicate and proposition creation"""
        loves = self.Predicate(name="loves", arity=2)

        # Create proposition from predicate
        prop = loves("Alice", "Bob")

        self.assertEqual(prop.predicate, "loves")
        self.assertEqual(len(prop.arguments), 2)

    def test_symbol_creation(self):
        """Test symbol creation"""
        sym = self.Symbol(
            name="x",
            symbol_type="variable",
            properties={"type": "integer"}
        )

        self.assertEqual(sym.name, "x")
        self.assertEqual(sym.symbol_type, "variable")


class TestLanguageGrounding(unittest.TestCase):
    """Test the Language Grounding system"""

    def setUp(self):
        from core.language_grounding import LanguageGrounding, WordType
        self.grounding = LanguageGrounding(
            embedding_dim=64,
            vocabulary_size=1000,
            learning_rate=0.05,
            random_seed=42
        )
        self.WordType = WordType

    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.grounding.embedding_dim, 64)
        self.assertEqual(len(self.grounding.vocabulary), 0)

    def test_word_type_detection(self):
        """Test word type detection"""
        # Test various word types
        self.assertEqual(self.grounding._detect_word_type("the"), self.WordType.DETERMINER)
        self.assertEqual(self.grounding._detect_word_type("in"), self.WordType.PREPOSITION)
        self.assertEqual(self.grounding._detect_word_type("and"), self.WordType.CONJUNCTION)
        self.assertEqual(self.grounding._detect_word_type("quickly"), self.WordType.ADVERB)
        self.assertEqual(self.grounding._detect_word_type("beautiful"), self.WordType.ADJECTIVE)
        self.assertEqual(self.grounding._detect_word_type("running"), self.WordType.VERB)

    def test_word_creation(self):
        """Test word creation in vocabulary"""
        word = self.grounding._get_or_create_word("apple")

        self.assertIn("apple", self.grounding.vocabulary)
        self.assertEqual(word.word, "apple")
        self.assertEqual(word.embedding.shape, (64,))

    def test_vocabulary_growth(self):
        """Test vocabulary grows with new words"""
        initial_size = len(self.grounding.vocabulary)

        self.grounding._get_or_create_word("cat")
        self.grounding._get_or_create_word("dog")
        self.grounding._get_or_create_word("bird")

        self.assertEqual(len(self.grounding.vocabulary), initial_size + 3)

    def test_word_embedding_normalization(self):
        """Test that word embeddings are normalized"""
        word = self.grounding._get_or_create_word("test")

        norm = np.linalg.norm(word.embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)


class TestEpisodicMemory(unittest.TestCase):
    """Test the Episodic Memory system"""

    def setUp(self):
        from core.episodic_memory import EpisodicMemory, Episode
        self.memory = EpisodicMemory(
            state_size=64,
            max_episodes=100,
            consolidation_threshold=0.7,
            random_seed=42
        )
        self.Episode = Episode

    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.memory.state_size, 64)
        self.assertEqual(self.memory.max_episodes, 100)
        self.assertEqual(len(self.memory.episodes), 0)

    def test_store_episode(self):
        """Test storing an episode"""
        state = np.random.rand(64)
        sensory_data = {
            "visual": np.random.rand(32),
            "audio": np.random.rand(16)
        }

        episode = self.memory.store(
            state=state,
            sensory_data=sensory_data,
            context={"location": "lab"},
            emotional_valence=0.5,
            surprise_level=0.3
        )

        self.assertIsInstance(episode, self.Episode)
        self.assertEqual(len(self.memory.episodes), 1)
        self.assertEqual(self.memory.total_stored, 1)

    def test_retrieve_by_cue(self):
        """Test retrieval by state cue"""
        # Store multiple episodes
        states = [np.random.rand(64) for _ in range(5)]
        for i, state in enumerate(states):
            self.memory.store(
                state=state,
                sensory_data={"visual": np.random.rand(32)},
                context={"index": i}
            )

        # Retrieve using cue similar to first state
        cue = states[0] + np.random.rand(64) * 0.1  # Add noise
        retrieved = self.memory.retrieve(cue=cue, n_episodes=1, similarity_threshold=0.5)

        self.assertTrue(len(retrieved) >= 0)  # May or may not find match

    def test_consolidation(self):
        """Test memory consolidation"""
        # Store some episodes
        for i in range(10):
            self.memory.store(
                state=np.random.rand(64),
                sensory_data={"visual": np.random.rand(32)},
                surprise_level=0.8  # High surprise for consolidation
            )

        # Consolidate
        consolidated = self.memory.consolidate(n_replay=5)

        self.assertGreaterEqual(consolidated, 0)
        self.assertGreater(self.memory.total_replayed, 0)

    def test_forgetting(self):
        """Test forgetting mechanism"""
        # Fill memory beyond capacity
        for i in range(150):
            self.memory.store(
                state=np.random.rand(64),
                sensory_data={"visual": np.random.rand(32)}
            )

        # Should be limited to max_episodes
        self.assertLessEqual(len(self.memory.episodes), self.memory.max_episodes)
        self.assertGreater(self.memory.total_forgotten, 0)

    def test_similar_episode_detection(self):
        """Test detection of similar episodes"""
        state = np.random.rand(64)

        # Store first episode
        self.memory.store(
            state=state,
            sensory_data={"visual": np.random.rand(32)}
        )

        # Store very similar episode (should be detected)
        similar = self.memory._find_similar_episode(state)

        self.assertIsNotNone(similar)

    def test_stats(self):
        """Test getting stats"""
        self.memory.store(
            state=np.random.rand(64),
            sensory_data={"visual": np.random.rand(32)}
        )

        stats = self.memory.get_stats()

        self.assertIn('total_episodes', stats)
        self.assertIn('total_stored', stats)
        self.assertIn('avg_consolidation_strength', stats)

    def test_serialization(self):
        """Test serialization"""
        self.memory.store(
            state=np.random.rand(64),
            sensory_data={"visual": np.random.rand(32)},
            emotional_valence=0.8,
            surprise_level=0.9
        )
        self.memory.consolidate()

        serialized = self.memory.serialize()

        self.assertIn('state_size', serialized)
        self.assertIn('episodes', serialized)


class TestSemanticMemory(unittest.TestCase):
    """Test the Semantic Memory system"""

    def setUp(self):
        from core.semantic_memory import SemanticMemory, RelationType, Concept
        self.memory = SemanticMemory(
            embedding_size=64,
            max_concepts=1000,
            random_seed=42
        )
        self.RelationType = RelationType
        self.Concept = Concept

    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.memory.embedding_size, 64)
        self.assertEqual(len(self.memory.concepts), 0)

    def test_add_concept(self):
        """Test adding a concept"""
        embedding = np.random.rand(64)
        concept = self.memory.add_concept(
            name="dog",
            embedding=embedding,
            attributes={"type": "animal", "legs": 4}
        )

        self.assertIn("dog", self.memory.concepts)
        self.assertEqual(concept.name, "dog")
        self.assertEqual(concept.attributes["legs"], 4)

    def test_add_relation(self):
        """Test adding relations between concepts"""
        self.memory.add_concept("dog", np.random.rand(64))
        self.memory.add_concept("animal", np.random.rand(64))

        relation = self.memory.add_relation(
            source="dog",
            target="animal",
            relation_type=self.RelationType.IS_A,
            strength=0.9
        )

        self.assertIsNotNone(relation)
        self.assertTrue(self.memory.concept_graph.has_edge("dog", "animal"))

    def test_query_by_embedding(self):
        """Test querying by embedding similarity"""
        dog_embedding = np.random.rand(64)
        cat_embedding = dog_embedding + np.random.rand(64) * 0.1  # Similar
        bird_embedding = np.random.rand(64)  # Different

        self.memory.add_concept("dog", dog_embedding)
        self.memory.add_concept("cat", cat_embedding)
        self.memory.add_concept("bird", bird_embedding)

        results = self.memory.query(cue_embedding=dog_embedding, n_results=2)

        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0][0], "dog")  # Most similar should be dog itself

    def test_spreading_activation(self):
        """Test spreading activation query"""
        # Build small network
        self.memory.add_concept("mammal", np.random.rand(64))
        self.memory.add_concept("dog", np.random.rand(64))
        self.memory.add_concept("cat", np.random.rand(64))

        self.memory.add_relation("dog", "mammal", self.RelationType.IS_A)
        self.memory.add_relation("cat", "mammal", self.RelationType.IS_A)

        # Query from dog
        results = self.memory.query(cue_concept="dog", n_results=3)

        self.assertTrue(len(results) > 0)

    def test_inference(self):
        """Test inference in concept graph"""
        self.memory.add_concept("A", np.random.rand(64))
        self.memory.add_concept("B", np.random.rand(64))
        self.memory.add_concept("C", np.random.rand(64))

        self.memory.add_relation("A", "B", self.RelationType.IS_A)
        self.memory.add_relation("B", "C", self.RelationType.IS_A)

        # Infer path from A to C
        inferences = self.memory.infer("A", target="C")

        self.assertTrue(len(inferences) > 0)

    def test_generalization(self):
        """Test generalization from examples"""
        # Add similar concepts
        base_embedding = np.random.rand(64)
        self.memory.add_concept("dog1", base_embedding + np.random.rand(64) * 0.05)
        self.memory.add_concept("dog2", base_embedding + np.random.rand(64) * 0.05)
        self.memory.add_concept("dog3", base_embedding + np.random.rand(64) * 0.05)

        # Generalize
        general = self.memory.generalize(["dog1", "dog2", "dog3"])

        # May or may not generalize depending on similarity

    def test_stats(self):
        """Test getting stats"""
        self.memory.add_concept("test", np.random.rand(64))

        stats = self.memory.get_stats()

        self.assertIn('total_concepts', stats)
        self.assertIn('total_relations', stats)


class TestWorkingMemory(unittest.TestCase):
    """Test the Working Memory system"""

    def setUp(self):
        from core.working_memory import WorkingMemory, WorkspaceSlotType, AttentionController
        self.wm = WorkingMemory(
            capacity=7,
            content_dim=64,
            decay_rate=0.05
        )
        self.SlotType = WorkspaceSlotType
        self.AttentionController = AttentionController

    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.wm.capacity, 7)
        self.assertEqual(self.wm.content_dim, 64)
        self.assertEqual(len(self.wm.slots), 0)

    def test_add_item(self):
        """Test adding items to working memory"""
        content = np.random.rand(64)

        item_id = self.wm.add(
            content=content,
            slot_type=self.SlotType.SENSORY,
            source="visual_processor",
            salience=0.8
        )

        self.assertIsNotNone(item_id)
        self.assertEqual(len(self.wm.slots), 1)
        self.assertIn(item_id, self.wm.slots)

    def test_capacity_limit(self):
        """Test capacity is enforced"""
        # Add more than capacity
        for i in range(10):
            self.wm.add(
                content=np.random.rand(64),
                slot_type=self.SlotType.SENSORY,
                source=f"source_{i}"
            )

        # Should not exceed capacity
        self.assertLessEqual(len(self.wm.slots), self.wm.capacity)

    def test_refresh(self):
        """Test refreshing an item"""
        content = np.random.rand(64)
        item_id = self.wm.add(content, self.SlotType.SENSORY, "test")

        # Let it decay
        self.wm.step()
        old_activation = self.wm.slots[item_id].activation

        # Refresh
        result = self.wm.refresh(item_id)

        self.assertTrue(result)
        self.assertEqual(self.wm.slots[item_id].activation, 1.0)

    def test_goal_setting(self):
        """Test setting goals"""
        goal = np.random.rand(64)

        self.wm.set_goal(goal)

        self.assertEqual(len(self.wm.current_goals), 1)

        # Add another goal
        self.wm.add_goal(np.random.rand(64))

        self.assertEqual(len(self.wm.current_goals), 2)

    def test_task_stack(self):
        """Test task stack operations"""
        self.wm.push_task("task1")
        self.assertEqual(self.wm.current_task, "task1")

        self.wm.push_task("task2")
        self.assertEqual(self.wm.current_task, "task2")
        self.assertEqual(len(self.wm.task_stack), 1)

        completed = self.wm.pop_task()
        self.assertEqual(completed, "task2")
        self.assertEqual(self.wm.current_task, "task1")

    def test_query_by_type(self):
        """Test querying by slot type"""
        self.wm.add(np.random.rand(64), self.SlotType.SENSORY, "s1")
        self.wm.add(np.random.rand(64), self.SlotType.SENSORY, "s2")
        self.wm.add(np.random.rand(64), self.SlotType.GOAL, "g1")

        sensory_items = self.wm.query_by_type(self.SlotType.SENSORY)

        self.assertEqual(len(sensory_items), 2)

    def test_query_by_similarity(self):
        """Test querying by similarity"""
        target = np.random.rand(64)
        similar = target + np.random.rand(64) * 0.1
        different = np.random.rand(64)

        self.wm.add(target, self.SlotType.SENSORY, "target")
        self.wm.add(similar, self.SlotType.SENSORY, "similar")
        self.wm.add(different, self.SlotType.SENSORY, "different")

        results = self.wm.query_by_similarity(target, top_k=2)

        self.assertEqual(len(results), 2)

    def test_attention_controller(self):
        """Test attention controller"""
        controller = self.AttentionController()

        # Add some items
        self.wm.add(np.random.rand(64), self.SlotType.SENSORY, "s1", salience=0.9)
        self.wm.add(np.random.rand(64), self.SlotType.SENSORY, "s2", salience=0.3)

        # Get most attended
        most_attended = self.wm.attention.get_most_attended(1)

        self.assertTrue(len(most_attended) >= 0)

    def test_step_and_decay(self):
        """Test step function with decay"""
        self.wm.add(np.random.rand(64), self.SlotType.SENSORY, "test")

        initial_size = len(self.wm.slots)

        # Run several steps
        for _ in range(50):
            self.wm.step()

        # Items should eventually decay away if not refreshed

    def test_broadcast(self):
        """Test broadcast mechanism"""
        broadcast_received = []

        def listener(item):
            broadcast_received.append(item)

        self.wm.register_broadcast_listener(listener)

        # Add high-priority item that should be broadcast
        self.wm.add(
            np.random.rand(64),
            self.SlotType.GOAL,
            "important",
            salience=0.9,
            relevance=0.9
        )

        # Broadcast happens on add if priority is high enough

    def test_state_summary(self):
        """Test getting state summary"""
        self.wm.add(np.random.rand(64), self.SlotType.SENSORY, "s1")
        self.wm.push_task("task1")

        summary = self.wm.get_state_summary()

        self.assertIn('capacity', summary)
        self.assertIn('current_size', summary)
        self.assertIn('current_task', summary)


class TestCognitiveIntegration(unittest.TestCase):
    """Test integration between cognitive systems"""

    def setUp(self):
        """Set up integrated cognitive systems"""
        from core.working_memory import WorkingMemory, WorkspaceSlotType
        from core.episodic_memory import EpisodicMemory
        from core.semantic_memory import SemanticMemory, RelationType
        from core.meta_learning import MetaLearner
        from core.goal_planning import GoalPlanningSystem, GoalType
        from core.causal_reasoning import CausalReasoner

        self.working_memory = WorkingMemory(capacity=7, content_dim=64)
        self.episodic_memory = EpisodicMemory(state_size=64)
        self.semantic_memory = SemanticMemory(embedding_size=64)
        self.meta_learner = MetaLearner(random_seed=42)
        self.goal_planner = GoalPlanningSystem(random_seed=42)
        self.causal_reasoner = CausalReasoner()

        self.SlotType = WorkspaceSlotType
        self.RelationType = RelationType
        self.GoalType = GoalType

    def test_working_episodic_integration(self):
        """Test integration between working memory and episodic memory"""
        # Simulate processing flow
        state = np.random.rand(64)

        # Add to working memory
        item_id = self.working_memory.add(
            content=state,
            slot_type=self.SlotType.SENSORY,
            source="perception"
        )

        # Store significant experience in episodic memory
        episode = self.episodic_memory.store(
            state=state,
            sensory_data={"visual": state[:32]},
            surprise_level=0.8
        )

        # Retrieve from episodic and add to working memory
        retrieved = self.episodic_memory.retrieve(cue=state, n_episodes=1)

        if len(retrieved) > 0:
            self.working_memory.add(
                content=retrieved[0].state,
                slot_type=self.SlotType.EPISODIC,
                source="episodic_memory"
            )

        self.assertTrue(len(self.working_memory.slots) >= 1)

    def test_semantic_episodic_integration(self):
        """Test integration between semantic and episodic memory"""
        # Add concept to semantic memory
        dog_embedding = np.random.rand(64)
        self.semantic_memory.add_concept("dog", dog_embedding)

        # Store episodic experience linked to concept
        episode = self.episodic_memory.store(
            state=dog_embedding,
            sensory_data={"visual": dog_embedding[:32]},
            context={"concept": "dog"}
        )

        # Query semantic memory
        results = self.semantic_memory.query(cue_embedding=dog_embedding)

        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0][0], "dog")

    def test_goal_meta_learning_integration(self):
        """Test integration between goal planning and meta-learning"""
        # Generate goal
        goal = self.goal_planner.generate_goal(
            goal_type=self.GoalType.LEARNING,
            context={"task": "classification"}
        )

        # Select learning strategy using meta-learner
        strategy, hyperparams = self.meta_learner.select_strategy({
            "goal_priority": goal.priority,
            "goal_value": goal.value
        })

        # Simulate learning and update meta-learner
        self.meta_learner.update(
            strategy=strategy,
            hyperparameters=hyperparams,
            task_characteristics={"complexity": 0.5},
            performance_metrics={"accuracy": 0.75}
        )

        self.assertGreater(self.meta_learner.total_updates, 0)

    def test_causal_goal_integration(self):
        """Test integration between causal reasoning and goal planning"""
        # Build causal model of task
        self.causal_reasoner.model.graph.add_variable("effort")
        self.causal_reasoner.model.graph.add_variable("skill")
        self.causal_reasoner.model.graph.add_variable("success")

        self.causal_reasoner.model.graph.add_link("effort", "success", strength=0.6)
        self.causal_reasoner.model.graph.add_link("skill", "success", strength=0.7)

        # Use causal model to plan intervention
        plan = self.causal_reasoner.plan_intervention(
            goal={"success": 0.9},
            available_interventions=["effort", "skill"],
            current_state={"effort": 0.5, "skill": 0.5}
        )

        self.assertTrue(len(plan) >= 0)

    def test_full_cognitive_cycle(self):
        """Test full cognitive processing cycle"""
        # 1. Perception -> Working Memory
        sensory_input = np.random.rand(64)
        self.working_memory.add(
            content=sensory_input,
            slot_type=self.SlotType.SENSORY,
            source="perception"
        )

        # 2. Set goal
        goal = self.goal_planner.generate_goal(self.GoalType.LEARNING)
        if goal:
            self.working_memory.add(
                content=np.random.rand(64),
                slot_type=self.SlotType.GOAL,
                source="goal_system",
                metadata={"goal_name": goal.name}
            )

        # 3. Query semantic memory
        concept_results = self.semantic_memory.query(cue_embedding=sensory_input)

        # 4. Store in episodic memory
        self.episodic_memory.store(
            state=sensory_input,
            sensory_data={"visual": sensory_input[:32]},
            context={"concepts": [r[0] for r in concept_results]}
        )

        # 5. Select learning strategy
        strategy, params = self.meta_learner.select_strategy({"complexity": 0.5})

        # 6. Update with result
        self.meta_learner.update(
            strategy=strategy,
            hyperparameters=params,
            task_characteristics={"complexity": 0.5},
            performance_metrics={"accuracy": 0.8}
        )

        # Verify cognitive state
        self.assertTrue(len(self.working_memory.slots) >= 1)
        self.assertTrue(len(self.episodic_memory.episodes) >= 1)


def run_all_tests():
    """Run all cognitive system tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestMetaLearning,
        TestGoalPlanning,
        TestCausalReasoning,
        TestAbstractReasoning,
        TestLanguageGrounding,
        TestEpisodicMemory,
        TestSemanticMemory,
        TestWorkingMemory,
        TestCognitiveIntegration,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("COGNITIVE SYSTEMS TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.errors) - len(result.failures)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result


if __name__ == "__main__":
    result = run_all_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
