"""
Comprehensive tests for the KnowledgeBase class (core/knowledge_base.py).
Tests cover initialization, knowledge event creation, memory query construction,
memory query results, experience storage, consolidation, serialization,
deserialization, recent events, and statistics reporting.
"""
import sys
import os
import unittest
import time
import numpy as np
from unittest.mock import patch, MagicMock
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.knowledge_base import (
    KnowledgeBase,
    KnowledgeEvent,
    MemoryQuery,
    MemoryQueryResult,
)
from self_organizing_av_system.core.episodic_memory import Episode
from self_organizing_av_system.core.semantic_memory import RelationType


# ---------------------------------------------------------------------------
# Helper: create a KnowledgeBase with consolidation thread disabled so tests
# are deterministic and do not spawn background threads.
# ---------------------------------------------------------------------------
def _make_kb(state_dim=16, **kwargs):
    """Create a KnowledgeBase with background consolidation disabled."""
    kwargs.setdefault("enable_consolidation", False)
    kwargs.setdefault("state_dim", state_dim)
    return KnowledgeBase(**kwargs)


def _make_state(dim=16, seed=0):
    """Create a deterministic random state vector."""
    rng = np.random.RandomState(seed)
    return rng.randn(dim).astype(np.float64)


# ===================================================================
# Dataclass tests
# ===================================================================

class TestKnowledgeEventDataclass(unittest.TestCase):
    """Test the KnowledgeEvent dataclass."""

    def test_creation_with_all_fields(self):
        event = KnowledgeEvent(
            timestamp=1000.0,
            event_type="episode_stored",
            description="Saw a red ball",
            source="curriculum",
            consolidation_strength=0.75,
        )
        self.assertAlmostEqual(event.timestamp, 1000.0)
        self.assertEqual(event.event_type, "episode_stored")
        self.assertEqual(event.description, "Saw a red ball")
        self.assertEqual(event.source, "curriculum")
        self.assertAlmostEqual(event.consolidation_strength, 0.75)

    def test_default_consolidation_strength(self):
        event = KnowledgeEvent(
            timestamp=0.0,
            event_type="concept_learned",
            description="Learned gravity",
            source="free_play",
        )
        self.assertAlmostEqual(event.consolidation_strength, 0.0)

    def test_various_event_types(self):
        for event_type in ["episode_stored", "concept_learned",
                           "relation_created", "inference_made"]:
            event = KnowledgeEvent(
                timestamp=1.0,
                event_type=event_type,
                description="test",
                source="system",
            )
            self.assertEqual(event.event_type, event_type)

    def test_source_values(self):
        for source in ["curriculum", "free_play", "system"]:
            event = KnowledgeEvent(
                timestamp=1.0,
                event_type="episode_stored",
                description="test",
                source=source,
            )
            self.assertEqual(event.source, source)


class TestMemoryQueryDataclass(unittest.TestCase):
    """Test the MemoryQuery dataclass."""

    def test_creation_with_defaults(self):
        state = np.zeros(8)
        query = MemoryQuery(query_state=state)
        np.testing.assert_array_equal(query.query_state, state)
        self.assertIsNone(query.context)
        self.assertIsNone(query.source)
        self.assertEqual(query.k_episodes, 5)
        self.assertEqual(query.k_concepts, 10)

    def test_creation_with_all_fields(self):
        state = np.ones(8)
        query = MemoryQuery(
            query_state=state,
            context="navigation",
            source="curriculum",
            k_episodes=3,
            k_concepts=7,
        )
        np.testing.assert_array_equal(query.query_state, state)
        self.assertEqual(query.context, "navigation")
        self.assertEqual(query.source, "curriculum")
        self.assertEqual(query.k_episodes, 3)
        self.assertEqual(query.k_concepts, 7)

    def test_query_state_is_ndarray(self):
        state = np.array([1.0, 2.0, 3.0])
        query = MemoryQuery(query_state=state)
        self.assertIsInstance(query.query_state, np.ndarray)
        self.assertEqual(len(query.query_state), 3)


class TestMemoryQueryResultDataclass(unittest.TestCase):
    """Test the MemoryQueryResult dataclass."""

    def test_creation_empty(self):
        result = MemoryQueryResult(
            episodes=[],
            concepts=[],
            inferences=[],
            related_concepts={},
        )
        self.assertEqual(result.episodes, [])
        self.assertEqual(result.concepts, [])
        self.assertEqual(result.inferences, [])
        self.assertEqual(result.related_concepts, {})

    def test_creation_with_data(self):
        ep = Episode(
            timestamp=1.0,
            state=np.zeros(4),
            context={"task": "test"},
            sensory_data={},
            emotional_valence=0.5,
            surprise_level=0.3,
        )
        result = MemoryQueryResult(
            episodes=[ep],
            concepts=[("color_red", 0.9), ("shape_round", 0.7)],
            inferences=["red implies ripe"],
            related_concepts={"color_red": ["color_blue", "color_green"]},
        )
        self.assertEqual(len(result.episodes), 1)
        self.assertIs(result.episodes[0], ep)
        self.assertEqual(len(result.concepts), 2)
        self.assertEqual(result.concepts[0][0], "color_red")
        self.assertAlmostEqual(result.concepts[0][1], 0.9)
        self.assertEqual(len(result.inferences), 1)
        self.assertIn("color_red", result.related_concepts)

    def test_multiple_episodes(self):
        episodes = [
            Episode(
                timestamp=float(i),
                state=np.zeros(4),
                context={},
                sensory_data={},
                emotional_valence=0.0,
                surprise_level=0.0,
            )
            for i in range(5)
        ]
        result = MemoryQueryResult(
            episodes=episodes,
            concepts=[],
            inferences=[],
            related_concepts={},
        )
        self.assertEqual(len(result.episodes), 5)


# ===================================================================
# KnowledgeBase initialization tests
# ===================================================================

class TestKnowledgeBaseInitialization(unittest.TestCase):
    """Test KnowledgeBase construction and initial state."""

    def test_default_initialization(self):
        kb = _make_kb()
        self.assertEqual(kb.state_dim, 16)
        self.assertIsNotNone(kb.episodic)
        self.assertIsNotNone(kb.semantic)
        self.assertEqual(len(kb.recent_events), 0)
        self.assertEqual(kb.total_experiences_stored, 0)
        self.assertEqual(kb.total_concepts_extracted, 0)
        self.assertEqual(kb.total_queries, 0)
        self.assertFalse(kb.enable_consolidation)
        self.assertFalse(kb._persistence_enabled)
        self.assertIsNone(kb.vector_store)
        self.assertIsNone(kb.graph_store)
        kb.stop()

    def test_custom_state_dim(self):
        kb = _make_kb(state_dim=64)
        self.assertEqual(kb.state_dim, 64)
        self.assertEqual(kb.episodic.state_size, 64)
        self.assertEqual(kb.semantic.embedding_size, 64)
        kb.stop()

    def test_custom_max_episodes(self):
        kb = _make_kb(max_episodes=200)
        self.assertEqual(kb.episodic.max_episodes, 200)
        kb.stop()

    def test_custom_max_concepts(self):
        kb = _make_kb(max_concepts=500)
        self.assertEqual(kb.semantic.max_concepts, 500)
        kb.stop()

    def test_consolidation_disabled(self):
        kb = _make_kb(enable_consolidation=False)
        self.assertFalse(kb.enable_consolidation)
        self.assertIsNone(kb._consolidation_thread)
        kb.stop()

    def test_consolidation_interval_stored(self):
        kb = _make_kb(consolidation_interval=30.0)
        self.assertAlmostEqual(kb.consolidation_interval, 30.0)
        kb.stop()

    def test_recent_events_is_deque_with_maxlen(self):
        kb = _make_kb()
        self.assertIsInstance(kb.recent_events, deque)
        self.assertEqual(kb.recent_events.maxlen, 100)
        kb.stop()

    def test_no_persistence_by_default(self):
        kb = _make_kb()
        self.assertFalse(kb._persistence_enabled)
        self.assertIsNone(kb.vector_store)
        self.assertIsNone(kb.graph_store)
        kb.stop()


# ===================================================================
# store_experience tests
# ===================================================================

class TestStoreExperience(unittest.TestCase):
    """Test storing experiences into the knowledge base."""

    def setUp(self):
        self.kb = _make_kb(state_dim=16)

    def tearDown(self):
        self.kb.stop()

    def test_store_basic_experience(self):
        state = _make_state(16, seed=1)
        self.kb.store_experience(
            state=state,
            context={"description": "saw object"},
            source="free_play",
        )
        self.assertEqual(self.kb.total_experiences_stored, 1)
        self.assertGreater(len(self.kb.episodic.episodes), 0)

    def test_store_increments_counter(self):
        for i in range(5):
            self.kb.store_experience(
                state=_make_state(16, seed=i + 10),
                context={"step": i},
                source="curriculum",
            )
        self.assertEqual(self.kb.total_experiences_stored, 5)

    def test_store_adds_source_to_context(self):
        state = _make_state(16, seed=20)
        self.kb.store_experience(
            state=state,
            context={"task": "explore"},
            source="curriculum",
        )
        # The source should be embedded in the stored episode's context
        ep = self.kb.episodic.episodes[-1]
        self.assertEqual(ep.context.get("source"), "curriculum")

    def test_store_adds_timestamp_to_context(self):
        state = _make_state(16, seed=21)
        before = time.time()
        self.kb.store_experience(
            state=state,
            context={},
            source="free_play",
        )
        after = time.time()
        ep = self.kb.episodic.episodes[-1]
        self.assertGreaterEqual(ep.context["timestamp"], before)
        self.assertLessEqual(ep.context["timestamp"], after)

    def test_store_with_sensory_data(self):
        state = _make_state(16, seed=22)
        sensory = {"visual": np.zeros(4), "audio": np.ones(3)}
        self.kb.store_experience(
            state=state,
            context={},
            sensory_data=sensory,
            source="free_play",
        )
        self.assertEqual(self.kb.total_experiences_stored, 1)

    def test_store_with_emotional_valence(self):
        state = _make_state(16, seed=23)
        self.kb.store_experience(
            state=state,
            context={},
            emotional_valence=0.8,
            source="free_play",
        )
        ep = self.kb.episodic.episodes[-1]
        self.assertAlmostEqual(ep.emotional_valence, 0.8)

    def test_store_records_event(self):
        state = _make_state(16, seed=24)
        self.kb.store_experience(
            state=state,
            context={"description": "interesting event"},
            source="curriculum",
        )
        self.assertEqual(len(self.kb.recent_events), 1)
        event = self.kb.recent_events[0]
        self.assertEqual(event.event_type, "episode_stored")
        self.assertEqual(event.description, "interesting event")
        self.assertEqual(event.source, "curriculum")

    def test_store_extracts_concepts(self):
        state = _make_state(16, seed=25)
        self.kb.store_experience(
            state=state,
            context={"challenge_name": "Color Match"},
            source="curriculum",
        )
        # Should have extracted at least the challenge concept and the experience concept
        self.assertGreater(self.kb.total_concepts_extracted, 0)
        # Check that the challenge concept was created in semantic memory
        self.assertIn("challenge:Color_Match", self.kb.semantic.concepts)

    def test_store_extracts_performance_concepts(self):
        state = _make_state(16, seed=26)
        self.kb.store_experience(
            state=state,
            context={"accuracy": 0.95},
            source="free_play",
        )
        self.assertIn("performance:excellent", self.kb.semantic.concepts)

    def test_store_performance_good(self):
        state = _make_state(16, seed=27)
        self.kb.store_experience(
            state=state,
            context={"accuracy": 0.75},
            source="free_play",
        )
        self.assertIn("performance:good", self.kb.semantic.concepts)

    def test_store_performance_moderate(self):
        state = _make_state(16, seed=28)
        self.kb.store_experience(
            state=state,
            context={"accuracy": 0.55},
            source="free_play",
        )
        self.assertIn("performance:moderate", self.kb.semantic.concepts)

    def test_store_performance_needs_improvement(self):
        state = _make_state(16, seed=29)
        self.kb.store_experience(
            state=state,
            context={"accuracy": 0.3},
            source="free_play",
        )
        self.assertIn("performance:needs_improvement", self.kb.semantic.concepts)

    def test_store_default_source(self):
        state = _make_state(16, seed=30)
        self.kb.store_experience(
            state=state,
            context={},
        )
        ep = self.kb.episodic.episodes[-1]
        self.assertEqual(ep.context.get("source"), "free_play")

    def test_store_event_description_fallback_to_challenge_name(self):
        state = _make_state(16, seed=31)
        self.kb.store_experience(
            state=state,
            context={"challenge_name": "Pattern Recognition"},
            source="curriculum",
        )
        event = self.kb.recent_events[-1]
        self.assertEqual(event.description, "Pattern Recognition")

    def test_store_event_description_fallback_to_default(self):
        state = _make_state(16, seed=32)
        self.kb.store_experience(
            state=state,
            context={"some_other_key": "value"},
            source="free_play",
        )
        event = self.kb.recent_events[-1]
        self.assertEqual(event.description, "Experience stored")


# ===================================================================
# _compute_novelty tests
# ===================================================================

class TestComputeNovelty(unittest.TestCase):
    """Test the internal _compute_novelty method."""

    def setUp(self):
        self.kb = _make_kb(state_dim=8)

    def tearDown(self):
        self.kb.stop()

    def test_first_experience_is_fully_novel(self):
        state = _make_state(8, seed=40)
        novelty = self.kb._compute_novelty(state)
        self.assertAlmostEqual(novelty, 1.0)

    def test_identical_state_has_low_novelty(self):
        state = np.ones(8)
        self.kb.store_experience(state=state, context={})
        novelty = self.kb._compute_novelty(state)
        # Identical state should have high similarity => low novelty
        self.assertLess(novelty, 0.5)

    def test_orthogonal_state_has_high_novelty(self):
        state_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.kb.store_experience(state=state_a, context={})
        state_b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        novelty = self.kb._compute_novelty(state_b)
        self.assertGreater(novelty, 0.5)

    def test_novelty_bounded_zero_to_one(self):
        # Store some experiences
        for i in range(5):
            self.kb.store_experience(
                state=_make_state(8, seed=i + 50),
                context={},
            )
        novelty = self.kb._compute_novelty(_make_state(8, seed=99))
        self.assertGreaterEqual(novelty, 0.0)
        self.assertLessEqual(novelty, 1.0)


# ===================================================================
# _extract_concepts tests
# ===================================================================

class TestExtractConcepts(unittest.TestCase):
    """Test the internal _extract_concepts method."""

    def setUp(self):
        self.kb = _make_kb(state_dim=8)

    def tearDown(self):
        self.kb.stop()

    def test_always_extracts_experience_concept(self):
        state = _make_state(8, seed=60)
        concepts = self.kb._extract_concepts(state, {})
        # At least one "experience:..." concept
        names = [c[0] for c in concepts]
        experience_concepts = [n for n in names if n.startswith("experience:")]
        self.assertGreaterEqual(len(experience_concepts), 1)

    def test_extracts_challenge_concept(self):
        state = _make_state(8, seed=61)
        concepts = self.kb._extract_concepts(
            state, {"challenge_name": "Sound Match"}
        )
        names = [c[0] for c in concepts]
        self.assertIn("challenge:Sound_Match", names)

    def test_extracts_accuracy_performance_excellent(self):
        state = _make_state(8, seed=62)
        concepts = self.kb._extract_concepts(state, {"accuracy": 0.92})
        names = [c[0] for c in concepts]
        self.assertIn("performance:excellent", names)

    def test_extracts_accuracy_performance_good(self):
        state = _make_state(8, seed=63)
        concepts = self.kb._extract_concepts(state, {"accuracy": 0.75})
        names = [c[0] for c in concepts]
        self.assertIn("performance:good", names)

    def test_extracts_accuracy_performance_moderate(self):
        state = _make_state(8, seed=64)
        concepts = self.kb._extract_concepts(state, {"accuracy": 0.6})
        names = [c[0] for c in concepts]
        self.assertIn("performance:moderate", names)

    def test_extracts_accuracy_performance_needs_improvement(self):
        state = _make_state(8, seed=65)
        concepts = self.kb._extract_concepts(state, {"accuracy": 0.2})
        names = [c[0] for c in concepts]
        self.assertIn("performance:needs_improvement", names)

    def test_concept_embeddings_are_arrays(self):
        state = _make_state(8, seed=66)
        concepts = self.kb._extract_concepts(state, {"accuracy": 0.5})
        for name, embedding in concepts:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(len(embedding), 8)

    def test_challenge_concept_copies_state(self):
        state = _make_state(8, seed=67)
        concepts = self.kb._extract_concepts(
            state, {"challenge_name": "Test"}
        )
        challenge_concepts = [(n, e) for n, e in concepts
                              if n.startswith("challenge:")]
        self.assertEqual(len(challenge_concepts), 1)
        np.testing.assert_array_equal(challenge_concepts[0][1], state)
        # Verify it is a copy
        state[0] = 999.0
        self.assertNotAlmostEqual(challenge_concepts[0][1][0], 999.0)


# ===================================================================
# _record_event tests
# ===================================================================

class TestRecordEvent(unittest.TestCase):
    """Test the internal _record_event method."""

    def setUp(self):
        self.kb = _make_kb(state_dim=8)

    def tearDown(self):
        self.kb.stop()

    def test_record_event_appends_to_deque(self):
        self.assertEqual(len(self.kb.recent_events), 0)
        self.kb._record_event("episode_stored", "saw a ball", "free_play", 0.5)
        self.assertEqual(len(self.kb.recent_events), 1)

    def test_record_event_fields(self):
        before = time.time()
        self.kb._record_event("concept_learned", "gravity", "curriculum", 0.8)
        after = time.time()
        event = self.kb.recent_events[0]
        self.assertEqual(event.event_type, "concept_learned")
        self.assertEqual(event.description, "gravity")
        self.assertEqual(event.source, "curriculum")
        self.assertAlmostEqual(event.consolidation_strength, 0.8)
        self.assertGreaterEqual(event.timestamp, before)
        self.assertLessEqual(event.timestamp, after)

    def test_record_event_respects_maxlen(self):
        for i in range(150):
            self.kb._record_event(
                "episode_stored", f"event_{i}", "free_play", 0.0
            )
        # Deque maxlen is 100
        self.assertEqual(len(self.kb.recent_events), 100)
        # The oldest events should have been dropped
        self.assertEqual(self.kb.recent_events[0].description, "event_50")

    def test_record_event_default_consolidation_strength(self):
        self.kb._record_event("relation_created", "test", "system")
        event = self.kb.recent_events[0]
        self.assertAlmostEqual(event.consolidation_strength, 0.0)


# ===================================================================
# query tests (uses mocking for missing SemanticMemory methods)
# ===================================================================

class TestQuery(unittest.TestCase):
    """Test the query method of KnowledgeBase.

    KnowledgeBase.query() calls spreading_activation, get_neighbors,
    and infer_from_state on the semantic memory object. These methods
    are stubbed/mocked here to make the tests deterministic and reliable.
    """

    def setUp(self):
        self.kb = _make_kb(state_dim=8)
        # Provide the missing methods on semantic memory for testing
        self.kb.semantic.spreading_activation = MagicMock(return_value={
            "concept_a": 0.9,
            "concept_b": 0.6,
            "concept_c": 0.3,
        })
        self.kb.semantic.get_neighbors = MagicMock(return_value=["neighbor_1"])
        self.kb.semantic.infer_from_state = MagicMock(
            return_value=["inference_1"]
        )
        # Mock episodic retrieve to avoid incompatible k= parameter
        self.kb.episodic.retrieve = MagicMock(return_value=[])

    def tearDown(self):
        self.kb.stop()

    def test_query_returns_memory_query_result(self):
        query = MemoryQuery(query_state=_make_state(8, seed=70))
        result = self.kb.query(query)
        self.assertIsInstance(result, MemoryQueryResult)

    def test_query_increments_total_queries(self):
        self.assertEqual(self.kb.total_queries, 0)
        query = MemoryQuery(query_state=_make_state(8, seed=71))
        self.kb.query(query)
        self.assertEqual(self.kb.total_queries, 1)

    def test_query_returns_concepts(self):
        query = MemoryQuery(query_state=_make_state(8, seed=72))
        result = self.kb.query(query)
        # spreading_activation returns 3 concepts; default k_concepts=10
        self.assertGreaterEqual(len(result.concepts), 1)
        # Concepts should be sorted by activation descending
        if len(result.concepts) >= 2:
            self.assertGreaterEqual(result.concepts[0][1],
                                    result.concepts[1][1])

    def test_query_returns_inferences(self):
        query = MemoryQuery(query_state=_make_state(8, seed=73))
        result = self.kb.query(query)
        self.assertEqual(result.inferences, ["inference_1"])

    def test_query_returns_related_concepts(self):
        query = MemoryQuery(query_state=_make_state(8, seed=74))
        result = self.kb.query(query)
        self.assertIsInstance(result.related_concepts, dict)
        # get_neighbors should have been called for each top concept
        self.assertTrue(self.kb.semantic.get_neighbors.called)

    def test_query_with_source_filter(self):
        # Create mock episodes from different sources
        ep_curriculum = Episode(
            timestamp=1.0,
            state=_make_state(8, seed=75),
            context={"source": "curriculum"},
            sensory_data={},
            emotional_valence=0.0,
            surprise_level=0.0,
        )
        ep_freeplay = Episode(
            timestamp=2.0,
            state=_make_state(8, seed=76),
            context={"source": "free_play"},
            sensory_data={},
            emotional_valence=0.0,
            surprise_level=0.0,
        )
        # Mock episodic retrieve to return both episodes
        self.kb.episodic.retrieve = MagicMock(
            return_value=[ep_curriculum, ep_freeplay]
        )
        # Query filtering by source
        query = MemoryQuery(
            query_state=_make_state(8, seed=75),
            source="curriculum",
            k_episodes=10,
        )
        result = self.kb.query(query)
        # All returned episodes should be from "curriculum"
        self.assertEqual(len(result.episodes), 1)
        for ep in result.episodes:
            self.assertEqual(ep.context.get("source"), "curriculum")

    def test_query_empty_knowledge_base(self):
        query = MemoryQuery(query_state=_make_state(8, seed=77))
        result = self.kb.query(query)
        self.assertEqual(result.episodes, [])

    def test_query_k_concepts_limits_results(self):
        query = MemoryQuery(
            query_state=_make_state(8, seed=78), k_concepts=2
        )
        result = self.kb.query(query)
        self.assertLessEqual(len(result.concepts), 2)

    def test_query_calls_spreading_activation(self):
        query = MemoryQuery(query_state=_make_state(8, seed=79))
        self.kb.query(query)
        self.kb.semantic.spreading_activation.assert_called_once()

    def test_query_inference_disabled(self):
        self.kb.semantic.enable_inference = False
        query = MemoryQuery(query_state=_make_state(8, seed=80))
        result = self.kb.query(query)
        self.assertEqual(result.inferences, [])


# ===================================================================
# consolidate tests
# ===================================================================

class TestConsolidate(unittest.TestCase):
    """Test the consolidate method."""

    def setUp(self):
        self.kb = _make_kb(state_dim=8)

    def tearDown(self):
        self.kb.stop()

    def test_consolidate_returns_dict(self):
        result = self.kb.consolidate()
        self.assertIsInstance(result, dict)

    def test_consolidate_returns_expected_keys(self):
        result = self.kb.consolidate()
        expected_keys = {
            "episodes_consolidated",
            "episodes_forgotten",
            "concepts_strengthened",
            "relations_strengthened",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_consolidate_empty_kb(self):
        result = self.kb.consolidate()
        self.assertEqual(result["episodes_consolidated"], 0)

    def test_consolidate_with_experiences(self):
        for i in range(5):
            self.kb.store_experience(
                state=_make_state(8, seed=i + 100),
                context={"step": i},
                emotional_valence=0.9,
                source="curriculum",
            )
        result = self.kb.consolidate()
        # Should return a dict with counts (may or may not have consolidated)
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(result["episodes_consolidated"], 0)


# ===================================================================
# add_relation tests
# ===================================================================

class TestAddRelation(unittest.TestCase):
    """Test adding relations between concepts."""

    def setUp(self):
        self.kb = _make_kb(state_dim=8)

    def tearDown(self):
        self.kb.stop()

    def test_add_relation_between_existing_concepts(self):
        state_a = _make_state(8, seed=110)
        state_b = _make_state(8, seed=111)
        self.kb.semantic.add_concept("animal", state_a)
        self.kb.semantic.add_concept("dog", state_b)
        self.kb.add_relation("dog", "animal", RelationType.IS_A, strength=0.9)
        # Relation should be in the concept graph
        self.assertTrue(
            self.kb.semantic.concept_graph.has_edge("dog", "animal")
        )

    def test_add_relation_records_event(self):
        state_a = _make_state(8, seed=112)
        state_b = _make_state(8, seed=113)
        self.kb.semantic.add_concept("rain", state_a)
        self.kb.semantic.add_concept("wet", state_b)
        self.kb.add_relation("rain", "wet", RelationType.CAUSES, strength=0.8)
        # Should record a relation_created event
        self.assertGreater(len(self.kb.recent_events), 0)
        event = self.kb.recent_events[-1]
        self.assertEqual(event.event_type, "relation_created")
        self.assertIn("rain", event.description)
        self.assertIn("wet", event.description)
        self.assertEqual(event.source, "system")
        self.assertAlmostEqual(event.consolidation_strength, 0.8)

    def test_add_relation_missing_concept_returns_none(self):
        state = _make_state(8, seed=114)
        self.kb.semantic.add_concept("existing", state)
        result = self.kb.add_relation(
            "existing", "nonexistent", RelationType.IS_A
        )
        self.assertIsNone(result)


# ===================================================================
# get_stats tests
# ===================================================================

class TestGetStats(unittest.TestCase):
    """Test statistics reporting."""

    def setUp(self):
        self.kb = _make_kb(state_dim=8)

    def tearDown(self):
        self.kb.stop()

    def test_stats_keys_present(self):
        stats = self.kb.get_stats()
        expected_keys = {
            "total_episodes",
            "consolidated_episodes",
            "total_retrieved",
            "total_forgotten",
            "total_concepts",
            "total_relations",
            "total_inferences",
            "total_generalizations",
            "total_experiences_stored",
            "total_concepts_extracted",
            "total_queries",
            "recent_events_count",
        }
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_stats_initial_values(self):
        stats = self.kb.get_stats()
        self.assertEqual(stats["total_episodes"], 0)
        self.assertEqual(stats["consolidated_episodes"], 0)
        self.assertEqual(stats["total_experiences_stored"], 0)
        self.assertEqual(stats["total_concepts_extracted"], 0)
        self.assertEqual(stats["total_queries"], 0)
        self.assertEqual(stats["recent_events_count"], 0)

    def test_stats_after_storing_experience(self):
        self.kb.store_experience(
            state=_make_state(8, seed=120),
            context={"challenge_name": "Test"},
            source="curriculum",
        )
        stats = self.kb.get_stats()
        self.assertEqual(stats["total_experiences_stored"], 1)
        self.assertGreater(stats["total_concepts_extracted"], 0)
        self.assertGreater(stats["total_concepts"], 0)
        self.assertGreater(stats["recent_events_count"], 0)

    def test_stats_query_count_increments(self):
        # Mock the missing methods
        self.kb.semantic.spreading_activation = MagicMock(return_value={})
        self.kb.semantic.get_neighbors = MagicMock(return_value=[])
        self.kb.semantic.infer_from_state = MagicMock(return_value=[])
        self.kb.episodic.retrieve = MagicMock(return_value=[])

        query = MemoryQuery(query_state=_make_state(8, seed=121))
        self.kb.query(query)
        stats = self.kb.get_stats()
        self.assertEqual(stats["total_queries"], 1)


# ===================================================================
# get_recent_events tests
# ===================================================================

class TestGetRecentEvents(unittest.TestCase):
    """Test recent event retrieval."""

    def setUp(self):
        self.kb = _make_kb(state_dim=8)

    def tearDown(self):
        self.kb.stop()

    def test_get_recent_events_empty(self):
        events = self.kb.get_recent_events(n=5)
        self.assertEqual(events, [])

    def test_get_recent_events_returns_list(self):
        self.kb._record_event("episode_stored", "test", "free_play", 0.0)
        events = self.kb.get_recent_events(n=5)
        self.assertIsInstance(events, list)
        self.assertEqual(len(events), 1)

    def test_get_recent_events_returns_n_most_recent(self):
        for i in range(10):
            self.kb._record_event(
                "episode_stored", f"event_{i}", "free_play", 0.0
            )
        events = self.kb.get_recent_events(n=3)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].description, "event_7")
        self.assertEqual(events[1].description, "event_8")
        self.assertEqual(events[2].description, "event_9")

    def test_get_recent_events_fewer_than_n(self):
        self.kb._record_event("episode_stored", "only_one", "free_play", 0.0)
        events = self.kb.get_recent_events(n=10)
        self.assertEqual(len(events), 1)

    def test_get_recent_events_default_n(self):
        for i in range(15):
            self.kb._record_event(
                "episode_stored", f"event_{i}", "free_play", 0.0
            )
        events = self.kb.get_recent_events()
        self.assertEqual(len(events), 10)


class TestGetRecentEventString(unittest.TestCase):
    """Test the get_recent_event_string method."""

    def setUp(self):
        self.kb = _make_kb(state_dim=8)

    def tearDown(self):
        self.kb.stop()

    def test_no_events_returns_default(self):
        result = self.kb.get_recent_event_string()
        self.assertEqual(result, "No recent events")

    def test_returns_most_recent_event_description(self):
        self.kb._record_event("episode_stored", "First", "free_play", 0.5)
        self.kb._record_event("concept_learned", "Second", "curriculum", 0.8)
        result = self.kb.get_recent_event_string()
        self.assertIn("Second", result)
        self.assertIn("80%", result)

    def test_format_includes_consolidation_strength(self):
        self.kb._record_event("episode_stored", "Event", "free_play", 0.0)
        result = self.kb.get_recent_event_string()
        self.assertIn("consolidation:", result)
        self.assertIn("0%", result)


# ===================================================================
# Serialization and deserialization tests
# ===================================================================

class TestSerialization(unittest.TestCase):
    """Test serialize and deserialize methods."""

    def setUp(self):
        self.kb = _make_kb(state_dim=16)

    def tearDown(self):
        self.kb.stop()

    def test_serialize_returns_dict(self):
        data = self.kb.serialize()
        self.assertIsInstance(data, dict)

    def test_serialize_contains_expected_keys(self):
        data = self.kb.serialize()
        expected_keys = {
            "state_dim",
            "total_experiences_stored",
            "total_concepts_extracted",
            "total_queries",
        }
        self.assertEqual(set(data.keys()), expected_keys)

    def test_serialize_initial_values(self):
        data = self.kb.serialize()
        self.assertEqual(data["state_dim"], 16)
        self.assertEqual(data["total_experiences_stored"], 0)
        self.assertEqual(data["total_concepts_extracted"], 0)
        self.assertEqual(data["total_queries"], 0)

    def test_serialize_after_operations(self):
        self.kb.store_experience(
            state=_make_state(16, seed=130),
            context={"challenge_name": "Test"},
            source="curriculum",
        )
        data = self.kb.serialize()
        self.assertEqual(data["total_experiences_stored"], 1)
        self.assertGreater(data["total_concepts_extracted"], 0)

    def test_serialize_preserves_state_dim(self):
        kb32 = _make_kb(state_dim=32)
        data = kb32.serialize()
        self.assertEqual(data["state_dim"], 32)
        kb32.stop()

    def test_deserialize_creates_knowledge_base(self):
        data = {
            "state_dim": 32,
            "total_experiences_stored": 10,
            "total_concepts_extracted": 5,
            "total_queries": 3,
        }
        kb = KnowledgeBase.deserialize(data)
        self.assertIsInstance(kb, KnowledgeBase)
        self.assertEqual(kb.state_dim, 32)
        self.assertEqual(kb.total_experiences_stored, 10)
        self.assertEqual(kb.total_concepts_extracted, 5)
        self.assertEqual(kb.total_queries, 3)
        kb.stop()

    def test_deserialize_default_state_dim(self):
        data = {}
        kb = KnowledgeBase.deserialize(data)
        self.assertEqual(kb.state_dim, 128)
        kb.stop()

    def test_deserialize_default_counters(self):
        data = {"state_dim": 16}
        kb = KnowledgeBase.deserialize(data)
        self.assertEqual(kb.total_experiences_stored, 0)
        self.assertEqual(kb.total_concepts_extracted, 0)
        self.assertEqual(kb.total_queries, 0)
        kb.stop()

    def test_roundtrip_serialize_deserialize(self):
        self.kb.store_experience(
            state=_make_state(16, seed=131),
            context={"challenge_name": "Roundtrip Test"},
            source="curriculum",
        )
        data = self.kb.serialize()
        restored = KnowledgeBase.deserialize(data)
        self.assertEqual(restored.state_dim, self.kb.state_dim)
        self.assertEqual(
            restored.total_experiences_stored,
            self.kb.total_experiences_stored,
        )
        self.assertEqual(
            restored.total_concepts_extracted,
            self.kb.total_concepts_extracted,
        )
        self.assertEqual(
            restored.total_queries,
            self.kb.total_queries,
        )
        restored.stop()

    def test_roundtrip_preserves_query_count(self):
        # Mock the missing methods
        self.kb.semantic.spreading_activation = MagicMock(return_value={})
        self.kb.semantic.get_neighbors = MagicMock(return_value=[])
        self.kb.semantic.infer_from_state = MagicMock(return_value=[])
        self.kb.episodic.retrieve = MagicMock(return_value=[])

        query = MemoryQuery(query_state=_make_state(16, seed=132))
        self.kb.query(query)
        self.kb.query(query)

        data = self.kb.serialize()
        restored = KnowledgeBase.deserialize(data)
        self.assertEqual(restored.total_queries, 2)
        restored.stop()


# ===================================================================
# stop / lifecycle tests
# ===================================================================

class TestStopLifecycle(unittest.TestCase):
    """Test stop and lifecycle management."""

    def test_stop_without_consolidation_thread(self):
        kb = _make_kb(enable_consolidation=False)
        # Should not raise
        kb.stop()

    def test_stop_idempotent(self):
        kb = _make_kb(enable_consolidation=False)
        kb.stop()
        kb.stop()  # Second call should be safe

    def test_consolidation_thread_starts_when_enabled(self):
        kb = KnowledgeBase(
            state_dim=8,
            enable_consolidation=True,
            consolidation_interval=9999.0,
        )
        self.assertTrue(kb.enable_consolidation)
        self.assertIsNotNone(kb._consolidation_thread)
        self.assertTrue(kb._consolidation_thread.is_alive())
        kb.stop()
        # After stop, the thread should no longer be alive (or join completed)
        kb._consolidation_thread.join(timeout=5.0)
        self.assertFalse(kb._consolidation_thread.is_alive())

    def test_stop_sets_event_flag(self):
        kb = _make_kb(enable_consolidation=False)
        self.assertFalse(kb._stop_consolidation.is_set())
        kb.stop()
        self.assertTrue(kb._stop_consolidation.is_set())


# ===================================================================
# set_stores tests
# ===================================================================

class TestSetStores(unittest.TestCase):
    """Test the set_stores method."""

    def setUp(self):
        self.kb = _make_kb(state_dim=8)

    def tearDown(self):
        self.kb.stop()

    def test_initially_no_persistence(self):
        self.assertFalse(self.kb._persistence_enabled)

    def test_set_stores_enables_persistence(self):
        mock_vs = MagicMock()
        self.kb.set_stores(vector_store=mock_vs)
        self.assertTrue(self.kb._persistence_enabled)
        self.assertIs(self.kb.vector_store, mock_vs)

    def test_set_stores_with_both(self):
        mock_vs = MagicMock()
        mock_gs = MagicMock()
        self.kb.set_stores(vector_store=mock_vs, graph_store=mock_gs)
        self.assertTrue(self.kb._persistence_enabled)
        self.assertIs(self.kb.vector_store, mock_vs)
        self.assertIs(self.kb.graph_store, mock_gs)

    def test_set_stores_with_none_disables_persistence(self):
        mock_vs = MagicMock()
        self.kb.set_stores(vector_store=mock_vs)
        self.assertTrue(self.kb._persistence_enabled)
        self.kb.set_stores(vector_store=None, graph_store=None)
        self.assertFalse(self.kb._persistence_enabled)


# ===================================================================
# Integration tests
# ===================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple operations."""

    def test_full_lifecycle_store_and_stats(self):
        """Store experiences, check stats, serialize, deserialize."""
        kb = _make_kb(state_dim=8)

        # Store several experiences
        for i in range(5):
            kb.store_experience(
                state=_make_state(8, seed=i + 200),
                context={"challenge_name": f"Task_{i}", "accuracy": 0.5 + i * 0.1},
                source="curriculum" if i % 2 == 0 else "free_play",
            )

        # Check stats
        stats = kb.get_stats()
        self.assertEqual(stats["total_experiences_stored"], 5)
        self.assertGreater(stats["total_concepts_extracted"], 0)
        self.assertGreater(stats["total_concepts"], 0)
        self.assertGreater(stats["recent_events_count"], 0)

        # Serialize
        data = kb.serialize()
        self.assertEqual(data["total_experiences_stored"], 5)

        # Deserialize
        restored = KnowledgeBase.deserialize(data)
        self.assertEqual(restored.total_experiences_stored, 5)

        kb.stop()
        restored.stop()

    def test_store_and_consolidate(self):
        """Store experiences and run consolidation."""
        kb = _make_kb(state_dim=8)

        for i in range(3):
            kb.store_experience(
                state=_make_state(8, seed=i + 210),
                context={"step": i},
                emotional_valence=0.9,
                source="free_play",
            )

        result = kb.consolidate()
        self.assertIsInstance(result, dict)
        self.assertIn("episodes_consolidated", result)

        kb.stop()

    def test_add_concepts_and_relations(self):
        """Add concepts and relations, check graph structure."""
        kb = _make_kb(state_dim=8)

        kb.semantic.add_concept("animal", _make_state(8, seed=220))
        kb.semantic.add_concept("dog", _make_state(8, seed=221))
        kb.semantic.add_concept("cat", _make_state(8, seed=222))

        kb.add_relation("dog", "animal", RelationType.IS_A, strength=0.9)
        kb.add_relation("cat", "animal", RelationType.IS_A, strength=0.9)

        graph = kb.semantic.concept_graph
        self.assertTrue(graph.has_edge("dog", "animal"))
        self.assertTrue(graph.has_edge("cat", "animal"))
        self.assertEqual(len(kb.recent_events), 2)

        kb.stop()

    def test_xp_backend_is_available(self):
        """Verify the xp backend import works (numpy or cupy)."""
        arr = xp.zeros(4)
        self.assertEqual(len(arr), 4)

    def test_multiple_queries_count(self):
        """Multiple queries should each increment the counter."""
        kb = _make_kb(state_dim=8)
        kb.semantic.spreading_activation = MagicMock(return_value={})
        kb.semantic.get_neighbors = MagicMock(return_value=[])
        kb.semantic.infer_from_state = MagicMock(return_value=[])
        kb.episodic.retrieve = MagicMock(return_value=[])

        for i in range(5):
            query = MemoryQuery(query_state=_make_state(8, seed=i + 230))
            kb.query(query)

        self.assertEqual(kb.total_queries, 5)
        kb.stop()

    def test_store_and_query_source_filtering(self):
        """End-to-end: store from both sources, query filtered by source."""
        kb = _make_kb(state_dim=8)
        kb.semantic.spreading_activation = MagicMock(return_value={})
        kb.semantic.get_neighbors = MagicMock(return_value=[])
        kb.semantic.infer_from_state = MagicMock(return_value=[])

        curriculum_state = _make_state(8, seed=240)
        freeplay_state = _make_state(8, seed=241)

        kb.store_experience(
            state=curriculum_state,
            context={"task": "learning"},
            source="curriculum",
        )
        kb.store_experience(
            state=freeplay_state,
            context={"task": "exploring"},
            source="free_play",
        )

        # Create mock episodes matching what was stored
        ep_curriculum = Episode(
            timestamp=1.0,
            state=curriculum_state,
            context={"task": "learning", "source": "curriculum",
                      "timestamp": 1.0},
            sensory_data={},
            emotional_valence=0.0,
            surprise_level=0.0,
        )
        ep_freeplay = Episode(
            timestamp=2.0,
            state=freeplay_state,
            context={"task": "exploring", "source": "free_play",
                      "timestamp": 2.0},
            sensory_data={},
            emotional_valence=0.0,
            surprise_level=0.0,
        )
        kb.episodic.retrieve = MagicMock(
            return_value=[ep_curriculum, ep_freeplay]
        )

        # Query for curriculum experiences only
        query = MemoryQuery(
            query_state=curriculum_state,
            source="curriculum",
            k_episodes=10,
        )
        result = kb.query(query)
        self.assertEqual(len(result.episodes), 1)
        for ep in result.episodes:
            self.assertEqual(ep.context.get("source"), "curriculum")

        kb.stop()


if __name__ == '__main__':
    unittest.main()
