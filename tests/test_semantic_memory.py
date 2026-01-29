"""
Comprehensive tests for the SemanticMemory module.

Tests cover concept creation, relation management across all RelationTypes,
query/retrieval, path finding via inference, spreading activation, and
serialization/deserialization.

All tests are deterministic (fixed random seeds) and use small data for speed.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.semantic_memory import (
    SemanticMemory,
    Concept,
    Relation,
    RelationType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_SIZE = 16
SEED = 42


def _make_memory(**kwargs) -> SemanticMemory:
    """Create a SemanticMemory instance with small, deterministic defaults."""
    defaults = dict(
        embedding_size=EMBEDDING_SIZE,
        random_seed=SEED,
        activation_threshold=0.1,
        spreading_decay=0.7,
        enable_inference=True,
        enable_generalization=True,
    )
    defaults.update(kwargs)
    return SemanticMemory(**defaults)


def _deterministic_embedding(index: int, size: int = EMBEDDING_SIZE) -> np.ndarray:
    """Return a deterministic embedding vector seeded by *index*."""
    rng = np.random.RandomState(index)
    return rng.randn(size).astype(np.float64)


def _unit_embedding(index: int, size: int = EMBEDDING_SIZE) -> np.ndarray:
    """Return a deterministic unit-norm embedding vector."""
    vec = _deterministic_embedding(index, size)
    return vec / (np.linalg.norm(vec) + 1e-10)


# ===========================================================================
# Test classes
# ===========================================================================


class TestSemanticMemoryInitialization(unittest.TestCase):
    """Tests for SemanticMemory.__init__."""

    def test_default_initialization(self):
        """SemanticMemory should initialise with correct parameters."""
        mem = _make_memory()
        self.assertEqual(mem.embedding_size, EMBEDDING_SIZE)
        self.assertEqual(mem.max_concepts, 100000)
        self.assertAlmostEqual(mem.activation_threshold, 0.1)
        self.assertAlmostEqual(mem.spreading_decay, 0.7)
        self.assertTrue(mem.enable_inference)
        self.assertTrue(mem.enable_generalization)

    def test_concepts_start_empty(self):
        """No concepts should exist immediately after construction."""
        mem = _make_memory()
        self.assertEqual(len(mem.concepts), 0)
        self.assertEqual(mem.concept_graph.number_of_nodes(), 0)
        self.assertEqual(mem.concept_graph.number_of_edges(), 0)

    def test_statistics_start_at_zero(self):
        """All counters should be zero on a fresh instance."""
        mem = _make_memory()
        self.assertEqual(mem.total_concepts_created, 0)
        self.assertEqual(mem.total_relations_created, 0)
        self.assertEqual(mem.total_inferences_made, 0)
        self.assertEqual(mem.total_generalizations, 0)

    def test_custom_parameters(self):
        """Custom constructor arguments should be stored correctly."""
        mem = SemanticMemory(
            embedding_size=32,
            max_concepts=500,
            activation_threshold=0.5,
            spreading_decay=0.8,
            learning_rate=0.05,
            consolidation_rate=0.01,
            similarity_threshold=0.9,
            enable_inference=False,
            enable_generalization=False,
            random_seed=99,
        )
        self.assertEqual(mem.embedding_size, 32)
        self.assertEqual(mem.max_concepts, 500)
        self.assertAlmostEqual(mem.activation_threshold, 0.5)
        self.assertAlmostEqual(mem.spreading_decay, 0.8)
        self.assertAlmostEqual(mem.learning_rate, 0.05)
        self.assertFalse(mem.enable_inference)
        self.assertFalse(mem.enable_generalization)

    def test_persistence_disabled_by_default(self):
        """Without stores, persistence should be disabled."""
        mem = _make_memory()
        self.assertFalse(mem._persistence_enabled)
        self.assertIsNone(mem.vector_store)
        self.assertIsNone(mem.graph_store)

    def test_activation_state_starts_empty(self):
        """Activation state dict should be empty at construction."""
        mem = _make_memory()
        self.assertEqual(len(mem.activation_state), 0)


class TestAddConcept(unittest.TestCase):
    """Tests for SemanticMemory.add_concept."""

    def setUp(self):
        self.mem = _make_memory()

    def test_add_single_concept(self):
        """Adding a concept should store it in concepts and the graph."""
        emb = _deterministic_embedding(0)
        concept = self.mem.add_concept("dog", emb)
        self.assertIsInstance(concept, Concept)
        self.assertEqual(concept.name, "dog")
        self.assertIn("dog", self.mem.concepts)
        self.assertTrue(self.mem.concept_graph.has_node("dog"))

    def test_concept_embedding_copied(self):
        """The stored embedding should be a copy, not a reference."""
        emb = _deterministic_embedding(1)
        concept = self.mem.add_concept("cat", emb)
        # Mutating the original should not affect the stored embedding.
        original = concept.embedding.copy()
        emb[0] = 999.0
        np.testing.assert_array_equal(concept.embedding, original)

    def test_concept_with_attributes(self):
        """Attributes passed at creation should be stored."""
        emb = _deterministic_embedding(2)
        attrs = {"color": "brown", "legs": 4}
        concept = self.mem.add_concept("horse", emb, attributes=attrs)
        self.assertEqual(concept.attributes["color"], "brown")
        self.assertEqual(concept.attributes["legs"], 4)

    def test_add_multiple_concepts(self):
        """Adding several concepts should increase counts correctly."""
        for i, name in enumerate(["a", "b", "c"]):
            self.mem.add_concept(name, _deterministic_embedding(i))
        self.assertEqual(len(self.mem.concepts), 3)
        self.assertEqual(self.mem.total_concepts_created, 3)
        self.assertEqual(self.mem.concept_graph.number_of_nodes(), 3)

    def test_update_existing_concept_blends_embedding(self):
        """Re-adding an existing concept should blend embeddings."""
        emb1 = np.ones(EMBEDDING_SIZE, dtype=np.float64)
        emb2 = np.zeros(EMBEDDING_SIZE, dtype=np.float64)
        self.mem.add_concept("x", emb1)
        self.mem.add_concept("x", emb2)
        # Blended: (1 - lr) * emb1 + lr * emb2
        lr = self.mem.learning_rate
        expected = (1 - lr) * emb1 + lr * emb2
        np.testing.assert_allclose(self.mem.concepts["x"].embedding, expected)

    def test_update_existing_concept_increments_activation_count(self):
        """Re-adding a concept should increase its activation_count."""
        emb = _deterministic_embedding(0)
        self.mem.add_concept("x", emb)
        self.assertEqual(self.mem.concepts["x"].activation_count, 0)
        self.mem.add_concept("x", emb)
        self.assertEqual(self.mem.concepts["x"].activation_count, 1)

    def test_update_existing_concept_merges_attributes(self):
        """Re-adding with new attributes should merge them."""
        emb = _deterministic_embedding(0)
        self.mem.add_concept("x", emb, attributes={"a": 1})
        self.mem.add_concept("x", emb, attributes={"b": 2})
        self.assertEqual(self.mem.concepts["x"].attributes["a"], 1)
        self.assertEqual(self.mem.concepts["x"].attributes["b"], 2)

    def test_concept_creation_time_set(self):
        """New concepts should have a non-zero creation_time."""
        emb = _deterministic_embedding(0)
        concept = self.mem.add_concept("t", emb)
        self.assertGreater(concept.creation_time, 0.0)

    def test_total_concepts_created_not_incremented_on_update(self):
        """Updating a concept should not increment total_concepts_created."""
        emb = _deterministic_embedding(0)
        self.mem.add_concept("x", emb)
        self.assertEqual(self.mem.total_concepts_created, 1)
        self.mem.add_concept("x", emb)
        self.assertEqual(self.mem.total_concepts_created, 1)

    def test_add_concept_returns_concept(self):
        """add_concept should return a Concept dataclass."""
        emb = _deterministic_embedding(0)
        result = self.mem.add_concept("test", emb)
        self.assertIsInstance(result, Concept)
        self.assertEqual(result.name, "test")


class TestAddRelation(unittest.TestCase):
    """Tests for SemanticMemory.add_relation with various RelationTypes."""

    def setUp(self):
        self.mem = _make_memory(enable_inference=False)
        # Pre-populate a handful of concepts.
        self.names = ["animal", "dog", "wheel", "car", "rain", "wet",
                       "event_a", "event_b", "parent", "child"]
        for i, name in enumerate(self.names):
            self.mem.add_concept(name, _deterministic_embedding(i))

    # -- IS_A (taxonomic / hierarchical) ------------------------------------

    def test_add_relation_is_a(self):
        """IS_A relation should create a directed edge."""
        rel = self.mem.add_relation("dog", "animal", RelationType.IS_A)
        self.assertIsInstance(rel, Relation)
        self.assertEqual(rel.relation_type, RelationType.IS_A)
        self.assertTrue(self.mem.concept_graph.has_edge("dog", "animal"))

    def test_is_a_direction(self):
        """IS_A should be directional: dog->animal but not animal->dog."""
        self.mem.add_relation("dog", "animal", RelationType.IS_A)
        self.assertTrue(self.mem.concept_graph.has_edge("dog", "animal"))
        self.assertFalse(self.mem.concept_graph.has_edge("animal", "dog"))

    # -- PART_OF ------------------------------------------------------------

    def test_add_relation_part_of(self):
        """PART_OF relation should create a directed edge."""
        rel = self.mem.add_relation("wheel", "car", RelationType.PART_OF)
        self.assertIsNotNone(rel)
        self.assertEqual(rel.relation_type, RelationType.PART_OF)
        self.assertTrue(self.mem.concept_graph.has_edge("wheel", "car"))

    def test_part_of_stored_type(self):
        """The edge data should record the correct RelationType."""
        self.mem.add_relation("wheel", "car", RelationType.PART_OF)
        data = self.mem.concept_graph.edges["wheel", "car"]
        self.assertEqual(data["type"], RelationType.PART_OF)

    # -- SIMILAR_TO ---------------------------------------------------------

    def test_add_relation_similar_to(self):
        """SIMILAR_TO relation should be storable."""
        rel = self.mem.add_relation("dog", "animal", RelationType.SIMILAR_TO)
        self.assertIsNotNone(rel)
        self.assertEqual(rel.relation_type, RelationType.SIMILAR_TO)

    def test_similar_to_bidirectional(self):
        """SIMILAR_TO created as bidirectional should add edges both ways."""
        self.mem.add_relation(
            "dog", "animal", RelationType.SIMILAR_TO, bidirectional=True,
        )
        self.assertTrue(self.mem.concept_graph.has_edge("dog", "animal"))
        self.assertTrue(self.mem.concept_graph.has_edge("animal", "dog"))

    # -- CAUSES (causal) ----------------------------------------------------

    def test_add_relation_causes(self):
        """CAUSES relation should link cause to effect."""
        rel = self.mem.add_relation("rain", "wet", RelationType.CAUSES)
        self.assertIsNotNone(rel)
        self.assertEqual(rel.relation_type, RelationType.CAUSES)
        self.assertTrue(self.mem.concept_graph.has_edge("rain", "wet"))

    def test_causes_strength(self):
        """Custom strength should be recorded on the edge."""
        self.mem.add_relation("rain", "wet", RelationType.CAUSES, strength=0.8)
        data = self.mem.concept_graph.edges["rain", "wet"]
        self.assertAlmostEqual(data["weight"], 0.8)

    # -- BEFORE / AFTER (temporal) ------------------------------------------

    def test_add_relation_before(self):
        """BEFORE relation should encode temporal ordering."""
        rel = self.mem.add_relation("event_a", "event_b", RelationType.BEFORE)
        self.assertIsNotNone(rel)
        self.assertEqual(rel.relation_type, RelationType.BEFORE)
        self.assertTrue(self.mem.concept_graph.has_edge("event_a", "event_b"))

    def test_add_relation_after(self):
        """AFTER relation should encode reverse temporal ordering."""
        rel = self.mem.add_relation("event_b", "event_a", RelationType.AFTER)
        self.assertIsNotNone(rel)
        self.assertEqual(rel.relation_type, RelationType.AFTER)
        self.assertTrue(self.mem.concept_graph.has_edge("event_b", "event_a"))

    # -- HAS_A (hierarchical / part-whole) ----------------------------------

    def test_add_relation_has_a(self):
        """HAS_A relation should link container to part."""
        rel = self.mem.add_relation("car", "wheel", RelationType.HAS_A)
        self.assertIsNotNone(rel)
        self.assertEqual(rel.relation_type, RelationType.HAS_A)
        self.assertTrue(self.mem.concept_graph.has_edge("car", "wheel"))

    # -- General relation properties ----------------------------------------

    def test_relation_returns_none_for_missing_source(self):
        """add_relation should return None when source does not exist."""
        result = self.mem.add_relation("ghost", "dog", RelationType.IS_A)
        self.assertIsNone(result)

    def test_relation_returns_none_for_missing_target(self):
        """add_relation should return None when target does not exist."""
        result = self.mem.add_relation("dog", "ghost", RelationType.IS_A)
        self.assertIsNone(result)

    def test_relation_with_string_type(self):
        """add_relation should accept a string relation type."""
        rel = self.mem.add_relation("dog", "animal", "is_a")
        self.assertIsNotNone(rel)
        self.assertEqual(rel.relation_type, RelationType.IS_A)

    def test_relation_increments_counter(self):
        """Each successful add_relation should increment total_relations_created."""
        self.mem.add_relation("dog", "animal", RelationType.IS_A)
        self.mem.add_relation("wheel", "car", RelationType.PART_OF)
        self.assertEqual(self.mem.total_relations_created, 2)

    def test_relation_metadata(self):
        """Metadata dict should be stored on the Relation object."""
        meta = {"source": "textbook", "confidence": 0.95}
        rel = self.mem.add_relation(
            "dog", "animal", RelationType.IS_A, metadata=meta,
        )
        self.assertEqual(rel.metadata["source"], "textbook")
        self.assertAlmostEqual(rel.metadata["confidence"], 0.95)

    def test_bidirectional_flag_creates_reverse_edge(self):
        """Setting bidirectional=True should add the reverse edge."""
        self.mem.add_relation(
            "parent", "child", RelationType.IS_A, bidirectional=True,
        )
        self.assertTrue(self.mem.concept_graph.has_edge("parent", "child"))
        self.assertTrue(self.mem.concept_graph.has_edge("child", "parent"))

    def test_edge_weight_matches_strength(self):
        """Edge weight stored in the graph should match the strength arg."""
        self.mem.add_relation("dog", "animal", RelationType.IS_A, strength=0.75)
        data = self.mem.concept_graph.edges["dog", "animal"]
        self.assertAlmostEqual(data["weight"], 0.75)


class TestQuery(unittest.TestCase):
    """Tests for SemanticMemory.query."""

    def setUp(self):
        self.mem = _make_memory(enable_inference=False)

    def test_query_empty_memory_returns_empty(self):
        """Querying empty memory should return an empty list."""
        result = self.mem.query(cue_embedding=_deterministic_embedding(0))
        self.assertEqual(result, [])

    def test_query_by_embedding_similarity(self):
        """Query with a cue embedding should return similar concepts."""
        # Add three concepts with known embeddings.
        emb_a = np.zeros(EMBEDDING_SIZE)
        emb_a[0] = 1.0
        emb_b = np.zeros(EMBEDDING_SIZE)
        emb_b[1] = 1.0
        emb_c = np.zeros(EMBEDDING_SIZE)
        emb_c[0] = 0.9
        emb_c[1] = 0.1

        self.mem.add_concept("a", emb_a)
        self.mem.add_concept("b", emb_b)
        self.mem.add_concept("c", emb_c)

        # Cue that is very close to 'a' and 'c'.
        cue = np.zeros(EMBEDDING_SIZE)
        cue[0] = 1.0
        results = self.mem.query(cue_embedding=cue, n_results=3)

        # Results should be a list of (name, score) tuples.
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, tuple) and len(r) == 2 for r in results))

        names = [name for name, _ in results]
        # 'a' should be the closest match.
        self.assertEqual(names[0], "a")

    def test_query_n_results_limits_output(self):
        """n_results should cap the number of returned concepts."""
        for i in range(10):
            self.mem.add_concept(f"c{i}", _deterministic_embedding(i))
        results = self.mem.query(
            cue_embedding=_deterministic_embedding(0), n_results=3,
        )
        self.assertLessEqual(len(results), 3)

    def test_query_by_concept_name_with_spreading(self):
        """Querying by concept name should use spreading activation."""
        self.mem.add_concept("a", _deterministic_embedding(0))
        self.mem.add_concept("b", _deterministic_embedding(1))
        self.mem.add_concept("c", _deterministic_embedding(2))
        self.mem.add_relation("a", "b", RelationType.IS_A)
        self.mem.add_relation("b", "c", RelationType.IS_A)

        results = self.mem.query(cue_concept="a", n_results=5)
        names = [name for name, _ in results]
        # 'a' should be activated (source) and 'b' should be reachable.
        self.assertIn("a", names)
        self.assertIn("b", names)

    def test_query_no_cue_returns_empty(self):
        """Querying without any cue should return an empty list."""
        self.mem.add_concept("x", _deterministic_embedding(0))
        results = self.mem.query()
        self.assertEqual(results, [])

    def test_query_with_relation_type_filter(self):
        """Spreading activation should respect relation_type filter."""
        self.mem.add_concept("a", _deterministic_embedding(0))
        self.mem.add_concept("b", _deterministic_embedding(1))
        self.mem.add_concept("c", _deterministic_embedding(2))
        self.mem.add_relation("a", "b", RelationType.IS_A)
        self.mem.add_relation("a", "c", RelationType.CAUSES)

        # Filter by IS_A: should reach b but not c.
        results = self.mem.query(
            cue_concept="a", relation_type=RelationType.IS_A, n_results=5,
        )
        names = [name for name, _ in results]
        self.assertIn("b", names)
        self.assertNotIn("c", names)

    def test_query_similarity_scores_are_floats(self):
        """Scores in query results should be Python floats."""
        self.mem.add_concept("x", _deterministic_embedding(0))
        results = self.mem.query(cue_embedding=_deterministic_embedding(0))
        for _, score in results:
            self.assertIsInstance(score, float)


class TestFindPaths(unittest.TestCase):
    """Tests for path-finding between concepts via SemanticMemory.infer."""

    def setUp(self):
        # enable_inference=True so infer() works; but we add relations in a
        # leaf-to-root order that avoids triggering transitive inference rules
        # (transitive IS_A inference only fires when *adding* a new IS_A edge
        # whose target already has an outgoing IS_A edge).
        self.mem = _make_memory(enable_inference=True)
        # Build a small chain: a -> b -> c -> d
        for i, name in enumerate(["a", "b", "c", "d"]):
            self.mem.add_concept(name, _deterministic_embedding(i))
        # Add edges root-first so transitive rule is not triggered
        self.mem.add_relation("c", "d", RelationType.IS_A)
        self.mem.add_relation("b", "c", RelationType.IS_A)
        self.mem.add_relation("a", "b", RelationType.IS_A)

    def test_find_path_between_two_concepts(self):
        """infer(source, target) should find the connecting path."""
        inferences = self.mem.infer("a", "d", max_steps=5)
        self.assertTrue(len(inferences) > 0)
        # Each inference is (target, path, confidence).
        target, path, confidence = inferences[0]
        self.assertEqual(target, "d")
        self.assertEqual(path[0], "a")
        self.assertEqual(path[-1], "d")

    def test_path_length_is_correct(self):
        """Path a->b->c->d should have 4 nodes."""
        inferences = self.mem.infer("a", "d", max_steps=5)
        _, path, _ = inferences[0]
        self.assertEqual(len(path), 4)
        self.assertEqual(path, ["a", "b", "c", "d"])

    def test_path_confidence_decays(self):
        """Confidence should be less than 1.0 for multi-hop paths."""
        inferences = self.mem.infer("a", "d", max_steps=5)
        _, _, confidence = inferences[0]
        self.assertLess(confidence, 1.0)
        self.assertGreater(confidence, 0.0)

    def test_no_path_returns_empty(self):
        """When no path exists, infer should return an empty list."""
        self.mem.add_concept("isolated", _deterministic_embedding(99))
        inferences = self.mem.infer("a", "isolated", max_steps=5)
        self.assertEqual(len(inferences), 0)

    def test_direct_path(self):
        """A direct 1-hop path should have exactly 2 nodes."""
        inferences = self.mem.infer("a", "b", max_steps=5)
        self.assertTrue(len(inferences) > 0)
        _, path, _ = inferences[0]
        self.assertEqual(path, ["a", "b"])

    def test_infer_with_no_target_finds_reachable(self):
        """infer without a target should find all reachable concepts."""
        inferences = self.mem.infer("a", target=None, max_steps=5)
        reached_names = {t for t, _, _ in inferences}
        self.assertIn("b", reached_names)
        self.assertIn("c", reached_names)
        self.assertIn("d", reached_names)

    def test_infer_nonexistent_source(self):
        """Inference from a nonexistent concept should return empty."""
        result = self.mem.infer("nonexistent", "a")
        self.assertEqual(result, [])

    def test_infer_disabled(self):
        """When enable_inference=False, infer should return empty."""
        mem = _make_memory(enable_inference=False)
        mem.add_concept("x", _deterministic_embedding(0))
        mem.add_concept("y", _deterministic_embedding(1))
        mem.add_relation("x", "y", RelationType.IS_A)
        mem.enable_inference = False
        result = mem.infer("x", "y")
        self.assertEqual(result, [])


class TestSpreadingActivation(unittest.TestCase):
    """Tests for the spreading activation mechanism."""

    def setUp(self):
        self.mem = _make_memory(
            enable_inference=False,
            activation_threshold=0.1,
            spreading_decay=0.7,
        )
        # Build a small network: hub -> spoke1, hub -> spoke2, spoke1 -> leaf
        for i, name in enumerate(["hub", "spoke1", "spoke2", "leaf"]):
            self.mem.add_concept(name, _deterministic_embedding(i))
        self.mem.add_relation("hub", "spoke1", RelationType.IS_A, strength=1.0)
        self.mem.add_relation("hub", "spoke2", RelationType.IS_A, strength=1.0)
        self.mem.add_relation("spoke1", "leaf", RelationType.IS_A, strength=1.0)

    def test_source_gets_full_activation(self):
        """The source concept should receive activation of 1.0."""
        self.mem._spread_activation("hub")
        self.assertAlmostEqual(self.mem.activation_state.get("hub", 0.0), 1.0)

    def test_direct_neighbors_activated(self):
        """Direct neighbors should receive decayed activation."""
        self.mem._spread_activation("hub")
        spoke1_act = self.mem.activation_state.get("spoke1", 0.0)
        spoke2_act = self.mem.activation_state.get("spoke2", 0.0)
        # Expected: 1.0 * decay * weight = 0.7
        self.assertAlmostEqual(spoke1_act, 0.7)
        self.assertAlmostEqual(spoke2_act, 0.7)

    def test_activation_decays_with_distance(self):
        """Two-hop neighbors should have less activation than one-hop."""
        self.mem._spread_activation("hub")
        spoke1_act = self.mem.activation_state.get("spoke1", 0.0)
        leaf_act = self.mem.activation_state.get("leaf", 0.0)
        self.assertGreater(spoke1_act, leaf_act)

    def test_two_hop_activation_value(self):
        """Leaf should get activation = 1.0 * 0.7 * 0.7 = 0.49."""
        self.mem._spread_activation("hub")
        leaf_act = self.mem.activation_state.get("leaf", 0.0)
        expected = 1.0 * 0.7 * 1.0 * 0.7 * 1.0  # decay * weight at each hop
        self.assertAlmostEqual(leaf_act, expected)

    def test_max_depth_limits_spreading(self):
        """Spreading with max_depth=2 should reach spokes but not the leaf."""
        # depth starts at 0: hub processed at depth 0, spokes at depth 1,
        # leaf would need depth 2 which is >= max_depth=2 so it stops.
        self.mem._spread_activation("hub", max_depth=2)
        self.assertIn("spoke1", self.mem.activation_state)
        self.assertIn("spoke2", self.mem.activation_state)
        self.assertNotIn("leaf", self.mem.activation_state)

    def test_activation_threshold_stops_weak_signals(self):
        """Activation below threshold should not propagate further."""
        mem = _make_memory(
            enable_inference=False,
            activation_threshold=0.5,
            spreading_decay=0.7,
        )
        for i, name in enumerate(["a", "b", "c", "d"]):
            mem.add_concept(name, _deterministic_embedding(i))
        mem.add_relation("a", "b", RelationType.IS_A, strength=1.0)
        mem.add_relation("b", "c", RelationType.IS_A, strength=1.0)
        mem.add_relation("c", "d", RelationType.IS_A, strength=1.0)

        mem._spread_activation("a")
        # a=1.0, b=0.7, c=0.49 < 0.5 threshold => c gets set but d unreachable.
        self.assertAlmostEqual(mem.activation_state.get("a", 0.0), 1.0)
        self.assertAlmostEqual(mem.activation_state.get("b", 0.0), 0.7)
        self.assertNotIn("d", mem.activation_state)

    def test_edge_strength_modulates_activation(self):
        """Weaker edge strengths should produce less activation."""
        mem = _make_memory(
            enable_inference=False,
            activation_threshold=0.01,
            spreading_decay=0.7,
        )
        mem.add_concept("a", _deterministic_embedding(0))
        mem.add_concept("strong_b", _deterministic_embedding(1))
        mem.add_concept("weak_b", _deterministic_embedding(2))
        mem.add_relation("a", "strong_b", RelationType.IS_A, strength=1.0)
        mem.add_relation("a", "weak_b", RelationType.IS_A, strength=0.5)

        mem._spread_activation("a")
        strong_act = mem.activation_state.get("strong_b", 0.0)
        weak_act = mem.activation_state.get("weak_b", 0.0)
        self.assertGreater(strong_act, weak_act)
        self.assertAlmostEqual(strong_act, 1.0 * 0.7 * 1.0)
        self.assertAlmostEqual(weak_act, 1.0 * 0.7 * 0.5)

    def test_spreading_activation_via_query(self):
        """query(cue_concept=...) should populate activation_state."""
        results = self.mem.query(cue_concept="hub", n_results=10)
        self.assertTrue(len(results) > 0)
        # Source should have highest activation.
        names_scores = dict(results)
        self.assertIn("hub", names_scores)
        for name in ["spoke1", "spoke2"]:
            if name in names_scores:
                self.assertGreaterEqual(names_scores["hub"], names_scores[name])


class TestGetStateAndSerialization(unittest.TestCase):
    """Tests for get_stats and serialize/deserialize."""

    def _build_small_memory(self) -> SemanticMemory:
        """Build a small memory with concepts and relations."""
        mem = _make_memory(enable_inference=False)
        mem.add_concept("animal", _deterministic_embedding(0))
        mem.add_concept("dog", _deterministic_embedding(1))
        mem.add_concept("cat", _deterministic_embedding(2))
        mem.add_relation("dog", "animal", RelationType.IS_A, strength=0.9)
        mem.add_relation("cat", "animal", RelationType.IS_A, strength=0.85)
        return mem

    def test_get_stats_returns_dict(self):
        """get_stats should return a dictionary."""
        mem = self._build_small_memory()
        stats = mem.get_stats()
        self.assertIsInstance(stats, dict)

    def test_get_stats_concept_count(self):
        """get_stats should report the correct concept count."""
        mem = self._build_small_memory()
        stats = mem.get_stats()
        self.assertEqual(stats["total_concepts"], 3)

    def test_get_stats_relation_count(self):
        """get_stats should report the correct relation/edge count."""
        mem = self._build_small_memory()
        stats = mem.get_stats()
        self.assertEqual(stats["total_relations"], 2)

    def test_get_stats_keys(self):
        """get_stats should contain all expected keys."""
        mem = self._build_small_memory()
        stats = mem.get_stats()
        expected_keys = {
            "total_concepts", "total_relations",
            "total_concepts_created", "total_relations_created",
            "total_inferences_made", "total_generalizations",
            "avg_degree", "graph_density",
        }
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_serialize_returns_dict(self):
        """serialize should return a dictionary."""
        mem = self._build_small_memory()
        data = mem.serialize()
        self.assertIsInstance(data, dict)

    def test_serialize_contains_embedding_size(self):
        """Serialized data should include embedding_size."""
        mem = self._build_small_memory()
        data = mem.serialize()
        self.assertEqual(data["embedding_size"], EMBEDDING_SIZE)

    def test_serialize_contains_concepts(self):
        """Serialized data should include all concepts."""
        mem = self._build_small_memory()
        data = mem.serialize()
        self.assertIn("concepts", data)
        self.assertEqual(len(data["concepts"]), 3)
        self.assertIn("dog", data["concepts"])

    def test_serialize_concept_embedding_is_list(self):
        """Concept embeddings should be serialized as plain lists."""
        mem = self._build_small_memory()
        data = mem.serialize()
        emb = data["concepts"]["dog"]["embedding"]
        self.assertIsInstance(emb, list)
        self.assertEqual(len(emb), EMBEDDING_SIZE)

    def test_serialize_contains_relations(self):
        """Serialized data should include relations."""
        mem = self._build_small_memory()
        data = mem.serialize()
        self.assertIn("relations", data)
        self.assertEqual(len(data["relations"]), 2)

    def test_serialize_relation_type_is_string(self):
        """Relation types should be serialized as string values."""
        mem = self._build_small_memory()
        data = mem.serialize()
        for rel in data["relations"]:
            self.assertIsInstance(rel["type"], str)

    def test_serialize_contains_stats(self):
        """Serialized data should include stats."""
        mem = self._build_small_memory()
        data = mem.serialize()
        self.assertIn("stats", data)
        self.assertIsInstance(data["stats"], dict)

    def test_deserialize_roundtrip(self):
        """Deserializing serialized data should restore concepts and relations."""
        mem = self._build_small_memory()
        data = mem.serialize()

        restored = SemanticMemory.deserialize(data)

        self.assertEqual(len(restored.concepts), 3)
        self.assertIn("dog", restored.concepts)
        self.assertIn("animal", restored.concepts)
        self.assertIn("cat", restored.concepts)

    def test_deserialize_restores_embeddings(self):
        """Deserialized concepts should have correct embedding shape."""
        mem = self._build_small_memory()
        data = mem.serialize()
        restored = SemanticMemory.deserialize(data)

        for name in ["dog", "animal", "cat"]:
            self.assertEqual(
                len(restored.concepts[name].embedding), EMBEDDING_SIZE,
            )

    def test_deserialize_restores_relations(self):
        """Deserialized memory should restore graph edges."""
        mem = self._build_small_memory()
        data = mem.serialize()
        restored = SemanticMemory.deserialize(data)

        self.assertTrue(restored.concept_graph.has_edge("dog", "animal"))
        self.assertTrue(restored.concept_graph.has_edge("cat", "animal"))

    def test_deserialize_restores_edge_weights(self):
        """Deserialized edge weights should match original strengths."""
        mem = self._build_small_memory()
        data = mem.serialize()
        restored = SemanticMemory.deserialize(data)

        dog_edge = restored.concept_graph.edges["dog", "animal"]
        self.assertAlmostEqual(dog_edge["weight"], 0.9)

    def test_serialize_empty_memory(self):
        """Serializing an empty memory should not raise."""
        mem = _make_memory()
        data = mem.serialize()
        self.assertEqual(data["embedding_size"], EMBEDDING_SIZE)
        self.assertEqual(len(data["concepts"]), 0)
        self.assertEqual(len(data["relations"]), 0)

    def test_deserialize_empty_memory(self):
        """Deserializing empty data should produce a valid empty memory."""
        mem = _make_memory()
        data = mem.serialize()
        restored = SemanticMemory.deserialize(data)
        self.assertEqual(len(restored.concepts), 0)


class TestRelationTypeEnum(unittest.TestCase):
    """Tests for the RelationType enum values."""

    def test_all_expected_types_exist(self):
        """All documented RelationTypes should be accessible."""
        expected = [
            "IS_A", "HAS_A", "CAUSES", "SIMILAR_TO", "OPPOSITE_OF",
            "MEMBER_OF", "ATTRIBUTE", "BEFORE", "AFTER", "ENABLES",
            "PREVENTS", "USED_FOR", "LOCATED_AT", "PART_OF", "CREATED_BY",
        ]
        for name in expected:
            self.assertTrue(
                hasattr(RelationType, name),
                f"RelationType.{name} is missing",
            )

    def test_relation_type_values_are_strings(self):
        """Each RelationType value should be a string."""
        for member in RelationType:
            self.assertIsInstance(member.value, str)

    def test_relation_type_from_string(self):
        """RelationType should be constructable from its string value."""
        for member in RelationType:
            reconstructed = RelationType(member.value)
            self.assertEqual(reconstructed, member)


class TestConceptAndRelationDataclasses(unittest.TestCase):
    """Tests for the Concept and Relation dataclass basics."""

    def test_concept_hash_by_name(self):
        """Two Concepts with the same name should hash equally."""
        c1 = Concept(name="test", embedding=np.zeros(4))
        c2 = Concept(name="test", embedding=np.ones(4))
        self.assertEqual(hash(c1), hash(c2))

    def test_concept_equality_by_name(self):
        """Concepts are equal iff they share the same name."""
        c1 = Concept(name="x", embedding=np.zeros(4))
        c2 = Concept(name="x", embedding=np.ones(4))
        c3 = Concept(name="y", embedding=np.zeros(4))
        self.assertEqual(c1, c2)
        self.assertNotEqual(c1, c3)

    def test_concept_not_equal_to_non_concept(self):
        """Concept should not equal a non-Concept object."""
        c = Concept(name="x", embedding=np.zeros(4))
        self.assertNotEqual(c, "x")
        self.assertNotEqual(c, 42)

    def test_relation_hash(self):
        """Relation hash should depend on source, target, and type."""
        r1 = Relation(source="a", target="b", relation_type=RelationType.IS_A)
        r2 = Relation(source="a", target="b", relation_type=RelationType.IS_A)
        r3 = Relation(source="a", target="c", relation_type=RelationType.IS_A)
        self.assertEqual(hash(r1), hash(r2))
        self.assertNotEqual(hash(r1), hash(r3))

    def test_relation_default_strength(self):
        """Relation default strength should be 1.0."""
        r = Relation(source="a", target="b", relation_type=RelationType.IS_A)
        self.assertAlmostEqual(r.strength, 1.0)

    def test_relation_default_bidirectional(self):
        """Relation should default to unidirectional."""
        r = Relation(source="a", target="b", relation_type=RelationType.IS_A)
        self.assertFalse(r.bidirectional)

    def test_relation_default_metadata(self):
        """Relation metadata should default to an empty dict."""
        r = Relation(source="a", target="b", relation_type=RelationType.IS_A)
        self.assertEqual(r.metadata, {})


class TestTransitiveInference(unittest.TestCase):
    """Tests for automatic transitive IS_A inference."""

    def test_transitive_is_a_inference(self):
        """Adding A IS_A B then B IS_A C should infer A IS_A C."""
        mem = _make_memory(enable_inference=True)
        mem.add_concept("poodle", _deterministic_embedding(0))
        mem.add_concept("dog", _deterministic_embedding(1))
        mem.add_concept("animal", _deterministic_embedding(2))

        # First add dog IS_A animal.
        mem.add_relation("dog", "animal", RelationType.IS_A)
        # Now add poodle IS_A dog -- this should trigger transitive inference.
        mem.add_relation("poodle", "dog", RelationType.IS_A)

        # The system should have inferred poodle IS_A animal.
        self.assertTrue(mem.concept_graph.has_edge("poodle", "animal"))

    def test_transitive_inference_has_reduced_strength(self):
        """Inferred transitive relations should have confidence < 1."""
        mem = _make_memory(enable_inference=True)
        mem.add_concept("a", _deterministic_embedding(0))
        mem.add_concept("b", _deterministic_embedding(1))
        mem.add_concept("c", _deterministic_embedding(2))

        mem.add_relation("b", "c", RelationType.IS_A, strength=1.0)
        mem.add_relation("a", "b", RelationType.IS_A, strength=1.0)

        # Inferred edge a -> c should exist with reduced weight.
        self.assertTrue(mem.concept_graph.has_edge("a", "c"))
        weight = mem.concept_graph.edges["a", "c"]["weight"]
        self.assertLess(weight, 1.0)
        self.assertGreater(weight, 0.0)


class TestPruning(unittest.TestCase):
    """Tests for concept pruning when capacity is exceeded."""

    def test_prune_when_over_capacity(self):
        """When exceeding max_concepts, the weakest concept should be pruned."""
        mem = _make_memory(max_concepts=3, enable_inference=False)
        for i in range(4):
            mem.add_concept(f"c{i}", _deterministic_embedding(i))
        # Should have pruned down to max_concepts.
        self.assertLessEqual(len(mem.concepts), 3)


class TestGeneralization(unittest.TestCase):
    """Tests for the generalize method."""

    def test_generalize_creates_new_concept(self):
        """Generalizing similar concepts should create a new abstract concept."""
        mem = _make_memory(enable_inference=False, enable_generalization=True)
        # Create two very similar embeddings.
        base = np.ones(EMBEDDING_SIZE, dtype=np.float64)
        emb1 = base + 0.01 * _deterministic_embedding(0)
        emb2 = base + 0.01 * _deterministic_embedding(1)
        mem.add_concept("example1", emb1)
        mem.add_concept("example2", emb2)

        result = mem.generalize(["example1", "example2"], min_similarity=0.5)
        self.assertIsNotNone(result)
        self.assertTrue(result.name.startswith("GENERAL_"))

    def test_generalize_needs_at_least_two_examples(self):
        """Generalization with fewer than 2 examples should return None."""
        mem = _make_memory(enable_inference=False, enable_generalization=True)
        mem.add_concept("solo", _deterministic_embedding(0))
        result = mem.generalize(["solo"])
        self.assertIsNone(result)

    def test_generalize_disabled(self):
        """When enable_generalization=False, generalize should return None."""
        mem = _make_memory(enable_inference=False, enable_generalization=False)
        mem.add_concept("a", _deterministic_embedding(0))
        mem.add_concept("b", _deterministic_embedding(1))
        result = mem.generalize(["a", "b"])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
