"""
Comprehensive tests for the CreativityEngine module.

Tests cover:
- CreativityEngine initialization (defaults and custom parameters)
- Conceptual blending (all blend modes, thresholds, edge cases)
- Imagination / novel idea generation (trajectories, goals, constraints)
- Creative recombination (divergent thinking, generation modes, learning)
- CreativeMode and NoveltyType enum values
- Serialization round-trip (serialize / deserialize)
- Statistics reporting (get_stats)
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.creativity import (
    CreativityEngine,
    CreativeMode,
    NoveltyType,
    Concept,
    Blend,
    Imagination,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestCreativeMode(unittest.TestCase):
    """Verify every CreativeMode variant and its string value."""

    def test_convergent(self):
        self.assertEqual(CreativeMode.CONVERGENT.value, "convergent")

    def test_divergent(self):
        self.assertEqual(CreativeMode.DIVERGENT.value, "divergent")

    def test_exploratory(self):
        self.assertEqual(CreativeMode.EXPLORATORY.value, "exploratory")

    def test_combinatorial(self):
        self.assertEqual(CreativeMode.COMBINATORIAL.value, "combinatorial")

    def test_transformational(self):
        self.assertEqual(CreativeMode.TRANSFORMATIONAL.value, "transformational")

    def test_total_mode_count(self):
        self.assertEqual(len(list(CreativeMode)), 5)

    def test_modes_are_unique(self):
        values = [m.value for m in CreativeMode]
        self.assertEqual(len(values), len(set(values)))


class TestNoveltyType(unittest.TestCase):
    """Verify every NoveltyType variant and its string value."""

    def test_statistical(self):
        self.assertEqual(NoveltyType.STATISTICAL.value, "statistical")

    def test_structural(self):
        self.assertEqual(NoveltyType.STRUCTURAL.value, "structural")

    def test_conceptual(self):
        self.assertEqual(NoveltyType.CONCEPTUAL.value, "conceptual")

    def test_functional(self):
        self.assertEqual(NoveltyType.FUNCTIONAL.value, "functional")

    def test_total_type_count(self):
        self.assertEqual(len(list(NoveltyType)), 4)

    def test_types_are_unique(self):
        values = [t.value for t in NoveltyType]
        self.assertEqual(len(values), len(set(values)))


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestConceptDataclass(unittest.TestCase):
    """Test the Concept dataclass construction and defaults."""

    def test_minimal_creation(self):
        emb = np.ones(64)
        c = Concept(concept_id="c0", name="test", embedding=emb)
        self.assertEqual(c.concept_id, "c0")
        self.assertEqual(c.name, "test")
        np.testing.assert_array_equal(c.embedding, emb)
        self.assertEqual(c.attributes, {})
        self.assertEqual(c.relations, [])
        self.assertAlmostEqual(c.abstraction_level, 0.5)

    def test_full_creation(self):
        emb = np.zeros(64)
        c = Concept(
            concept_id="c1",
            name="complex",
            embedding=emb,
            attributes={"color": "red"},
            relations=[("is_a", "thing")],
            abstraction_level=0.8,
        )
        self.assertEqual(c.attributes, {"color": "red"})
        self.assertEqual(c.relations, [("is_a", "thing")])
        self.assertAlmostEqual(c.abstraction_level, 0.8)


class TestBlendDataclass(unittest.TestCase):
    """Test the Blend dataclass construction."""

    def test_creation(self):
        emb = np.ones(64)
        b = Blend(
            blend_id="b0",
            source_concepts=["c0", "c1"],
            blend_embedding=emb,
            emergent_properties={"key": "val"},
            novelty_score=0.7,
            coherence_score=0.8,
        )
        self.assertEqual(b.blend_id, "b0")
        self.assertEqual(b.source_concepts, ["c0", "c1"])
        np.testing.assert_array_equal(b.blend_embedding, emb)
        self.assertAlmostEqual(b.novelty_score, 0.7)
        self.assertAlmostEqual(b.coherence_score, 0.8)
        self.assertEqual(b.emergent_properties, {"key": "val"})


class TestImaginationDataclass(unittest.TestCase):
    """Test the Imagination dataclass construction."""

    def test_creation(self):
        state = np.zeros(64)
        traj = [np.zeros(64), np.ones(64)]
        im = Imagination(
            imagination_id="im0",
            initial_state=state,
            trajectory=traj,
            constraints={},
            plausibility=0.5,
            novelty=0.6,
            utility=0.7,
        )
        self.assertEqual(im.imagination_id, "im0")
        self.assertAlmostEqual(im.plausibility, 0.5)
        self.assertAlmostEqual(im.novelty, 0.6)
        self.assertAlmostEqual(im.utility, 0.7)
        self.assertEqual(len(im.trajectory), 2)


# ---------------------------------------------------------------------------
# CreativityEngine initialisation
# ---------------------------------------------------------------------------

class TestCreativityEngineInit(unittest.TestCase):
    """Test CreativityEngine default and custom initialization."""

    def test_default_init(self):
        engine = CreativityEngine(random_seed=42)
        self.assertEqual(engine.embedding_dim, 64)
        self.assertEqual(engine.max_concepts, 1000)
        self.assertAlmostEqual(engine.novelty_threshold, 0.3)
        self.assertAlmostEqual(engine.coherence_threshold, 0.4)
        self.assertAlmostEqual(engine.temperature, 1.0)
        self.assertAlmostEqual(engine.learning_rate, 0.05)
        self.assertEqual(len(engine.concepts), 0)
        self.assertEqual(len(engine.blends), 0)
        self.assertEqual(len(engine.imaginations), 0)
        self.assertEqual(engine.concept_counter, 0)
        self.assertEqual(engine.blend_counter, 0)
        self.assertEqual(engine.imagination_counter, 0)
        self.assertAlmostEqual(engine.relaxation_level, 0.0)

    def test_custom_init(self):
        engine = CreativityEngine(
            embedding_dim=32,
            max_concepts=500,
            novelty_threshold=0.5,
            coherence_threshold=0.6,
            temperature=2.0,
            learning_rate=0.1,
            random_seed=99,
        )
        self.assertEqual(engine.embedding_dim, 32)
        self.assertEqual(engine.max_concepts, 500)
        self.assertAlmostEqual(engine.novelty_threshold, 0.5)
        self.assertAlmostEqual(engine.coherence_threshold, 0.6)
        self.assertAlmostEqual(engine.temperature, 2.0)
        self.assertAlmostEqual(engine.learning_rate, 0.1)

    def test_weight_shapes(self):
        engine = CreativityEngine(embedding_dim=32, random_seed=42)
        self.assertEqual(engine.generator_weights.shape, (32, 32))
        self.assertEqual(engine.generator_bias.shape, (32,))
        self.assertEqual(engine.discriminator_weights.shape, (32, 1))
        self.assertEqual(engine.association_matrix.shape, (1000, 1000))

    def test_initial_statistics_counters(self):
        engine = CreativityEngine(random_seed=42)
        self.assertEqual(engine.total_generations, 0)
        self.assertEqual(engine.successful_blends, 0)
        self.assertEqual(engine.total_imaginations, 0)
        self.assertEqual(len(engine.exposure_history), 0)

    def test_xp_is_numpy_compatible(self):
        """Confirm xp imported from backend is usable (numpy or cupy)."""
        arr = xp.array([1.0, 2.0, 3.0])
        self.assertEqual(arr.shape, (3,))


# ---------------------------------------------------------------------------
# Adding concepts
# ---------------------------------------------------------------------------

class TestAddConcept(unittest.TestCase):
    """Test adding concepts to the engine."""

    def setUp(self):
        self.engine = CreativityEngine(embedding_dim=16, random_seed=42)

    def test_add_single_concept(self):
        np.random.seed(100)
        emb = np.random.randn(16)
        cid = self.engine.add_concept("test_concept", emb)
        self.assertEqual(cid, "concept_0")
        self.assertIn(cid, self.engine.concepts)
        self.assertEqual(self.engine.concepts[cid].name, "test_concept")

    def test_embedding_is_normalized(self):
        emb = np.ones(16) * 5.0
        cid = self.engine.add_concept("normed", emb)
        stored = self.engine.concepts[cid].embedding
        self.assertAlmostEqual(np.linalg.norm(stored), 1.0, places=5)

    def test_short_embedding_is_padded(self):
        emb = np.ones(8)
        cid = self.engine.add_concept("short", emb)
        stored = self.engine.concepts[cid].embedding
        self.assertEqual(len(stored), 16)

    def test_long_embedding_is_truncated(self):
        emb = np.ones(32)
        cid = self.engine.add_concept("long", emb)
        stored = self.engine.concepts[cid].embedding
        self.assertEqual(len(stored), 16)

    def test_concept_counter_increments(self):
        np.random.seed(100)
        cid1 = self.engine.add_concept("c1", np.random.randn(16))
        cid2 = self.engine.add_concept("c2", np.random.randn(16))
        self.assertEqual(cid1, "concept_0")
        self.assertEqual(cid2, "concept_1")
        self.assertEqual(self.engine.concept_counter, 2)

    def test_max_concepts_eviction(self):
        engine = CreativityEngine(embedding_dim=16, max_concepts=3, random_seed=42)
        np.random.seed(200)
        for i in range(4):
            engine.add_concept(f"c{i}", np.random.randn(16))
        self.assertEqual(len(engine.concepts), 3)

    def test_add_with_attributes_and_relations(self):
        np.random.seed(100)
        cid = self.engine.add_concept(
            "rich",
            np.random.randn(16),
            attributes={"size": 10, "color": "blue"},
            relations=[("is_a", "object")],
        )
        concept = self.engine.concepts[cid]
        self.assertEqual(concept.attributes["size"], 10)
        self.assertEqual(concept.attributes["color"], "blue")
        self.assertEqual(concept.relations, [("is_a", "object")])

    def test_exposure_history_updated(self):
        np.random.seed(100)
        self.engine.add_concept("h", np.random.randn(16))
        self.assertEqual(len(self.engine.exposure_history), 1)


# ---------------------------------------------------------------------------
# Conceptual blending
# ---------------------------------------------------------------------------

class TestConceptualBlending(unittest.TestCase):
    """Test all facets of conceptual blending."""

    def setUp(self):
        # Use zero thresholds so that blends are never rejected by score.
        self.engine = CreativityEngine(
            embedding_dim=16,
            novelty_threshold=0.0,
            coherence_threshold=0.0,
            random_seed=42,
        )
        np.random.seed(300)
        self.cid1 = self.engine.add_concept("alpha", np.random.randn(16))
        self.cid2 = self.engine.add_concept("beta", np.random.randn(16))
        self.cid3 = self.engine.add_concept("gamma", np.random.randn(16))

    def test_blend_empty_ids_returns_none(self):
        result = self.engine.blend([])
        self.assertIsNone(result)

    def test_blend_single_concept_returns_none(self):
        result = self.engine.blend([self.cid1])
        self.assertIsNone(result)

    def test_blend_invalid_ids_returns_none(self):
        result = self.engine.blend(["nonexistent_0", "nonexistent_1"])
        self.assertIsNone(result)

    def test_blend_average_mode(self):
        result = self.engine.blend([self.cid1, self.cid2], blend_mode="average")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Blend)
        self.assertEqual(len(result.source_concepts), 2)
        self.assertEqual(result.blend_embedding.shape, (16,))

    def test_blend_max_mode(self):
        result = self.engine.blend([self.cid1, self.cid2], blend_mode="max")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Blend)

    def test_blend_interpolate_mode_normalized(self):
        result = self.engine.blend([self.cid1, self.cid2], blend_mode="interpolate")
        self.assertIsNotNone(result)
        norm = np.linalg.norm(result.blend_embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_blend_emergent_mode(self):
        result = self.engine.blend([self.cid1, self.cid2], blend_mode="emergent")
        self.assertIsNotNone(result)

    def test_blend_unknown_mode_falls_back_to_average(self):
        """Unknown mode falls through to the else branch (same as average)."""
        result = self.engine.blend(
            [self.cid1, self.cid2], blend_mode="unknown_mode"
        )
        self.assertIsNotNone(result)

    def test_blend_with_custom_weights(self):
        result = self.engine.blend(
            [self.cid1, self.cid2],
            blend_mode="average",
            weights=[0.8, 0.2],
        )
        self.assertIsNotNone(result)

    def test_blend_three_concepts(self):
        result = self.engine.blend(
            [self.cid1, self.cid2, self.cid3], blend_mode="average"
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result.source_concepts), 3)

    def test_blend_increments_counter(self):
        self.engine.blend([self.cid1, self.cid2], blend_mode="average")
        self.engine.blend([self.cid1, self.cid3], blend_mode="average")
        self.assertEqual(self.engine.blend_counter, 2)

    def test_blend_stored_in_engine(self):
        result = self.engine.blend([self.cid1, self.cid2], blend_mode="average")
        self.assertIn(result.blend_id, self.engine.blends)

    def test_blend_emergent_properties_present(self):
        result = self.engine.blend([self.cid1, self.cid2], blend_mode="average")
        self.assertIn("abstraction_level", result.emergent_properties)
        self.assertIn("emergent_component_norm", result.emergent_properties)

    def test_blend_novelty_coherence_types(self):
        result = self.engine.blend([self.cid1, self.cid2], blend_mode="average")
        self.assertIsInstance(result.novelty_score, float)
        self.assertIsInstance(result.coherence_score, float)
        self.assertGreaterEqual(result.novelty_score, 0.0)
        self.assertGreaterEqual(result.coherence_score, 0.0)
        self.assertLessEqual(result.coherence_score, 1.0)

    def test_blend_rejected_by_high_thresholds(self):
        strict_engine = CreativityEngine(
            embedding_dim=16,
            novelty_threshold=0.99,
            coherence_threshold=0.99,
            random_seed=42,
        )
        np.random.seed(400)
        a = strict_engine.add_concept("a", np.random.randn(16))
        b = strict_engine.add_concept("b", np.random.randn(16))
        result = strict_engine.blend([a, b], blend_mode="average")
        self.assertIsNone(result)

    def test_successful_blends_stat(self):
        self.engine.blend([self.cid1, self.cid2], blend_mode="average")
        self.assertEqual(self.engine.successful_blends, 1)

    def test_all_blend_modes_produce_correct_shape(self):
        for mode in ("average", "max", "interpolate", "emergent"):
            result = self.engine.blend(
                [self.cid1, self.cid2], blend_mode=mode
            )
            self.assertIsNotNone(result, f"Blend mode '{mode}' returned None")
            self.assertEqual(
                result.blend_embedding.shape,
                (16,),
                f"Wrong shape for mode '{mode}'",
            )


# ---------------------------------------------------------------------------
# Imagination / mental simulation
# ---------------------------------------------------------------------------

class TestImagination(unittest.TestCase):
    """Test imagination / mental simulation."""

    def setUp(self):
        self.engine = CreativityEngine(embedding_dim=16, random_seed=42)

    def test_basic_imagination(self):
        np.random.seed(500)
        state = np.random.randn(16)
        result = self.engine.imagine(state, steps=5)
        self.assertIsInstance(result, Imagination)
        # trajectory = initial state + 5 steps
        self.assertEqual(len(result.trajectory), 6)
        self.assertEqual(result.imagination_id, "imagination_0")

    def test_imagination_stored(self):
        np.random.seed(500)
        state = np.random.randn(16)
        result = self.engine.imagine(state, steps=3)
        self.assertIn(result.imagination_id, self.engine.imaginations)

    def test_imagination_counter_increments(self):
        np.random.seed(500)
        self.engine.imagine(np.random.randn(16), steps=2)
        self.engine.imagine(np.random.randn(16), steps=2)
        self.assertEqual(self.engine.imagination_counter, 2)

    def test_plausibility_in_valid_range(self):
        np.random.seed(500)
        result = self.engine.imagine(np.random.randn(16), steps=5)
        self.assertGreaterEqual(result.plausibility, 0.0)
        self.assertLessEqual(result.plausibility, 1.0)

    def test_imagination_with_goal_state(self):
        np.random.seed(500)
        state = np.random.randn(16)
        goal = np.random.randn(16)
        result = self.engine.imagine(state, steps=5, goal_state=goal)
        # utility = 1 / (1 + goal_distance), always > 0
        self.assertGreater(result.utility, 0.0)

    def test_short_embedding_padded(self):
        state = np.ones(8)
        result = self.engine.imagine(state, steps=3)
        self.assertEqual(result.initial_state.shape[0], 16)

    def test_imagination_with_constraints(self):
        np.random.seed(500)
        target = np.random.randn(16)
        target /= np.linalg.norm(target)
        constraints = {"similar_to": target, "min_similarity": 0.5}
        result = self.engine.imagine(
            np.random.randn(16), steps=3, constraints=constraints
        )
        self.assertIsInstance(result, Imagination)

    def test_no_goal_utility_equals_novelty_times_plausibility(self):
        np.random.seed(500)
        result = self.engine.imagine(np.random.randn(16), steps=3)
        expected = result.novelty * result.plausibility
        self.assertAlmostEqual(result.utility, expected, places=10)

    def test_trajectory_states_are_normalized(self):
        np.random.seed(500)
        result = self.engine.imagine(np.random.randn(16), steps=5)
        # All trajectory states after the first should be normalised
        for state in result.trajectory[1:]:
            norm = np.linalg.norm(state)
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_total_imaginations_counter(self):
        np.random.seed(500)
        self.engine.imagine(np.random.randn(16), steps=2)
        self.engine.imagine(np.random.randn(16), steps=2)
        self.assertEqual(self.engine.total_imaginations, 2)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class TestGenerate(unittest.TestCase):
    """Test novel idea generation via the generate() method."""

    def setUp(self):
        self.engine = CreativityEngine(embedding_dim=16, random_seed=42)

    def test_generate_returns_correct_count(self):
        np.random.seed(600)
        results = self.engine.generate(n_samples=3)
        self.assertEqual(len(results), 3)

    def test_generate_tuple_structure(self):
        np.random.seed(600)
        results = self.engine.generate(n_samples=2)
        for emb, novelty, coherence in results:
            self.assertEqual(emb.shape, (16,))
            self.assertIsInstance(float(novelty), float)
            self.assertIsInstance(float(coherence), float)

    def test_generate_divergent_mode(self):
        np.random.seed(600)
        results = self.engine.generate(
            mode=CreativeMode.DIVERGENT, n_samples=3
        )
        self.assertEqual(len(results), 3)

    def test_generate_exploratory_mode(self):
        np.random.seed(600)
        seed = np.random.randn(16)
        results = self.engine.generate(
            seed=seed, mode=CreativeMode.EXPLORATORY, n_samples=3
        )
        self.assertEqual(len(results), 3)

    def test_generate_transformational_mode(self):
        np.random.seed(600)
        seed = np.random.randn(16)
        results = self.engine.generate(
            seed=seed, mode=CreativeMode.TRANSFORMATIONAL, n_samples=2
        )
        self.assertEqual(len(results), 2)

    def test_generate_convergent_mode(self):
        """Convergent mode has no special transformation, just default."""
        np.random.seed(600)
        seed = np.random.randn(16)
        results = self.engine.generate(
            seed=seed, mode=CreativeMode.CONVERGENT, n_samples=2
        )
        self.assertEqual(len(results), 2)

    def test_generate_with_seed_embedding(self):
        np.random.seed(600)
        seed = np.random.randn(16)
        results = self.engine.generate(seed=seed, n_samples=3)
        self.assertEqual(len(results), 3)

    def test_generate_sorted_by_combined_score(self):
        np.random.seed(600)
        results = self.engine.generate(n_samples=5)
        scores = [0.5 * n + 0.5 * c for _, n, c in results]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i] + 1e-12, scores[i + 1])

    def test_generate_increments_total_generations(self):
        np.random.seed(600)
        self.engine.generate(n_samples=5)
        self.assertEqual(self.engine.total_generations, 5)
        self.engine.generate(n_samples=3)
        self.assertEqual(self.engine.total_generations, 8)

    def test_generate_with_constraints(self):
        np.random.seed(600)
        target = np.random.randn(16)
        target /= np.linalg.norm(target)
        constraints = {"similar_to": target, "min_similarity": 0.3}
        results = self.engine.generate(n_samples=2, constraints=constraints)
        self.assertEqual(len(results), 2)

    def test_generate_output_normalized(self):
        np.random.seed(600)
        results = self.engine.generate(n_samples=3)
        for emb, _, _ in results:
            self.assertAlmostEqual(np.linalg.norm(emb), 1.0, places=5)


# ---------------------------------------------------------------------------
# Divergent thinking
# ---------------------------------------------------------------------------

class TestDivergentThink(unittest.TestCase):
    """Test divergent thinking for diverse solution generation."""

    def setUp(self):
        self.engine = CreativityEngine(embedding_dim=16, random_seed=42)

    def test_basic_divergent_think(self):
        np.random.seed(700)
        problem = np.random.randn(16)
        results = self.engine.divergent_think(problem, n_solutions=5)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 5)

    def test_divergent_solutions_are_tuples(self):
        np.random.seed(700)
        problem = np.random.randn(16)
        results = self.engine.divergent_think(problem, n_solutions=3)
        for item in results:
            self.assertEqual(len(item), 2)
            emb, score = item
            self.assertEqual(emb.shape, (16,))

    def test_divergent_output_normalized(self):
        np.random.seed(700)
        problem = np.random.randn(16)
        results = self.engine.divergent_think(problem, n_solutions=3)
        for emb, _ in results:
            self.assertAlmostEqual(np.linalg.norm(emb), 1.0, places=5)

    def test_diversity_weight_effect(self):
        """Higher diversity weight should not crash and still produce results."""
        np.random.seed(700)
        problem = np.random.randn(16)
        results_low = self.engine.divergent_think(
            problem, n_solutions=5, diversity_weight=0.1
        )
        np.random.seed(700)
        results_high = self.engine.divergent_think(
            problem, n_solutions=5, diversity_weight=0.9
        )
        self.assertGreater(len(results_low), 0)
        self.assertGreater(len(results_high), 0)


# ---------------------------------------------------------------------------
# Engine settings
# ---------------------------------------------------------------------------

class TestSettings(unittest.TestCase):
    """Test relaxation level and temperature setters."""

    def setUp(self):
        self.engine = CreativityEngine(random_seed=42)

    def test_set_relaxation_level(self):
        self.engine.set_relaxation_level(0.5)
        self.assertAlmostEqual(self.engine.relaxation_level, 0.5)

    def test_set_relaxation_clamped_high(self):
        self.engine.set_relaxation_level(1.5)
        self.assertAlmostEqual(self.engine.relaxation_level, 1.0)

    def test_set_relaxation_clamped_low(self):
        self.engine.set_relaxation_level(-0.5)
        self.assertAlmostEqual(self.engine.relaxation_level, 0.0)

    def test_set_temperature(self):
        self.engine.set_temperature(2.0)
        self.assertAlmostEqual(self.engine.temperature, 2.0)

    def test_set_temperature_minimum(self):
        self.engine.set_temperature(0.01)
        self.assertAlmostEqual(self.engine.temperature, 0.1)


# ---------------------------------------------------------------------------
# Novelty and coherence internals
# ---------------------------------------------------------------------------

class TestNoveltyAndCoherence(unittest.TestCase):
    """Test internal novelty and coherence computations."""

    def setUp(self):
        self.engine = CreativityEngine(embedding_dim=16, random_seed=42)

    def test_novelty_with_empty_history_is_one(self):
        np.random.seed(800)
        emb = np.random.randn(16)
        emb /= np.linalg.norm(emb)
        novelty = self.engine._compute_novelty(emb)
        self.assertAlmostEqual(novelty, 1.0)

    def test_novelty_of_identical_concept_is_near_zero(self):
        np.random.seed(800)
        emb = np.random.randn(16)
        emb /= np.linalg.norm(emb)
        self.engine.add_concept("same", emb)
        novelty = self.engine._compute_novelty(emb)
        self.assertLess(novelty, 0.05)

    def test_coherence_in_valid_range(self):
        np.random.seed(800)
        emb = np.random.randn(16)
        emb /= np.linalg.norm(emb)
        coherence = self.engine._compute_coherence(emb)
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)

    def test_novelty_nonnegative(self):
        np.random.seed(800)
        emb = np.random.randn(16)
        emb /= np.linalg.norm(emb)
        self.engine.add_concept("x", emb)
        novelty = self.engine._compute_novelty(emb)
        self.assertGreaterEqual(novelty, 0.0)


# ---------------------------------------------------------------------------
# Constraint application
# ---------------------------------------------------------------------------

class TestApplyConstraints(unittest.TestCase):
    """Test the internal constraint application logic."""

    def setUp(self):
        self.engine = CreativityEngine(embedding_dim=16, random_seed=42)

    def test_similarity_constraint_moves_toward_target(self):
        np.random.seed(900)
        target = np.random.randn(16)
        target /= np.linalg.norm(target)
        emb = -target  # maximally far away
        constraints = {"similar_to": target, "min_similarity": 0.5}
        result = self.engine._apply_constraints(emb, constraints)
        orig_sim = np.dot(emb / (np.linalg.norm(emb) + 1e-8), target)
        new_sim = np.dot(result, target)
        self.assertGreater(new_sim, orig_sim)

    def test_dissimilarity_constraint_pushes_away(self):
        np.random.seed(900)
        target = np.random.randn(16)
        target /= np.linalg.norm(target)
        emb = target.copy()
        constraints = {"different_from": target, "max_similarity": 0.3}
        result = self.engine._apply_constraints(emb, constraints)
        new_sim = np.dot(result, target)
        self.assertLess(new_sim, 1.0)

    def test_constraint_output_is_normalized(self):
        np.random.seed(900)
        target = np.random.randn(16)
        target /= np.linalg.norm(target)
        emb = np.random.randn(16)
        constraints = {"similar_to": target, "min_similarity": 0.5}
        result = self.engine._apply_constraints(emb, constraints)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=5)

    def test_empty_constraints_still_normalizes(self):
        np.random.seed(900)
        emb = np.random.randn(16) * 5.0
        result = self.engine._apply_constraints(emb, {})
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=5)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestGetStats(unittest.TestCase):
    """Test get_stats() reporting."""

    def setUp(self):
        self.engine = CreativityEngine(
            embedding_dim=16,
            novelty_threshold=0.0,
            coherence_threshold=0.0,
            random_seed=42,
        )

    def test_initial_stats(self):
        stats = self.engine.get_stats()
        self.assertEqual(stats["total_concepts"], 0)
        self.assertEqual(stats["total_blends"], 0)
        self.assertEqual(stats["total_imaginations"], 0)
        self.assertEqual(stats["total_generations"], 0)
        self.assertEqual(stats["successful_blends"], 0)
        self.assertAlmostEqual(stats["relaxation_level"], 0.0)
        self.assertAlmostEqual(stats["temperature"], 1.0)
        self.assertAlmostEqual(stats["novelty_threshold"], 0.0)
        self.assertAlmostEqual(stats["coherence_threshold"], 0.0)
        self.assertEqual(stats["exposure_history_size"], 0)

    def test_stats_after_operations(self):
        np.random.seed(1000)
        cid1 = self.engine.add_concept("a", np.random.randn(16))
        cid2 = self.engine.add_concept("b", np.random.randn(16))
        self.engine.blend([cid1, cid2])
        self.engine.generate(n_samples=3)
        self.engine.imagine(np.random.randn(16), steps=2)

        stats = self.engine.get_stats()
        self.assertEqual(stats["total_concepts"], 2)
        self.assertEqual(stats["total_imaginations"], 1)
        self.assertEqual(stats["total_generations"], 3)
        self.assertGreater(stats["exposure_history_size"], 0)

    def test_stats_keys(self):
        stats = self.engine.get_stats()
        expected_keys = {
            "total_concepts",
            "total_blends",
            "total_imaginations",
            "total_generations",
            "successful_blends",
            "relaxation_level",
            "temperature",
            "novelty_threshold",
            "coherence_threshold",
            "exposure_history_size",
        }
        self.assertEqual(set(stats.keys()), expected_keys)


# ---------------------------------------------------------------------------
# Serialization / deserialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization(unittest.TestCase):
    """Test serialize() and deserialize() round-trip fidelity."""

    def setUp(self):
        self.engine = CreativityEngine(
            embedding_dim=16,
            max_concepts=100,
            novelty_threshold=0.0,
            coherence_threshold=0.0,
            random_seed=42,
        )
        np.random.seed(1100)
        self.cid1 = self.engine.add_concept(
            "alpha", np.random.randn(16), attributes={"size": 5}
        )
        self.cid2 = self.engine.add_concept(
            "beta", np.random.randn(16), attributes={"size": 10}
        )
        self.engine.blend([self.cid1, self.cid2], blend_mode="average")
        self.engine.set_relaxation_level(0.3)

    def test_serialize_has_expected_keys(self):
        data = self.engine.serialize()
        expected_keys = {
            "embedding_dim",
            "max_concepts",
            "novelty_threshold",
            "coherence_threshold",
            "temperature",
            "learning_rate",
            "relaxation_level",
            "concepts",
            "blends",
            "generator_weights",
            "generator_bias",
            "stats",
        }
        self.assertEqual(set(data.keys()), expected_keys)

    def test_serialize_scalar_values(self):
        data = self.engine.serialize()
        self.assertEqual(data["embedding_dim"], 16)
        self.assertEqual(data["max_concepts"], 100)
        self.assertAlmostEqual(data["relaxation_level"], 0.3)
        self.assertAlmostEqual(data["temperature"], 1.0)
        self.assertAlmostEqual(data["learning_rate"], 0.05)

    def test_serialize_concepts_structure(self):
        data = self.engine.serialize()
        self.assertEqual(len(data["concepts"]), 2)
        for cid, cdata in data["concepts"].items():
            self.assertIn("concept_id", cdata)
            self.assertIn("name", cdata)
            self.assertIn("embedding", cdata)
            self.assertIsInstance(cdata["embedding"], list)
            self.assertEqual(len(cdata["embedding"]), 16)

    def test_serialize_blends_structure(self):
        data = self.engine.serialize()
        self.assertGreaterEqual(len(data["blends"]), 1)
        for bid, bdata in data["blends"].items():
            self.assertIn("blend_id", bdata)
            self.assertIn("source_concepts", bdata)
            self.assertIn("blend_embedding", bdata)
            self.assertIsInstance(bdata["blend_embedding"], list)

    def test_serialize_weights_are_lists(self):
        data = self.engine.serialize()
        self.assertIsInstance(data["generator_weights"], list)
        self.assertIsInstance(data["generator_bias"], list)

    def test_roundtrip_scalars(self):
        data = self.engine.serialize()
        restored = CreativityEngine.deserialize(data)
        self.assertEqual(restored.embedding_dim, self.engine.embedding_dim)
        self.assertEqual(restored.max_concepts, self.engine.max_concepts)
        self.assertAlmostEqual(
            restored.novelty_threshold, self.engine.novelty_threshold
        )
        self.assertAlmostEqual(
            restored.coherence_threshold, self.engine.coherence_threshold
        )
        self.assertAlmostEqual(restored.temperature, self.engine.temperature)
        self.assertAlmostEqual(restored.learning_rate, self.engine.learning_rate)
        self.assertAlmostEqual(
            restored.relaxation_level, self.engine.relaxation_level
        )

    def test_roundtrip_concepts(self):
        data = self.engine.serialize()
        restored = CreativityEngine.deserialize(data)
        self.assertEqual(len(restored.concepts), len(self.engine.concepts))
        for cid in self.engine.concepts:
            self.assertIn(cid, restored.concepts)
            np.testing.assert_array_almost_equal(
                restored.concepts[cid].embedding,
                self.engine.concepts[cid].embedding,
            )
            self.assertEqual(
                restored.concepts[cid].name,
                self.engine.concepts[cid].name,
            )

    def test_roundtrip_blends(self):
        data = self.engine.serialize()
        restored = CreativityEngine.deserialize(data)
        self.assertEqual(len(restored.blends), len(self.engine.blends))
        for bid in self.engine.blends:
            self.assertIn(bid, restored.blends)
            np.testing.assert_array_almost_equal(
                restored.blends[bid].blend_embedding,
                self.engine.blends[bid].blend_embedding,
            )

    def test_roundtrip_generator_weights(self):
        data = self.engine.serialize()
        restored = CreativityEngine.deserialize(data)
        np.testing.assert_array_almost_equal(
            restored.generator_weights, self.engine.generator_weights
        )
        np.testing.assert_array_almost_equal(
            restored.generator_bias, self.engine.generator_bias
        )

    def test_deserialized_engine_is_functional(self):
        """The restored engine can generate, blend, and imagine."""
        data = self.engine.serialize()
        restored = CreativityEngine.deserialize(data)

        np.random.seed(1200)
        new_cid = restored.add_concept("new_concept", np.random.randn(16))
        self.assertIn(new_cid, restored.concepts)

        results = restored.generate(n_samples=2)
        self.assertEqual(len(results), 2)

        result = restored.imagine(np.random.randn(16), steps=3)
        self.assertIsInstance(result, Imagination)


# ---------------------------------------------------------------------------
# Creative recombination and Hebbian learning
# ---------------------------------------------------------------------------

class TestCreativeRecombination(unittest.TestCase):
    """Test that blending triggers Hebbian learning updates."""

    def setUp(self):
        self.engine = CreativityEngine(
            embedding_dim=16,
            novelty_threshold=0.0,
            coherence_threshold=0.0,
            random_seed=42,
        )

    def test_blend_updates_association_matrix(self):
        np.random.seed(1300)
        cid1 = self.engine.add_concept("a", np.random.randn(16))
        cid2 = self.engine.add_concept("b", np.random.randn(16))
        initial_sum = self.engine.association_matrix.sum()
        self.engine.blend([cid1, cid2], blend_mode="average")
        after_sum = self.engine.association_matrix.sum()
        self.assertGreater(after_sum, initial_sum)

    def test_blend_updates_generator_weights(self):
        np.random.seed(1300)
        cid1 = self.engine.add_concept("a", np.random.randn(16))
        cid2 = self.engine.add_concept("b", np.random.randn(16))
        initial_weights = self.engine.generator_weights.copy()
        self.engine.blend([cid1, cid2], blend_mode="average")
        self.assertFalse(
            np.allclose(self.engine.generator_weights, initial_weights)
        )

    def test_repeated_blends_strengthen_associations(self):
        np.random.seed(1300)
        cid1 = self.engine.add_concept("a", np.random.randn(16))
        cid2 = self.engine.add_concept("b", np.random.randn(16))
        self.engine.blend([cid1, cid2], blend_mode="average")
        first_sum = self.engine.association_matrix.sum()
        self.engine.blend([cid1, cid2], blend_mode="max")
        second_sum = self.engine.association_matrix.sum()
        self.assertGreater(second_sum, first_sum)


# ---------------------------------------------------------------------------
# Plausibility evaluation
# ---------------------------------------------------------------------------

class TestPlausibility(unittest.TestCase):
    """Test the plausibility evaluation of trajectories."""

    def setUp(self):
        self.engine = CreativityEngine(embedding_dim=16, random_seed=42)

    def test_single_state_trajectory(self):
        traj = [np.random.randn(16)]
        p = self.engine._evaluate_plausibility(traj)
        self.assertAlmostEqual(p, 1.0)

    def test_smooth_trajectory_plausible(self):
        np.random.seed(1400)
        base = np.random.randn(16)
        base /= np.linalg.norm(base)
        traj = [base + 0.01 * i * np.ones(16) for i in range(5)]
        p = self.engine._evaluate_plausibility(traj)
        self.assertGreater(p, 0.0)
        self.assertLessEqual(p, 1.0)


if __name__ == "__main__":
    unittest.main()
