"""
Comprehensive tests for the VectorStore class (database/vector_store.py).

Tests cover initialization, episode CRUD, concept CRUD, similarity search,
metadata serialization, embedding dimension handling, statistics, and cleanup.

All tests are deterministic and pass reliably whether or not ChromaDB is
installed, because VectorStore falls back to an in-memory brute-force
implementation when ChromaDB is unavailable.
"""

import os
import sys
import shutil
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.database.vector_store import (
    VectorStore,
    VectorSearchResult,
    CHROMADB_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_embedding(dim, seed=0):
    """Return a deterministic float32 embedding of the given dimension."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    # Normalize so cosine similarity is meaningful
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestVectorStoreInitialization(unittest.TestCase):
    """Test VectorStore construction and initial state."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_init_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_default_embedding_dim(self):
        vs = VectorStore(data_dir=self.tmp_dir)
        self.assertEqual(vs.embedding_dim, 128)
        vs.close()

    def test_custom_embedding_dim(self):
        vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=64)
        self.assertEqual(vs.embedding_dim, 64)
        vs.close()

    def test_data_dir_created(self):
        sub = os.path.join(self.tmp_dir, "nested", "store")
        vs = VectorStore(data_dir=sub)
        self.assertTrue(os.path.isdir(sub))
        vs.close()

    def test_using_chromadb_flag(self):
        """_using_chromadb should match CHROMADB_AVAILABLE (unless init fails)."""
        vs = VectorStore(data_dir=self.tmp_dir)
        if CHROMADB_AVAILABLE:
            self.assertTrue(vs._using_chromadb)
        else:
            self.assertFalse(vs._using_chromadb)
        vs.close()

    def test_fallback_has_empty_dicts(self):
        """When using the in-memory fallback, internal dicts should start empty."""
        vs = VectorStore(data_dir=self.tmp_dir)
        if not vs._using_chromadb:
            self.assertEqual(len(vs._episodes), 0)
            self.assertEqual(len(vs._concepts), 0)
        vs.close()

    def test_initial_counts_are_zero(self):
        vs = VectorStore(data_dir=self.tmp_dir)
        self.assertEqual(vs.count_episodes(), 0)
        self.assertEqual(vs.count_concepts(), 0)
        vs.close()


# ---------------------------------------------------------------------------
# Episode operations
# ---------------------------------------------------------------------------

class TestAddEpisode(unittest.TestCase):
    """Test adding episodes to the VectorStore."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_ep_")
        self.dim = 32
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=self.dim)

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_add_single_episode(self):
        emb = _random_embedding(self.dim, seed=1)
        meta = {"timestamp": 1000.0, "context": "test"}
        result = self.vs.add_episode("ep_001", emb, meta)
        self.assertTrue(result)
        self.assertEqual(self.vs.count_episodes(), 1)

    def test_add_multiple_episodes(self):
        for i in range(5):
            emb = _random_embedding(self.dim, seed=i + 10)
            self.vs.add_episode(f"ep_{i:03d}", emb, {"idx": i})
        self.assertEqual(self.vs.count_episodes(), 5)

    def test_upsert_overwrites_existing(self):
        emb1 = _random_embedding(self.dim, seed=20)
        emb2 = _random_embedding(self.dim, seed=21)
        self.vs.add_episode("ep_dup", emb1, {"version": 1})
        self.vs.add_episode("ep_dup", emb2, {"version": 2})
        # Count should still be 1 (upsert, not duplicate)
        self.assertEqual(self.vs.count_episodes(), 1)

    def test_add_episode_pads_short_embedding(self):
        """Embeddings shorter than embedding_dim should be zero-padded."""
        short = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = self.vs.add_episode("ep_short", short, {"note": "short"})
        self.assertTrue(result)
        retrieved = self.vs.get_episode("ep_short")
        self.assertIsNotNone(retrieved)
        emb_out, _ = retrieved
        self.assertEqual(len(emb_out.flatten()), self.dim)

    def test_add_episode_truncates_long_embedding(self):
        """Embeddings longer than embedding_dim should be truncated."""
        long_emb = np.ones(self.dim * 2, dtype=np.float32)
        result = self.vs.add_episode("ep_long", long_emb, {"note": "long"})
        self.assertTrue(result)
        retrieved = self.vs.get_episode("ep_long")
        self.assertIsNotNone(retrieved)
        emb_out, _ = retrieved
        self.assertEqual(len(emb_out.flatten()), self.dim)

    def test_add_episode_exact_dim(self):
        """Embeddings matching embedding_dim should pass through unchanged."""
        emb = _random_embedding(self.dim, seed=30)
        self.vs.add_episode("ep_exact", emb, {})
        retrieved = self.vs.get_episode("ep_exact")
        self.assertIsNotNone(retrieved)
        emb_out, _ = retrieved
        np.testing.assert_allclose(emb_out.flatten(), emb, atol=1e-6)

    def test_add_episode_with_complex_metadata(self):
        """Metadata with nested structures should be handled."""
        meta = {
            "timestamp": 42.5,
            "tags": ["a", "b"],
            "nested": {"x": 1, "y": 2},
            "flag": True,
            "count": 7,
            "score": 0.95,
            "empty_val": None,
        }
        result = self.vs.add_episode(
            "ep_complex", _random_embedding(self.dim, seed=40), meta
        )
        self.assertTrue(result)
        self.assertEqual(self.vs.count_episodes(), 1)

    def test_add_episode_with_numpy_metadata(self):
        """Metadata containing numpy arrays and scalars should serialize."""
        meta = {
            "vec": np.array([1.0, 2.0, 3.0]),
            "np_int": np.int64(42),
            "np_float": np.float64(3.14),
            "np_bool": np.bool_(True),
        }
        result = self.vs.add_episode(
            "ep_numpy", _random_embedding(self.dim, seed=50), meta
        )
        self.assertTrue(result)


class TestGetEpisode(unittest.TestCase):
    """Test retrieving a single episode by ID."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_get_ep_")
        self.dim = 16
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=self.dim)

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_existing_episode(self):
        emb = _random_embedding(self.dim, seed=60)
        self.vs.add_episode("ep_get", emb, {"label": "hello"})
        result = self.vs.get_episode("ep_get")
        self.assertIsNotNone(result)
        emb_out, meta_out = result
        self.assertEqual(len(emb_out.flatten()), self.dim)

    def test_get_nonexistent_episode(self):
        result = self.vs.get_episode("does_not_exist")
        self.assertIsNone(result)

    def test_get_preserves_simple_metadata(self):
        meta = {"label": "test", "score": 0.75, "count": 3, "active": True}
        self.vs.add_episode("ep_meta", _random_embedding(self.dim, seed=61), meta)
        result = self.vs.get_episode("ep_meta")
        self.assertIsNotNone(result)
        _, meta_out = result
        self.assertEqual(meta_out["label"], "test")
        self.assertAlmostEqual(meta_out["score"], 0.75, places=2)
        self.assertEqual(meta_out["count"], 3)


class TestDeleteEpisode(unittest.TestCase):
    """Test deleting episodes."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_del_ep_")
        self.dim = 16
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=self.dim)

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_delete_existing_episode(self):
        self.vs.add_episode("ep_del", _random_embedding(self.dim, seed=70), {})
        self.assertEqual(self.vs.count_episodes(), 1)
        result = self.vs.delete_episode("ep_del")
        self.assertTrue(result)
        self.assertEqual(self.vs.count_episodes(), 0)

    def test_delete_nonexistent_episode(self):
        result = self.vs.delete_episode("ghost")
        if not self.vs._using_chromadb:
            # Fallback returns False for missing keys
            self.assertFalse(result)

    def test_delete_then_get_returns_none(self):
        self.vs.add_episode("ep_vanish", _random_embedding(self.dim, seed=71), {})
        self.vs.delete_episode("ep_vanish")
        self.assertIsNone(self.vs.get_episode("ep_vanish"))


# ---------------------------------------------------------------------------
# Episode search
# ---------------------------------------------------------------------------

class TestSearchEpisodes(unittest.TestCase):
    """Test similarity search over episodes."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_search_ep_")
        self.dim = 16
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=self.dim)
        # Populate with deterministic embeddings
        self.seeds = list(range(100, 110))
        for i, seed in enumerate(self.seeds):
            emb = _random_embedding(self.dim, seed=seed)
            self.vs.add_episode(f"ep_{i:03d}", emb, {"idx": i})

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_search_returns_list(self):
        query = _random_embedding(self.dim, seed=100)
        results = self.vs.search_episodes(query, n_results=5)
        self.assertIsInstance(results, list)

    def test_search_returns_vector_search_results(self):
        query = _random_embedding(self.dim, seed=100)
        results = self.vs.search_episodes(query, n_results=3)
        for r in results:
            self.assertIsInstance(r, VectorSearchResult)

    def test_search_result_fields(self):
        query = _random_embedding(self.dim, seed=100)
        results = self.vs.search_episodes(query, n_results=1)
        self.assertGreaterEqual(len(results), 1)
        r = results[0]
        self.assertIsInstance(r.id, str)
        self.assertIsInstance(r.embedding, np.ndarray)
        self.assertIsInstance(r.metadata, dict)
        self.assertTrue(isinstance(r.distance, (float, np.floating)),
                        f"distance should be float-like, got {type(r.distance)}")
        self.assertTrue(isinstance(r.similarity, (float, np.floating)),
                        f"similarity should be float-like, got {type(r.similarity)}")

    def test_search_exact_match_is_most_similar(self):
        """Querying with the exact embedding should return it as the top hit."""
        query = _random_embedding(self.dim, seed=100)  # Same seed as ep_000
        results = self.vs.search_episodes(query, n_results=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].id, "ep_000")

    def test_search_respects_n_results(self):
        query = _random_embedding(self.dim, seed=105)
        results = self.vs.search_episodes(query, n_results=3)
        self.assertLessEqual(len(results), 3)

    def test_search_on_empty_store(self):
        tmp2 = tempfile.mkdtemp(prefix="atlas_vs_empty_")
        vs2 = VectorStore(data_dir=tmp2, embedding_dim=self.dim)
        query = _random_embedding(self.dim, seed=999)
        results = vs2.search_episodes(query, n_results=5)
        self.assertEqual(len(results), 0)
        vs2.close()
        shutil.rmtree(tmp2, ignore_errors=True)

    def test_search_similarity_between_zero_and_one(self):
        query = _random_embedding(self.dim, seed=103)
        results = self.vs.search_episodes(query, n_results=5)
        for r in results:
            self.assertGreaterEqual(r.similarity, -0.1)  # Allow small numerical slack
            self.assertLessEqual(r.similarity, 1.1)

    def test_search_distance_non_negative(self):
        query = _random_embedding(self.dim, seed=104)
        results = self.vs.search_episodes(query, n_results=5)
        for r in results:
            self.assertGreaterEqual(r.distance, -0.1)

    def test_search_pads_short_query(self):
        """A query shorter than embedding_dim should still work (padded)."""
        short_query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = self.vs.search_episodes(short_query, n_results=3)
        self.assertIsInstance(results, list)

    def test_search_truncates_long_query(self):
        """A query longer than embedding_dim should still work (truncated)."""
        long_query = np.ones(self.dim * 3, dtype=np.float32)
        results = self.vs.search_episodes(long_query, n_results=3)
        self.assertIsInstance(results, list)

    def test_search_results_sorted_by_similarity(self):
        """In fallback mode, results should be sorted by descending similarity."""
        if not self.vs._using_chromadb:
            query = _random_embedding(self.dim, seed=100)
            results = self.vs.search_episodes(query, n_results=10)
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i].similarity, results[i + 1].similarity)


# ---------------------------------------------------------------------------
# Concept operations
# ---------------------------------------------------------------------------

class TestAddConcept(unittest.TestCase):
    """Test adding concepts to the VectorStore."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_concept_")
        self.dim = 24
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=self.dim)

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_add_single_concept(self):
        emb = _random_embedding(self.dim, seed=200)
        result = self.vs.add_concept("color_red", emb, {"category": "color"})
        self.assertTrue(result)
        self.assertEqual(self.vs.count_concepts(), 1)

    def test_add_multiple_concepts(self):
        names = ["cat", "dog", "bird", "fish"]
        for i, name in enumerate(names):
            emb = _random_embedding(self.dim, seed=210 + i)
            self.vs.add_concept(name, emb, {"type": "animal"})
        self.assertEqual(self.vs.count_concepts(), 4)

    def test_upsert_concept(self):
        emb1 = _random_embedding(self.dim, seed=220)
        emb2 = _random_embedding(self.dim, seed=221)
        self.vs.add_concept("gravity", emb1, {"version": 1})
        self.vs.add_concept("gravity", emb2, {"version": 2})
        self.assertEqual(self.vs.count_concepts(), 1)

    def test_add_concept_pads_short_embedding(self):
        short = np.array([1.0, 2.0], dtype=np.float32)
        result = self.vs.add_concept("tiny", short, {})
        self.assertTrue(result)
        retrieved = self.vs.get_concept("tiny")
        self.assertIsNotNone(retrieved)
        emb_out, _ = retrieved
        self.assertEqual(len(emb_out.flatten()), self.dim)

    def test_add_concept_truncates_long_embedding(self):
        long_emb = np.ones(self.dim * 2, dtype=np.float32)
        result = self.vs.add_concept("big", long_emb, {})
        self.assertTrue(result)
        retrieved = self.vs.get_concept("big")
        self.assertIsNotNone(retrieved)
        emb_out, _ = retrieved
        self.assertEqual(len(emb_out.flatten()), self.dim)

    def test_add_concept_with_metadata(self):
        meta = {
            "category": "physics",
            "activation_count": 42,
            "strength": 0.8,
            "attributes": ["mass", "acceleration"],
        }
        result = self.vs.add_concept(
            "force", _random_embedding(self.dim, seed=230), meta
        )
        self.assertTrue(result)


class TestGetConcept(unittest.TestCase):
    """Test retrieving a single concept by name."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_get_con_")
        self.dim = 16
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=self.dim)

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_existing_concept(self):
        emb = _random_embedding(self.dim, seed=240)
        self.vs.add_concept("light", emb, {"wavelength": 550})
        result = self.vs.get_concept("light")
        self.assertIsNotNone(result)
        emb_out, meta_out = result
        self.assertEqual(len(emb_out.flatten()), self.dim)

    def test_get_nonexistent_concept(self):
        result = self.vs.get_concept("dark_matter")
        self.assertIsNone(result)


class TestDeleteConcept(unittest.TestCase):
    """Test deleting concepts."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_del_con_")
        self.dim = 16
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=self.dim)

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_delete_existing_concept(self):
        self.vs.add_concept("temp", _random_embedding(self.dim, seed=250), {})
        self.assertEqual(self.vs.count_concepts(), 1)
        result = self.vs.delete_concept("temp")
        self.assertTrue(result)
        self.assertEqual(self.vs.count_concepts(), 0)

    def test_delete_nonexistent_concept(self):
        result = self.vs.delete_concept("phantom")
        if not self.vs._using_chromadb:
            self.assertFalse(result)

    def test_delete_then_get_returns_none(self):
        self.vs.add_concept("ephemeral", _random_embedding(self.dim, seed=251), {})
        self.vs.delete_concept("ephemeral")
        self.assertIsNone(self.vs.get_concept("ephemeral"))


class TestGetAllConceptNames(unittest.TestCase):
    """Test listing all concept names."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_names_")
        self.dim = 16
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=self.dim)

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_empty_store(self):
        names = self.vs.get_all_concept_names()
        self.assertEqual(names, [])

    def test_populated_store(self):
        for name in ["alpha", "beta", "gamma"]:
            self.vs.add_concept(name, _random_embedding(self.dim, seed=260), {})
        names = self.vs.get_all_concept_names()
        self.assertEqual(sorted(names), ["alpha", "beta", "gamma"])


# ---------------------------------------------------------------------------
# Concept search
# ---------------------------------------------------------------------------

class TestSearchConcepts(unittest.TestCase):
    """Test similarity search over concepts."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_search_con_")
        self.dim = 16
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=self.dim)
        # Populate with deterministic embeddings
        self.concept_names = ["apple", "banana", "cherry", "date", "elderberry"]
        for i, name in enumerate(self.concept_names):
            emb = _random_embedding(self.dim, seed=300 + i)
            self.vs.add_concept(name, emb, {"idx": i, "type": "fruit"})

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_search_returns_list(self):
        query = _random_embedding(self.dim, seed=300)
        results = self.vs.search_concepts(query, n_results=3)
        self.assertIsInstance(results, list)

    def test_search_returns_vector_search_results(self):
        query = _random_embedding(self.dim, seed=300)
        results = self.vs.search_concepts(query, n_results=3)
        for r in results:
            self.assertIsInstance(r, VectorSearchResult)

    def test_search_exact_match_is_most_similar(self):
        """Querying with the same embedding as 'apple' should return it first."""
        query = _random_embedding(self.dim, seed=300)  # Same as "apple"
        results = self.vs.search_concepts(query, n_results=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].id, "apple")

    def test_search_respects_n_results(self):
        query = _random_embedding(self.dim, seed=302)
        results = self.vs.search_concepts(query, n_results=2)
        self.assertLessEqual(len(results), 2)

    def test_search_on_empty_concepts(self):
        tmp2 = tempfile.mkdtemp(prefix="atlas_vs_empty_con_")
        vs2 = VectorStore(data_dir=tmp2, embedding_dim=self.dim)
        query = _random_embedding(self.dim, seed=999)
        results = vs2.search_concepts(query, n_results=5)
        self.assertEqual(len(results), 0)
        vs2.close()
        shutil.rmtree(tmp2, ignore_errors=True)

    def test_search_result_metadata(self):
        query = _random_embedding(self.dim, seed=301)  # Same as "banana"
        results = self.vs.search_concepts(query, n_results=1)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("type", results[0].metadata)

    def test_search_results_sorted_by_similarity_fallback(self):
        """In fallback mode, results should be sorted by descending similarity."""
        if not self.vs._using_chromadb:
            query = _random_embedding(self.dim, seed=300)
            results = self.vs.search_concepts(query, n_results=5)
            for i in range(len(results) - 1):
                self.assertGreaterEqual(
                    results[i].similarity, results[i + 1].similarity
                )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestGetStats(unittest.TestCase):
    """Test get_stats() method."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_stats_")
        self.dim = 16
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=self.dim)

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_stats_returns_dict(self):
        stats = self.vs.get_stats()
        self.assertIsInstance(stats, dict)

    def test_stats_keys(self):
        stats = self.vs.get_stats()
        expected_keys = {
            "using_chromadb",
            "data_dir",
            "embedding_dim",
            "episode_count",
            "concept_count",
        }
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_stats_empty_store(self):
        stats = self.vs.get_stats()
        self.assertEqual(stats["episode_count"], 0)
        self.assertEqual(stats["concept_count"], 0)
        self.assertEqual(stats["embedding_dim"], self.dim)
        self.assertEqual(stats["data_dir"], str(self.vs.data_dir))

    def test_stats_using_chromadb_is_bool(self):
        stats = self.vs.get_stats()
        self.assertIsInstance(stats["using_chromadb"], bool)

    def test_stats_after_adding_data(self):
        for i in range(3):
            self.vs.add_episode(
                f"ep_{i}", _random_embedding(self.dim, seed=400 + i), {}
            )
        for i in range(2):
            self.vs.add_concept(
                f"con_{i}", _random_embedding(self.dim, seed=500 + i), {}
            )
        stats = self.vs.get_stats()
        self.assertEqual(stats["episode_count"], 3)
        self.assertEqual(stats["concept_count"], 2)

    def test_stats_after_deletion(self):
        self.vs.add_episode("ep_x", _random_embedding(self.dim, seed=600), {})
        self.vs.add_concept("con_x", _random_embedding(self.dim, seed=601), {})
        self.vs.delete_episode("ep_x")
        self.vs.delete_concept("con_x")
        stats = self.vs.get_stats()
        self.assertEqual(stats["episode_count"], 0)
        self.assertEqual(stats["concept_count"], 0)


# ---------------------------------------------------------------------------
# Metadata serialization / deserialization
# ---------------------------------------------------------------------------

class TestMetadataSerialization(unittest.TestCase):
    """Test the internal _serialize_metadata and _deserialize_metadata."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_ser_")
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=8)

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_primitive_types_pass_through(self):
        meta = {"s": "hello", "i": 42, "f": 3.14, "b": True}
        ser = self.vs._serialize_metadata(meta)
        self.assertEqual(ser["s"], "hello")
        self.assertEqual(ser["i"], 42)
        self.assertAlmostEqual(ser["f"], 3.14)
        self.assertEqual(ser["b"], True)

    def test_none_becomes_empty_string(self):
        meta = {"key": None}
        ser = self.vs._serialize_metadata(meta)
        self.assertEqual(ser["key"], "")

    def test_list_serialized_as_json(self):
        meta = {"tags": [1, 2, 3]}
        ser = self.vs._serialize_metadata(meta)
        self.assertIn("tags_json", ser)

    def test_dict_serialized_as_json(self):
        meta = {"nested": {"a": 1, "b": 2}}
        ser = self.vs._serialize_metadata(meta)
        self.assertIn("nested_json", ser)

    def test_numpy_array_serialized_as_json(self):
        meta = {"vec": np.array([1.0, 2.0, 3.0])}
        ser = self.vs._serialize_metadata(meta)
        self.assertIn("vec_json", ser)

    def test_numpy_scalar_converted(self):
        meta = {
            "np_int": np.int64(10),
            "np_float": np.float64(2.5),
            "np_bool": np.bool_(False),
        }
        ser = self.vs._serialize_metadata(meta)
        self.assertEqual(ser["np_int"], 10)
        self.assertAlmostEqual(ser["np_float"], 2.5)
        self.assertEqual(ser["np_bool"], False)

    def test_roundtrip_serialization(self):
        meta = {
            "name": "test",
            "count": 5,
            "tags": ["a", "b"],
            "info": {"nested": True},
        }
        ser = self.vs._serialize_metadata(meta)
        deser = self.vs._deserialize_metadata(ser)
        self.assertEqual(deser["name"], "test")
        self.assertEqual(deser["count"], 5)
        self.assertEqual(deser["tags"], ["a", "b"])
        self.assertEqual(deser["info"], {"nested": True})


# ---------------------------------------------------------------------------
# VectorSearchResult dataclass
# ---------------------------------------------------------------------------

class TestVectorSearchResult(unittest.TestCase):
    """Test the VectorSearchResult dataclass."""

    def test_creation(self):
        r = VectorSearchResult(
            id="test_001",
            embedding=np.zeros(8),
            metadata={"key": "value"},
            distance=0.3,
            similarity=0.7,
        )
        self.assertEqual(r.id, "test_001")
        self.assertEqual(r.metadata["key"], "value")
        self.assertAlmostEqual(r.distance, 0.3)
        self.assertAlmostEqual(r.similarity, 0.7)

    def test_embedding_is_array(self):
        r = VectorSearchResult(
            id="x",
            embedding=np.array([1.0, 2.0]),
            metadata={},
            distance=0.0,
            similarity=1.0,
        )
        self.assertIsInstance(r.embedding, np.ndarray)
        self.assertEqual(len(r.embedding), 2)


# ---------------------------------------------------------------------------
# Fallback brute-force search
# ---------------------------------------------------------------------------

class TestFallbackSearch(unittest.TestCase):
    """Test the in-memory fallback brute-force search directly."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_fb_")
        self.dim = 8
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=self.dim)

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_fallback_search_empty_storage(self):
        """Searching an empty dict should return an empty list."""
        results = self.vs._fallback_search({}, np.ones(self.dim), 5)
        self.assertEqual(results, [])

    def test_fallback_search_returns_correct_count(self):
        storage = {}
        for i in range(5):
            emb = _random_embedding(self.dim, seed=700 + i)
            storage[f"item_{i}"] = (emb, {"idx": i})
        query = _random_embedding(self.dim, seed=700)
        results = self.vs._fallback_search(storage, query, 3)
        self.assertLessEqual(len(results), 3)

    def test_fallback_search_sorted_descending(self):
        storage = {}
        for i in range(10):
            emb = _random_embedding(self.dim, seed=800 + i)
            storage[f"item_{i}"] = (emb, {"idx": i})
        query = _random_embedding(self.dim, seed=800)
        results = self.vs._fallback_search(storage, query, 10)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].similarity, results[i + 1].similarity)

    def test_fallback_search_identical_vector_has_highest_similarity(self):
        emb = _random_embedding(self.dim, seed=900)
        other = _random_embedding(self.dim, seed=901)
        storage = {
            "exact": (emb.copy(), {"match": True}),
            "other": (other, {"match": False}),
        }
        results = self.vs._fallback_search(storage, emb, 2)
        self.assertEqual(results[0].id, "exact")
        self.assertAlmostEqual(results[0].similarity, 1.0, places=5)

    def test_fallback_search_zero_norm_embedding(self):
        """A zero-vector should produce similarity 0 for all entries."""
        storage = {"a": (np.ones(self.dim, dtype=np.float32), {"name": "a"})}
        query = np.zeros(self.dim, dtype=np.float32)
        results = self.vs._fallback_search(storage, query, 1)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].similarity, 0.0)

    def test_fallback_search_zero_norm_stored(self):
        """A zero-vector stored should produce similarity 0."""
        storage = {"a": (np.zeros(self.dim, dtype=np.float32), {"name": "a"})}
        query = np.ones(self.dim, dtype=np.float32)
        results = self.vs._fallback_search(storage, query, 1)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].similarity, 0.0)


# ---------------------------------------------------------------------------
# Cleanup / close
# ---------------------------------------------------------------------------

class TestCloseAndCleanup(unittest.TestCase):
    """Test that close() and cleanup work without errors."""

    def test_close_without_error(self):
        tmp = tempfile.mkdtemp(prefix="atlas_vs_close_")
        vs = VectorStore(data_dir=tmp, embedding_dim=16)
        try:
            vs.close()
        except Exception as e:
            self.fail(f"close() raised {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_close_idempotent(self):
        """Calling close() multiple times should not raise."""
        tmp = tempfile.mkdtemp(prefix="atlas_vs_close2_")
        vs = VectorStore(data_dir=tmp, embedding_dim=16)
        vs.close()
        vs.close()
        vs.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def test_close_after_operations(self):
        tmp = tempfile.mkdtemp(prefix="atlas_vs_close3_")
        vs = VectorStore(data_dir=tmp, embedding_dim=16)
        vs.add_episode("ep1", _random_embedding(16, seed=1000), {"a": 1})
        vs.add_concept("c1", _random_embedding(16, seed=1001), {"b": 2})
        vs.search_episodes(_random_embedding(16, seed=1002), n_results=1)
        vs.search_concepts(_random_embedding(16, seed=1003), n_results=1)
        try:
            vs.close()
        except Exception as e:
            self.fail(f"close() after operations raised {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_tempdir_cleanup(self):
        """Verify that shutil.rmtree works on the data directory after close."""
        tmp = tempfile.mkdtemp(prefix="atlas_vs_cleanup_")
        vs = VectorStore(data_dir=tmp, embedding_dim=16)
        vs.add_episode("ep1", _random_embedding(16, seed=1010), {})
        vs.close()
        shutil.rmtree(tmp, ignore_errors=True)
        self.assertFalse(os.path.isdir(tmp))


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple operations."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_integ_")
        self.dim = 32
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=self.dim)

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_full_episode_lifecycle(self):
        """add -> get -> search -> delete -> get returns None."""
        emb = _random_embedding(self.dim, seed=1100)
        meta = {"context": "test_lifecycle"}

        # Add
        self.assertTrue(self.vs.add_episode("lc_ep", emb, meta))
        self.assertEqual(self.vs.count_episodes(), 1)

        # Get
        result = self.vs.get_episode("lc_ep")
        self.assertIsNotNone(result)

        # Search
        search_results = self.vs.search_episodes(emb, n_results=1)
        self.assertGreaterEqual(len(search_results), 1)
        self.assertEqual(search_results[0].id, "lc_ep")

        # Delete
        self.assertTrue(self.vs.delete_episode("lc_ep"))
        self.assertEqual(self.vs.count_episodes(), 0)

        # Verify gone
        self.assertIsNone(self.vs.get_episode("lc_ep"))

    def test_full_concept_lifecycle(self):
        """add -> get -> search -> delete -> get returns None."""
        emb = _random_embedding(self.dim, seed=1200)
        meta = {"category": "test"}

        # Add
        self.assertTrue(self.vs.add_concept("lc_con", emb, meta))
        self.assertEqual(self.vs.count_concepts(), 1)

        # Get
        result = self.vs.get_concept("lc_con")
        self.assertIsNotNone(result)

        # Search
        search_results = self.vs.search_concepts(emb, n_results=1)
        self.assertGreaterEqual(len(search_results), 1)
        self.assertEqual(search_results[0].id, "lc_con")

        # Delete
        self.assertTrue(self.vs.delete_concept("lc_con"))
        self.assertEqual(self.vs.count_concepts(), 0)

        # Verify gone
        self.assertIsNone(self.vs.get_concept("lc_con"))

    def test_mixed_episodes_and_concepts(self):
        """Episodes and concepts are stored in separate namespaces."""
        emb = _random_embedding(self.dim, seed=1300)
        self.vs.add_episode("shared_id", emb, {"type": "episode"})
        self.vs.add_concept("shared_id", emb, {"type": "concept"})

        self.assertEqual(self.vs.count_episodes(), 1)
        self.assertEqual(self.vs.count_concepts(), 1)

        ep_result = self.vs.get_episode("shared_id")
        con_result = self.vs.get_concept("shared_id")
        self.assertIsNotNone(ep_result)
        self.assertIsNotNone(con_result)

    def test_stats_reflect_all_operations(self):
        """Stats should correctly report counts after mixed operations."""
        for i in range(4):
            self.vs.add_episode(
                f"ep_{i}", _random_embedding(self.dim, seed=1400 + i), {}
            )
        for i in range(3):
            self.vs.add_concept(
                f"con_{i}", _random_embedding(self.dim, seed=1500 + i), {}
            )
        self.vs.delete_episode("ep_0")

        stats = self.vs.get_stats()
        self.assertEqual(stats["episode_count"], 3)
        self.assertEqual(stats["concept_count"], 3)
        self.assertEqual(stats["embedding_dim"], self.dim)

    def test_xp_backend_creates_compatible_embeddings(self):
        """Embeddings created with the xp backend should be storable."""
        emb = xp.random.RandomState(42).randn(self.dim).astype(xp.float32)
        # Convert to numpy if needed for the vector store
        emb_np = np.asarray(emb)
        result = self.vs.add_episode("xp_ep", emb_np, {"backend": "xp"})
        self.assertTrue(result)
        self.assertEqual(self.vs.count_episodes(), 1)

    def test_large_batch_add_and_search(self):
        """Add many items and verify search still works."""
        n_items = 50
        for i in range(n_items):
            emb = _random_embedding(self.dim, seed=2000 + i)
            self.vs.add_episode(f"batch_ep_{i:04d}", emb, {"batch_idx": i})

        self.assertEqual(self.vs.count_episodes(), n_items)

        # Search for one of them
        query = _random_embedding(self.dim, seed=2025)  # Same as batch_ep_0025
        results = self.vs.search_episodes(query, n_results=5)
        self.assertGreaterEqual(len(results), 1)
        # The exact match should be the top result
        self.assertEqual(results[0].id, "batch_ep_0025")


# ---------------------------------------------------------------------------
# Count methods
# ---------------------------------------------------------------------------

class TestCountMethods(unittest.TestCase):
    """Test count_episodes and count_concepts."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="atlas_vs_count_")
        self.dim = 8
        self.vs = VectorStore(data_dir=self.tmp_dir, embedding_dim=self.dim)

    def tearDown(self):
        self.vs.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_count_episodes_starts_at_zero(self):
        self.assertEqual(self.vs.count_episodes(), 0)

    def test_count_concepts_starts_at_zero(self):
        self.assertEqual(self.vs.count_concepts(), 0)

    def test_count_episodes_increments(self):
        for i in range(3):
            self.vs.add_episode(
                f"ep_{i}", _random_embedding(self.dim, seed=3000 + i), {}
            )
            self.assertEqual(self.vs.count_episodes(), i + 1)

    def test_count_concepts_increments(self):
        for i in range(3):
            self.vs.add_concept(
                f"con_{i}", _random_embedding(self.dim, seed=3100 + i), {}
            )
            self.assertEqual(self.vs.count_concepts(), i + 1)

    def test_count_episodes_after_delete(self):
        self.vs.add_episode("ep_a", _random_embedding(self.dim, seed=3200), {})
        self.vs.add_episode("ep_b", _random_embedding(self.dim, seed=3201), {})
        self.assertEqual(self.vs.count_episodes(), 2)
        self.vs.delete_episode("ep_a")
        self.assertEqual(self.vs.count_episodes(), 1)

    def test_count_concepts_after_delete(self):
        self.vs.add_concept("con_a", _random_embedding(self.dim, seed=3300), {})
        self.vs.add_concept("con_b", _random_embedding(self.dim, seed=3301), {})
        self.assertEqual(self.vs.count_concepts(), 2)
        self.vs.delete_concept("con_a")
        self.assertEqual(self.vs.count_concepts(), 1)


if __name__ == "__main__":
    unittest.main()
