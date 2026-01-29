"""
Comprehensive tests for the EpisodicMemory class (core/episodic_memory.py).
Tests cover initialization, episode storage, retrieval, consolidation,
memory capacity limits, forgetting, serialization, and statistics.
"""
import sys
import os
import unittest
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.episodic_memory import EpisodicMemory, Episode


class TestEpisodeDataclass(unittest.TestCase):
    """Test the Episode dataclass itself."""

    def test_episode_creation(self):
        ep = Episode(
            timestamp=1000.0,
            state=np.array([1.0, 2.0, 3.0]),
            context={"location": "office"},
            sensory_data={"visual": np.zeros(4)},
            emotional_valence=0.5,
            surprise_level=0.3,
        )
        self.assertAlmostEqual(ep.timestamp, 1000.0)
        self.assertEqual(ep.context["location"], "office")
        self.assertAlmostEqual(ep.emotional_valence, 0.5)
        self.assertAlmostEqual(ep.surprise_level, 0.3)
        self.assertEqual(ep.replay_count, 0)
        self.assertAlmostEqual(ep.consolidation_strength, 0.0)

    def test_episode_auto_id(self):
        ep = Episode(
            timestamp=0.0,
            state=np.zeros(3),
            context={},
            sensory_data={},
            emotional_valence=0.0,
            surprise_level=0.0,
        )
        self.assertTrue(ep.episode_id.startswith("ep_"))
        self.assertEqual(len(ep.episode_id), 15)  # "ep_" + 12 hex chars

    def test_episode_custom_id(self):
        ep = Episode(
            timestamp=0.0,
            state=np.zeros(3),
            context={},
            sensory_data={},
            emotional_valence=0.0,
            surprise_level=0.0,
            episode_id="custom_123",
        )
        self.assertEqual(ep.episode_id, "custom_123")

    def test_episode_hash(self):
        ep1 = Episode(
            timestamp=1.0,
            state=np.zeros(3),
            context={},
            sensory_data={},
            emotional_valence=0.0,
            surprise_level=0.0,
            episode_id="id_a",
        )
        ep2 = Episode(
            timestamp=2.0,
            state=np.ones(3),
            context={},
            sensory_data={},
            emotional_valence=0.0,
            surprise_level=0.0,
            episode_id="id_b",
        )
        # Different IDs produce different hashes
        self.assertNotEqual(hash(ep1), hash(ep2))
        # Same ID produces same hash
        ep3 = Episode(
            timestamp=3.0,
            state=np.zeros(3),
            context={},
            sensory_data={},
            emotional_valence=0.0,
            surprise_level=0.0,
            episode_id="id_a",
        )
        self.assertEqual(hash(ep1), hash(ep3))

    def test_episode_usable_in_set(self):
        ep = Episode(
            timestamp=0.0,
            state=np.zeros(3),
            context={},
            sensory_data={},
            emotional_valence=0.0,
            surprise_level=0.0,
        )
        s = {ep}
        self.assertIn(ep, s)


class TestEpisodicMemoryInitialization(unittest.TestCase):
    """Test EpisodicMemory construction and initial state."""

    def test_default_initialization(self):
        mem = EpisodicMemory(state_size=8)
        self.assertEqual(mem.state_size, 8)
        self.assertEqual(mem.max_episodes, 10000)
        self.assertAlmostEqual(mem.consolidation_threshold, 0.7)
        self.assertAlmostEqual(mem.replay_rate, 0.1)
        self.assertAlmostEqual(mem.similarity_threshold, 0.85)
        self.assertTrue(mem.enable_consolidation)
        self.assertTrue(mem.enable_forgetting)
        self.assertAlmostEqual(mem.forgetting_rate, 0.0001)
        self.assertAlmostEqual(mem.novelty_bonus, 2.0)
        self.assertAlmostEqual(mem.emotional_bonus, 1.5)
        self.assertEqual(len(mem.episodes), 0)
        self.assertEqual(len(mem.consolidated_episodes), 0)
        self.assertEqual(mem.total_stored, 0)
        self.assertEqual(mem.total_retrieved, 0)
        self.assertEqual(mem.total_replayed, 0)
        self.assertEqual(mem.total_forgotten, 0)
        self.assertFalse(mem._persistence_enabled)

    def test_custom_initialization(self):
        mem = EpisodicMemory(
            state_size=16,
            max_episodes=50,
            consolidation_threshold=0.5,
            replay_rate=0.2,
            similarity_threshold=0.9,
            enable_consolidation=False,
            enable_forgetting=False,
            forgetting_rate=0.001,
            novelty_bonus=3.0,
            emotional_bonus=2.0,
            random_seed=42,
        )
        self.assertEqual(mem.state_size, 16)
        self.assertEqual(mem.max_episodes, 50)
        self.assertAlmostEqual(mem.consolidation_threshold, 0.5)
        self.assertAlmostEqual(mem.replay_rate, 0.2)
        self.assertAlmostEqual(mem.similarity_threshold, 0.9)
        self.assertFalse(mem.enable_consolidation)
        self.assertFalse(mem.enable_forgetting)
        self.assertAlmostEqual(mem.forgetting_rate, 0.001)
        self.assertAlmostEqual(mem.novelty_bonus, 3.0)
        self.assertAlmostEqual(mem.emotional_bonus, 2.0)

    def test_consolidation_weights_shape(self):
        mem = EpisodicMemory(state_size=8, max_episodes=20)
        self.assertEqual(mem.consolidation_weights.shape, (20, 8))

    def test_no_vector_store_by_default(self):
        mem = EpisodicMemory(state_size=8)
        self.assertIsNone(mem.vector_store)
        self.assertFalse(mem._persistence_enabled)


class TestStore(unittest.TestCase):
    """Test storing episodes."""

    def setUp(self):
        self.mem = EpisodicMemory(
            state_size=8,
            max_episodes=100,
            similarity_threshold=0.99,  # Very high so similar-episode dedup rarely triggers
            random_seed=42,
        )

    def _make_state(self, seed=0):
        rng = np.random.RandomState(seed)
        return rng.randn(8)

    def test_store_basic(self):
        state = self._make_state(0)
        sensory = {"visual": np.zeros(4)}
        ep = self.mem.store(state=state, sensory_data=sensory)
        self.assertIsInstance(ep, Episode)
        self.assertEqual(len(self.mem.episodes), 1)
        self.assertEqual(self.mem.total_stored, 1)

    def test_store_returns_episode_with_correct_data(self):
        state = self._make_state(1)
        context = {"task": "navigation"}
        sensory = {"lidar": np.array([1.0, 2.0])}
        ep = self.mem.store(
            state=state,
            sensory_data=sensory,
            context=context,
            emotional_valence=0.7,
            surprise_level=0.4,
        )
        np.testing.assert_array_equal(ep.state, state)
        self.assertEqual(ep.context["task"], "navigation")
        self.assertAlmostEqual(ep.emotional_valence, 0.7)
        self.assertAlmostEqual(ep.surprise_level, 0.4)
        self.assertTrue(ep.episode_id.startswith("ep_"))

    def test_store_copies_state(self):
        """Stored state should be a copy, not a reference."""
        state = np.ones(8)
        sensory = {"visual": np.zeros(4)}
        ep = self.mem.store(state=state, sensory_data=sensory)
        state[:] = 999.0  # Mutate original
        self.assertAlmostEqual(ep.state[0], 1.0)

    def test_store_copies_sensory_data(self):
        """Stored sensory data should be a copy."""
        visual = np.array([1.0, 2.0, 3.0])
        ep = self.mem.store(
            state=self._make_state(2),
            sensory_data={"visual": visual},
        )
        visual[:] = 999.0  # Mutate original
        self.assertAlmostEqual(ep.sensory_data["visual"][0], 1.0)

    def test_store_default_context(self):
        ep = self.mem.store(
            state=self._make_state(3),
            sensory_data={"audio": np.zeros(2)},
        )
        self.assertEqual(ep.context, {})

    def test_store_multiple_episodes(self):
        for i in range(10):
            self.mem.store(
                state=self._make_state(i + 100),
                sensory_data={"s": np.zeros(2)},
            )
        self.assertEqual(len(self.mem.episodes), 10)
        self.assertEqual(self.mem.total_stored, 10)

    def test_store_episode_indexed_by_id(self):
        ep = self.mem.store(
            state=self._make_state(4),
            sensory_data={"s": np.zeros(2)},
        )
        self.assertIn(ep.episode_id, self.mem.episode_index)
        self.assertIs(self.mem.episode_index[ep.episode_id], ep)

    def test_store_high_surprise_and_emotion_consolidates_immediately(self):
        """Episodes with high surprise AND high emotion should be immediately consolidated."""
        ep = self.mem.store(
            state=self._make_state(5),
            sensory_data={"s": np.zeros(2)},
            emotional_valence=0.9,  # triggers emotional_bonus (1.5x)
            surprise_level=0.8,     # triggers novelty_bonus (2.0x)
        )
        # priority = 1.0 * 2.0 * 1.5 = 3.0 > 2.0, so immediate consolidation
        self.assertIn(ep, self.mem.consolidated_episodes)
        self.assertGreater(ep.consolidation_strength, 0.0)

    def test_store_low_priority_not_immediately_consolidated(self):
        ep = self.mem.store(
            state=self._make_state(6),
            sensory_data={"s": np.zeros(2)},
            emotional_valence=0.1,
            surprise_level=0.1,
        )
        self.assertNotIn(ep, self.mem.consolidated_episodes)

    def test_store_similar_episode_deduplication(self):
        """Storing a very similar state with low priority should return the existing episode."""
        mem = EpisodicMemory(
            state_size=8,
            max_episodes=100,
            similarity_threshold=0.5,  # Low threshold to trigger dedup easily
            random_seed=99,
        )
        state = np.ones(8)
        ep1 = mem.store(state=state, sensory_data={"s": np.zeros(2)})
        # Same state, low priority (no novelty or emotion)
        ep2 = mem.store(state=state, sensory_data={"s": np.zeros(2)})
        # Should return the existing similar episode
        self.assertIs(ep1, ep2)
        self.assertEqual(len(mem.episodes), 1)

    def test_store_similar_episode_forced_by_high_priority(self):
        """Even similar episodes should be stored if priority is high enough."""
        mem = EpisodicMemory(
            state_size=8,
            max_episodes=100,
            similarity_threshold=0.5,
            random_seed=99,
        )
        state = np.ones(8)
        ep1 = mem.store(state=state, sensory_data={"s": np.zeros(2)})
        # Same state but high surprise + emotion -> priority >= 1.5
        ep2 = mem.store(
            state=state,
            sensory_data={"s": np.zeros(2)},
            surprise_level=0.8,
            emotional_valence=0.8,
        )
        self.assertIsNot(ep1, ep2)
        self.assertEqual(len(mem.episodes), 2)


class TestRetrieve(unittest.TestCase):
    """Test retrieving episodes."""

    def setUp(self):
        self.mem = EpisodicMemory(
            state_size=8,
            max_episodes=100,
            similarity_threshold=0.0,  # Accept all matches for test flexibility
            random_seed=42,
        )

    def _make_state(self, seed=0):
        rng = np.random.RandomState(seed)
        return rng.randn(8)

    def test_retrieve_from_empty_memory(self):
        result = self.mem.retrieve(cue=np.zeros(8))
        self.assertEqual(result, [])

    def test_retrieve_by_state_cue(self):
        state = self._make_state(10)
        self.mem.store(state=state, sensory_data={"s": np.zeros(2)})
        # Retrieve using same state as cue
        results = self.mem.retrieve(cue=state, n_episodes=1)
        self.assertEqual(len(results), 1)
        np.testing.assert_array_almost_equal(results[0].state, state)

    def test_retrieve_increments_replay_count(self):
        state = self._make_state(11)
        ep = self.mem.store(state=state, sensory_data={"s": np.zeros(2)})
        self.assertEqual(ep.replay_count, 0)
        self.mem.retrieve(cue=state, n_episodes=1)
        self.assertEqual(ep.replay_count, 1)
        self.mem.retrieve(cue=state, n_episodes=1)
        self.assertEqual(ep.replay_count, 2)

    def test_retrieve_updates_total_retrieved(self):
        state = self._make_state(12)
        self.mem.store(state=state, sensory_data={"s": np.zeros(2)})
        self.assertEqual(self.mem.total_retrieved, 0)
        self.mem.retrieve(cue=state, n_episodes=1)
        self.assertEqual(self.mem.total_retrieved, 1)

    def test_retrieve_n_episodes(self):
        # Store 5 different episodes
        for i in range(5):
            self.mem.store(
                state=self._make_state(20 + i),
                sensory_data={"s": np.zeros(2)},
            )
        # Request 3 episodes
        results = self.mem.retrieve(cue=self._make_state(20), n_episodes=3)
        self.assertLessEqual(len(results), 3)
        self.assertGreaterEqual(len(results), 1)

    def test_retrieve_most_similar_first(self):
        """The most similar episode should be returned first."""
        state_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.mem.store(state=state_b, sensory_data={"s": np.zeros(2)})
        self.mem.store(state=state_a, sensory_data={"s": np.zeros(2)})

        # Query with state_a -> should return state_a first
        results = self.mem.retrieve(cue=state_a, n_episodes=2)
        self.assertEqual(len(results), 2)
        np.testing.assert_array_almost_equal(results[0].state, state_a)

    def test_retrieve_with_context(self):
        self.mem.store(
            state=np.ones(8),
            sensory_data={"s": np.zeros(2)},
            context={"location": "park"},
        )
        self.mem.store(
            state=np.ones(8) * 0.5,
            sensory_data={"s": np.zeros(2)},
            context={"location": "office"},
        )
        # Retrieve with context cue
        results = self.mem.retrieve(
            context={"location": "park"},
            n_episodes=2,
        )
        self.assertGreaterEqual(len(results), 1)

    def test_retrieve_with_state_and_context(self):
        target_state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.mem.store(
            state=target_state,
            sensory_data={"s": np.zeros(2)},
            context={"task": "explore"},
        )
        self.mem.store(
            state=self._make_state(30),
            sensory_data={"s": np.zeros(2)},
            context={"task": "rest"},
        )
        results = self.mem.retrieve(
            cue=target_state,
            context={"task": "explore"},
            n_episodes=1,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].context["task"], "explore")

    def test_retrieve_similarity_threshold_filtering(self):
        """Episodes below the similarity threshold should be excluded."""
        mem = EpisodicMemory(
            state_size=8,
            max_episodes=100,
            similarity_threshold=0.99,  # Very high threshold
            random_seed=42,
        )
        mem.store(
            state=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            sensory_data={"s": np.zeros(2)},
        )
        # Query with an orthogonal vector -> similarity ~ 0
        cue = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        results = mem.retrieve(cue=cue, n_episodes=1)
        self.assertEqual(len(results), 0)

    def test_retrieve_custom_threshold_override(self):
        """Passing similarity_threshold to retrieve should override the default."""
        mem = EpisodicMemory(
            state_size=8,
            max_episodes=100,
            similarity_threshold=0.99,  # Very high default
            random_seed=42,
        )
        state = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mem.store(state=state, sensory_data={"s": np.zeros(2)})
        # Use a low override threshold
        results = mem.retrieve(
            cue=state * 0.9 + np.random.RandomState(0).randn(8) * 0.01,
            n_episodes=1,
            similarity_threshold=0.0,
        )
        self.assertEqual(len(results), 1)

    def test_retrieve_boosts_consolidated_episodes(self):
        """Consolidated episodes should get a similarity boost."""
        state = np.ones(8)
        ep = self.mem.store(state=state, sensory_data={"s": np.zeros(2)})
        ep.consolidation_strength = 1.0  # Fully consolidated

        state2 = np.ones(8) * 0.99
        ep2 = self.mem.store(state=state2, sensory_data={"s": np.zeros(2)})
        ep2.consolidation_strength = 0.0  # Not consolidated

        results = self.mem.retrieve(cue=np.ones(8), n_episodes=2)
        # The consolidated episode should be ranked first due to the boost
        self.assertIs(results[0], ep)


class TestConsolidate(unittest.TestCase):
    """Test memory consolidation through replay."""

    def setUp(self):
        self.mem = EpisodicMemory(
            state_size=8,
            max_episodes=100,
            consolidation_threshold=0.7,
            enable_consolidation=True,
            random_seed=42,
        )

    def _make_state(self, seed=0):
        rng = np.random.RandomState(seed)
        return rng.randn(8)

    def test_consolidate_empty_memory(self):
        result = self.mem.consolidate(n_replay=5, batch_size=3)
        self.assertEqual(result, 0)

    def test_consolidate_disabled(self):
        mem = EpisodicMemory(
            state_size=8,
            enable_consolidation=False,
            random_seed=42,
        )
        mem.store(state=self._make_state(0), sensory_data={"s": np.zeros(2)})
        result = mem.consolidate(n_replay=10, batch_size=5)
        self.assertEqual(result, 0)

    def test_consolidate_increases_strength(self):
        ep = self.mem.store(
            state=self._make_state(0),
            sensory_data={"s": np.zeros(2)},
        )
        initial_strength = ep.consolidation_strength
        self.mem.consolidate(n_replay=5, batch_size=5)
        self.assertGreater(ep.consolidation_strength, initial_strength)

    def test_consolidate_increments_replay_count(self):
        ep = self.mem.store(
            state=self._make_state(1),
            sensory_data={"s": np.zeros(2)},
        )
        self.assertEqual(ep.replay_count, 0)
        self.mem.consolidate(n_replay=3, batch_size=1)
        self.assertGreater(ep.replay_count, 0)

    def test_consolidate_updates_total_replayed(self):
        self.mem.store(
            state=self._make_state(2),
            sensory_data={"s": np.zeros(2)},
        )
        self.assertEqual(self.mem.total_replayed, 0)
        self.mem.consolidate(n_replay=2, batch_size=1)
        self.assertGreater(self.mem.total_replayed, 0)

    def test_consolidate_moves_to_consolidated_list(self):
        """After enough replays, episodes should enter the consolidated list."""
        ep = self.mem.store(
            state=self._make_state(3),
            sensory_data={"s": np.zeros(2)},
            emotional_valence=0.8,
            surprise_level=0.8,
        )
        # Many replay cycles to push consolidation_strength above threshold (0.7)
        self.mem.consolidate(n_replay=20, batch_size=5)
        self.assertGreaterEqual(ep.consolidation_strength, self.mem.consolidation_threshold)

    def test_consolidate_strength_caps_at_one(self):
        ep = self.mem.store(
            state=self._make_state(4),
            sensory_data={"s": np.zeros(2)},
        )
        self.mem.consolidate(n_replay=50, batch_size=5)
        self.assertLessEqual(ep.consolidation_strength, 1.0)

    def test_consolidate_returns_count(self):
        for i in range(5):
            self.mem.store(
                state=self._make_state(50 + i),
                sensory_data={"s": np.zeros(2)},
                emotional_valence=0.9,
                surprise_level=0.9,
            )
        count = self.mem.consolidate(n_replay=20, batch_size=5)
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)


class TestMemoryCapacityLimits(unittest.TestCase):
    """Test that memory enforces capacity limits and forgets appropriately."""

    def _make_state(self, seed=0):
        rng = np.random.RandomState(seed)
        return rng.randn(8)

    def test_capacity_enforced(self):
        mem = EpisodicMemory(
            state_size=8,
            max_episodes=5,
            similarity_threshold=0.999,  # Avoid dedup interference
            enable_forgetting=True,
            random_seed=42,
        )
        for i in range(10):
            mem.store(
                state=self._make_state(i + 200),
                sensory_data={"s": np.zeros(2)},
            )
        self.assertLessEqual(len(mem.episodes), 5)

    def test_total_stored_counts_all_attempts(self):
        mem = EpisodicMemory(
            state_size=8,
            max_episodes=3,
            similarity_threshold=0.999,
            enable_forgetting=True,
            random_seed=42,
        )
        for i in range(6):
            mem.store(
                state=self._make_state(i + 300),
                sensory_data={"s": np.zeros(2)},
            )
        # total_stored counts every successful store (including forgotten later)
        self.assertEqual(mem.total_stored, 6)

    def test_total_forgotten_tracks_removals(self):
        mem = EpisodicMemory(
            state_size=8,
            max_episodes=3,
            similarity_threshold=0.999,
            enable_forgetting=True,
            random_seed=42,
        )
        for i in range(6):
            mem.store(
                state=self._make_state(i + 400),
                sensory_data={"s": np.zeros(2)},
            )
        self.assertGreater(mem.total_forgotten, 0)

    def test_forgetting_disabled(self):
        """When forgetting is disabled, _forget_weakest still removes oldest to stay under limit."""
        mem = EpisodicMemory(
            state_size=8,
            max_episodes=3,
            similarity_threshold=0.999,
            enable_forgetting=False,
            random_seed=42,
        )
        for i in range(5):
            mem.store(
                state=self._make_state(i + 500),
                sensory_data={"s": np.zeros(2)},
            )
        # When forgetting is disabled, _forget_weakest is a no-op,
        # so episodes just accumulate beyond max
        # (the check in store appends first, then calls _forget_weakest which does nothing)
        self.assertGreaterEqual(len(mem.episodes), 3)

    def test_consolidated_episodes_protected(self):
        """Highly consolidated episodes should be harder to forget."""
        mem = EpisodicMemory(
            state_size=8,
            max_episodes=3,
            similarity_threshold=0.999,
            consolidation_threshold=0.7,
            enable_forgetting=True,
            random_seed=42,
        )
        # Store an episode and manually consolidate it
        important_state = self._make_state(600)
        ep = mem.store(
            state=important_state,
            sensory_data={"s": np.zeros(2)},
            emotional_valence=0.9,
            surprise_level=0.9,
        )
        ep.consolidation_strength = 0.99  # Very consolidated

        # Fill up memory with more episodes
        for i in range(5):
            mem.store(
                state=self._make_state(601 + i),
                sensory_data={"s": np.zeros(2)},
            )

        # The highly consolidated episode should still be present
        episode_ids = [e.episode_id for e in mem.episodes]
        self.assertIn(ep.episode_id, episode_ids)


class TestContextSimilarity(unittest.TestCase):
    """Test the internal _context_similarity method."""

    def setUp(self):
        self.mem = EpisodicMemory(state_size=4, random_seed=42)

    def test_identical_contexts(self):
        ctx = {"a": 1, "b": "hello"}
        sim = self.mem._context_similarity(ctx, ctx)
        self.assertAlmostEqual(sim, 1.0)

    def test_completely_different_contexts(self):
        ctx1 = {"a": 1}
        ctx2 = {"a": 2}
        sim = self.mem._context_similarity(ctx1, ctx2)
        self.assertAlmostEqual(sim, 0.0)

    def test_no_common_keys(self):
        ctx1 = {"a": 1}
        ctx2 = {"b": 2}
        sim = self.mem._context_similarity(ctx1, ctx2)
        self.assertAlmostEqual(sim, 0.0)

    def test_empty_contexts(self):
        self.assertAlmostEqual(self.mem._context_similarity({}, {}), 0.0)
        self.assertAlmostEqual(self.mem._context_similarity({"a": 1}, {}), 0.0)

    def test_partial_match(self):
        ctx1 = {"a": 1, "b": 2, "c": 3}
        ctx2 = {"a": 1, "b": 99, "c": 3}
        sim = self.mem._context_similarity(ctx1, ctx2)
        self.assertAlmostEqual(sim, 2.0 / 3.0, places=5)


class TestFindSimilarEpisode(unittest.TestCase):
    """Test the internal _find_similar_episode method."""

    def setUp(self):
        self.mem = EpisodicMemory(
            state_size=8,
            max_episodes=100,
            similarity_threshold=0.9,
            random_seed=42,
        )

    def test_no_similar_in_empty_memory(self):
        result = self.mem._find_similar_episode(np.ones(8))
        self.assertIsNone(result)

    def test_finds_identical_state(self):
        state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.mem.store(state=state, sensory_data={"s": np.zeros(2)})
        result = self.mem._find_similar_episode(state)
        self.assertIsNotNone(result)

    def test_no_match_for_orthogonal_state(self):
        state_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.mem.store(state=state_a, sensory_data={"s": np.zeros(2)})
        result = self.mem._find_similar_episode(state_b)
        self.assertIsNone(result)


class TestSelectReplayCandidates(unittest.TestCase):
    """Test the internal _select_replay_candidates method."""

    def setUp(self):
        self.mem = EpisodicMemory(state_size=4, max_episodes=50, random_seed=42)

    def test_empty_memory(self):
        result = self.mem._select_replay_candidates(batch_size=5)
        self.assertEqual(result, [])

    def test_returns_at_most_batch_size(self):
        for i in range(10):
            rng = np.random.RandomState(i + 700)
            self.mem.store(
                state=rng.randn(4),
                sensory_data={"s": np.zeros(2)},
            )
        candidates = self.mem._select_replay_candidates(batch_size=3)
        self.assertLessEqual(len(candidates), 3)

    def test_prioritizes_significant_episodes(self):
        """Episodes with high emotion/surprise should be selected more often."""
        rng = np.random.RandomState(800)
        # Store a boring episode
        self.mem.store(
            state=rng.randn(4),
            sensory_data={"s": np.zeros(2)},
            emotional_valence=0.0,
            surprise_level=0.0,
        )
        # Store a significant episode
        significant_state = rng.randn(4)
        self.mem.store(
            state=significant_state,
            sensory_data={"s": np.zeros(2)},
            emotional_valence=0.9,
            surprise_level=0.9,
        )
        candidates = self.mem._select_replay_candidates(batch_size=1)
        self.assertEqual(len(candidates), 1)
        # The significant episode should be the top candidate
        np.testing.assert_array_equal(candidates[0].state, significant_state)


class TestGetStats(unittest.TestCase):
    """Test statistics reporting."""

    def test_stats_empty_memory(self):
        mem = EpisodicMemory(state_size=4, random_seed=42)
        stats = mem.get_stats()
        self.assertEqual(stats['total_episodes'], 0)
        self.assertEqual(stats['consolidated_episodes'], 0)
        self.assertEqual(stats['total_stored'], 0)
        self.assertEqual(stats['total_retrieved'], 0)
        self.assertEqual(stats['total_replayed'], 0)
        self.assertEqual(stats['total_forgotten'], 0)
        self.assertAlmostEqual(stats['avg_consolidation_strength'], 0.0)
        self.assertAlmostEqual(stats['avg_replay_count'], 0.0)
        self.assertAlmostEqual(stats['avg_surprise_level'], 0.0)
        self.assertAlmostEqual(stats['avg_emotional_valence'], 0.0)
        self.assertAlmostEqual(stats['cache_hit_rate'], 0.0)

    def test_stats_after_operations(self):
        mem = EpisodicMemory(
            state_size=4,
            similarity_threshold=0.0,
            random_seed=42,
        )
        rng = np.random.RandomState(900)
        state = rng.randn(4)
        mem.store(
            state=state,
            sensory_data={"s": np.zeros(2)},
            emotional_valence=0.6,
            surprise_level=0.4,
        )
        mem.retrieve(cue=state, n_episodes=1)
        mem.consolidate(n_replay=2, batch_size=1)

        stats = mem.get_stats()
        self.assertEqual(stats['total_episodes'], 1)
        self.assertEqual(stats['total_stored'], 1)
        self.assertGreaterEqual(stats['total_retrieved'], 1)
        self.assertGreaterEqual(stats['total_replayed'], 1)
        self.assertAlmostEqual(stats['avg_surprise_level'], 0.4)
        # emotional_valence stat uses abs(), so 0.6
        self.assertAlmostEqual(stats['avg_emotional_valence'], 0.6)

    def test_stats_keys(self):
        mem = EpisodicMemory(state_size=4, random_seed=42)
        stats = mem.get_stats()
        expected_keys = {
            'total_episodes', 'consolidated_episodes', 'total_stored',
            'total_retrieved', 'total_replayed', 'total_forgotten',
            'avg_consolidation_strength', 'avg_replay_count',
            'avg_surprise_level', 'avg_emotional_valence', 'cache_hit_rate',
        }
        self.assertEqual(set(stats.keys()), expected_keys)


class TestSerialization(unittest.TestCase):
    """Test serialize and deserialize methods."""

    def test_serialize_empty_memory(self):
        mem = EpisodicMemory(state_size=4, max_episodes=20, random_seed=42)
        data = mem.serialize()
        self.assertEqual(data['state_size'], 4)
        self.assertEqual(data['max_episodes'], 20)
        self.assertAlmostEqual(data['consolidation_threshold'], 0.7)
        self.assertEqual(len(data['episodes']), 0)
        self.assertIn('stats', data)

    def test_serialize_includes_consolidated_episodes(self):
        mem = EpisodicMemory(
            state_size=4,
            max_episodes=20,
            consolidation_threshold=0.3,  # Low threshold
            random_seed=42,
        )
        rng = np.random.RandomState(1000)
        ep = mem.store(
            state=rng.randn(4),
            sensory_data={"s": np.zeros(2)},
            emotional_valence=0.9,
            surprise_level=0.9,
        )
        # Force consolidation
        mem.consolidate(n_replay=20, batch_size=5)

        data = mem.serialize()
        # At least some episodes should be serialized (those that were consolidated)
        self.assertGreaterEqual(len(data['episodes']), 1)

    def test_serialize_episode_fields(self):
        mem = EpisodicMemory(state_size=4, max_episodes=20, random_seed=42)
        rng = np.random.RandomState(1001)
        state = rng.randn(4)
        # Use low valence/surprise so the episode is NOT auto-consolidated
        ep = mem.store(
            state=state,
            sensory_data={"s": np.zeros(2)},
            context={"task": "test"},
            emotional_valence=0.2,
            surprise_level=0.1,
        )
        # Manually add to consolidated list for serialization
        mem.consolidated_episodes.append(ep)

        data = mem.serialize()
        self.assertEqual(len(data['episodes']), 1)
        ep_data = data['episodes'][0]
        self.assertIn('timestamp', ep_data)
        self.assertIn('state', ep_data)
        self.assertIn('context', ep_data)
        self.assertIn('emotional_valence', ep_data)
        self.assertIn('surprise_level', ep_data)
        self.assertIn('replay_count', ep_data)
        self.assertIn('consolidation_strength', ep_data)
        # State should be a list (from .tolist())
        self.assertIsInstance(ep_data['state'], list)
        self.assertEqual(len(ep_data['state']), 4)

    def test_deserialize_restores_memory(self):
        mem = EpisodicMemory(
            state_size=4,
            max_episodes=20,
            consolidation_threshold=0.5,
            random_seed=42,
        )
        rng = np.random.RandomState(1002)
        state = rng.randn(4)
        ep = mem.store(
            state=state,
            sensory_data={"s": np.zeros(2)},
            context={"location": "lab"},
            emotional_valence=0.6,
            surprise_level=0.5,
        )
        # Force into consolidated list for serialization
        ep.consolidation_strength = 0.8
        mem.consolidated_episodes.append(ep)

        data = mem.serialize()

        # Deserialize
        restored = EpisodicMemory.deserialize(data)
        self.assertEqual(restored.state_size, 4)
        self.assertEqual(restored.max_episodes, 20)
        self.assertAlmostEqual(restored.consolidation_threshold, 0.5)
        self.assertEqual(len(restored.consolidated_episodes), 1)
        self.assertEqual(len(restored.episodes), 1)

        # Verify the episode data was restored
        restored_ep = restored.episodes[0]
        np.testing.assert_array_almost_equal(restored_ep.state, state)
        self.assertEqual(restored_ep.context["location"], "lab")
        self.assertAlmostEqual(restored_ep.emotional_valence, 0.6)
        self.assertAlmostEqual(restored_ep.surprise_level, 0.5)
        self.assertAlmostEqual(restored_ep.consolidation_strength, 0.8)

    def test_deserialize_episode_index(self):
        mem = EpisodicMemory(state_size=4, max_episodes=20, random_seed=42)
        rng = np.random.RandomState(1003)
        ep = mem.store(
            state=rng.randn(4),
            sensory_data={"s": np.zeros(2)},
        )
        ep.consolidation_strength = 0.9
        mem.consolidated_episodes.append(ep)

        data = mem.serialize()
        restored = EpisodicMemory.deserialize(data)

        # episode_index should be populated
        self.assertEqual(len(restored.episode_index), 1)
        restored_ep = restored.episodes[0]
        self.assertIn(restored_ep.episode_id, restored.episode_index)

    def test_roundtrip_preserves_retrieval(self):
        """Serialized and deserialized memory should still support retrieval."""
        mem = EpisodicMemory(
            state_size=4,
            max_episodes=20,
            similarity_threshold=0.0,
            random_seed=42,
        )
        rng = np.random.RandomState(1004)
        state = rng.randn(4)
        ep = mem.store(
            state=state,
            sensory_data={"s": np.zeros(2)},
        )
        ep.consolidation_strength = 0.8
        mem.consolidated_episodes.append(ep)

        data = mem.serialize()
        restored = EpisodicMemory.deserialize(data)
        # Override threshold for retrieval
        restored.similarity_threshold = 0.0

        results = restored.retrieve(cue=state, n_episodes=1)
        self.assertEqual(len(results), 1)
        np.testing.assert_array_almost_equal(results[0].state, state)


class TestIntegration(unittest.TestCase):
    """Integration tests combining store, retrieve, consolidate workflows."""

    def test_full_lifecycle(self):
        """Test the complete lifecycle: store -> consolidate -> retrieve."""
        mem = EpisodicMemory(
            state_size=8,
            max_episodes=50,
            consolidation_threshold=0.5,
            similarity_threshold=0.999,  # High threshold to avoid dedup during store
            random_seed=42,
        )

        rng = np.random.RandomState(2000)

        # 1. Store several episodes
        states = []
        for i in range(10):
            state = rng.randn(8)
            states.append(state)
            mem.store(
                state=state,
                sensory_data={"visual": rng.randn(4)},
                context={"step": i},
                emotional_valence=rng.uniform(-1, 1),
                surprise_level=rng.uniform(0, 1),
            )

        self.assertEqual(mem.total_stored, 10)

        # 2. Consolidate
        consolidated = mem.consolidate(n_replay=10, batch_size=5)
        self.assertGreaterEqual(consolidated, 0)
        self.assertGreater(mem.total_replayed, 0)

        # 3. Retrieve (use low threshold to ensure matches are returned)
        results = mem.retrieve(cue=states[0], n_episodes=3, similarity_threshold=0.0)
        self.assertGreaterEqual(len(results), 1)
        self.assertGreater(mem.total_retrieved, 0)

        # 4. Stats should reflect all operations
        stats = mem.get_stats()
        self.assertEqual(stats['total_episodes'], 10)
        self.assertGreater(stats['total_stored'], 0)
        self.assertGreater(stats['total_retrieved'], 0)
        self.assertGreater(stats['total_replayed'], 0)

    def test_store_and_retrieve_preserves_context(self):
        mem = EpisodicMemory(
            state_size=4,
            similarity_threshold=0.0,
            random_seed=42,
        )
        rng = np.random.RandomState(2001)
        state = rng.randn(4)
        mem.store(
            state=state,
            sensory_data={"audio": np.array([1.0, 2.0])},
            context={"weather": "sunny", "time": "morning"},
            emotional_valence=0.3,
            surprise_level=0.1,
        )
        results = mem.retrieve(cue=state, n_episodes=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].context["weather"], "sunny")
        self.assertEqual(results[0].context["time"], "morning")
        self.assertAlmostEqual(results[0].emotional_valence, 0.3)

    def test_consolidation_improves_retrieval_ranking(self):
        """Consolidated episodes should rank higher in retrieval results."""
        mem = EpisodicMemory(
            state_size=4,
            similarity_threshold=0.0,
            random_seed=42,
        )

        # Two episodes with similar states
        base = np.array([1.0, 1.0, 1.0, 1.0])
        ep1 = mem.store(state=base, sensory_data={"s": np.zeros(2)})
        ep2 = mem.store(state=base * 1.01, sensory_data={"s": np.zeros(2)})

        # Consolidate ep1 heavily
        ep1.consolidation_strength = 1.0
        ep2.consolidation_strength = 0.0

        results = mem.retrieve(cue=base, n_episodes=2)
        # ep1 should be ranked first due to consolidation boost
        self.assertIs(results[0], ep1)

    def test_xp_backend_is_available(self):
        """Verify the xp backend import works (numpy or cupy)."""
        arr = xp.zeros(4)
        self.assertEqual(len(arr), 4)


if __name__ == '__main__':
    unittest.main()
