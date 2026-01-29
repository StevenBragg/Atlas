"""
Comprehensive tests for the NetworkStore persistence module.

Tests cover:
- NetworkStore initialization and database creation
- Weight save/load round-trip with various array shapes and dtypes
- Snapshot creation, loading, listing, and deletion
- Snapshot limit enforcement (MAX_SNAPSHOTS_PER_TYPE = 10)
- Learning event logging and history retrieval
- Utility methods (has_weights, delete_weights, get_all_network_types, get_stats)
- Proper cleanup of temporary files

All tests are deterministic and pass reliably in CPU-only environments.
"""

import os
import sys
import json
import shutil
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.database.network_store import NetworkStore, NetworkSnapshot


class TestNetworkStoreInit(unittest.TestCase):
    """Tests for NetworkStore initialization and database setup."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="atlas_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_init_creates_data_directory(self):
        """NetworkStore should create the data directory if it does not exist."""
        sub_dir = os.path.join(self.test_dir, "nested", "data")
        store = NetworkStore(data_dir=sub_dir)
        self.assertTrue(os.path.isdir(sub_dir))
        store.close()

    def test_init_creates_database_file(self):
        """NetworkStore should create the SQLite database file on init."""
        store = NetworkStore(data_dir=self.test_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, "atlas_network.db")))
        store.close()

    def test_db_path_attribute(self):
        """The db_path attribute should point inside the data directory."""
        store = NetworkStore(data_dir=self.test_dir)
        self.assertEqual(str(store.db_path), os.path.join(self.test_dir, "atlas_network.db"))
        store.close()

    def test_init_idempotent(self):
        """Initializing twice on the same directory should not raise."""
        store1 = NetworkStore(data_dir=self.test_dir)
        store1.close()
        store2 = NetworkStore(data_dir=self.test_dir)
        store2.close()

    def test_max_snapshots_constant(self):
        """MAX_SNAPSHOTS_PER_TYPE should be 10."""
        self.assertEqual(NetworkStore.MAX_SNAPSHOTS_PER_TYPE, 10)


class TestSaveAndLoadWeights(unittest.TestCase):
    """Tests for save_weights() and load_weights() round-trip."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="atlas_test_")
        self.store = NetworkStore(data_dir=self.test_dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_returns_true(self):
        """save_weights should return True on success."""
        weights = np.ones((4, 4), dtype=np.float64)
        result = self.store.save_weights("test_net", weights)
        self.assertTrue(result)

    def test_round_trip_1d(self):
        """Saving and loading a 1-D array should preserve values exactly."""
        original = np.arange(10, dtype=np.float32)
        self.store.save_weights("net_1d", original)
        loaded, metadata = self.store.load_weights("net_1d")
        np.testing.assert_array_equal(loaded, original)
        self.assertEqual(loaded.dtype, original.dtype)

    def test_round_trip_2d(self):
        """Saving and loading a 2-D array should preserve shape and values."""
        original = np.eye(5, dtype=np.float64)
        self.store.save_weights("net_2d", original)
        loaded, metadata = self.store.load_weights("net_2d")
        np.testing.assert_array_equal(loaded, original)
        self.assertEqual(loaded.shape, (5, 5))

    def test_round_trip_3d(self):
        """Saving and loading a 3-D array should preserve all dimensions."""
        original = np.zeros((2, 3, 4), dtype=np.float32)
        original[0, 1, 2] = 42.0
        self.store.save_weights("net_3d", original)
        loaded, _ = self.store.load_weights("net_3d")
        np.testing.assert_array_equal(loaded, original)

    def test_round_trip_with_metadata(self):
        """Metadata should be preserved through save/load."""
        weights = np.array([1.0, 2.0, 3.0])
        meta = {"learning_rate": 0.01, "epoch": 5, "tags": ["a", "b"]}
        self.store.save_weights("meta_net", weights, metadata=meta)
        _, loaded_meta = self.store.load_weights("meta_net")
        self.assertEqual(loaded_meta["learning_rate"], 0.01)
        self.assertEqual(loaded_meta["epoch"], 5)
        self.assertEqual(loaded_meta["tags"], ["a", "b"])

    def test_save_without_metadata(self):
        """Saving without metadata should load as empty dict."""
        weights = np.array([1.0])
        self.store.save_weights("no_meta", weights)
        _, loaded_meta = self.store.load_weights("no_meta")
        self.assertEqual(loaded_meta, {})

    def test_load_nonexistent_returns_none(self):
        """Loading a network type that was never saved should return None."""
        result = self.store.load_weights("nonexistent_network")
        self.assertIsNone(result)

    def test_overwrite_weights(self):
        """Saving the same network type twice should overwrite the first."""
        first = np.array([1.0, 2.0, 3.0])
        second = np.array([10.0, 20.0, 30.0, 40.0])
        self.store.save_weights("overwrite_net", first)
        self.store.save_weights("overwrite_net", second)
        loaded, _ = self.store.load_weights("overwrite_net")
        np.testing.assert_array_equal(loaded, second)

    def test_multiple_network_types(self):
        """Different network types should be stored independently."""
        w1 = np.array([1.0, 2.0])
        w2 = np.array([3.0, 4.0, 5.0])
        self.store.save_weights("type_a", w1)
        self.store.save_weights("type_b", w2)
        loaded_a, _ = self.store.load_weights("type_a")
        loaded_b, _ = self.store.load_weights("type_b")
        np.testing.assert_array_equal(loaded_a, w1)
        np.testing.assert_array_equal(loaded_b, w2)

    def test_integer_dtype_round_trip(self):
        """Integer array dtypes should be preserved through save/load."""
        original = np.array([1, 2, 3, 4], dtype=np.int32)
        self.store.save_weights("int_net", original)
        loaded, _ = self.store.load_weights("int_net")
        np.testing.assert_array_equal(loaded, original)
        self.assertEqual(loaded.dtype, np.int32)


class TestHasAndDeleteWeights(unittest.TestCase):
    """Tests for has_weights(), delete_weights(), and get_all_network_types()."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="atlas_test_")
        self.store = NetworkStore(data_dir=self.test_dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_has_weights_false_when_empty(self):
        """has_weights should return False for a network that was never saved."""
        self.assertFalse(self.store.has_weights("absent"))

    def test_has_weights_true_after_save(self):
        """has_weights should return True after saving weights."""
        self.store.save_weights("present", np.array([1.0]))
        self.assertTrue(self.store.has_weights("present"))

    def test_delete_weights(self):
        """delete_weights should remove the network state."""
        self.store.save_weights("to_delete", np.array([1.0]))
        self.assertTrue(self.store.has_weights("to_delete"))
        result = self.store.delete_weights("to_delete")
        self.assertTrue(result)
        self.assertFalse(self.store.has_weights("to_delete"))

    def test_delete_nonexistent_returns_true(self):
        """Deleting a nonexistent network type should still return True (no error)."""
        result = self.store.delete_weights("never_existed")
        self.assertTrue(result)

    def test_get_all_network_types_empty(self):
        """get_all_network_types should return empty list when nothing is saved."""
        self.assertEqual(self.store.get_all_network_types(), [])

    def test_get_all_network_types(self):
        """get_all_network_types should return all saved network types."""
        self.store.save_weights("alpha", np.array([1.0]))
        self.store.save_weights("beta", np.array([2.0]))
        self.store.save_weights("gamma", np.array([3.0]))
        types = self.store.get_all_network_types()
        self.assertEqual(sorted(types), ["alpha", "beta", "gamma"])


class TestCreateAndLoadSnapshot(unittest.TestCase):
    """Tests for create_snapshot() and load_snapshot()."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="atlas_test_")
        self.store = NetworkStore(data_dir=self.test_dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_create_snapshot_returns_id(self):
        """create_snapshot should return a non-empty snapshot ID string."""
        weights = np.array([1.0, 2.0, 3.0])
        snapshot_id = self.store.create_snapshot("test_net", weights)
        self.assertIsInstance(snapshot_id, str)
        self.assertTrue(len(snapshot_id) > 0)

    def test_snapshot_id_contains_network_type(self):
        """Snapshot ID should start with the network type as a prefix."""
        snapshot_id = self.store.create_snapshot("my_network", np.zeros(5))
        self.assertTrue(snapshot_id.startswith("my_network_"))

    def test_load_snapshot_round_trip(self):
        """Loading a snapshot should return a NetworkSnapshot with correct data."""
        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        meta = {"version": 2}
        snapshot_id = self.store.create_snapshot(
            "snap_net", original, metadata=meta, description="test snapshot"
        )
        snapshot = self.store.load_snapshot(snapshot_id)

        self.assertIsInstance(snapshot, NetworkSnapshot)
        self.assertEqual(snapshot.snapshot_id, snapshot_id)
        self.assertEqual(snapshot.network_type, "snap_net")
        np.testing.assert_array_equal(snapshot.weights, original)
        self.assertEqual(snapshot.weights_shape, (2, 2))
        self.assertEqual(snapshot.metadata, {"version": 2})
        self.assertIsInstance(snapshot.timestamp, float)

    def test_load_nonexistent_snapshot_returns_none(self):
        """Loading a snapshot ID that does not exist should return None."""
        result = self.store.load_snapshot("nonexistent_id_12345678")
        self.assertIsNone(result)

    def test_get_latest_snapshot(self):
        """get_latest_snapshot should return the most recently created snapshot."""
        w1 = np.array([1.0])
        w2 = np.array([2.0])
        self.store.create_snapshot("latest_net", w1, description="first")
        sid2 = self.store.create_snapshot("latest_net", w2, description="second")
        latest = self.store.get_latest_snapshot("latest_net")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.snapshot_id, sid2)
        np.testing.assert_array_equal(latest.weights, w2)

    def test_get_latest_snapshot_none_when_empty(self):
        """get_latest_snapshot should return None when no snapshots exist."""
        result = self.store.get_latest_snapshot("empty_type")
        self.assertIsNone(result)

    def test_list_snapshots_by_type(self):
        """list_snapshots should filter by network type."""
        self.store.create_snapshot("type_x", np.zeros(3))
        self.store.create_snapshot("type_x", np.ones(3))
        self.store.create_snapshot("type_y", np.zeros(5))

        x_snaps = self.store.list_snapshots(network_type="type_x")
        y_snaps = self.store.list_snapshots(network_type="type_y")
        all_snaps = self.store.list_snapshots()

        self.assertEqual(len(x_snaps), 2)
        self.assertEqual(len(y_snaps), 1)
        self.assertEqual(len(all_snaps), 3)

    def test_list_snapshots_fields(self):
        """Each entry from list_snapshots should have expected keys."""
        self.store.create_snapshot("field_net", np.zeros(4), description="desc")
        snaps = self.store.list_snapshots(network_type="field_net")
        self.assertEqual(len(snaps), 1)
        entry = snaps[0]
        self.assertIn("snapshot_id", entry)
        self.assertIn("network_type", entry)
        self.assertIn("timestamp", entry)
        self.assertIn("weights_shape", entry)
        self.assertIn("description", entry)
        self.assertEqual(entry["network_type"], "field_net")
        self.assertEqual(entry["description"], "desc")
        self.assertEqual(entry["weights_shape"], [4])

    def test_delete_snapshot(self):
        """delete_snapshot should remove the snapshot so it cannot be loaded."""
        sid = self.store.create_snapshot("del_net", np.ones(3))
        self.assertIsNotNone(self.store.load_snapshot(sid))
        result = self.store.delete_snapshot(sid)
        self.assertTrue(result)
        self.assertIsNone(self.store.load_snapshot(sid))

    def test_snapshot_with_metadata(self):
        """Snapshot metadata should survive the round-trip."""
        meta = {"lr": 0.001, "optimizer": "adam", "layers": [64, 32]}
        sid = self.store.create_snapshot("meta_snap", np.zeros(2), metadata=meta)
        loaded = self.store.load_snapshot(sid)
        self.assertEqual(loaded.metadata["lr"], 0.001)
        self.assertEqual(loaded.metadata["optimizer"], "adam")
        self.assertEqual(loaded.metadata["layers"], [64, 32])

    def test_snapshot_without_metadata(self):
        """A snapshot created with no metadata should load with an empty dict."""
        sid = self.store.create_snapshot("bare_snap", np.zeros(2))
        loaded = self.store.load_snapshot(sid)
        self.assertEqual(loaded.metadata, {})


class TestSnapshotLimit(unittest.TestCase):
    """Tests for the MAX_SNAPSHOTS_PER_TYPE cleanup behaviour."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="atlas_test_")
        self.store = NetworkStore(data_dir=self.test_dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_snapshot_limit_enforced(self):
        """Creating more than MAX_SNAPSHOTS_PER_TYPE should prune the oldest."""
        max_snaps = NetworkStore.MAX_SNAPSHOTS_PER_TYPE  # 10
        total_created = max_snaps + 5  # 15

        snapshot_ids = []
        for i in range(total_created):
            sid = self.store.create_snapshot(
                "limited_net",
                np.full(3, float(i)),
                description=f"snapshot_{i}"
            )
            snapshot_ids.append(sid)

        snaps = self.store.list_snapshots(network_type="limited_net")
        self.assertEqual(len(snaps), max_snaps)

    def test_oldest_snapshots_removed(self):
        """After exceeding the limit, the earliest snapshots should be gone."""
        max_snaps = NetworkStore.MAX_SNAPSHOTS_PER_TYPE
        total_created = max_snaps + 3

        snapshot_ids = []
        for i in range(total_created):
            sid = self.store.create_snapshot(
                "prune_net",
                np.full(2, float(i)),
                description=f"snap_{i}"
            )
            snapshot_ids.append(sid)

        # The first 3 should have been pruned
        for old_id in snapshot_ids[:3]:
            self.assertIsNone(self.store.load_snapshot(old_id))

        # The last max_snaps should still exist
        for recent_id in snapshot_ids[total_created - max_snaps:]:
            self.assertIsNotNone(self.store.load_snapshot(recent_id))

    def test_snapshot_limit_per_network_type(self):
        """Snapshot limits should be enforced independently per network type."""
        max_snaps = NetworkStore.MAX_SNAPSHOTS_PER_TYPE

        # Create max+2 for type_a
        for i in range(max_snaps + 2):
            self.store.create_snapshot("type_a", np.zeros(2))

        # Create 3 for type_b
        for i in range(3):
            self.store.create_snapshot("type_b", np.ones(2))

        a_snaps = self.store.list_snapshots(network_type="type_a")
        b_snaps = self.store.list_snapshots(network_type="type_b")
        self.assertEqual(len(a_snaps), max_snaps)
        self.assertEqual(len(b_snaps), 3)


class TestLearningHistory(unittest.TestCase):
    """Tests for log_learning_event() and get_learning_history()."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="atlas_test_")
        self.store = NetworkStore(data_dir=self.test_dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_log_event_returns_true(self):
        """log_learning_event should return True on success."""
        result = self.store.log_learning_event(
            "train_net", "training", metrics={"loss": 0.5}
        )
        self.assertTrue(result)

    def test_get_learning_history_single_event(self):
        """A single logged event should appear in the history."""
        self.store.log_learning_event(
            "hist_net", "training",
            metrics={"accuracy": 0.95},
            description="epoch 1"
        )
        history = self.store.get_learning_history(network_type="hist_net")
        self.assertEqual(len(history), 1)
        event = history[0]
        self.assertEqual(event["network_type"], "hist_net")
        self.assertEqual(event["event_type"], "training")
        self.assertEqual(event["metrics"]["accuracy"], 0.95)
        self.assertEqual(event["description"], "epoch 1")
        self.assertIsInstance(event["timestamp"], float)

    def test_get_learning_history_multiple_events(self):
        """Multiple events should be returned in reverse chronological order."""
        for i in range(5):
            self.store.log_learning_event(
                "multi_net", "training",
                metrics={"epoch": i},
                description=f"event_{i}"
            )
        history = self.store.get_learning_history(network_type="multi_net")
        self.assertEqual(len(history), 5)
        # Most recent first (ORDER BY timestamp DESC)
        epochs = [e["metrics"]["epoch"] for e in history]
        self.assertEqual(epochs, [4, 3, 2, 1, 0])

    def test_get_learning_history_filter_by_type(self):
        """History retrieval should filter by network type when specified."""
        self.store.log_learning_event("net_a", "training")
        self.store.log_learning_event("net_b", "adaptation")
        self.store.log_learning_event("net_a", "consolidation")

        a_history = self.store.get_learning_history(network_type="net_a")
        b_history = self.store.get_learning_history(network_type="net_b")
        all_history = self.store.get_learning_history()

        self.assertEqual(len(a_history), 2)
        self.assertEqual(len(b_history), 1)
        self.assertEqual(len(all_history), 3)

    def test_get_learning_history_limit(self):
        """The limit parameter should cap the number of returned events."""
        for i in range(20):
            self.store.log_learning_event("limited_net", "training", metrics={"i": i})

        limited = self.store.get_learning_history(network_type="limited_net", limit=5)
        self.assertEqual(len(limited), 5)

    def test_get_learning_history_empty(self):
        """History should be empty for a network type with no events."""
        history = self.store.get_learning_history(network_type="no_events_net")
        self.assertEqual(history, [])

    def test_log_event_without_metrics(self):
        """Logging an event with no metrics should store an empty dict."""
        self.store.log_learning_event("bare_net", "adaptation")
        history = self.store.get_learning_history(network_type="bare_net")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["metrics"], {})

    def test_log_event_various_event_types(self):
        """Different event types should be preserved correctly."""
        event_types = ["training", "consolidation", "adaptation", "evaluation"]
        for et in event_types:
            self.store.log_learning_event("varied_net", et)
        history = self.store.get_learning_history(network_type="varied_net")
        stored_types = sorted([e["event_type"] for e in history])
        self.assertEqual(stored_types, sorted(event_types))


class TestGetStats(unittest.TestCase):
    """Tests for the get_stats() utility method."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="atlas_test_")
        self.store = NetworkStore(data_dir=self.test_dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_stats_empty_store(self):
        """Stats on a fresh store should show all zeroes."""
        stats = self.store.get_stats()
        self.assertEqual(stats["stored_networks"], 0)
        self.assertEqual(stats["total_snapshots"], 0)
        self.assertEqual(stats["history_events"], 0)
        self.assertIn("db_path", stats)
        self.assertIn("networks", stats)
        self.assertEqual(stats["networks"], {})

    def test_stats_after_operations(self):
        """Stats should reflect the number of saved items."""
        self.store.save_weights("net_1", np.zeros(3))
        self.store.save_weights("net_2", np.ones(3))
        self.store.create_snapshot("net_1", np.zeros(3))
        self.store.create_snapshot("net_1", np.ones(3))
        self.store.create_snapshot("net_2", np.zeros(3))
        self.store.log_learning_event("net_1", "training")

        stats = self.store.get_stats()
        self.assertEqual(stats["stored_networks"], 2)
        self.assertEqual(stats["total_snapshots"], 3)
        self.assertEqual(stats["history_events"], 1)
        self.assertIn("net_1", stats["networks"])
        self.assertIn("net_2", stats["networks"])


class TestCloseAndCleanup(unittest.TestCase):
    """Tests for close() and teardown behaviour."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="atlas_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_close_does_not_raise(self):
        """Calling close() should not raise an exception."""
        store = NetworkStore(data_dir=self.test_dir)
        store.close()

    def test_double_close_does_not_raise(self):
        """Calling close() twice should be safe."""
        store = NetworkStore(data_dir=self.test_dir)
        store.close()
        store.close()

    def test_temporary_directory_cleanup(self):
        """After rmtree, the temporary directory should be gone."""
        tmp = tempfile.mkdtemp(prefix="atlas_cleanup_test_")
        store = NetworkStore(data_dir=tmp)
        store.save_weights("cleanup_net", np.array([1.0, 2.0]))
        store.close()
        shutil.rmtree(tmp)
        self.assertFalse(os.path.exists(tmp))


class TestNetworkSnapshotDataclass(unittest.TestCase):
    """Tests for the NetworkSnapshot dataclass."""

    def test_create_network_snapshot(self):
        """NetworkSnapshot should be constructable with the expected fields."""
        weights = np.array([1.0, 2.0, 3.0])
        snap = NetworkSnapshot(
            snapshot_id="test_abc123",
            timestamp=1000.0,
            network_type="test_net",
            metadata={"key": "value"},
            weights_shape=(3,),
            weights=weights,
        )
        self.assertEqual(snap.snapshot_id, "test_abc123")
        self.assertEqual(snap.timestamp, 1000.0)
        self.assertEqual(snap.network_type, "test_net")
        self.assertEqual(snap.metadata, {"key": "value"})
        self.assertEqual(snap.weights_shape, (3,))
        np.testing.assert_array_equal(snap.weights, weights)

    def test_snapshot_fields_are_accessible(self):
        """All fields defined in the dataclass should be accessible attributes."""
        snap = NetworkSnapshot(
            snapshot_id="id",
            timestamp=0.0,
            network_type="type",
            metadata={},
            weights_shape=(1,),
            weights=np.zeros(1),
        )
        expected_fields = {"snapshot_id", "timestamp", "network_type",
                           "metadata", "weights_shape", "weights"}
        for field in expected_fields:
            self.assertTrue(hasattr(snap, field), f"Missing field: {field}")


class TestBackendIntegration(unittest.TestCase):
    """Tests that weights created via xp backend survive round-trip storage."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="atlas_test_")
        self.store = NetworkStore(data_dir=self.test_dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_xp_created_weights_round_trip(self):
        """Weights created with xp (numpy or cupy) should save and load correctly."""
        original = xp.linspace(0.0, 1.0, 50)
        # NetworkStore uses numpy internally; convert to numpy for storage
        original_np = np.asarray(original)
        self.store.save_weights("xp_net", original_np)
        loaded, _ = self.store.load_weights("xp_net")
        np.testing.assert_allclose(loaded, original_np, atol=1e-12)

    def test_xp_random_weights_round_trip(self):
        """Random weights from xp should round-trip through the store."""
        rng = np.random.RandomState(42)
        original = rng.randn(10, 10).astype(np.float32)
        self.store.save_weights("random_net", original)
        loaded, _ = self.store.load_weights("random_net")
        np.testing.assert_array_equal(loaded, original)


if __name__ == "__main__":
    unittest.main()
