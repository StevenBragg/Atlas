"""
Network Store - Persistence for Atlas Self-Organizing Network Weights

Stores the learned weights that make up Atlas's "brain":
- Self-organizing network connections
- Canvas generation weights
- Neuron states and topology
- Learning history snapshots
"""

import logging
import sqlite3
import json
import numpy as np
import threading
import time
import io
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class NetworkSnapshot:
    """A snapshot of network state at a point in time."""
    snapshot_id: str
    timestamp: float
    network_type: str  # 'self_organizing', 'canvas', 'meta_learner', etc.
    metadata: Dict[str, Any]
    weights_shape: Tuple[int, ...]
    weights: np.ndarray


class NetworkStore:
    """
    SQLite-backed storage for Atlas network weights and state.

    Stores:
    - Current network weights (live state)
    - Historical snapshots (for rollback/analysis)
    - Network metadata and configuration
    """

    MAX_SNAPSHOTS_PER_TYPE = 10  # Keep last N snapshots per network type

    def __init__(self, data_dir: str = "atlas_data"):
        """
        Initialize the network store.

        Args:
            data_dir: Directory for persistent storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "atlas_network.db"
        self._local = threading.local()

        self._init_database()
        logger.info(f"NetworkStore initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    def _init_database(self):
        """Initialize database schema."""
        with self._transaction() as conn:
            # Current network state table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS network_state (
                    network_type TEXT PRIMARY KEY,
                    weights_blob BLOB,
                    weights_shape TEXT,
                    weights_dtype TEXT,
                    metadata_json TEXT DEFAULT '{}',
                    updated_at REAL
                )
            """)

            # Snapshots table for history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS network_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_id TEXT UNIQUE,
                    network_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    weights_blob BLOB,
                    weights_shape TEXT,
                    weights_dtype TEXT,
                    metadata_json TEXT DEFAULT '{}',
                    description TEXT
                )
            """)

            # Learning history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    network_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    metrics_json TEXT DEFAULT '{}',
                    description TEXT
                )
            """)

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_type ON network_snapshots(network_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_time ON network_snapshots(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_type ON learning_history(network_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_time ON learning_history(timestamp)")

    # ==================== Weight Serialization ====================

    def _serialize_weights(self, weights: np.ndarray) -> Tuple[bytes, str, str]:
        """Serialize numpy array to bytes for storage."""
        buffer = io.BytesIO()
        np.save(buffer, weights, allow_pickle=False)
        return buffer.getvalue(), json.dumps(weights.shape), str(weights.dtype)

    def _deserialize_weights(self, blob: bytes, shape_str: str, dtype_str: str) -> np.ndarray:
        """Deserialize bytes back to numpy array."""
        buffer = io.BytesIO(blob)
        weights = np.load(buffer, allow_pickle=False)
        return weights

    # ==================== Current State Operations ====================

    def save_weights(self, network_type: str, weights: np.ndarray,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save current network weights.

        Args:
            network_type: Type of network ('self_organizing', 'canvas', etc.)
            weights: Network weights as numpy array
            metadata: Additional metadata to store

        Returns:
            True if successful
        """
        blob, shape_str, dtype_str = self._serialize_weights(weights)

        with self._transaction() as conn:
            try:
                conn.execute("""
                    INSERT INTO network_state
                    (network_type, weights_blob, weights_shape, weights_dtype, metadata_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(network_type) DO UPDATE SET
                        weights_blob = excluded.weights_blob,
                        weights_shape = excluded.weights_shape,
                        weights_dtype = excluded.weights_dtype,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                """, (
                    network_type,
                    blob,
                    shape_str,
                    dtype_str,
                    json.dumps(metadata or {}),
                    time.time()
                ))
                logger.debug(f"Saved weights for {network_type}: shape={weights.shape}")
                return True
            except Exception as e:
                logger.error(f"Failed to save weights for {network_type}: {e}")
                return False

    def load_weights(self, network_type: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Load current network weights.

        Args:
            network_type: Type of network to load

        Returns:
            Tuple of (weights, metadata) or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT weights_blob, weights_shape, weights_dtype, metadata_json FROM network_state WHERE network_type = ?",
            (network_type,)
        )
        row = cursor.fetchone()

        if row and row['weights_blob']:
            try:
                weights = self._deserialize_weights(
                    row['weights_blob'],
                    row['weights_shape'],
                    row['weights_dtype']
                )
                metadata = json.loads(row['metadata_json'])
                logger.debug(f"Loaded weights for {network_type}: shape={weights.shape}")
                return weights, metadata
            except Exception as e:
                logger.error(f"Failed to load weights for {network_type}: {e}")

        return None

    def delete_weights(self, network_type: str) -> bool:
        """Delete current weights for a network type."""
        with self._transaction() as conn:
            try:
                conn.execute("DELETE FROM network_state WHERE network_type = ?", (network_type,))
                return True
            except Exception as e:
                logger.error(f"Failed to delete weights for {network_type}: {e}")
                return False

    def has_weights(self, network_type: str) -> bool:
        """Check if weights exist for a network type."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT 1 FROM network_state WHERE network_type = ? LIMIT 1",
            (network_type,)
        )
        return cursor.fetchone() is not None

    def get_all_network_types(self) -> List[str]:
        """Get all stored network types."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT network_type FROM network_state")
        return [row[0] for row in cursor.fetchall()]

    # ==================== Snapshot Operations ====================

    def create_snapshot(self, network_type: str, weights: np.ndarray,
                       metadata: Optional[Dict[str, Any]] = None,
                       description: str = "") -> str:
        """
        Create a snapshot of network weights.

        Args:
            network_type: Type of network
            weights: Network weights
            metadata: Additional metadata
            description: Human-readable description

        Returns:
            Snapshot ID
        """
        import uuid
        snapshot_id = f"{network_type}_{uuid.uuid4().hex[:8]}"
        blob, shape_str, dtype_str = self._serialize_weights(weights)

        with self._transaction() as conn:
            try:
                conn.execute("""
                    INSERT INTO network_snapshots
                    (snapshot_id, network_type, timestamp, weights_blob, weights_shape,
                     weights_dtype, metadata_json, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot_id,
                    network_type,
                    time.time(),
                    blob,
                    shape_str,
                    dtype_str,
                    json.dumps(metadata or {}),
                    description
                ))

                # Cleanup old snapshots
                self._cleanup_old_snapshots(conn, network_type)

                logger.info(f"Created snapshot {snapshot_id} for {network_type}")
                return snapshot_id
            except Exception as e:
                logger.error(f"Failed to create snapshot: {e}")
                return ""

    def _cleanup_old_snapshots(self, conn: sqlite3.Connection, network_type: str):
        """Remove old snapshots beyond the maximum limit."""
        conn.execute("""
            DELETE FROM network_snapshots
            WHERE network_type = ?
            AND id NOT IN (
                SELECT id FROM network_snapshots
                WHERE network_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            )
        """, (network_type, network_type, self.MAX_SNAPSHOTS_PER_TYPE))

    def load_snapshot(self, snapshot_id: str) -> Optional[NetworkSnapshot]:
        """Load a specific snapshot."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM network_snapshots WHERE snapshot_id = ?",
            (snapshot_id,)
        )
        row = cursor.fetchone()

        if row and row['weights_blob']:
            try:
                weights = self._deserialize_weights(
                    row['weights_blob'],
                    row['weights_shape'],
                    row['weights_dtype']
                )
                return NetworkSnapshot(
                    snapshot_id=row['snapshot_id'],
                    timestamp=row['timestamp'],
                    network_type=row['network_type'],
                    metadata=json.loads(row['metadata_json']),
                    weights_shape=tuple(json.loads(row['weights_shape'])),
                    weights=weights
                )
            except Exception as e:
                logger.error(f"Failed to load snapshot {snapshot_id}: {e}")

        return None

    def get_latest_snapshot(self, network_type: str) -> Optional[NetworkSnapshot]:
        """Get the most recent snapshot for a network type."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT snapshot_id FROM network_snapshots
            WHERE network_type = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (network_type,))
        row = cursor.fetchone()

        if row:
            return self.load_snapshot(row['snapshot_id'])
        return None

    def list_snapshots(self, network_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available snapshots."""
        conn = self._get_connection()

        if network_type:
            cursor = conn.execute("""
                SELECT snapshot_id, network_type, timestamp, weights_shape, description
                FROM network_snapshots
                WHERE network_type = ?
                ORDER BY timestamp DESC
            """, (network_type,))
        else:
            cursor = conn.execute("""
                SELECT snapshot_id, network_type, timestamp, weights_shape, description
                FROM network_snapshots
                ORDER BY timestamp DESC
            """)

        return [
            {
                'snapshot_id': row['snapshot_id'],
                'network_type': row['network_type'],
                'timestamp': row['timestamp'],
                'weights_shape': json.loads(row['weights_shape']),
                'description': row['description']
            }
            for row in cursor.fetchall()
        ]

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a specific snapshot."""
        with self._transaction() as conn:
            try:
                conn.execute("DELETE FROM network_snapshots WHERE snapshot_id = ?", (snapshot_id,))
                return True
            except Exception as e:
                logger.error(f"Failed to delete snapshot {snapshot_id}: {e}")
                return False

    # ==================== Learning History ====================

    def log_learning_event(self, network_type: str, event_type: str,
                          metrics: Optional[Dict[str, Any]] = None,
                          description: str = "") -> bool:
        """
        Log a learning event.

        Args:
            network_type: Type of network
            event_type: Type of event ('training', 'consolidation', 'adaptation', etc.)
            metrics: Event metrics (accuracy, loss, etc.)
            description: Human-readable description

        Returns:
            True if successful
        """
        with self._transaction() as conn:
            try:
                conn.execute("""
                    INSERT INTO learning_history
                    (network_type, timestamp, event_type, metrics_json, description)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    network_type,
                    time.time(),
                    event_type,
                    json.dumps(metrics or {}),
                    description
                ))
                return True
            except Exception as e:
                logger.error(f"Failed to log learning event: {e}")
                return False

    def get_learning_history(self, network_type: Optional[str] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get learning history events."""
        conn = self._get_connection()

        if network_type:
            cursor = conn.execute("""
                SELECT * FROM learning_history
                WHERE network_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (network_type, limit))
        else:
            cursor = conn.execute("""
                SELECT * FROM learning_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        return [
            {
                'network_type': row['network_type'],
                'timestamp': row['timestamp'],
                'event_type': row['event_type'],
                'metrics': json.loads(row['metrics_json']),
                'description': row['description']
            }
            for row in cursor.fetchall()
        ]

    # ==================== Utility Methods ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        conn = self._get_connection()

        cursor = conn.execute("""
            SELECT
                (SELECT COUNT(*) FROM network_state) as state_count,
                (SELECT COUNT(*) FROM network_snapshots) as snapshot_count,
                (SELECT COUNT(*) FROM learning_history) as history_count
        """)
        row = cursor.fetchone()

        # Get network types with their last update time
        cursor = conn.execute(
            "SELECT network_type, updated_at FROM network_state"
        )
        networks = {r['network_type']: r['updated_at'] for r in cursor.fetchall()}

        return {
            'db_path': str(self.db_path),
            'stored_networks': state_count if (state_count := row[0]) else 0,
            'total_snapshots': row[1],
            'history_events': row[2],
            'networks': networks
        }

    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
        logger.info("NetworkStore closed")
