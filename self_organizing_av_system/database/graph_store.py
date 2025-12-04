"""
Graph Store - SQLite-backed graph storage for Atlas

Provides persistent storage for:
- Concept nodes and their metadata
- Relationships between concepts (edges)
- Graph structure for semantic memory
"""

import logging
import sqlite3
import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of relationships between concepts."""
    IS_A = "is_a"
    HAS_A = "has_a"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    CAUSES = "causes"
    ENABLES = "enables"
    PREVENTS = "prevents"
    USED_FOR = "used_for"
    ATTRIBUTE = "attribute"
    MEMBER_OF = "member_of"
    BEFORE = "before"
    AFTER = "after"
    LOCATED_AT = "located_at"
    CREATED_BY = "created_by"


@dataclass
class GraphNode:
    """A node in the concept graph."""
    name: str
    node_type: str
    attributes: Dict[str, Any]
    activation_count: int
    consolidation_strength: float
    created_at: float
    last_activated: float


@dataclass
class GraphEdge:
    """An edge (relationship) in the concept graph."""
    source: str
    target: str
    relation_type: str
    weight: float
    bidirectional: bool
    metadata: Dict[str, Any]


class GraphStore:
    """
    SQLite-backed graph store for Atlas semantic memory.

    Stores concept nodes and their relationships using adjacency lists.
    Supports efficient graph traversal and relationship queries.
    """

    def __init__(self, data_dir: str = "atlas_data"):
        """
        Initialize the graph store.

        Args:
            data_dir: Directory for persistent storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "atlas_graph.db"
        self._local = threading.local()

        self._init_database()
        logger.info(f"GraphStore initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
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
            # Nodes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    name TEXT PRIMARY KEY,
                    node_type TEXT DEFAULT 'concept',
                    attributes_json TEXT DEFAULT '{}',
                    activation_count INTEGER DEFAULT 0,
                    consolidation_strength REAL DEFAULT 0.0,
                    created_at REAL,
                    last_activated REAL
                )
            """)

            # Edges table (adjacency list)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    bidirectional INTEGER DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY (source) REFERENCES nodes(name) ON DELETE CASCADE,
                    FOREIGN KEY (target) REFERENCES nodes(name) ON DELETE CASCADE,
                    UNIQUE(source, target, relation_type)
                )
            """)

            # Indexes for efficient lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(relation_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_strength ON nodes(consolidation_strength)")

    # ==================== Node Operations ====================

    def add_node(self, name: str, node_type: str = "concept",
                 attributes: Optional[Dict[str, Any]] = None,
                 consolidation_strength: float = 0.0,
                 created_at: Optional[float] = None,
                 last_activated: Optional[float] = None) -> bool:
        """
        Add or update a node in the graph.

        Args:
            name: Unique node identifier
            node_type: Type of node (concept, entity, etc.)
            attributes: Node attributes as dictionary
            consolidation_strength: Consolidation strength (0-1)
            created_at: Creation timestamp
            last_activated: Last activation timestamp

        Returns:
            True if successful
        """
        import time
        now = time.time()

        with self._transaction() as conn:
            try:
                conn.execute("""
                    INSERT INTO nodes (name, node_type, attributes_json, activation_count,
                                      consolidation_strength, created_at, last_activated)
                    VALUES (?, ?, ?, 0, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        node_type = excluded.node_type,
                        attributes_json = excluded.attributes_json,
                        consolidation_strength = excluded.consolidation_strength,
                        last_activated = excluded.last_activated
                """, (
                    name,
                    node_type,
                    json.dumps(attributes or {}),
                    consolidation_strength,
                    created_at or now,
                    last_activated or now
                ))
                return True
            except Exception as e:
                logger.error(f"Failed to add node {name}: {e}")
                return False

    def get_node(self, name: str) -> Optional[GraphNode]:
        """Get a node by name."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM nodes WHERE name = ?", (name,)
        )
        row = cursor.fetchone()

        if row:
            return GraphNode(
                name=row['name'],
                node_type=row['node_type'],
                attributes=json.loads(row['attributes_json']),
                activation_count=row['activation_count'],
                consolidation_strength=row['consolidation_strength'],
                created_at=row['created_at'],
                last_activated=row['last_activated']
            )
        return None

    def update_node_activation(self, name: str) -> bool:
        """Increment activation count and update last_activated timestamp."""
        import time
        with self._transaction() as conn:
            try:
                conn.execute("""
                    UPDATE nodes
                    SET activation_count = activation_count + 1,
                        last_activated = ?
                    WHERE name = ?
                """, (time.time(), name))
                return True
            except Exception as e:
                logger.error(f"Failed to update node activation {name}: {e}")
                return False

    def update_node_strength(self, name: str, strength: float) -> bool:
        """Update consolidation strength for a node."""
        with self._transaction() as conn:
            try:
                conn.execute("""
                    UPDATE nodes SET consolidation_strength = ? WHERE name = ?
                """, (strength, name))
                return True
            except Exception as e:
                logger.error(f"Failed to update node strength {name}: {e}")
                return False

    def delete_node(self, name: str) -> bool:
        """Delete a node and all its edges."""
        with self._transaction() as conn:
            try:
                conn.execute("DELETE FROM nodes WHERE name = ?", (name,))
                return True
            except Exception as e:
                logger.error(f"Failed to delete node {name}: {e}")
                return False

    def node_exists(self, name: str) -> bool:
        """Check if a node exists."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT 1 FROM nodes WHERE name = ? LIMIT 1", (name,)
        )
        return cursor.fetchone() is not None

    def count_nodes(self) -> int:
        """Get total node count."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM nodes")
        return cursor.fetchone()[0]

    def get_all_node_names(self) -> List[str]:
        """Get all node names."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT name FROM nodes")
        return [row[0] for row in cursor.fetchall()]

    def get_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM nodes WHERE node_type = ?", (node_type,)
        )
        return [GraphNode(
            name=row['name'],
            node_type=row['node_type'],
            attributes=json.loads(row['attributes_json']),
            activation_count=row['activation_count'],
            consolidation_strength=row['consolidation_strength'],
            created_at=row['created_at'],
            last_activated=row['last_activated']
        ) for row in cursor.fetchall()]

    def get_weak_nodes(self, threshold: float = 0.3, limit: int = 100) -> List[str]:
        """Get nodes with low consolidation strength."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT name FROM nodes
            WHERE consolidation_strength < ?
            ORDER BY consolidation_strength ASC
            LIMIT ?
        """, (threshold, limit))
        return [row[0] for row in cursor.fetchall()]

    # ==================== Edge Operations ====================

    def add_edge(self, source: str, target: str, relation_type: str,
                 weight: float = 1.0, bidirectional: bool = False,
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add or update an edge in the graph.

        Args:
            source: Source node name
            target: Target node name
            relation_type: Type of relationship
            weight: Edge weight (strength)
            bidirectional: Whether the relationship is bidirectional
            metadata: Additional edge metadata

        Returns:
            True if successful
        """
        with self._transaction() as conn:
            try:
                conn.execute("""
                    INSERT INTO edges (source, target, relation_type, weight, bidirectional, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source, target, relation_type) DO UPDATE SET
                        weight = excluded.weight,
                        bidirectional = excluded.bidirectional,
                        metadata_json = excluded.metadata_json
                """, (
                    source,
                    target,
                    relation_type,
                    weight,
                    1 if bidirectional else 0,
                    json.dumps(metadata or {})
                ))
                return True
            except Exception as e:
                logger.error(f"Failed to add edge {source}->{target}: {e}")
                return False

    def get_edge(self, source: str, target: str,
                 relation_type: Optional[str] = None) -> Optional[GraphEdge]:
        """Get an edge between two nodes."""
        conn = self._get_connection()

        if relation_type:
            cursor = conn.execute("""
                SELECT * FROM edges
                WHERE source = ? AND target = ? AND relation_type = ?
            """, (source, target, relation_type))
        else:
            cursor = conn.execute("""
                SELECT * FROM edges WHERE source = ? AND target = ?
            """, (source, target))

        row = cursor.fetchone()
        if row:
            return GraphEdge(
                source=row['source'],
                target=row['target'],
                relation_type=row['relation_type'],
                weight=row['weight'],
                bidirectional=bool(row['bidirectional']),
                metadata=json.loads(row['metadata_json'])
            )
        return None

    def get_outgoing_edges(self, source: str) -> List[GraphEdge]:
        """Get all edges from a source node."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM edges WHERE source = ?", (source,)
        )
        return [GraphEdge(
            source=row['source'],
            target=row['target'],
            relation_type=row['relation_type'],
            weight=row['weight'],
            bidirectional=bool(row['bidirectional']),
            metadata=json.loads(row['metadata_json'])
        ) for row in cursor.fetchall()]

    def get_incoming_edges(self, target: str) -> List[GraphEdge]:
        """Get all edges to a target node."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM edges WHERE target = ?", (target,)
        )
        return [GraphEdge(
            source=row['source'],
            target=row['target'],
            relation_type=row['relation_type'],
            weight=row['weight'],
            bidirectional=bool(row['bidirectional']),
            metadata=json.loads(row['metadata_json'])
        ) for row in cursor.fetchall()]

    def get_neighbors(self, node: str, include_incoming: bool = True) -> Set[str]:
        """Get all neighboring nodes."""
        neighbors = set()
        conn = self._get_connection()

        # Outgoing
        cursor = conn.execute("SELECT target FROM edges WHERE source = ?", (node,))
        neighbors.update(row[0] for row in cursor.fetchall())

        # Incoming (if requested)
        if include_incoming:
            cursor = conn.execute("SELECT source FROM edges WHERE target = ?", (node,))
            neighbors.update(row[0] for row in cursor.fetchall())

        return neighbors

    def get_node_degree(self, node: str) -> Tuple[int, int]:
        """Get in-degree and out-degree of a node."""
        conn = self._get_connection()

        cursor = conn.execute("SELECT COUNT(*) FROM edges WHERE source = ?", (node,))
        out_degree = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM edges WHERE target = ?", (node,))
        in_degree = cursor.fetchone()[0]

        return in_degree, out_degree

    def update_edge_weight(self, source: str, target: str,
                          relation_type: str, weight: float) -> bool:
        """Update the weight of an edge."""
        with self._transaction() as conn:
            try:
                conn.execute("""
                    UPDATE edges SET weight = ?
                    WHERE source = ? AND target = ? AND relation_type = ?
                """, (weight, source, target, relation_type))
                return True
            except Exception as e:
                logger.error(f"Failed to update edge weight: {e}")
                return False

    def delete_edge(self, source: str, target: str,
                   relation_type: Optional[str] = None) -> bool:
        """Delete an edge."""
        with self._transaction() as conn:
            try:
                if relation_type:
                    conn.execute("""
                        DELETE FROM edges
                        WHERE source = ? AND target = ? AND relation_type = ?
                    """, (source, target, relation_type))
                else:
                    conn.execute(
                        "DELETE FROM edges WHERE source = ? AND target = ?",
                        (source, target)
                    )
                return True
            except Exception as e:
                logger.error(f"Failed to delete edge: {e}")
                return False

    def count_edges(self) -> int:
        """Get total edge count."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM edges")
        return cursor.fetchone()[0]

    # ==================== Graph Traversal ====================

    def find_path(self, source: str, target: str,
                  max_depth: int = 5) -> Optional[List[str]]:
        """
        Find a path between two nodes using BFS.

        Args:
            source: Source node name
            target: Target node name
            max_depth: Maximum path length

        Returns:
            List of node names forming the path, or None if no path exists
        """
        if source == target:
            return [source]

        from collections import deque

        visited = {source}
        queue = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            for neighbor in self.get_neighbors(current, include_incoming=True):
                if neighbor == target:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def get_subgraph(self, center: str, depth: int = 2) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Get a subgraph around a center node.

        Args:
            center: Center node name
            depth: How many hops to include

        Returns:
            Tuple of (nodes, edges) in the subgraph
        """
        visited_nodes = set()
        current_level = {center}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                if node not in visited_nodes:
                    visited_nodes.add(node)
                    next_level.update(self.get_neighbors(node, include_incoming=True))
            current_level = next_level - visited_nodes

        visited_nodes.update(current_level)

        # Get all nodes
        nodes = []
        for name in visited_nodes:
            node = self.get_node(name)
            if node:
                nodes.append(node)

        # Get all edges between visited nodes
        edges = []
        conn = self._get_connection()
        placeholders = ','.join('?' * len(visited_nodes))
        cursor = conn.execute(f"""
            SELECT * FROM edges
            WHERE source IN ({placeholders}) AND target IN ({placeholders})
        """, list(visited_nodes) + list(visited_nodes))

        for row in cursor.fetchall():
            edges.append(GraphEdge(
                source=row['source'],
                target=row['target'],
                relation_type=row['relation_type'],
                weight=row['weight'],
                bidirectional=bool(row['bidirectional']),
                metadata=json.loads(row['metadata_json'])
            ))

        return nodes, edges

    # ==================== Utility Methods ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        conn = self._get_connection()

        cursor = conn.execute("""
            SELECT
                (SELECT COUNT(*) FROM nodes) as node_count,
                (SELECT COUNT(*) FROM edges) as edge_count,
                (SELECT AVG(consolidation_strength) FROM nodes) as avg_strength
        """)
        row = cursor.fetchone()

        return {
            'db_path': str(self.db_path),
            'node_count': row[0],
            'edge_count': row[1],
            'avg_consolidation_strength': row[2] or 0.0,
        }

    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
        logger.info("GraphStore closed")
