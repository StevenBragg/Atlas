"""
Comprehensive tests for GraphStore, GraphNode, GraphEdge, and RelationType.

Tests cover:
- GraphStore initialization with temporary database
- Node CRUD operations (add, get, update, delete)
- Edge CRUD operations with various RelationTypes
- Graph traversal: get_neighbors, find_path, get_subgraph
- Weak node detection via get_weak_nodes
- Proper cleanup of temporary files
"""

import os
import sys
import shutil
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.database.graph_store import (
    GraphStore,
    GraphNode,
    GraphEdge,
    RelationType,
)


class TestRelationType(unittest.TestCase):
    """Tests for the RelationType enum."""

    def test_relation_type_values(self):
        """Verify that key RelationType members have expected string values."""
        self.assertEqual(RelationType.IS_A.value, "is_a")
        self.assertEqual(RelationType.PART_OF.value, "part_of")
        self.assertEqual(RelationType.SIMILAR_TO.value, "similar_to")
        self.assertEqual(RelationType.CAUSES.value, "causes")
        self.assertEqual(RelationType.BEFORE.value, "before")
        self.assertEqual(RelationType.AFTER.value, "after")
        self.assertEqual(RelationType.HAS_A.value, "has_a")
        self.assertEqual(RelationType.MEMBER_OF.value, "member_of")

    def test_relation_type_enum_members(self):
        """All expected members should be present in the enum."""
        expected = {
            "IS_A", "HAS_A", "PART_OF", "SIMILAR_TO", "OPPOSITE_OF",
            "CAUSES", "ENABLES", "PREVENTS", "USED_FOR", "ATTRIBUTE",
            "MEMBER_OF", "BEFORE", "AFTER", "LOCATED_AT", "CREATED_BY",
        }
        actual = set(RelationType.__members__.keys())
        self.assertEqual(actual, expected)


class TestGraphNodeDataclass(unittest.TestCase):
    """Tests for the GraphNode dataclass."""

    def test_graph_node_creation(self):
        """GraphNode should hold all expected fields."""
        now = time.time()
        node = GraphNode(
            name="test_node",
            node_type="concept",
            attributes={"color": "blue"},
            activation_count=5,
            consolidation_strength=0.8,
            created_at=now,
            last_activated=now,
        )
        self.assertEqual(node.name, "test_node")
        self.assertEqual(node.node_type, "concept")
        self.assertEqual(node.attributes, {"color": "blue"})
        self.assertEqual(node.activation_count, 5)
        self.assertAlmostEqual(node.consolidation_strength, 0.8)
        self.assertEqual(node.created_at, now)
        self.assertEqual(node.last_activated, now)


class TestGraphEdgeDataclass(unittest.TestCase):
    """Tests for the GraphEdge dataclass."""

    def test_graph_edge_creation(self):
        """GraphEdge should hold all expected fields."""
        edge = GraphEdge(
            source="A",
            target="B",
            relation_type=RelationType.IS_A.value,
            weight=0.9,
            bidirectional=True,
            metadata={"context": "test"},
        )
        self.assertEqual(edge.source, "A")
        self.assertEqual(edge.target, "B")
        self.assertEqual(edge.relation_type, "is_a")
        self.assertAlmostEqual(edge.weight, 0.9)
        self.assertTrue(edge.bidirectional)
        self.assertEqual(edge.metadata, {"context": "test"})


class TestGraphStoreInit(unittest.TestCase):
    """Tests for GraphStore initialization and database creation."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_initialization_creates_directory_and_db(self):
        """GraphStore should create data directory and database file on init."""
        data_dir = os.path.join(self.tmp_dir, "graph_data")
        store = GraphStore(data_dir=data_dir)
        try:
            self.assertTrue(os.path.isdir(data_dir))
            self.assertTrue(os.path.isfile(os.path.join(data_dir, "atlas_graph.db")))
        finally:
            store.close()

    def test_initialization_empty_graph(self):
        """A freshly initialized graph store should have zero nodes and edges."""
        store = GraphStore(data_dir=self.tmp_dir)
        try:
            self.assertEqual(store.count_nodes(), 0)
            self.assertEqual(store.count_edges(), 0)
        finally:
            store.close()

    def test_get_stats_empty_graph(self):
        """Stats on an empty graph should report zeros."""
        store = GraphStore(data_dir=self.tmp_dir)
        try:
            stats = store.get_stats()
            self.assertEqual(stats['node_count'], 0)
            self.assertEqual(stats['edge_count'], 0)
            self.assertAlmostEqual(stats['avg_consolidation_strength'], 0.0)
            self.assertIn('db_path', stats)
        finally:
            store.close()


class TestGraphStoreAddNode(unittest.TestCase):
    """Tests for the add_node method."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store = GraphStore(data_dir=self.tmp_dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_add_node_returns_true(self):
        """add_node should return True on success."""
        result = self.store.add_node("animal")
        self.assertTrue(result)

    def test_add_node_increments_count(self):
        """Adding a node should increase node count by one."""
        self.store.add_node("alpha")
        self.assertEqual(self.store.count_nodes(), 1)
        self.store.add_node("beta")
        self.assertEqual(self.store.count_nodes(), 2)

    def test_add_node_default_values(self):
        """A node added with defaults should have type 'concept' and strength 0."""
        self.store.add_node("default_node")
        node = self.store.get_node("default_node")
        self.assertIsNotNone(node)
        self.assertEqual(node.name, "default_node")
        self.assertEqual(node.node_type, "concept")
        self.assertEqual(node.attributes, {})
        self.assertEqual(node.activation_count, 0)
        self.assertAlmostEqual(node.consolidation_strength, 0.0)

    def test_add_node_with_custom_attributes(self):
        """Attributes dict should be stored and retrievable as JSON."""
        attrs = {"color": "red", "size": 42, "nested": {"key": "val"}}
        self.store.add_node("rich_node", node_type="entity", attributes=attrs)
        node = self.store.get_node("rich_node")
        self.assertIsNotNone(node)
        self.assertEqual(node.node_type, "entity")
        self.assertEqual(node.attributes, attrs)

    def test_add_node_with_custom_strength(self):
        """consolidation_strength should be stored correctly."""
        self.store.add_node("strong_node", consolidation_strength=0.95)
        node = self.store.get_node("strong_node")
        self.assertAlmostEqual(node.consolidation_strength, 0.95)

    def test_add_node_with_timestamps(self):
        """created_at and last_activated should be stored when provided."""
        ts = 1700000000.0
        self.store.add_node("ts_node", created_at=ts, last_activated=ts + 100)
        node = self.store.get_node("ts_node")
        self.assertAlmostEqual(node.created_at, ts)
        self.assertAlmostEqual(node.last_activated, ts + 100)

    def test_add_duplicate_node_updates(self):
        """Re-adding a node with same name should update, not duplicate."""
        self.store.add_node("dup", node_type="concept")
        self.store.add_node("dup", node_type="entity")
        self.assertEqual(self.store.count_nodes(), 1)
        node = self.store.get_node("dup")
        self.assertEqual(node.node_type, "entity")

    def test_node_exists(self):
        """node_exists should return True for existing nodes, False otherwise."""
        self.assertFalse(self.store.node_exists("ghost"))
        self.store.add_node("ghost")
        self.assertTrue(self.store.node_exists("ghost"))

    def test_get_node_nonexistent(self):
        """get_node should return None for a name that does not exist."""
        self.assertIsNone(self.store.get_node("nonexistent"))

    def test_get_all_node_names(self):
        """get_all_node_names should return all stored names."""
        for name in ["x", "y", "z"]:
            self.store.add_node(name)
        names = self.store.get_all_node_names()
        self.assertEqual(sorted(names), ["x", "y", "z"])

    def test_get_nodes_by_type(self):
        """Filtering by node_type should return only matching nodes."""
        self.store.add_node("a", node_type="concept")
        self.store.add_node("b", node_type="entity")
        self.store.add_node("c", node_type="concept")
        concepts = self.store.get_nodes_by_type("concept")
        concept_names = sorted([n.name for n in concepts])
        self.assertEqual(concept_names, ["a", "c"])

    def test_delete_node(self):
        """Deleting a node should remove it from the store."""
        self.store.add_node("to_delete")
        self.assertTrue(self.store.node_exists("to_delete"))
        result = self.store.delete_node("to_delete")
        self.assertTrue(result)
        self.assertFalse(self.store.node_exists("to_delete"))
        self.assertEqual(self.store.count_nodes(), 0)

    def test_update_node_activation(self):
        """update_node_activation should increment count and update timestamp."""
        self.store.add_node("active_node")
        node_before = self.store.get_node("active_node")
        self.assertEqual(node_before.activation_count, 0)

        self.store.update_node_activation("active_node")
        node_after = self.store.get_node("active_node")
        self.assertEqual(node_after.activation_count, 1)
        self.assertGreaterEqual(node_after.last_activated, node_before.last_activated)

    def test_update_node_strength(self):
        """update_node_strength should change the consolidation_strength."""
        self.store.add_node("str_node", consolidation_strength=0.1)
        self.store.update_node_strength("str_node", 0.75)
        node = self.store.get_node("str_node")
        self.assertAlmostEqual(node.consolidation_strength, 0.75)


class TestGraphStoreAddEdge(unittest.TestCase):
    """Tests for the add_edge method with various RelationTypes."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store = GraphStore(data_dir=self.tmp_dir)
        # Create a set of nodes for edge tests
        for name in ["animal", "dog", "leg", "cat", "bark", "morning", "evening",
                      "mammal", "pet", "tail"]:
            self.store.add_node(name)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_add_edge_is_a(self):
        """Edge with IS_A relation should be stored correctly."""
        result = self.store.add_edge("dog", "animal", RelationType.IS_A.value)
        self.assertTrue(result)
        edge = self.store.get_edge("dog", "animal", RelationType.IS_A.value)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.source, "dog")
        self.assertEqual(edge.target, "animal")
        self.assertEqual(edge.relation_type, RelationType.IS_A.value)

    def test_add_edge_part_of(self):
        """Edge with PART_OF relation should be stored correctly."""
        result = self.store.add_edge("leg", "dog", RelationType.PART_OF.value, weight=0.8)
        self.assertTrue(result)
        edge = self.store.get_edge("leg", "dog", RelationType.PART_OF.value)
        self.assertIsNotNone(edge)
        self.assertAlmostEqual(edge.weight, 0.8)

    def test_add_edge_similar_to(self):
        """Edge with SIMILAR_TO should be stored; test bidirectional flag."""
        result = self.store.add_edge(
            "dog", "cat", RelationType.SIMILAR_TO.value,
            weight=0.7, bidirectional=True,
        )
        self.assertTrue(result)
        edge = self.store.get_edge("dog", "cat", RelationType.SIMILAR_TO.value)
        self.assertIsNotNone(edge)
        self.assertTrue(edge.bidirectional)
        self.assertAlmostEqual(edge.weight, 0.7)

    def test_add_edge_causal_using_causes(self):
        """Edge with CAUSES (causal) relation type."""
        result = self.store.add_edge(
            "dog", "bark", RelationType.CAUSES.value, weight=0.9,
        )
        self.assertTrue(result)
        edge = self.store.get_edge("dog", "bark", RelationType.CAUSES.value)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.relation_type, "causes")

    def test_add_edge_temporal_using_before(self):
        """Edge with BEFORE (temporal) relation type."""
        result = self.store.add_edge(
            "morning", "evening", RelationType.BEFORE.value, weight=1.0,
        )
        self.assertTrue(result)
        edge = self.store.get_edge("morning", "evening", RelationType.BEFORE.value)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.relation_type, "before")

    def test_add_edge_hierarchical_using_member_of(self):
        """Edge with MEMBER_OF (hierarchical) relation type."""
        result = self.store.add_edge(
            "dog", "mammal", RelationType.MEMBER_OF.value, weight=1.0,
        )
        self.assertTrue(result)
        edge = self.store.get_edge("dog", "mammal", RelationType.MEMBER_OF.value)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.relation_type, "member_of")

    def test_add_edge_with_metadata(self):
        """Metadata dictionary should be stored and retrievable."""
        meta = {"source": "textbook", "confidence": 0.95}
        self.store.add_edge(
            "dog", "pet", RelationType.IS_A.value,
            weight=1.0, metadata=meta,
        )
        edge = self.store.get_edge("dog", "pet", RelationType.IS_A.value)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.metadata, meta)

    def test_add_edge_default_weight(self):
        """Default weight should be 1.0."""
        self.store.add_edge("tail", "dog", RelationType.PART_OF.value)
        edge = self.store.get_edge("tail", "dog", RelationType.PART_OF.value)
        self.assertAlmostEqual(edge.weight, 1.0)

    def test_add_edge_default_not_bidirectional(self):
        """Default bidirectional should be False."""
        self.store.add_edge("cat", "animal", RelationType.IS_A.value)
        edge = self.store.get_edge("cat", "animal", RelationType.IS_A.value)
        self.assertFalse(edge.bidirectional)

    def test_add_duplicate_edge_updates(self):
        """Re-adding an edge with same (source, target, relation_type) should update."""
        self.store.add_edge("dog", "animal", RelationType.IS_A.value, weight=0.5)
        self.store.add_edge("dog", "animal", RelationType.IS_A.value, weight=0.9)
        self.assertEqual(self.store.count_edges(), 1)
        edge = self.store.get_edge("dog", "animal", RelationType.IS_A.value)
        self.assertAlmostEqual(edge.weight, 0.9)

    def test_multiple_edge_types_between_same_nodes(self):
        """Two nodes can have multiple edges with different relation types."""
        self.store.add_edge("dog", "cat", RelationType.SIMILAR_TO.value)
        self.store.add_edge("dog", "cat", RelationType.CAUSES.value)
        self.assertEqual(self.store.count_edges(), 2)

    def test_get_edge_without_relation_type(self):
        """get_edge without relation_type should return the first match."""
        self.store.add_edge("dog", "animal", RelationType.IS_A.value)
        edge = self.store.get_edge("dog", "animal")
        self.assertIsNotNone(edge)
        self.assertEqual(edge.source, "dog")

    def test_get_edge_nonexistent(self):
        """get_edge should return None for edges that don't exist."""
        self.assertIsNone(self.store.get_edge("dog", "animal"))

    def test_get_outgoing_edges(self):
        """get_outgoing_edges should return all edges from a source."""
        self.store.add_edge("dog", "animal", RelationType.IS_A.value)
        self.store.add_edge("dog", "mammal", RelationType.IS_A.value)
        self.store.add_edge("cat", "animal", RelationType.IS_A.value)
        edges = self.store.get_outgoing_edges("dog")
        targets = sorted([e.target for e in edges])
        self.assertEqual(targets, ["animal", "mammal"])

    def test_get_incoming_edges(self):
        """get_incoming_edges should return all edges to a target."""
        self.store.add_edge("dog", "animal", RelationType.IS_A.value)
        self.store.add_edge("cat", "animal", RelationType.IS_A.value)
        edges = self.store.get_incoming_edges("animal")
        sources = sorted([e.source for e in edges])
        self.assertEqual(sources, ["cat", "dog"])

    def test_update_edge_weight(self):
        """update_edge_weight should change the weight of an existing edge."""
        self.store.add_edge("dog", "animal", RelationType.IS_A.value, weight=0.5)
        result = self.store.update_edge_weight(
            "dog", "animal", RelationType.IS_A.value, 0.99,
        )
        self.assertTrue(result)
        edge = self.store.get_edge("dog", "animal", RelationType.IS_A.value)
        self.assertAlmostEqual(edge.weight, 0.99)

    def test_delete_edge_with_relation_type(self):
        """delete_edge with relation_type should remove only that edge."""
        self.store.add_edge("dog", "cat", RelationType.SIMILAR_TO.value)
        self.store.add_edge("dog", "cat", RelationType.CAUSES.value)
        self.assertEqual(self.store.count_edges(), 2)
        result = self.store.delete_edge("dog", "cat", RelationType.SIMILAR_TO.value)
        self.assertTrue(result)
        self.assertEqual(self.store.count_edges(), 1)
        remaining = self.store.get_edge("dog", "cat", RelationType.CAUSES.value)
        self.assertIsNotNone(remaining)

    def test_delete_edge_without_relation_type(self):
        """delete_edge without relation_type should remove all edges between nodes."""
        self.store.add_edge("dog", "cat", RelationType.SIMILAR_TO.value)
        self.store.add_edge("dog", "cat", RelationType.CAUSES.value)
        result = self.store.delete_edge("dog", "cat")
        self.assertTrue(result)
        self.assertEqual(self.store.count_edges(), 0)

    def test_count_edges(self):
        """count_edges should reflect the total number of edges."""
        self.assertEqual(self.store.count_edges(), 0)
        self.store.add_edge("dog", "animal", RelationType.IS_A.value)
        self.assertEqual(self.store.count_edges(), 1)
        self.store.add_edge("cat", "animal", RelationType.IS_A.value)
        self.assertEqual(self.store.count_edges(), 2)

    def test_get_node_degree(self):
        """get_node_degree should return correct (in_degree, out_degree)."""
        self.store.add_edge("dog", "animal", RelationType.IS_A.value)
        self.store.add_edge("dog", "mammal", RelationType.IS_A.value)
        self.store.add_edge("cat", "dog", RelationType.SIMILAR_TO.value)
        in_deg, out_deg = self.store.get_node_degree("dog")
        self.assertEqual(in_deg, 1)   # cat -> dog
        self.assertEqual(out_deg, 2)  # dog -> animal, dog -> mammal


class TestGraphStoreGetNeighbors(unittest.TestCase):
    """Tests for the get_neighbors method."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store = GraphStore(data_dir=self.tmp_dir)
        # Build a small graph:  A -> B -> C,  D -> B
        for name in ["A", "B", "C", "D"]:
            self.store.add_node(name)
        self.store.add_edge("A", "B", RelationType.IS_A.value)
        self.store.add_edge("B", "C", RelationType.PART_OF.value)
        self.store.add_edge("D", "B", RelationType.SIMILAR_TO.value)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_neighbors_include_incoming(self):
        """With include_incoming=True, B's neighbors include A, C, and D."""
        neighbors = self.store.get_neighbors("B", include_incoming=True)
        self.assertEqual(neighbors, {"A", "C", "D"})

    def test_get_neighbors_exclude_incoming(self):
        """With include_incoming=False, B's neighbors include only outgoing (C)."""
        neighbors = self.store.get_neighbors("B", include_incoming=False)
        self.assertEqual(neighbors, {"C"})

    def test_get_neighbors_leaf_node(self):
        """A leaf node with no outgoing edges should still have incoming neighbors."""
        neighbors = self.store.get_neighbors("C", include_incoming=True)
        self.assertEqual(neighbors, {"B"})

    def test_get_neighbors_no_edges(self):
        """A node with no edges should return an empty set."""
        self.store.add_node("isolated")
        neighbors = self.store.get_neighbors("isolated", include_incoming=True)
        self.assertEqual(neighbors, set())


class TestGraphStoreFindPath(unittest.TestCase):
    """Tests for the find_path BFS method."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store = GraphStore(data_dir=self.tmp_dir)
        # Build a chain: A -> B -> C -> D -> E
        for name in ["A", "B", "C", "D", "E", "X"]:
            self.store.add_node(name)
        self.store.add_edge("A", "B", RelationType.IS_A.value)
        self.store.add_edge("B", "C", RelationType.IS_A.value)
        self.store.add_edge("C", "D", RelationType.IS_A.value)
        self.store.add_edge("D", "E", RelationType.IS_A.value)
        # X is isolated -- no edges

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_find_path_same_node(self):
        """Path from a node to itself should be [node]."""
        path = self.store.find_path("A", "A")
        self.assertEqual(path, ["A"])

    def test_find_path_direct_neighbors(self):
        """Path between directly connected nodes."""
        path = self.store.find_path("A", "B")
        self.assertEqual(path, ["A", "B"])

    def test_find_path_multi_hop(self):
        """Path over multiple hops should follow the chain."""
        path = self.store.find_path("A", "D")
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "D")
        # Path length should be 4: A, B, C, D
        self.assertEqual(len(path), 4)

    def test_find_path_reverse_direction(self):
        """BFS uses get_neighbors with include_incoming=True, so reverse is found."""
        path = self.store.find_path("E", "A")
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "E")
        self.assertEqual(path[-1], "A")

    def test_find_path_no_path(self):
        """No path should exist to an isolated node."""
        path = self.store.find_path("A", "X")
        self.assertIsNone(path)

    def test_find_path_max_depth_exceeded(self):
        """Path longer than max_depth should return None."""
        # A->B->C->D->E is length 5 (5 nodes), so max_depth=3 should fail
        path = self.store.find_path("A", "E", max_depth=3)
        self.assertIsNone(path)

    def test_find_path_max_depth_exact(self):
        """Path exactly at max_depth should be found."""
        # A->B->C->D->E has 5 nodes in the path; the BFS checks len(path) > max_depth
        # So we need max_depth=5 to allow paths of length 5
        path = self.store.find_path("A", "E", max_depth=5)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "E")


class TestGraphStoreGetSubgraph(unittest.TestCase):
    """Tests for the get_subgraph method."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store = GraphStore(data_dir=self.tmp_dir)
        # Build a star graph around "center":
        #   center -> n1, center -> n2, center -> n3
        #   n1 -> n1a
        #   n3 -> n3a
        for name in ["center", "n1", "n2", "n3", "n1a", "n3a", "far_away"]:
            self.store.add_node(name)
        self.store.add_edge("center", "n1", RelationType.IS_A.value)
        self.store.add_edge("center", "n2", RelationType.PART_OF.value)
        self.store.add_edge("center", "n3", RelationType.SIMILAR_TO.value)
        self.store.add_edge("n1", "n1a", RelationType.CAUSES.value)
        self.store.add_edge("n3", "n3a", RelationType.BEFORE.value)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_subgraph_depth_zero(self):
        """Depth 0 should return only the center node itself."""
        nodes, edges = self.store.get_subgraph("center", depth=0)
        node_names = {n.name for n in nodes}
        # Depth 0: no expansion iterations, only the center is added in
        # current_level and then visited_nodes.update(current_level)
        self.assertIn("center", node_names)

    def test_get_subgraph_depth_one(self):
        """Depth 1 should return center and its direct neighbors."""
        nodes, edges = self.store.get_subgraph("center", depth=1)
        node_names = {n.name for n in nodes}
        self.assertIn("center", node_names)
        self.assertIn("n1", node_names)
        self.assertIn("n2", node_names)
        self.assertIn("n3", node_names)
        # n1a and n3a should NOT be included at depth 1
        self.assertNotIn("n1a", node_names)
        self.assertNotIn("n3a", node_names)

    def test_get_subgraph_depth_two(self):
        """Depth 2 should include second-level neighbors."""
        nodes, edges = self.store.get_subgraph("center", depth=2)
        node_names = {n.name for n in nodes}
        self.assertIn("center", node_names)
        self.assertIn("n1", node_names)
        self.assertIn("n1a", node_names)
        self.assertIn("n3a", node_names)

    def test_get_subgraph_excludes_disconnected(self):
        """Disconnected nodes should not appear in any subgraph."""
        nodes, edges = self.store.get_subgraph("center", depth=5)
        node_names = {n.name for n in nodes}
        self.assertNotIn("far_away", node_names)

    def test_get_subgraph_edges_between_visited(self):
        """Edges returned should only connect visited nodes."""
        nodes, edges = self.store.get_subgraph("center", depth=1)
        node_names = {n.name for n in nodes}
        for edge in edges:
            self.assertIn(edge.source, node_names)
            self.assertIn(edge.target, node_names)

    def test_get_subgraph_correct_edge_count_depth_one(self):
        """At depth 1 from center, only edges among {center, n1, n2, n3} qualify."""
        nodes, edges = self.store.get_subgraph("center", depth=1)
        # The three edges from center to n1, n2, n3 should be included.
        # The edges n1->n1a and n3->n3a should NOT be included (n1a, n3a not visited).
        self.assertEqual(len(edges), 3)


class TestGraphStoreGetWeakNodes(unittest.TestCase):
    """Tests for the get_weak_nodes method."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store = GraphStore(data_dir=self.tmp_dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_weak_nodes_empty(self):
        """No weak nodes should be returned from an empty store."""
        weak = self.store.get_weak_nodes()
        self.assertEqual(weak, [])

    def test_get_weak_nodes_all_weak(self):
        """All nodes below threshold should be returned."""
        self.store.add_node("low1", consolidation_strength=0.1)
        self.store.add_node("low2", consolidation_strength=0.2)
        self.store.add_node("high", consolidation_strength=0.9)
        weak = self.store.get_weak_nodes(threshold=0.3)
        self.assertIn("low1", weak)
        self.assertIn("low2", weak)
        self.assertNotIn("high", weak)

    def test_get_weak_nodes_threshold(self):
        """Only nodes strictly below the threshold should be returned."""
        self.store.add_node("at_threshold", consolidation_strength=0.3)
        self.store.add_node("below", consolidation_strength=0.29)
        weak = self.store.get_weak_nodes(threshold=0.3)
        self.assertIn("below", weak)
        self.assertNotIn("at_threshold", weak)

    def test_get_weak_nodes_ordered_ascending(self):
        """Weak nodes should be ordered by consolidation_strength ascending."""
        self.store.add_node("w1", consolidation_strength=0.2)
        self.store.add_node("w2", consolidation_strength=0.05)
        self.store.add_node("w3", consolidation_strength=0.15)
        weak = self.store.get_weak_nodes(threshold=0.3)
        self.assertEqual(weak, ["w2", "w3", "w1"])

    def test_get_weak_nodes_limit(self):
        """Limit parameter should cap the number of returned nodes."""
        for i in range(10):
            self.store.add_node(f"node_{i}", consolidation_strength=0.01 * i)
        weak = self.store.get_weak_nodes(threshold=0.5, limit=3)
        self.assertEqual(len(weak), 3)

    def test_get_weak_nodes_zero_strength(self):
        """Nodes with zero strength should be returned as weak."""
        self.store.add_node("zero_node", consolidation_strength=0.0)
        weak = self.store.get_weak_nodes(threshold=0.3)
        self.assertIn("zero_node", weak)


class TestGraphStoreCleanup(unittest.TestCase):
    """Tests for cleanup and teardown with temporary files."""

    def test_close_allows_directory_removal(self):
        """After close(), the temp directory should be removable without error."""
        tmp_dir = tempfile.mkdtemp()
        store = GraphStore(data_dir=tmp_dir)
        store.add_node("temp")
        store.close()
        # Should not raise; the DB connection is closed.
        shutil.rmtree(tmp_dir)
        self.assertFalse(os.path.exists(tmp_dir))

    def test_close_idempotent(self):
        """Calling close() multiple times should not raise."""
        tmp_dir = tempfile.mkdtemp()
        try:
            store = GraphStore(data_dir=tmp_dir)
            store.close()
            store.close()  # second call should be safe
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_data_persists_after_reopen(self):
        """Data written should persist when store is closed and reopened."""
        tmp_dir = tempfile.mkdtemp()
        try:
            store = GraphStore(data_dir=tmp_dir)
            store.add_node("persistent", node_type="entity",
                           consolidation_strength=0.5)
            store.add_edge("persistent", "persistent",
                           RelationType.IS_A.value, weight=0.7)
            store.close()

            # Reopen with same data_dir
            store2 = GraphStore(data_dir=tmp_dir)
            node = store2.get_node("persistent")
            self.assertIsNotNone(node)
            self.assertEqual(node.name, "persistent")
            self.assertEqual(node.node_type, "entity")
            self.assertAlmostEqual(node.consolidation_strength, 0.5)

            edge = store2.get_edge("persistent", "persistent",
                                   RelationType.IS_A.value)
            self.assertIsNotNone(edge)
            self.assertAlmostEqual(edge.weight, 0.7)
            store2.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestGraphStoreStats(unittest.TestCase):
    """Tests for get_stats with populated graph."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store = GraphStore(data_dir=self.tmp_dir)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_stats_reflect_additions(self):
        """Stats should accurately count nodes and edges."""
        self.store.add_node("s1", consolidation_strength=0.4)
        self.store.add_node("s2", consolidation_strength=0.6)
        self.store.add_edge("s1", "s2", RelationType.IS_A.value)

        stats = self.store.get_stats()
        self.assertEqual(stats['node_count'], 2)
        self.assertEqual(stats['edge_count'], 1)
        self.assertAlmostEqual(stats['avg_consolidation_strength'], 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
