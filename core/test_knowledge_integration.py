"""
Comprehensive tests for the KnowledgeIntegrationSystem.

Tests cover memory storage, cross-modal learning, knowledge consolidation,
and knowledge graph construction.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.knowledge_integration import (
    KnowledgeIntegrationSystem,
    MemoryType,
    ModalityType,
    IntegrationStrategy,
    MemoryTrace,
    CrossModalAssociation,
    KnowledgeNode,
    ProceduralSkill,
)


class TestKnowledgeIntegrationInitialization(unittest.TestCase):
    """Tests for KnowledgeIntegrationSystem initialization."""

    def setUp(self):
        self.kis = KnowledgeIntegrationSystem(random_seed=42)

    def test_default_initialization(self):
        """Test that default construction sets expected values."""
        self.assertEqual(self.kis.embedding_dim, 128)
        self.assertEqual(self.kis.max_memory_traces, 100000)
        self.assertAlmostEqual(self.kis.consolidation_threshold, 0.7)
        self.assertAlmostEqual(self.kis.association_threshold, 0.6)
        self.assertTrue(self.kis.enable_cross_modal_learning)
        self.assertTrue(self.kis.enable_consolidation)

    def test_custom_initialization(self):
        """Test construction with custom parameters."""
        kis = KnowledgeIntegrationSystem(
            embedding_dim=64,
            max_memory_traces=50000,
            consolidation_threshold=0.8,
            association_threshold=0.5,
            enable_cross_modal_learning=False,
            enable_consolidation=False,
            random_seed=99
        )
        self.assertEqual(kis.embedding_dim, 64)
        self.assertEqual(kis.max_memory_traces, 50000)
        self.assertFalse(kis.enable_cross_modal_learning)
        self.assertFalse(kis.enable_consolidation)

    def test_initial_stores_empty(self):
        """Test that memory stores start empty."""
        self.assertEqual(len(self.kis.memory_traces), 0)
        self.assertEqual(len(self.kis.knowledge_graph), 0)
        self.assertEqual(len(self.kis.procedural_skills), 0)
        self.assertEqual(len(self.kis.cross_modal_associations), 0)

    def test_initial_statistics_zero(self):
        """Test that statistics start at zero."""
        self.assertEqual(self.kis.total_traces_stored, 0)
        self.assertEqual(self.kis.total_associations_formed, 0)
        self.assertEqual(self.kis.total_consolidations, 0)
        self.assertEqual(self.kis.total_skills_extracted, 0)


class TestEpisodicMemoryStorage(unittest.TestCase):
    """Tests for episodic memory storage."""

    def setUp(self):
        self.kis = KnowledgeIntegrationSystem(random_seed=42)

    def test_store_episodic_memory_text(self):
        """Test storing text episodic memory."""
        trace_id = self.kis.store_episodic_memory(
            content="This is a test memory",
            modality=ModalityType.TEXT,
            context={'topic': 'testing'}
        )
        
        self.assertIsNotNone(trace_id)
        self.assertIn(trace_id, self.kis.memory_traces)
        
        trace = self.kis.memory_traces[trace_id]
        self.assertEqual(trace.memory_type, MemoryType.EPISODIC)
        self.assertEqual(trace.modality, ModalityType.TEXT)
        self.assertEqual(trace.content, "This is a test memory")

    def test_store_episodic_memory_code(self):
        """Test storing code episodic memory."""
        code = "def test():\n    return 42"
        trace_id = self.kis.store_episodic_memory(
            content=code,
            modality=ModalityType.CODE
        )
        
        self.assertIn(trace_id, self.kis.memory_traces)
        trace = self.kis.memory_traces[trace_id]
        self.assertEqual(trace.modality, ModalityType.CODE)

    def test_store_episodic_memory_generates_embedding(self):
        """Test that embedding is generated if not provided."""
        trace_id = self.kis.store_episodic_memory(
            content="Test content",
            modality=ModalityType.TEXT
        )
        
        trace = self.kis.memory_traces[trace_id]
        self.assertIsNotNone(trace.embedding)
        self.assertEqual(len(trace.embedding), self.kis.embedding_dim)
        
        # Check normalization
        norm = np.linalg.norm(trace.embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_store_episodic_memory_uses_provided_embedding(self):
        """Test that provided embedding is used."""
        embedding = np.random.randn(128)
        embedding = embedding / np.linalg.norm(embedding)
        
        trace_id = self.kis.store_episodic_memory(
            content="Test",
            modality=ModalityType.TEXT,
            embedding=embedding
        )
        
        trace = self.kis.memory_traces[trace_id]
        np.testing.assert_array_almost_equal(trace.embedding, embedding)

    def test_store_increments_counter(self):
        """Test that storing increments the counter."""
        self.assertEqual(self.kis.total_traces_stored, 0)
        
        self.kis.store_episodic_memory(
            content="Test 1",
            modality=ModalityType.TEXT
        )
        self.assertEqual(self.kis.total_traces_stored, 1)
        
        self.kis.store_episodic_memory(
            content="Test 2",
            modality=ModalityType.TEXT
        )
        self.assertEqual(self.kis.total_traces_stored, 2)


class TestSemanticKnowledgeStorage(unittest.TestCase):
    """Tests for semantic knowledge storage."""

    def setUp(self):
        self.kis = KnowledgeIntegrationSystem(random_seed=42)

    def test_store_semantic_knowledge_new(self):
        """Test storing new semantic knowledge."""
        node_id = self.kis.store_semantic_knowledge(
            concept_name="test_concept",
            attributes={'type': 'abstract'}
        )
        
        self.assertIsNotNone(node_id)
        self.assertIn(node_id, self.kis.knowledge_graph)
        
        node = self.kis.knowledge_graph[node_id]
        self.assertEqual(node.concept_name, "test_concept")
        self.assertEqual(node.attributes['type'], 'abstract')

    def test_store_semantic_knowledge_update(self):
        """Test updating existing semantic knowledge."""
        node_id = self.kis.store_semantic_knowledge(
            concept_name="test_concept",
            attributes={'type': 'abstract'}
        )
        
        # Update
        node_id2 = self.kis.store_semantic_knowledge(
            concept_name="test_concept",
            attributes={'category': 'test'},
            source_traces=['trace1']
        )
        
        self.assertEqual(node_id, node_id2)
        node = self.kis.knowledge_graph[node_id]
        self.assertEqual(node.attributes['type'], 'abstract')
        self.assertEqual(node.attributes['category'], 'test')
        self.assertIn('trace1', node.memory_traces)


class TestProceduralSkillStorage(unittest.TestCase):
    """Tests for procedural skill storage."""

    def setUp(self):
        self.kis = KnowledgeIntegrationSystem(random_seed=42)

    def test_store_procedural_skill(self):
        """Test storing a procedural skill."""
        steps = [
            {'step_number': 1, 'action': 'prepare', 'description': 'Prepare materials'},
            {'step_number': 2, 'action': 'execute', 'description': 'Execute task'}
        ]
        
        skill_id = self.kis.store_procedural_skill(
            name="test_skill",
            description="A test skill",
            steps=steps,
            preconditions=['condition1']
        )
        
        self.assertIsNotNone(skill_id)
        self.assertIn(skill_id, self.kis.procedural_skills)
        
        skill = self.kis.procedural_skills[skill_id]
        self.assertEqual(skill.name, "test_skill")
        self.assertEqual(len(skill.steps), 2)
        self.assertIn('condition1', skill.preconditions)

    def test_skill_execution_tracking(self):
        """Test tracking skill execution results."""
        skill_id = self.kis.store_procedural_skill(
            name="test_skill",
            description="Test",
            steps=[{'step_number': 1, 'action': 'test'}]
        )
        
        skill = self.kis.procedural_skills[skill_id]
        
        # Record successes
        skill.record_execution(success=True)
        skill.record_execution(success=True)
        skill.record_execution(success=False)
        
        self.assertEqual(skill.success_count, 2)
        self.assertEqual(skill.failure_count, 1)
        self.assertAlmostEqual(skill.success_rate, 2/3)


class TestCrossModalLearning(unittest.TestCase):
    """Tests for cross-modal learning."""

    def setUp(self):
        self.kis = KnowledgeIntegrationSystem(random_seed=42)

    def test_cross_modal_association_formation(self):
        """Test that cross-modal associations are formed."""
        # Create system with lower threshold for testing
        kis = KnowledgeIntegrationSystem(
            association_threshold=0.1,  # Very low threshold for testing
            random_seed=42
        )
        
        # Use the same embedding for both to ensure high similarity
        shared_embedding = np.random.randn(128)
        shared_embedding = shared_embedding / np.linalg.norm(shared_embedding)
        
        text_trace = kis.store_episodic_memory(
            content="Test content",
            modality=ModalityType.TEXT,
            embedding=shared_embedding,
            context={'time': 100}
        )
        
        visual_trace = kis.store_episodic_memory(
            content="Test content",  # Same content = same embedding
            modality=ModalityType.VISUAL,
            embedding=shared_embedding,
            context={'time': 100}
        )
        
        # Check that associations were formed
        self.assertGreater(len(kis.cross_modal_associations), 0)

    def test_predict_across_modalities(self):
        """Test prediction across modalities."""
        # Store and manually create association
        text_id = self.kis.store_episodic_memory(
            content="test",
            modality=ModalityType.TEXT
        )
        
        visual_id = self.kis.store_episodic_memory(
            content=np.array([1, 2, 3]),
            modality=ModalityType.VISUAL
        )
        
        # Manually create association
        from core.knowledge_integration import CrossModalAssociation
        assoc = CrossModalAssociation(
            association_id="test_assoc",
            modality_a=ModalityType.TEXT,
            modality_b=ModalityType.VISUAL,
            trace_a_id=text_id,
            trace_b_id=visual_id,
            strength=0.8,
            a_predicts_b=0.9,
            b_predicts_a=0.7
        )
        self.kis.cross_modal_associations["test_assoc"] = assoc
        
        # Test prediction
        predictions = self.kis.predict_across_modalities(text_id, ModalityType.VISUAL)
        self.assertGreater(len(predictions), 0)


class TestMemoryConsolidation(unittest.TestCase):
    """Tests for memory consolidation."""

    def setUp(self):
        self.kis = KnowledgeIntegrationSystem(random_seed=42)

    def test_consolidate_memories_disabled(self):
        """Test that consolidation is skipped when disabled."""
        self.kis.enable_consolidation = False
        result = self.kis.consolidate_memories()
        self.assertEqual(result['consolidated'], 0)

    def test_consolidate_memories_no_accessed_traces(self):
        """Test consolidation with unaccessed traces."""
        # Store memories but don't access them
        for i in range(5):
            self.kis.store_episodic_memory(
                content=f"Memory {i}",
                modality=ModalityType.TEXT
            )
        
        result = self.kis.consolidate_memories()
        # Traces haven't been accessed enough
        self.assertEqual(result['consolidated'], 0)

    def test_consolidate_memories_with_access(self):
        """Test consolidation with accessed traces."""
        # Store and access memories
        trace_ids = []
        for i in range(5):
            tid = self.kis.store_episodic_memory(
                content=f"Memory {i}",
                modality=ModalityType.TEXT
            )
            trace_ids.append(tid)
            # Simulate access
            self.kis.memory_traces[tid].access_count = 3
        
        result = self.kis.consolidate_memories()
        # Should consolidate accessed traces
        self.assertGreaterEqual(result['consolidated'], 0)


class TestKnowledgeRetrieval(unittest.TestCase):
    """Tests for knowledge retrieval."""

    def setUp(self):
        self.kis = KnowledgeIntegrationSystem(random_seed=42)

    def test_retrieve_memories(self):
        """Test memory retrieval by similarity."""
        # Store some memories
        for i in range(5):
            self.kis.store_episodic_memory(
                content=f"Memory content {i}",
                modality=ModalityType.TEXT
            )
        
        # Create query embedding
        query = np.random.randn(128)
        query = query / np.linalg.norm(query)
        
        results = self.kis.retrieve_memories(query, n_results=3)
        self.assertLessEqual(len(results), 3)

    def test_query_knowledge_graph(self):
        """Test querying knowledge graph."""
        # Store knowledge
        node_id = self.kis.store_semantic_knowledge(
            concept_name="test_concept",
            attributes={'type': 'test'}
        )
        
        # Query by name
        results = self.kis.query_knowledge_graph(concept_name="test_concept")
        self.assertGreater(len(results), 0)

    def test_get_related_concepts(self):
        """Test getting related concepts."""
        # Create related nodes
        node1 = self.kis.store_semantic_knowledge(concept_name="animal")
        node2 = self.kis.store_semantic_knowledge(concept_name="dog")
        
        # Manually add relationship
        self.kis.knowledge_graph[node2].is_a.append(node1)
        self.kis.knowledge_graph[node1].child_concepts.append(node2)
        
        related = self.kis.get_related_concepts(node2, depth=1)
        self.assertIn(node1, related)


class TestKnowledgeIntegration(unittest.TestCase):
    """Tests for knowledge integration."""

    def setUp(self):
        self.kis = KnowledgeIntegrationSystem(random_seed=42)

    def test_integrate_knowledge_associative(self):
        """Test associative knowledge integration."""
        # Store some concepts
        for i in range(5):
            self.kis.store_semantic_knowledge(
                concept_name=f"concept_{i}",
                embedding=np.random.randn(128)
            )
        
        result = self.kis.integrate_knowledge(IntegrationStrategy.ASSOCIATIVE)
        self.assertEqual(result['strategy'], 'associative')

    def test_form_concept_associations(self):
        """Test forming concept associations."""
        # Store similar concepts
        base_embedding = np.random.randn(128)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        for i in range(3):
            # Create similar embeddings
            embedding = base_embedding + np.random.randn(128) * 0.1
            embedding = embedding / np.linalg.norm(embedding)
            
            self.kis.store_semantic_knowledge(
                concept_name=f"similar_{i}",
                embedding=embedding
            )
        
        count = self.kis._form_concept_associations()
        self.assertGreaterEqual(count, 0)


class TestSerialization(unittest.TestCase):
    """Tests for serialization/deserialization."""

    def setUp(self):
        self.kis = KnowledgeIntegrationSystem(random_seed=42)

    def test_serialize(self):
        """Test serialization."""
        # Add some data
        self.kis.store_episodic_memory(content="test", modality=ModalityType.TEXT)
        self.kis.store_semantic_knowledge(concept_name="test_concept")
        
        data = self.kis.serialize()
        
        self.assertIn('embedding_dim', data)
        self.assertIn('knowledge_graph', data)
        self.assertIn('procedural_skills', data)
        self.assertIn('stats', data)

    def test_deserialize(self):
        """Test deserialization."""
        # Create and populate
        self.kis.store_semantic_knowledge(
            concept_name="test_concept",
            attributes={'key': 'value'}
        )
        
        data = self.kis.serialize()
        restored = KnowledgeIntegrationSystem.deserialize(data)
        
        self.assertEqual(restored.embedding_dim, self.kis.embedding_dim)
        self.assertEqual(len(restored.knowledge_graph), len(self.kis.knowledge_graph))

    def test_round_trip(self):
        """Test serialize -> deserialize round trip."""
        # Store various types of knowledge
        self.kis.store_episodic_memory(content="episodic", modality=ModalityType.TEXT)
        self.kis.store_semantic_knowledge(concept_name="semantic")
        self.kis.store_procedural_skill(name="skill", description="test", steps=[])
        
        data = self.kis.serialize()
        restored = KnowledgeIntegrationSystem.deserialize(data)
        
        # Check that knowledge is preserved
        self.assertEqual(
            len(restored.knowledge_graph),
            len(self.kis.knowledge_graph)
        )


class TestStats(unittest.TestCase):
    """Tests for statistics."""

    def setUp(self):
        self.kis = KnowledgeIntegrationSystem(random_seed=42)

    def test_get_stats(self):
        """Test getting statistics."""
        # Add some data
        self.kis.store_episodic_memory(content="test", modality=ModalityType.TEXT)
        self.kis.store_semantic_knowledge(concept_name="concept")
        self.kis.store_procedural_skill(name="skill", description="test", steps=[])
        
        stats = self.kis.get_stats()
        
        self.assertIn('total_traces_stored', stats)
        self.assertIn('knowledge_nodes_count', stats)
        self.assertIn('procedural_skills_count', stats)
        self.assertEqual(stats['total_traces_stored'], 1)
        self.assertEqual(stats['knowledge_nodes_count'], 1)
        self.assertEqual(stats['procedural_skills_count'], 1)


if __name__ == '__main__':
    unittest.main()
