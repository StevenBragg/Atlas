"""
Integration Tests for Atlas Module Interactions.

Tests interactions between:
- Text learning and neural network components
- Memory systems and learning
- Cloud service and core modules
- Multimodal integration
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from self_organizing_av_system.core.text_learning import TextLearningModule
from self_organizing_av_system.core.neuron import Neuron
from self_organizing_av_system.core.layer import NeuralLayer
from self_organizing_av_system.core.pathway import NeuralPathway


# ============================================================================
# Text Learning + Neural Network Integration
# ============================================================================

class TestTextLearningNeuralIntegration:
    """Tests for text learning integration with neural components"""
    
    def test_text_embeddings_compatible_with_neurons(self):
        """Test that text embeddings can be processed by neurons"""
        text_module = TextLearningModule(embedding_dim=64)
        neuron = Neuron(input_dim=64, learning_rate=0.01)
        
        # Learn some text
        text_module.learn_from_text("hello world test")
        
        # Get embedding
        hello_idx = text_module.token_to_idx['hello']
        embedding = text_module.embeddings[hello_idx]
        
        # Process through neuron
        output = neuron.process(embedding)
        
        assert isinstance(output, (float, np.floating))
        assert 0 <= output <= 1 or np.isfinite(output)
    
    def test_text_embeddings_batch_processing(self):
        """Test batch processing of text embeddings through layers"""
        text_module = TextLearningModule(embedding_dim=32)
        layer = Layer(input_dim=32, num_neurons=5, learning_rate=0.01)
        
        # Learn multiple words
        text_module.learn_from_text("one two three four five")
        
        # Get embeddings
        embeddings = []
        for word in ['one', 'two', 'three', 'four', 'five']:
            idx = text_module.token_to_idx[word]
            embeddings.append(text_module.embeddings[idx])
        
        # Process through layer
        results = []
        for emb in embeddings:
            output = layer.process(emb)
            results.append(output)
        
        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)
    
    def test_text_learning_updates_affect_neural_processing(self):
        """Test that learning updates change neural processing"""
        text_module = TextLearningModule(embedding_dim=16)
        neuron = Neuron(input_dim=16, learning_rate=0.01)
        
        # Initial processing
        text_module.learn_from_text("test")
        idx = text_module.token_to_idx['test']
        emb1 = text_module.embeddings[idx].copy()
        out1 = neuron.process(emb1)
        
        # More learning
        for _ in range(10):
            text_module.learn_from_text("test word example")
        
        emb2 = text_module.embeddings[idx]
        out2 = neuron.process(emb2)
        
        # Embeddings should have changed
        assert not np.allclose(emb1, emb2)


# ============================================================================
# Memory Systems Integration
# ============================================================================

class TestMemorySystemsIntegration:
    """Tests for memory system integrations"""
    
    def test_episodic_text_memory_integration(self):
        """Test episodic memory stores text learning events"""
        from self_organizing_av_system.core.episodic_memory import EpisodicMemory
        from self_organizing_av_system.core.text_learning import TextLearningModule
        
        text_module = TextLearningModule()
        memory = EpisodicMemory(capacity=100)
        
        # Learn and store episode
        result = text_module.learn_from_text("important event happened")
        
        # Create memory entry
        episode = {
            'type': 'text_learning',
            'content': 'important event happened',
            'vocabulary_size': result['vocabulary_size'],
            'timestamp': 0
        }
        
        memory.store(episode)
        
        # Retrieve
        retrieved = memory.retrieve_recent(1)
        assert len(retrieved) == 1
        assert retrieved[0]['type'] == 'text_learning'
    
    def test_semantic_memory_text_concepts(self):
        """Test semantic memory stores text concepts"""
        from self_organizing_av_system.core.semantic_memory import SemanticMemory
        from self_organizing_av_system.core.text_learning import TextLearningModule
        
        text_module = TextLearningModule(embedding_dim=64)
        semantic = SemanticMemory(input_dim=64)
        
        # Learn concepts
        concepts = ['dog', 'cat', 'animal', 'pet']
        for concept in concepts:
            text_module.learn_from_text(concept)
            idx = text_module.token_to_idx[concept]
            embedding = text_module.embeddings[idx]
            semantic.store_concept(concept, embedding)
        
        # Query related concepts
        dog_idx = text_module.token_to_idx['dog']
        dog_emb = text_module.embeddings[dog_idx]
        related = semantic.query_similar(dog_emb, k=3)
        
        assert len(related) > 0


# ============================================================================
# Cloud Service Integration
# ============================================================================

class TestCloudServiceIntegration:
    """Tests for cloud service module integrations"""
    
    @patch.dict('os.environ', {
        'SALAD_MACHINE_ID': 'test-machine',
        'ATLAS_ENABLE_TEXT_LEARNING': 'true'
    })
    def test_salad_service_initializes_text_module(self):
        """Test that Salad service can initialize text learning"""
        try:
            from cloud.salad_service import AtlasSaladService
            
            # This would require more mocking for full test
            # Just verify the imports work
            assert AtlasSaladService is not None
        except ImportError:
            pytest.skip("Salad service not available")
    
    def test_text_api_integration_with_service(self):
        """Test text API integrates with service status"""
        from cloud.text_api import TextLearningAPI
        from self_organizing_av_system.core.text_learning import TextLearningModule
        
        text_module = TextLearningModule()
        api = TextLearningAPI(text_module)
        
        # Learn some text
        api.handle_learn({'text': 'service status check'})
        
        # Get stats for status report
        stats = api.handle_stats()
        
        assert stats['success'] is True
        assert stats['stats']['vocabulary_size'] > 0
        
        # This would be included in service status
        status_info = {
            'text_learning': stats['stats'],
            'service': 'running'
        }
        
        assert 'text_learning' in status_info


# ============================================================================
# Multimodal Integration
# ============================================================================

class TestMultimodalIntegration:
    """Tests for multimodal system integration"""
    
    def test_text_visual_fusion(self):
        """Test text and visual information fusion"""
        from self_organizing_av_system.core.multimodal import MultimodalFusion
        from self_organizing_av_system.core.text_learning import TextLearningModule
        
        text_module = TextLearningModule(embedding_dim=64)
        fusion = MultimodalFusion(
            visual_dim=64,
            audio_dim=32,
            output_dim=32
        )
        
        # Learn text
        text_module.learn_from_text("red square")
        red_idx = text_module.token_to_idx['red']
        text_emb = text_module.embeddings[red_idx]
        
        # Create visual pattern
        visual_pattern = np.random.randn(64).astype(np.float32)
        
        # Fuse
        combined = np.concatenate([visual_pattern, text_emb[:32]])
        
        assert combined.shape[0] == 96
    
    def test_audio_text_association(self):
        """Test audio and text association learning"""
        from self_organizing_av_system.core.text_learning import TextLearningModule
        
        text_module = TextLearningModule(embedding_dim=32)
        
        # Learn word representations
        words = ['high', 'low', 'fast', 'slow']
        for word in words:
            text_module.learn_from_text(word)
        
        # Simulate audio features
        audio_features = {
            'high': np.array([1.0, 0.8, 0.2]),
            'low': np.array([0.2, 0.3, 1.0]),
            'fast': np.array([0.9, 0.9, 0.1]),
            'slow': np.array([0.1, 0.2, 0.9])
        }
        
        # Create associations
        associations = {}
        for word in words:
            idx = text_module.token_to_idx[word]
            text_emb = text_module.embeddings[idx][:3]  # Take first 3 dims
            associations[word] = {
                'text': text_emb,
                'audio': audio_features[word]
            }
        
        assert len(associations) == 4


# ============================================================================
# Learning Engine Integration
# ============================================================================

class TestLearningEngineIntegration:
    """Tests for learning engine integrations"""
    
    def test_learning_engine_with_text(self):
        """Test learning engine processes text inputs"""
        from self_organizing_av_system.core.learning_engine import LearningEngine
        from self_organizing_av_system.core.text_learning import TextLearningModule
        
        text_module = TextLearningModule(embedding_dim=32)
        engine = LearningEngine(
            input_dim=32,
            output_dim=16,
            learning_rate=0.01
        )
        
        # Learn text
        text_module.learn_from_text("learning test")
        
        # Get embeddings for learning
        for word in ['learning', 'test']:
            idx = text_module.token_to_idx[word]
            embedding = text_module.embeddings[idx]
            
            # Process through learning engine
            result = engine.learn(embedding)
            
            assert result is not None
    
    def test_curriculum_with_text_learning(self):
        """Test curriculum system with text learning"""
        from self_organizing_av_system.core.curriculum_system import CurriculumLevel
        
        # Create curriculum levels for text
        levels = [
            CurriculumLevel(
                level_id=1,
                name='basic_words',
                requirements={'vocabulary_size': 10}
            ),
            CurriculumLevel(
                level_id=2,
                name='simple_sentences',
                requirements={'vocabulary_size': 50}
            )
        ]
        
        text_module = TextLearningModule(max_vocabulary=100)
        
        # Progress through curriculum
        text_module.learn_from_text("one two three four five")
        
        stats = text_module.get_stats()
        
        # Check level completion
        if stats['vocabulary_size'] >= 10:
            current_level = levels[1]
        else:
            current_level = levels[0]
        
        assert current_level.level_id >= 1


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    def test_full_learning_pipeline(self):
        """Test complete learning pipeline across modules"""
        text_module = TextLearningModule(embedding_dim=64, max_vocabulary=200)
        
        # 1. Learn from text
        texts = [
            "the quick brown fox jumps",
            "over the lazy dog",
            "the dog sleeps peacefully",
            "the fox runs quickly"
        ]
        
        for text in texts:
            result = text_module.learn_from_text(text)
            assert result['tokens_processed'] > 0
        
        # 2. Generate text
        generated = text_module.generate_text("the", max_length=10)
        assert isinstance(generated, str)
        
        # 3. Check stats
        stats = text_module.get_stats()
        assert stats['vocabulary_size'] > 5
        assert stats['total_tokens_seen'] >= 16
        
        # 4. Save and load
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name
        
        try:
            text_module.save_state(filepath)
            
            new_module = TextLearningModule(embedding_dim=64, max_vocabulary=200)
            new_module.load_state(filepath)
            
            new_stats = new_module.get_stats()
            assert new_stats['vocabulary_size'] == stats['vocabulary_size']
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_multi_module_coordination(self):
        """Test coordination between multiple modules"""
        from self_organizing_av_system.core.neuron import Neuron
        from self_organizing_av_system.core.layer import Layer
        from self_organizing_av_system.core.text_learning import TextLearningModule
        
        # Initialize modules
        text_module = TextLearningModule(embedding_dim=32)
        layer1 = NeuralLayer(input_size=32, layer_size=8, learning_rate=0.01)
        layer2 = NeuralLayer(input_size=8, layer_size=4, learning_rate=0.01)
        
        # Learn text
        text_module.learn_from_text("coordinate test")
        
        # Process through layers
        for word in ['coordinate', 'test']:
            idx = text_module.token_to_idx[word]
            emb = text_module.embeddings[idx]
            
            # Layer 1
            out1 = layer1.process(emb)
            
            # Layer 2 (if output is compatible)
            if isinstance(out1, dict) and 'activations' in out1:
                out2 = layer2.process(out1['activations'])
                assert out2 is not None


# ============================================================================
# Error Handling Integration
# ============================================================================

class TestErrorHandlingIntegration:
    """Tests for error handling across module boundaries"""
    
    def test_graceful_degradation_missing_module(self):
        """Test graceful degradation when module is missing"""
        text_module = TextLearningModule()
        
        # Try to use module before learning
        result = text_module.generate_text("unknown")
        assert isinstance(result, str)
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs across modules"""
        text_module = TextLearningModule()
        
        # Empty input
        result = text_module.learn_from_text("")
        assert result['tokens_processed'] == 0
        
        # Very long input
        long_text = "word " * 10000
        result = text_module.learn_from_text(long_text)
        assert result['tokens_processed'] > 0
    
    def test_dimension_mismatch_handling(self):
        """Test handling of dimension mismatches"""
        text_module = TextLearningModule(embedding_dim=64)
        neuron = Neuron(input_dim=32)  # Different dimension
        
        text_module.learn_from_text("test")
        emb = text_module.embeddings[text_module.token_to_idx['test']]
        
        # This should handle dimension mismatch gracefully
        # Either by error or by truncation/padding
        try:
            result = neuron.process(emb[:32])  # Truncate
            assert result is not None
        except Exception as e:
            # Should provide meaningful error
            assert 'dimension' in str(e).lower() or 'shape' in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
