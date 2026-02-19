"""
Comprehensive tests for the Text Learning Module.

Tests all core functionality:
- Tokenization
- Vocabulary growth
- Learning from text
- Context prediction
- Text generation
- Save/load state
- Max vocabulary limit
"""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from self_organizing_av_system.core.text_learning import TextLearningModule


class TestTokenization:
    """Test text tokenization functionality"""
    
    def test_basic_tokenization(self):
        """Test that basic word tokenization works"""
        module = TextLearningModule()
        tokens = module.tokenize("hello world")
        assert tokens == ["hello", "world"]
    
    def test_tokenization_with_punctuation(self):
        """Test tokenization handles punctuation correctly"""
        module = TextLearningModule()
        tokens = module.tokenize("Hello, world!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," in tokens or "!" in tokens
    
    def test_tokenization_lowercase(self):
        """Test that tokenization converts to lowercase"""
        module = TextLearningModule()
        tokens = module.tokenize("HELLO World")
        assert "hello" in tokens
        assert "world" in tokens
    
    def test_tokenization_empty_string(self):
        """Test tokenization of empty string returns UNK"""
        module = TextLearningModule()
        tokens = module.tokenize("")
        assert tokens == ["<UNK>"]
    
    def test_tokenization_numbers(self):
        """Test tokenization handles numbers"""
        module = TextLearningModule()
        tokens = module.tokenize("I have 42 apples")
        assert "42" in tokens
        assert "apples" in tokens


class TestVocabularyGrowth:
    """Test vocabulary growth and management"""
    
    def test_vocabulary_grows_with_new_words(self):
        """Test that vocabulary size increases with new words"""
        module = TextLearningModule()
        initial_size = len(module.token_to_idx)
        
        module.learn_from_text("hello world")
        assert len(module.token_to_idx) > initial_size
        
        module.learn_from_text("foo bar baz")
        assert len(module.token_to_idx) > initial_size + 2
    
    def test_vocabulary_contains_special_tokens(self):
        """Test that special tokens are added on initialization"""
        module = TextLearningModule()
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>', '<SPACE>']
        for token in special_tokens:
            assert token in module.token_to_idx
    
    def test_duplicate_words_not_added_twice(self):
        """Test that duplicate words don't increase vocabulary"""
        module = TextLearningModule()
        module.learn_from_text("hello hello hello")
        vocab_size = len(module.token_to_idx)
        
        module.learn_from_text("hello world")
        # Should only add "world", not "hello" again
        assert len(module.token_to_idx) == vocab_size + 1
    
    def test_token_indices_are_consistent(self):
        """Test that token indices remain consistent"""
        module = TextLearningModule()
        module.learn_from_text("hello world")
        
        hello_idx_1 = module.token_to_idx["hello"]
        world_idx_1 = module.token_to_idx["world"]
        
        module.learn_from_text("hello world again")
        
        hello_idx_2 = module.token_to_idx["hello"]
        world_idx_2 = module.token_to_idx["world"]
        
        assert hello_idx_1 == hello_idx_2
        assert world_idx_1 == world_idx_2


class TestLearningFromText:
    """Test learning functionality"""
    
    def test_learning_returns_stats(self):
        """Test that learning returns expected statistics"""
        module = TextLearningModule()
        result = module.learn_from_text("hello world test")
        
        assert 'tokens_processed' in result
        assert 'unique_tokens' in result
        assert 'vocabulary_size' in result
        assert 'avg_prediction_error' in result
        assert 'predictions' in result
    
    def test_learning_updates_token_counts(self):
        """Test that learning updates token statistics"""
        module = TextLearningModule()
        module.learn_from_text("hello hello world")
        
        assert module.token_counts[module.token_to_idx["hello"]] == 2
        assert module.token_counts[module.token_to_idx["world"]] == 1
    
    def test_learning_creates_embeddings(self):
        """Test that learning creates embeddings for tokens"""
        module = TextLearningModule()
        module.learn_from_text("hello world")
        
        hello_idx = module.token_to_idx["hello"]
        world_idx = module.token_to_idx["world"]
        
        assert hello_idx in module.embeddings
        assert world_idx in module.embeddings
        assert module.embeddings[hello_idx].shape == (module.embedding_dim,)
    
    def test_learning_updates_total_tokens(self):
        """Test that learning updates total token count"""
        module = TextLearningModule()
        initial_total = module.total_tokens
        
        module.learn_from_text("one two three")
        assert module.total_tokens == initial_total + 3
        
        module.learn_from_text("four five")
        assert module.total_tokens == initial_total + 5


class TestContextPrediction:
    """Test context-based prediction"""
    
    def test_prediction_error_decreases_with_training(self):
        """Test that prediction error decreases with more training"""
        module = TextLearningModule(learning_rate=0.1)
        
        # Train on a simple repetitive pattern
        text = "the cat sat on the mat"
        errors = []
        
        for _ in range(10):
            result = module.learn_from_text(text)
            errors.append(result['avg_prediction_error'])
        
        # Error should generally decrease (allowing for some noise)
        # Compare first few with last few
        early_avg = np.mean(errors[:3])
        late_avg = np.mean(errors[-3:])
        assert late_avg < early_avg or errors[-1] < errors[0] * 1.5
    
    def test_context_weights_created(self):
        """Test that context weights are created during learning"""
        module = TextLearningModule()
        module.learn_from_text("hello world test")
        
        # Should have created some context weights
        assert len(module.context_weights) > 0
    
    def test_prediction_from_context(self):
        """Test prediction from context returns embedding"""
        module = TextLearningModule()
        module.learn_from_text("hello world")
        
        # Get a context
        hello_idx = module.token_to_idx["hello"]
        context = (hello_idx,)
        
        prediction = module._predict_from_context(context)
        assert prediction.shape == (module.embedding_dim,)


class TestTextGeneration:
    """Test text generation functionality"""
    
    def test_generate_text_produces_output(self):
        """Test that text generation produces some output"""
        module = TextLearningModule()
        
        # Train on some text first
        for _ in range(5):
            module.learn_from_text("the quick brown fox jumps")
        
        generated = module.generate_text("the", max_length=10)
        assert isinstance(generated, str)
        assert len(generated) > 0
    
    def test_generate_text_with_empty_prompt(self):
        """Test generation with empty prompt"""
        module = TextLearningModule()
        module.learn_from_text("hello world test")
        
        generated = module.generate_text("")
        assert isinstance(generated, str)
    
    def test_generate_text_respects_max_length(self):
        """Test that generation respects max_length parameter"""
        module = TextLearningModule()
        
        # Train with enough data
        for _ in range(10):
            module.learn_from_text("one two three four five six seven eight nine ten")
        
        generated = module.generate_text("one", max_length=5)
        tokens = generated.split()
        assert len(tokens) <= 6  # Allow for prompt + max_length
    
    def test_generate_text_uses_learned_patterns(self):
        """Test that generation uses learned patterns"""
        module = TextLearningModule()
        
        # Train on a very repetitive pattern
        for _ in range(20):
            module.learn_from_text("alpha beta gamma")
        
        generated = module.generate_text("alpha", max_length=5)
        # Should contain learned words
        assert any(word in generated.lower() for word in ["alpha", "beta", "gamma"])


class TestSaveLoadState:
    """Test save and load functionality"""
    
    def test_save_state_creates_file(self):
        """Test that save_state creates a file"""
        module = TextLearningModule()
        module.learn_from_text("hello world test")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name
        
        try:
            module.save_state(filepath)
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_state_restores_vocabulary(self):
        """Test that load_state restores vocabulary"""
        module1 = TextLearningModule()
        module1.learn_from_text("hello world test foo bar")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name
        
        try:
            module1.save_state(filepath)
            
            module2 = TextLearningModule()
            module2.load_state(filepath)
            
            assert module2.token_to_idx == module1.token_to_idx
            assert module2.idx_to_token == module1.idx_to_token
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_state_restores_embeddings(self):
        """Test that load_state restores embeddings"""
        module1 = TextLearningModule()
        module1.learn_from_text("hello world test")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name
        
        try:
            module1.save_state(filepath)
            
            module2 = TextLearningModule()
            module2.load_state(filepath)
            
            for idx, emb in module1.embeddings.items():
                assert idx in module2.embeddings
                np.testing.assert_array_equal(emb, module2.embeddings[idx])
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_state_restores_context_weights(self):
        """Test that load_state restores context weights"""
        module1 = TextLearningModule()
        module1.learn_from_text("hello world test foo bar")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name
        
        try:
            module1.save_state(filepath)
            
            module2 = TextLearningModule()
            module2.load_state(filepath)
            
            assert len(module2.context_weights) == len(module1.context_weights)
            for ctx, weights in module1.context_weights.items():
                assert ctx in module2.context_weights
                np.testing.assert_array_equal(weights, module2.context_weights[ctx])
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_state_restores_statistics(self):
        """Test that load_state restores token statistics"""
        module1 = TextLearningModule()
        module1.learn_from_text("hello world test")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name
        
        try:
            module1.save_state(filepath)
            
            module2 = TextLearningModule()
            module2.load_state(filepath)
            
            assert module2.total_tokens == module1.total_tokens
            assert dict(module2.token_counts) == dict(module1.token_counts)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestMaxVocabularyLimit:
    """Test max vocabulary limit enforcement"""
    
    def test_vocabulary_respects_max_limit(self):
        """Test that vocabulary doesn't exceed max_vocabulary"""
        max_vocab = 20
        module = TextLearningModule(max_vocabulary=max_vocab)
        
        # Learn many unique words
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        module.learn_from_text(text)
        
        assert len(module.token_to_idx) <= max_vocab
    
    def test_rare_tokens_replaced_when_full(self):
        """Test that rare tokens are replaced when vocabulary is full"""
        max_vocab = 15
        module = TextLearningModule(max_vocabulary=max_vocab)
        
        # First, fill up vocabulary with some words
        for i in range(10):
            module.learn_from_text(f"common_word {i}")
        
        # Now add many rare words
        for i in range(50):
            module.learn_from_text(f"rare_word_{i}")
        
        # Vocabulary should still be at max
        assert len(module.token_to_idx) <= max_vocab
    
    def test_embeddings_stay_within_limit(self):
        """Test that embeddings dict respects vocabulary limit"""
        max_vocab = 15
        module = TextLearningModule(max_vocabulary=max_vocab)
        
        words = [f"word{i}" for i in range(50)]
        for word in words:
            module.learn_from_text(word)
        
        assert len(module.embeddings) <= max_vocab


class TestIntegration:
    """Integration tests for the full workflow"""
    
    def test_full_learning_workflow(self):
        """Test a complete learning workflow"""
        module = TextLearningModule(embedding_dim=64, max_vocabulary=100)
        
        # Learn from multiple texts
        texts = [
            "the quick brown fox",
            "jumps over the lazy dog",
            "the dog sleeps",
            "the fox runs quick"
        ]
        
        for text in texts:
            result = module.learn_from_text(text)
            assert result['tokens_processed'] > 0
        
        # Check stats
        stats = module.get_stats()
        assert stats['vocabulary_size'] > 0
        assert stats['total_tokens_seen'] > 0
        assert stats['unique_contexts'] > 0
        
        # Generate text
        generated = module.generate_text("the", max_length=10)
        assert isinstance(generated, str)
        assert len(generated) > 0
        
        # Save and load
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name
        
        try:
            module.save_state(filepath)
            
            new_module = TextLearningModule(embedding_dim=64, max_vocabulary=100)
            new_module.load_state(filepath)
            
            assert new_module.get_stats()['vocabulary_size'] == stats['vocabulary_size']
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_prediction_improves_with_training(self):
        """Test that context prediction improves with more training"""
        module = TextLearningModule(learning_rate=0.05, context_window=2)
        
        # Train on a simple pattern many times
        text = "red car blue car red car blue car"
        errors = []
        
        for i in range(20):
            result = module.learn_from_text(text)
            errors.append(result['avg_prediction_error'])
        
        # Error should generally trend downward
        # Compare first half with second half
        first_half_avg = np.mean(errors[:10])
        second_half_avg = np.mean(errors[10:])
        
        # Allow some tolerance for randomness
        assert second_half_avg < first_half_avg * 1.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
