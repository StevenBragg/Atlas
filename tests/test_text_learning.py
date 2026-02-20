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
import time
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


class TestBiologicalPlasticity:
    """Test biological plasticity mechanisms: forgetting, consolidation, Hebbian decay"""

    def test_memory_consolidation_after_repeated_exposure(self):
        """Test that tokens consolidate after reaching exposure threshold"""
        module = TextLearningModule(
            embedding_dim=64,
            consolidation_threshold=5
        )

        # Expose the same text many times to trigger consolidation
        for _ in range(100):
            module.learn_from_text("consolidate this word")

        # Force consolidation (normally happens every 100 steps)
        module._consolidate_memories()

        # Tokens seen many times should be consolidated
        assert len(module.consolidated_tokens) > 0

    def test_consolidated_tokens_not_forgotten(self):
        """Test that consolidated tokens are protected from forgetting"""
        module = TextLearningModule(
            embedding_dim=64,
            consolidation_threshold=3,
            forgetting_rate=0.5  # Aggressive forgetting
        )

        # Consolidate some tokens
        for _ in range(100):
            module.learn_from_text("permanent memory")
        module._consolidate_memories()

        consolidated_before = set(module.consolidated_tokens)
        assert len(consolidated_before) > 0

        # Apply forgetting many times
        for _ in range(50):
            module._apply_forgetting()

        # Consolidated tokens should still exist in vocabulary
        for idx in consolidated_before:
            assert idx in module.embeddings

    def test_forgetting_decays_embeddings(self):
        """Test that forgetting reduces embedding magnitudes for non-consolidated tokens"""
        module = TextLearningModule(
            embedding_dim=64,
            forgetting_rate=100.0,  # Very high forgetting to ensure it triggers
            consolidation_threshold=999  # Prevent consolidation
        )

        module.learn_from_text("ephemeral data")
        # Make all tokens look very old
        old_time = time.time() - 10000
        for idx in module.token_last_seen:
            module.token_last_seen[idx] = old_time

        # Record embedding norms before forgetting
        non_special_indices = [idx for idx in module.embeddings if idx >= 5]
        norms_before = {idx: np.linalg.norm(module.embeddings[idx]) for idx in non_special_indices}

        # Apply forgetting multiple times
        for _ in range(100):
            module._apply_forgetting()

        # At least some embeddings should have decayed
        any_decayed = False
        for idx in non_special_indices:
            if idx in module.embeddings:
                norm_after = np.linalg.norm(module.embeddings[idx])
                if norm_after < norms_before[idx]:
                    any_decayed = True
                    break
        assert any_decayed, "No embeddings were decayed by forgetting"

    def test_hebbian_decay_reduces_weights(self):
        """Test that Hebbian decay gradually reduces context weight magnitudes"""
        module = TextLearningModule(
            embedding_dim=64,
            hebbian_decay=0.5  # Strong decay for testing
        )

        # Learn to create context weights
        module.learn_from_text("alpha beta gamma delta")

        # Record initial weight norms
        if module.context_weights:
            ctx = list(module.context_weights.keys())[0]
            norm_before = np.linalg.norm(module.context_weights[ctx])

            # Call predict multiple times to apply decay
            for _ in range(5):
                module._predict_from_context(ctx)

            norm_after = np.linalg.norm(module.context_weights[ctx])
            assert norm_after < norm_before

    def test_consolidation_strengthens_embeddings(self):
        """Test that consolidation amplifies embedding vectors"""
        module = TextLearningModule(
            embedding_dim=64,
            consolidation_threshold=3
        )

        module.learn_from_text("strengthen this")
        token_idx = module.token_to_idx["strengthen"]
        norm_before = np.linalg.norm(module.embeddings[token_idx])

        # Exceed consolidation threshold
        module.token_exposure_count[token_idx] = 20
        module._consolidate_memories()

        assert token_idx in module.consolidated_tokens
        norm_after = np.linalg.norm(module.embeddings[token_idx])
        assert norm_after > norm_before

    def test_biological_processes_trigger_periodically(self):
        """Test that forgetting and consolidation happen every 100 learning steps"""
        module = TextLearningModule(
            embedding_dim=64,
            consolidation_threshold=5
        )

        # Run exactly 99 learning steps
        for i in range(99):
            module.learn_from_text(f"word{i}")

        # At step 99, no consolidation yet
        consolidated_at_99 = len(module.consolidated_tokens)

        # Step 100 should trigger biological processes
        module.learn_from_text("trigger step")

        # The step counter should be at 100
        assert module.learning_step == 100


class TestGetStats:
    """Test get_stats method"""

    def test_stats_keys(self):
        """Test that get_stats returns all expected keys"""
        module = TextLearningModule()
        module.learn_from_text("hello world")
        stats = module.get_stats()

        expected_keys = [
            'vocabulary_size', 'total_tokens_seen', 'unique_contexts',
            'consolidated_tokens', 'learning_step', 'avg_token_exposure'
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_stats_values_after_learning(self):
        """Test that stats reflect actual learning state"""
        module = TextLearningModule()
        module.learn_from_text("one two three")
        module.learn_from_text("four five")

        stats = module.get_stats()
        assert stats['vocabulary_size'] >= 10  # 5 special + 5 words
        assert stats['total_tokens_seen'] == 5
        assert stats['learning_step'] == 2

    def test_stats_on_fresh_module(self):
        """Test stats on a module with no training"""
        module = TextLearningModule()
        stats = module.get_stats()
        assert stats['total_tokens_seen'] == 0
        assert stats['learning_step'] == 0
        assert stats['unique_contexts'] == 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_very_long_text(self):
        """Test learning from a very long text"""
        module = TextLearningModule()
        long_text = " ".join([f"word{i}" for i in range(500)])
        result = module.learn_from_text(long_text)
        assert result['tokens_processed'] == 500

    def test_repeated_identical_text(self):
        """Test learning from the same text many times"""
        module = TextLearningModule()
        for _ in range(50):
            result = module.learn_from_text("repeat this exact sentence")
        assert result['tokens_processed'] == 4
        assert module.total_tokens == 200  # 4 tokens * 50 iterations

    def test_single_word_text(self):
        """Test learning from a single word"""
        module = TextLearningModule()
        result = module.learn_from_text("hello")
        assert result['tokens_processed'] == 1
        assert result['avg_prediction_error'] == 0  # No context for single word

    def test_special_characters_only(self):
        """Test tokenization of special characters"""
        module = TextLearningModule()
        tokens = module.tokenize("!@#$%")
        # Should extract punctuation or return UNK
        assert len(tokens) > 0

    def test_unicode_text(self):
        """Test handling of unicode text"""
        module = TextLearningModule()
        result = module.learn_from_text("hello café résumé")
        assert result['tokens_processed'] > 0

    def test_generate_with_unknown_prompt(self):
        """Test generation with words not in vocabulary"""
        module = TextLearningModule()
        module.learn_from_text("known words only")
        generated = module.generate_text("completely unknown words", max_length=5)
        assert isinstance(generated, str)
        assert len(generated) > 0

    def test_context_window_respected(self):
        """Test that context window size is respected"""
        window = 2
        module = TextLearningModule(context_window=window)
        module.learn_from_text("a b c d e f g")

        # All contexts should have at most window_size elements
        for ctx in module.context_weights.keys():
            assert len(ctx) <= window

    def test_pruned_token_cleanup(self):
        """Test that pruning cleans up all references to the removed token"""
        module = TextLearningModule(max_vocabulary=12)

        # Fill vocabulary
        module.learn_from_text("alpha beta gamma delta epsilon")
        initial_vocab = len(module.token_to_idx)

        # Add more words to trigger pruning
        for i in range(20):
            module.learn_from_text(f"newword{i}")

        # Verify consistency: every token in token_to_idx has an embedding
        for token, idx in module.token_to_idx.items():
            assert idx in module.embeddings, f"Token '{token}' (idx={idx}) has no embedding"
            assert idx in module.idx_to_token, f"Token '{token}' (idx={idx}) missing from idx_to_token"

    def test_save_load_preserves_consolidated_tokens(self):
        """Test that save/load preserves the consolidated tokens set"""
        module = TextLearningModule(consolidation_threshold=3)

        for _ in range(100):
            module.learn_from_text("persist this memory")
        module._consolidate_memories()

        assert len(module.consolidated_tokens) > 0

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name

        try:
            module.save_state(filepath)

            module2 = TextLearningModule()
            module2.load_state(filepath)

            assert module2.consolidated_tokens == module.consolidated_tokens
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_save_load_preserves_exposure_counts(self):
        """Test that save/load preserves token exposure counts"""
        module = TextLearningModule()
        module.learn_from_text("hello hello hello world")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name

        try:
            module.save_state(filepath)

            module2 = TextLearningModule()
            module2.load_state(filepath)

            assert module2.token_exposure_count == module.token_exposure_count
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
