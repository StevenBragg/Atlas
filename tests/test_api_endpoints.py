"""
API Endpoint Tests for Atlas Salad Cloud Service.

Tests HTTP endpoints including:
- Health checks (/health, /ready)
- Status endpoint (/status)
- Metrics endpoint (/metrics)
- Text learning endpoints (/text/learn, /text/generate, /text/stats, /chat)
"""

import pytest
import json
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import with proper path setup
from self_organizing_av_system.core.text_learning import TextLearningModule

# Try to import cloud modules
try:
    from cloud.text_api import TextLearningAPI, create_text_handler
    CLOUD_IMPORTS_AVAILABLE = True
except ImportError as e:
    CLOUD_IMPORTS_AVAILABLE = False
    print(f"Cloud imports not available: {e}")


# ============================================================================
# Text API Endpoint Tests
# ============================================================================

@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestTextLearnEndpoint:
    """Tests for POST /text/learn endpoint"""
    
    def test_learn_endpoint_success(self):
        """Test successful text learning via API"""
        text_module = TextLearningModule()
        api = TextLearningAPI(text_module)
        
        result = api.handle_learn({'text': 'hello world test'})
        
        assert result['success'] is True
        assert 'result' in result
        assert result['result']['tokens_processed'] == 3
        assert result['result']['vocabulary_size'] >= 3
    
    def test_learn_endpoint_empty_text(self):
        """Test learning endpoint with empty text"""
        text_module = TextLearningModule()
        api = TextLearningAPI(text_module)
        
        result = api.handle_learn({'text': ''})
        
        assert 'error' in result
        assert result['error'] == 'No text provided'
    
    def test_learn_endpoint_missing_text_key(self):
        """Test learning endpoint with missing text key"""
        text_module = TextLearningModule()
        api = TextLearningAPI(text_module)
        
        result = api.handle_learn({})
        
        assert 'error' in result
        assert result['error'] == 'No text provided'
    
    def test_learn_endpoint_updates_vocabulary(self):
        """Test that learning endpoint updates vocabulary"""
        text_module = TextLearningModule()
        api = TextLearningAPI(text_module)
        
        initial_vocab = len(text_module.token_to_idx)
        result = api.handle_learn({'text': 'uniqueword123 test'})
        
        assert len(text_module.token_to_idx) > initial_vocab
        assert 'uniqueword123' in text_module.token_to_idx


@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestTextGenerateEndpoint:
    """Tests for POST /text/generate endpoint"""
    
    def test_generate_endpoint_success(self):
        """Test successful text generation via API"""
        text_module = TextLearningModule()
        # Train first
        for _ in range(5):
            text_module.learn_from_text('the quick brown fox')
        
        api = TextLearningAPI(text_module)
        result = api.handle_generate({'prompt': 'the', 'max_length': 10})
        
        assert result['success'] is True
        assert 'generated' in result
        assert isinstance(result['generated'], str)
        assert result['prompt'] == 'the'
    
    def test_generate_endpoint_default_max_length(self):
        """Test generation with default max_length"""
        text_module = TextLearningModule()
        text_module.learn_from_text('hello world test')
        
        api = TextLearningAPI(text_module)
        result = api.handle_generate({'prompt': 'hello'})
        
        assert result['success'] is True
        assert 'generated' in result
    
    def test_generate_endpoint_empty_prompt(self):
        """Test generation with empty prompt"""
        text_module = TextLearningModule()
        text_module.learn_from_text('hello world')
        
        api = TextLearningAPI(text_module)
        result = api.handle_generate({'prompt': ''})
        
        assert result['success'] is True
        assert isinstance(result['generated'], str)


@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestTextStatsEndpoint:
    """Tests for GET /text/stats endpoint"""
    
    def test_stats_endpoint_success(self):
        """Test successful stats retrieval"""
        text_module = TextLearningModule()
        text_module.learn_from_text('hello world test foo bar')
        
        api = TextLearningAPI(text_module)
        result = api.handle_stats()
        
        assert result['success'] is True
        assert 'stats' in result
        stats = result['stats']
        assert 'vocabulary_size' in stats
        assert 'total_tokens_seen' in stats
        assert 'unique_contexts' in stats
        assert stats['vocabulary_size'] > 0
    
    def test_stats_endpoint_empty_module(self):
        """Test stats with empty module"""
        text_module = TextLearningModule()
        
        api = TextLearningAPI(text_module)
        result = api.handle_stats()
        
        assert result['success'] is True
        assert result['stats']['total_tokens_seen'] == 0
        assert result['stats']['vocabulary_size'] == 5  # Special tokens only


@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestChatEndpoint:
    """Tests for POST /chat endpoint"""
    
    def test_chat_endpoint_success(self):
        """Test successful chat interaction"""
        text_module = TextLearningModule()
        # Pre-train with some data
        for _ in range(5):
            text_module.learn_from_text('hello how are you today')
        
        api = TextLearningAPI(text_module)
        result = api.handle_chat({'message': 'hello'})
        
        assert result['success'] is True
        assert 'response' in result
        assert 'learned_tokens' in result
        assert result['learned_tokens'] > 0
    
    def test_chat_endpoint_learns_from_message(self):
        """Test that chat learns from the message"""
        text_module = TextLearningModule()
        api = TextLearningAPI(text_module)
        
        initial_vocab = len(text_module.token_to_idx)
        result = api.handle_chat({'message': 'uniquechatword123'})
        
        assert 'uniquechatword123' in text_module.token_to_idx
        assert len(text_module.token_to_idx) > initial_vocab


# ============================================================================
# HTTP Handler Tests
# ============================================================================

@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestTextHandlerMixin:
    """Tests for the TextHandlerMixin HTTP handler"""
    
    def test_handler_mixin_creation(self):
        """Test that handler mixin can be created"""
        text_module = TextLearningModule()
        api = TextLearningAPI(text_module)
        
        HandlerMixin = create_text_handler(api)
        assert HandlerMixin is not None
    
    @patch('builtins.open', mock_open(read_data=b'test'))
    def test_handler_routes_text_learn(self):
        """Test handler routes /text/learn correctly"""
        text_module = TextLearningModule()
        api = TextLearningAPI(text_module)
        
        HandlerMixin = create_text_handler(api)
        
        # Create a mock handler instance
        mock_handler = MagicMock()
        mock_handler.path = '/text/learn'
        mock_handler.headers = {'Content-Length': '25'}
        mock_handler.rfile.read.return_value = json.dumps({'text': 'hello world'}).encode()
        
        # Bind mixin methods to mock
        mock_handler.handle_text_endpoints = lambda: HandlerMixin.handle_text_endpoints(mock_handler)
        mock_handler._handle_text_learn = lambda: HandlerMixin._handle_text_learn(mock_handler)
        
        # Call the handler
        result = mock_handler._handle_text_learn()
        
        assert result is True
        mock_handler.send_response.assert_called_with(200)
    
    @patch('builtins.open', mock_open(read_data=b'test'))
    def test_handler_routes_text_stats(self):
        """Test handler routes /text/stats correctly"""
        text_module = TextLearningModule()
        api = TextLearningAPI(text_module)
        
        HandlerMixin = create_text_handler(api)
        
        mock_handler = MagicMock()
        mock_handler.path = '/text/stats'
        
        # Bind mixin method
        mock_handler._handle_text_stats = lambda: HandlerMixin._handle_text_stats(mock_handler)
        
        result = mock_handler._handle_text_stats()
        
        assert result is True
        mock_handler.send_response.assert_called_with(200)
    
    @patch('builtins.open', mock_open(read_data=b'test'))
    def test_handler_handles_errors(self):
        """Test handler properly handles errors"""
        text_module = TextLearningModule()
        api = TextLearningAPI(text_module)
        
        HandlerMixin = create_text_handler(api)
        
        mock_handler = MagicMock()
        mock_handler.path = '/text/learn'
        mock_handler.headers = {'Content-Length': '5'}
        # Invalid JSON to trigger error
        mock_handler.rfile.read.return_value = b'invalid'
        
        # Bind mixin method
        mock_handler._handle_text_learn = lambda: HandlerMixin._handle_text_learn(mock_handler)
        
        result = mock_handler._handle_text_learn()
        
        assert result is True
        mock_handler.send_response.assert_called_with(500)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestTextAPIIntegration:
    """Integration tests for the full Text API workflow"""
    
    def test_full_learning_workflow(self):
        """Test complete learning workflow through API"""
        text_module = TextLearningModule(max_vocabulary=50)
        api = TextLearningAPI(text_module)
        
        # Learn from multiple texts
        texts = [
            'the quick brown fox',
            'jumps over the lazy dog',
            'the dog sleeps well',
            'the fox runs fast'
        ]
        
        for text in texts:
            result = api.handle_learn({'text': text})
            assert result['success'] is True
        
        # Check stats
        stats_result = api.handle_stats()
        assert stats_result['stats']['vocabulary_size'] > 5
        assert stats_result['stats']['total_tokens_seen'] >= 16
        
        # Generate text
        gen_result = api.handle_generate({'prompt': 'the', 'max_length': 10})
        assert gen_result['success'] is True
        assert len(gen_result['generated']) > 0
        
        # Chat
        chat_result = api.handle_chat({'message': 'hello fox'})
        assert chat_result['success'] is True
        assert 'response' in chat_result
    
    def test_concurrent_learning_and_generation(self):
        """Test learning and generation work together"""
        text_module = TextLearningModule()
        api = TextLearningAPI(text_module)
        
        # Initial learning
        api.handle_learn({'text': 'hello world foo bar'})
        
        # Generate
        gen1 = api.handle_generate({'prompt': 'hello'})
        assert gen1['success'] is True
        
        # More learning
        api.handle_learn({'text': 'hello world baz qux'})
        
        # Generate again
        gen2 = api.handle_generate({'prompt': 'hello'})
        assert gen2['success'] is True
        
        # Stats should reflect all learning
        stats = api.handle_stats()
        assert stats['stats']['total_tokens_seen'] == 8


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
@pytest.mark.slow
class TestTextAPIPerformance:
    """Performance tests for Text API"""
    
    def test_large_text_learning_performance(self):
        """Test learning from large text"""
        text_module = TextLearningModule(max_vocabulary=1000)
        api = TextLearningAPI(text_module)
        
        # Generate large text
        large_text = ' '.join([f'word{i}' for i in range(500)])
        
        import time
        start = time.time()
        result = api.handle_learn({'text': large_text})
        elapsed = time.time() - start
        
        assert result['success'] is True
        assert elapsed < 5.0  # Should complete in under 5 seconds
    
    def test_rapid_api_calls(self):
        """Test handling rapid sequential API calls"""
        text_module = TextLearningModule()
        api = TextLearningAPI(text_module)
        
        import time
        start = time.time()
        
        for i in range(100):
            api.handle_learn({'text': f'test message {i}'})
        
        elapsed = time.time() - start
        
        assert elapsed < 10.0  # 100 calls should complete in under 10 seconds
        stats = api.handle_stats()
        assert stats['stats']['total_tokens_seen'] == 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
