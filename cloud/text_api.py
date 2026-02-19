"""
Text Learning API Extension for Atlas Salad Cloud Service

Adds HTTP endpoints for text-based learning and communication.
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.text_learning import TextLearningModule

logger = logging.getLogger(__name__)


class TextLearningAPI:
    """API handler for text learning endpoints."""
    
    def __init__(self, text_module: TextLearningModule):
        self.text_module = text_module
    
    def handle_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text learning request"""
        text = data.get('text', '')
        if not text:
            return {'error': 'No text provided'}
        
        result = self.text_module.learn_from_text(text)
        return {
            'success': True,
            'result': result
        }
    
    def handle_generate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text generation request"""
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 50)
        
        generated = self.text_module.generate_text(prompt, max_length)
        return {
            'success': True,
            'generated': generated,
            'prompt': prompt
        }
    
    def handle_stats(self) -> Dict[str, Any]:
        """Get text learning statistics"""
        return {
            'success': True,
            'stats': self.text_module.get_stats()
        }
    
    def handle_chat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle conversational interaction"""
        message = data.get('message', '')
        
        # Learn from the message
        self.text_module.learn_from_text(message)
        
        # Generate response
        response = self.text_module.generate_text(message, max_length=100)
        
        return {
            'success': True,
            'response': response,
            'learned_tokens': len(self.text_module.tokenize(message))
        }


def create_text_handler(text_api: TextLearningAPI):
    """Create HTTP request handler with text endpoints."""
    
    class TextHandlerMixin:
        """Mixin to add text endpoints to existing handler"""
        
        def handle_text_endpoints(self):
            """Route text-related endpoints"""
            path = self.path
            
            if path == '/text/learn':
                return self._handle_text_learn()
            elif path == '/text/generate':
                return self._handle_text_generate()
            elif path == '/text/stats':
                return self._handle_text_stats()
            elif path == '/chat':
                return self._handle_chat()
            
            return None  # Not a text endpoint
        
        def _handle_text_learn(self):
            """POST /text/learn - Learn from text"""
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode())
                
                result = text_api.handle_learn(data)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                return True
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
                return True
        
        def _handle_text_generate(self):
            """POST /text/generate - Generate text"""
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode())
                
                result = text_api.handle_generate(data)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                return True
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
                return True
        
        def _handle_text_stats(self):
            """GET /text/stats - Get learning statistics"""
            try:
                result = text_api.handle_stats()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                return True
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
                return True
        
        def _handle_chat(self):
            """POST /chat - Conversational interface"""
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode())
                
                result = text_api.handle_chat(data)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                return True
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
                return True
    
    return TextHandlerMixin


# Example usage / test
if __name__ == "__main__":
    # Create text module
    text_module = TextLearningModule()
    
    # Learn some example text
    examples = [
        "Hello world this is a test",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "Neural networks process information"
    ]
    
    for text in examples:
        result = text_module.learn_from_text(text)
        print(f"Learned: {text}")
        print(f"  Tokens: {result['tokens_processed']}, Vocab: {result['vocabulary_size']}")
    
    # Generate text
    print("\nGenerating text:")
    generated = text_module.generate_text("Hello", max_length=20)
    print(f"Prompt: 'Hello' -> Generated: '{generated}'")
    
    # Stats
    print("\nStats:")
    print(text_module.get_stats())
