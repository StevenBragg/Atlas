"""
Test fixtures and utilities for Atlas testing.

Provides common fixtures, mocks, and utilities for testing
the self-organizing AV system.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from self_organizing_av_system.core.text_learning import TextLearningModule
from self_organizing_av_system.core.neuron import Neuron
from self_organizing_av_system.core.layer import NeuralLayer
from self_organizing_av_system.core.pathway import NeuralPathway


# ============================================================================
# Text Learning Fixtures
# ============================================================================

@pytest.fixture
def text_module():
    """Create a basic TextLearningModule for testing."""
    return TextLearningModule(
        embedding_dim=64,
        max_vocabulary=100,
        context_window=3,
        learning_rate=0.01
    )


@pytest.fixture
def trained_text_module():
    """Create a pre-trained TextLearningModule."""
    module = TextLearningModule(
        embedding_dim=64,
        max_vocabulary=100,
        context_window=3,
        learning_rate=0.05
    )
    # Train on some sample text
    texts = [
        "the quick brown fox",
        "jumps over the lazy dog",
        "the dog sleeps",
        "the fox runs quick"
    ]
    for text in texts:
        module.learn_from_text(text)
    return module


@pytest.fixture
def temp_checkpoint_file():
    """Create a temporary file for checkpoint testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        filepath = f.name
    yield filepath
    if os.path.exists(filepath):
        os.unlink(filepath)


# ============================================================================
# Neural Network Fixtures
# ============================================================================

@pytest.fixture
def sample_neuron():
    """Create a sample neuron for testing."""
    return Neuron(
        input_dim=10,
        learning_rate=0.01,
        threshold=0.5
    )


@pytest.fixture
def sample_layer():
    """Create a sample layer for testing."""
    layer = NeuralLayer(
        input_size=10,
        layer_size=5,
        learning_rate=0.01
    )
    return layer


@pytest.fixture
def sample_pathway():
    """Create a sample pathway for testing."""
    pathway = NeuralPathway(
        input_size=10,
        output_size=5,
        learning_rate=0.01
    )
    return pathway


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_visual_data():
    """Generate sample visual data for testing."""
    return np.random.randint(0, 256, (64, 64), dtype=np.uint8)


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    return np.random.randn(1024).astype(np.float32)


@pytest.fixture
def sample_embedding():
    """Generate a sample embedding vector."""
    return np.random.randn(128).astype(np.float32)


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_backend():
    """Mock the backend module for testing without GPU."""
    with patch('self_organizing_av_system.core.backend') as mock:
        mock.HAS_GPU = False
        mock.get_backend_info.return_value = {
            'device_name': 'CPU',
            'device_memory_gb': 0,
            'backend': 'numpy'
        }
        yield mock


@pytest.fixture
def mock_gpu_backend():
    """Mock the backend module with GPU enabled."""
    with patch('self_organizing_av_system.core.backend') as mock:
        mock.HAS_GPU = True
        mock.get_backend_info.return_value = {
            'device_name': 'NVIDIA GeForce RTX 4090',
            'device_memory_gb': 24.0,
            'backend': 'cupy'
        }
        yield mock


@pytest.fixture
def mock_prometheus():
    """Mock Prometheus client for metrics testing."""
    with patch.dict('sys.modules', {'prometheus_client': MagicMock()}):
        import prometheus_client
        prometheus_client.Counter = MagicMock
        prometheus_client.Gauge = MagicMock
        prometheus_client.Histogram = MagicMock
        prometheus_client.generate_latest = MagicMock(return_value=b'metrics')
        prometheus_client.CONTENT_TYPE_LATEST = 'text/plain'
        yield prometheus_client


# ============================================================================
# Cloud Service Fixtures
# ============================================================================

@pytest.fixture
def mock_salad_environment():
    """Set up mock Salad Cloud environment variables."""
    env_vars = {
        'SALAD_MACHINE_ID': 'test-machine-123',
        'SALAD_CONTAINER_GROUP_ID': 'test-group-456',
        'ATLAS_CHECKPOINT_DIR': '/tmp/test_checkpoints',
        'ATLAS_LOG_LEVEL': 'DEBUG',
        'ATLAS_HTTP_PORT': '8888',
        'ATLAS_ENABLE_TEXT_LEARNING': 'true',
        'ATLAS_ENABLE_UNIFIED_INTELLIGENCE': 'false'
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Helper Functions
# ============================================================================

def create_mock_http_request(data: dict, method: str = 'POST') -> MagicMock:
    """Create a mock HTTP request for testing API endpoints."""
    request = MagicMock()
    request.method = method
    request.headers = {'Content-Length': str(len(str(data)))}
    request.rfile.read.return_value = str(data).encode()
    return request


def create_mock_http_handler() -> MagicMock:
    """Create a mock HTTP handler for testing."""
    handler = MagicMock()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.wfile.write = MagicMock()
    return handler


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def minimal_config():
    """Minimal configuration for testing."""
    return {
        'system': {
            'multimodal_size': 64,
            'learning_rate': 0.01,
            'prune_interval': 1000,
            'structural_plasticity_interval': 5000
        },
        'visual': {
            'input_width': 64,
            'input_height': 64,
            'use_grayscale': True,
            'patch_size': 8,
            'stride': 4,
            'contrast_normalize': True,
            'layer_sizes': [64, 32]
        },
        'audio': {
            'sample_rate': 16000,
            'window_size': 512,
            'hop_length': 256,
            'n_mels': 40,
            'min_freq': 20,
            'max_freq': 8000,
            'normalize': True,
            'layer_sizes': [40, 20]
        },
        'checkpointing': {
            'enabled': True,
            'checkpoint_dir': '/tmp/test_checkpoints',
            'checkpoint_interval': 100,
            'max_checkpoints': 3,
            'load_latest': False
        }
    }
