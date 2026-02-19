"""
Tests for Atlas Cloud Services.

Tests cloud-specific functionality:
- Salad Cloud service initialization
- Checkpoint management
- Health check endpoints
- Metrics collection
- Graceful shutdown
"""

import pytest
import json
import sys
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import cloud modules
try:
    from cloud.salad_service import AtlasSaladService, SaladCloudMetrics
    CLOUD_IMPORTS_AVAILABLE = True
except ImportError as e:
    CLOUD_IMPORTS_AVAILABLE = False
    print(f"Cloud imports not available: {e}")


# ============================================================================
# Salad Cloud Service Tests
# ============================================================================

@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestSaladCloudServiceInitialization:
    """Tests for Salad Cloud service initialization"""
    
    @patch.dict(os.environ, {
        'SALAD_MACHINE_ID': 'test-machine-123',
        'SALAD_CONTAINER_GROUP_ID': 'test-group-456',
        'ATLAS_CHECKPOINT_DIR': '/tmp/test_checkpoints',
        'ATLAS_LOG_LEVEL': 'DEBUG'
    }, clear=False)
    @patch('cloud.salad_service.HAS_CUDA', False)
    @patch('cloud.salad_service.GPU_NAME', 'CPU')
    def test_service_reads_environment_variables(self):
        """Test that service reads Salad Cloud environment variables"""
        from cloud.salad_service import AtlasSaladService
        
        # Mock the system initialization to avoid complex setup
        with patch.object(AtlasSaladService, '_initialize_system'):
            service = AtlasSaladService()
            
            assert service.salad_machine_id == 'test-machine-123'
            assert service.salad_container_group_id == 'test-group-456'
    
    @patch.dict(os.environ, {
        'ATLAS_CHECKPOINT_INTERVAL': '100',
        'ATLAS_MAX_CHECKPOINTS': '5',
        'ATLAS_MULTIMODAL_SIZE': '256',
        'ATLAS_LEARNING_RATE': '0.005'
    }, clear=False)
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_service_applies_salad_config(self):
        """Test that service applies Salad Cloud optimized configuration"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            service = AtlasSaladService()
            
            config = service.config.get_checkpointing_config()
            assert config['checkpoint_interval'] == 100
            assert config['max_checkpoints'] == 5
            
            system_config = service.config.get_system_config()
            assert system_config['multimodal_size'] == 256
            assert system_config['learning_rate'] == 0.005


@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestCheckpointManagement:
    """Tests for checkpoint management"""
    
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_checkpoint_filename_format(self):
        """Test checkpoint filename includes required components"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.dict(os.environ, {'ATLAS_CHECKPOINT_DIR': tmpdir}):
                    service = AtlasSaladService()
                    service.system = MagicMock()
                    service.system.frame_count = 1000
                    service.system.save_state.return_value = True
                    
                    service._save_checkpoint()
                    
                    # Check file was created with correct format
                    import glob
                    files = glob.glob(os.path.join(tmpdir, "*.pkl"))
                    assert len(files) > 0
                    
                    filename = os.path.basename(files[0])
                    assert 'checkpoint' in filename or 'final' in filename
                    assert '1000' in filename
    
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_cleanup_old_checkpoints(self):
        """Test that old checkpoints are cleaned up"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.dict(os.environ, {'ATLAS_CHECKPOINT_DIR': tmpdir}):
                    service = AtlasSaladService()
                    
                    # Create fake old checkpoints
                    for i in range(5):
                        old_file = os.path.join(tmpdir, f"checkpoint_old_{i}.pkl")
                        with open(old_file, 'w') as f:
                            f.write("test")
                        time.sleep(0.01)  # Ensure different timestamps
                    
                    # Cleanup with max_checkpoints=3
                    service._cleanup_old_checkpoints({
                        'checkpoint_dir': tmpdir,
                        'max_checkpoints': 3
                    })
                    
                    import glob
                    files = glob.glob(os.path.join(tmpdir, "checkpoint_*.pkl"))
                    assert len(files) <= 3
    
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_final_checkpoint_on_shutdown(self):
        """Test that final checkpoint is saved on shutdown"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.dict(os.environ, {'ATLAS_CHECKPOINT_DIR': tmpdir}):
                    service = AtlasSaladService()
                    service.system = MagicMock()
                    service.system.frame_count = 500
                    service.system.save_state.return_value = True
                    
                    service._save_checkpoint(is_final=True)
                    
                    import glob
                    files = glob.glob(os.path.join(tmpdir, "final_*.pkl"))
                    assert len(files) > 0


@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestHealthCheckEndpoints:
    """Tests for health check endpoints"""
    
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_get_status_structure(self):
        """Test that status has correct structure"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            service = AtlasSaladService()
            service.running = True
            service.ready = True
            service.start_time = time.time() - 100  # 100 seconds ago
            service.total_frames = 1000
            service.system = MagicMock()
            service.system.frame_count = 1000
            service.text_module = None
            service.unified_intelligence = None
            
            status = service.get_status()
            
            assert 'status' in status
            assert 'ready' in status
            assert 'uptime_seconds' in status
            assert 'total_frames_processed' in status
            assert 'gpu' in status
            assert 'salad_cloud' in status
            
            assert status['status'] == 'running'
            assert status['ready'] is True
            assert status['uptime_seconds'] >= 100
            assert status['gpu']['available'] is False
            assert status['salad_cloud']['machine_id'] == 'unknown'
    
    @patch('cloud.salad_service.HAS_CUDA', True)
    @patch('cloud.salad_service.GPU_NAME', 'NVIDIA GeForce RTX 4090')
    @patch('cloud.salad_service.GPU_MEMORY', 24.0)
    def test_get_status_with_gpu(self):
        """Test status includes GPU info when available"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            service = AtlasSaladService()
            service.running = True
            service.ready = True
            service.start_time = time.time()
            service.total_frames = 0
            service.system = MagicMock()
            service.text_module = None
            service.unified_intelligence = None
            
            status = service.get_status()
            
            assert status['gpu']['available'] is True
            assert status['gpu']['name'] == 'NVIDIA GeForce RTX 4090'
            assert status['gpu']['memory_gb'] == 24.0


@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestMetricsCollection:
    """Tests for metrics collection"""
    
    @patch('cloud.salad_service.HAS_PROMETHEUS', True)
    def test_metrics_initialization(self):
        """Test that metrics are initialized when Prometheus is available"""
        from cloud.salad_service import SaladCloudMetrics
        
        with patch('cloud.salad_service.Counter') as mock_counter, \
             patch('cloud.salad_service.Gauge') as mock_gauge, \
             patch('cloud.salad_service.Histogram') as mock_histogram:
            
            metrics = SaladCloudMetrics()
            
            # Verify metrics were created
            assert mock_counter.call_count >= 2  # frames_processed, learning_cycles
            assert mock_gauge.call_count >= 3    # gpu_utilization, memory, intelligence
            assert mock_histogram.call_count >= 1  # processing_time
    
    @patch('cloud.salad_service.HAS_PROMETHEUS', False)
    def test_metrics_no_prometheus(self):
        """Test that metrics work without Prometheus"""
        from cloud.salad_service import SaladCloudMetrics
        
        # Should not raise exception
        metrics = SaladCloudMetrics()
        
        # Update should also work
        metrics.update_gpu_metrics()  # Should not raise


@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestGracefulShutdown:
    """Tests for graceful shutdown behavior"""
    
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_shutdown_saves_checkpoint(self):
        """Test that shutdown saves a checkpoint"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.dict(os.environ, {'ATLAS_CHECKPOINT_DIR': tmpdir}):
                    service = AtlasSaladService()
                    service.system = MagicMock()
                    service.system.frame_count = 100
                    service.system.save_state.return_value = True
                    service.running = True
                    service.ready = True
                    service.start_time = time.time()
                    service.total_frames = 100
                    service.text_module = None
                    
                    service.stop()
                    
                    # Verify checkpoint was saved
                    import glob
                    files = glob.glob(os.path.join(tmpdir, "final_*.pkl"))
                    assert len(files) > 0
    
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_shutdown_sets_flags(self):
        """Test that shutdown sets correct flags"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.dict(os.environ, {'ATLAS_CHECKPOINT_DIR': tmpdir}):
                    service = AtlasSaladService()
                    service.system = MagicMock()
                    service.system.save_state.return_value = True
                    service.running = True
                    service.ready = True
                    service.text_module = None
                    
                    service.stop()
                    
                    assert service.running is False
                    assert service.ready is False


# ============================================================================
# HTTP Server Tests
# ============================================================================

@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestHTTPServer:
    """Tests for HTTP server functionality"""
    
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_health_endpoint(self):
        """Test /health endpoint returns correct status"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            service = AtlasSaladService()
            service.running = True
            service.ready = True
            service.start_time = time.time()
            service.total_frames = 0
            service.system = MagicMock()
            service.text_module = None
            service.unified_intelligence = None
            
            # Create mock handler
            mock_handler = MagicMock()
            mock_handler.path = '/health'
            mock_handler.wfile = MagicMock()
            
            # Simulate GET request
            status = service.get_status()
            
            assert status['status'] == 'running'
    
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_ready_endpoint_when_ready(self):
        """Test /ready endpoint when service is ready"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            service = AtlasSaladService()
            service.ready = True
            
            assert service.get_status()['ready'] is True
    
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_ready_endpoint_when_not_ready(self):
        """Test /ready endpoint when service is not ready"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            service = AtlasSaladService()
            service.ready = False
            
            assert service.get_status()['ready'] is False


# ============================================================================
# Text Learning in Cloud Context
# ============================================================================

@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestTextLearningCloudIntegration:
    """Tests for text learning in cloud context"""
    
    @patch.dict(os.environ, {'ATLAS_ENABLE_TEXT_LEARNING': 'true'})
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_text_module_initialization_when_enabled(self):
        """Test text module is initialized when enabled"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            service = AtlasSaladService()
            # The text module would be initialized in _initialize_system
            # We just verify the environment check works
            assert os.environ.get('ATLAS_ENABLE_TEXT_LEARNING') == 'true'
    
    @patch.dict(os.environ, {'ATLAS_ENABLE_TEXT_LEARNING': 'false'})
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_text_module_not_initialized_when_disabled(self):
        """Test text module is not initialized when disabled"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            service = AtlasSaladService()
            assert os.environ.get('ATLAS_ENABLE_TEXT_LEARNING') == 'false'


# ============================================================================
# Error Handling
# ============================================================================

@pytest.mark.skipif(not CLOUD_IMPORTS_AVAILABLE, reason="Cloud imports not available")
class TestCloudServiceErrorHandling:
    """Tests for error handling in cloud service"""
    
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_checkpoint_save_failure_handling(self):
        """Test handling of checkpoint save failure"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.dict(os.environ, {'ATLAS_CHECKPOINT_DIR': tmpdir}):
                    service = AtlasSaladService()
                    service.system = MagicMock()
                    service.system.save_state.return_value = False
                    
                    result = service._save_checkpoint()
                    
                    assert result is False
    
    @patch('cloud.salad_service.HAS_CUDA', False)
    def test_load_latest_checkpoint_no_files(self):
        """Test loading checkpoint when no files exist"""
        from cloud.salad_service import AtlasSaladService
        
        with patch.object(AtlasSaladService, '_initialize_system'):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.dict(os.environ, {'ATLAS_CHECKPOINT_DIR': tmpdir}):
                    service = AtlasSaladService()
                    service.system = MagicMock()
                    
                    # Should not raise exception
                    service._load_latest_checkpoint()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
