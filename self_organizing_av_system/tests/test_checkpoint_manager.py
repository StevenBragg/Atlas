"""
Tests for the checkpoint manager module.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from self_organizing_av_system.core.checkpoint_manager import (
    CheckpointManager, CheckpointInfo, get_checkpoint_manager, reset_checkpoint_manager
)


class TestCheckpointInfo(unittest.TestCase):
    """Test CheckpointInfo dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = CheckpointInfo(
            name="test",
            path="/tmp/test",
            created_at="2024-01-01T00:00:00",
            size_bytes=1000,
            version=1
        )
        
        data = info.to_dict()
        self.assertEqual(data["name"], "test")
        self.assertEqual(data["version"], 1)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "test",
            "path": "/tmp/test",
            "created_at": "2024-01-01T00:00:00",
            "size_bytes": 1000,
            "version": 1,
            "parent_version": None,
            "compressed": False,
            "cloud_synced": False,
            "cloud_url": None,
            "checksum": None,
            "metadata": None
        }
        
        info = CheckpointInfo.from_dict(data)
        self.assertEqual(info.name, "test")
        self.assertEqual(info.version, 1)


class TestCheckpointManager(unittest.TestCase):
    """Test CheckpointManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "checkpoint_dir": self.temp_dir,
            "compression": False,
            "cloud_sync": False,
            "max_checkpoints": 5
        }
        reset_checkpoint_manager()
        self.manager = CheckpointManager(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        reset_checkpoint_manager()
    
    def test_initialization(self):
        """Test manager initialization."""
        self.assertEqual(self.manager.checkpoint_dir, Path(self.temp_dir))
        self.assertTrue(self.manager.checkpoint_dir.exists())
    
    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        # Create a mock system
        mock_system = Mock()
        mock_system.save_checkpoint.return_value = os.path.join(self.temp_dir, "test_checkpoint")
        
        # Create the checkpoint directory
        os.makedirs(os.path.join(self.temp_dir, "test_checkpoint"), exist_ok=True)
        
        # Save checkpoint
        info = self.manager.save_checkpoint(mock_system, name="test_checkpoint")
        
        self.assertEqual(info.name, "test_checkpoint")
        self.assertEqual(info.version, 1)
        mock_system.save_checkpoint.assert_called_once()
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        # Create a mock system
        mock_system = Mock()
        mock_system.save_checkpoint.return_value = os.path.join(self.temp_dir, "test_checkpoint")
        
        # Create the checkpoint directory
        os.makedirs(os.path.join(self.temp_dir, "test_checkpoint"), exist_ok=True)
        
        # Save a checkpoint
        self.manager.save_checkpoint(mock_system, name="test_checkpoint")
        
        # List checkpoints
        checkpoints = self.manager.list_checkpoints()
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["name"], "test_checkpoint")
    
    def test_get_stats(self):
        """Test getting statistics."""
        stats = self.manager.get_stats()
        
        self.assertEqual(stats["total_checkpoints"], 0)
        self.assertEqual(stats["current_version"], 0)
        self.assertFalse(stats["auto_checkpoint_running"])
    
    def test_compute_checksum(self):
        """Test checksum computation."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        checksum = self.manager._compute_checksum(test_file)
        self.assertIsNotNone(checksum)
        self.assertEqual(len(checksum), 32)  # MD5 is 32 hex chars
    
    def test_rollback(self):
        """Test rollback functionality."""
        # Create a mock system
        mock_system = Mock()
        
        # Create multiple checkpoints
        for i in range(3):
            checkpoint_dir = os.path.join(self.temp_dir, f"checkpoint_{i}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            mock_system.save_checkpoint.return_value = checkpoint_dir
            self.manager.save_checkpoint(mock_system, name=f"checkpoint_{i}")
        
        # Rollback
        info = self.manager.rollback(steps=1)
        self.assertEqual(info.version, 2)
    
    def test_delete_checkpoint(self):
        """Test deleting a checkpoint."""
        # Create a mock system
        mock_system = Mock()
        checkpoint_dir = os.path.join(self.temp_dir, "test_checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        mock_system.save_checkpoint.return_value = checkpoint_dir
        
        # Save and delete
        info = self.manager.save_checkpoint(mock_system, name="test_checkpoint")
        success = self.manager.delete_checkpoint(info.version)
        
        self.assertTrue(success)
        self.assertEqual(len(self.manager.versions), 0)


class TestCompression(unittest.TestCase):
    """Test compression functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "checkpoint_dir": self.temp_dir,
            "compression": True,
            "compression_level": 6,
            "cloud_sync": False
        }
        reset_checkpoint_manager()
        self.manager = CheckpointManager(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        reset_checkpoint_manager()
    
    def test_compress_decompress(self):
        """Test compression and decompression."""
        # Create a test file
        source = Path(self.temp_dir) / "source.txt"
        source.write_text("test content for compression" * 100)
        
        dest = Path(self.temp_dir) / "compressed.txt.gz"
        
        # Compress
        result = self.manager._compress_checkpoint(source, dest)
        self.assertTrue(result)
        self.assertTrue(dest.exists())
        
        # Decompress
        decompressed = Path(self.temp_dir) / "decompressed.txt"
        result = self.manager._decompress_checkpoint(dest, decompressed)
        self.assertTrue(result)
        self.assertEqual(decompressed.read_text(), source.read_text())


class TestSingleton(unittest.TestCase):
    """Test singleton functionality."""
    
    def tearDown(self):
        """Clean up."""
        reset_checkpoint_manager()
    
    def test_get_checkpoint_manager(self):
        """Test singleton getter."""
        config = {"checkpoint_dir": "/tmp/test_checkpoints"}
        
        manager1 = get_checkpoint_manager(config)
        manager2 = get_checkpoint_manager()  # Should return same instance
        
        self.assertIs(manager1, manager2)


if __name__ == '__main__':
    unittest.main()
