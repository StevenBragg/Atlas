"""
Checkpoint Manager - Enhanced Data Persistence and Checkpointing

This module provides:
- Automatic checkpointing on timer
- Cloud storage sync (S3-compatible)
- Checkpoint compression
- Checkpoint versioning and rollback
"""

import os
import time
import json
import gzip
import shutil
import logging
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, asdict
from threading import Lock, Thread
import pickle

import numpy as np
import yaml

# Optional S3 support
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Metadata for a checkpoint."""
    name: str
    path: str
    created_at: str
    size_bytes: int
    version: int
    parent_version: Optional[int] = None
    compressed: bool = False
    cloud_synced: bool = False
    cloud_url: Optional[str] = None
    checksum: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointInfo':
        return cls(**data)


class CheckpointManager:
    """
    Manages system checkpoints with support for:
    - Automatic periodic checkpointing
    - Compression to save space
    - Versioning with rollback capability
    - Cloud storage sync (S3-compatible)
    """
    
    DEFAULT_CONFIG = {
        "checkpoint_dir": "checkpoints",
        "auto_checkpoint": True,
        "checkpoint_interval_minutes": 10,
        "max_checkpoints": 10,
        "max_versions": 5,
        "compression": True,
        "compression_level": 6,
        "cloud_sync": False,
        "cloud_provider": "s3",  # s3, minio, etc.
        "s3_bucket": None,
        "s3_region": "us-east-1",
        "s3_endpoint": None,  # For MinIO compatibility
        "s3_access_key": None,
        "s3_secret_key": None,
        "sync_on_save": False,
        "backup_retention_days": 30,
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the checkpoint manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.checkpoint_dir = Path(self.config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Version tracking
        self.version_file = self.checkpoint_dir / "versions.json"
        self.versions: Dict[int, CheckpointInfo] = {}
        self.current_version = 0
        self._load_versions()
        
        # Threading
        self._lock = Lock()
        self._auto_checkpoint_thread: Optional[Thread] = None
        self._stop_auto_checkpoint = False
        self._system_reference: Optional[Any] = None
        
        # Cloud storage client
        self._s3_client = None
        self._init_cloud_storage()
        
        # Callbacks
        self._pre_save_callbacks: List[Callable] = []
        self._post_save_callbacks: List[Callable] = []
        
        logger.info(f"CheckpointManager initialized: dir={self.checkpoint_dir}")
    
    def _load_versions(self):
        """Load version history from disk."""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    data = json.load(f)
                    self.current_version = data.get("current_version", 0)
                    versions_data = data.get("versions", {})
                    self.versions = {
                        int(k): CheckpointInfo.from_dict(v) 
                        for k, v in versions_data.items()
                    }
                logger.info(f"Loaded {len(self.versions)} checkpoint versions")
            except Exception as e:
                logger.error(f"Error loading versions: {e}")
                self.versions = {}
                self.current_version = 0
    
    def _save_versions(self):
        """Save version history to disk."""
        try:
            with self._lock:
                data = {
                    "current_version": self.current_version,
                    "versions": {
                        str(k): v.to_dict() 
                        for k, v in self.versions.items()
                    }
                }
                with open(self.version_file, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving versions: {e}")
    
    def _init_cloud_storage(self):
        """Initialize S3-compatible cloud storage client."""
        if not S3_AVAILABLE or not self.config.get("cloud_sync"):
            return
        
        try:
            session_kwargs = {
                "region_name": self.config.get("s3_region", "us-east-1"),
            }
            
            # Support for MinIO and other S3-compatible services
            endpoint_url = self.config.get("s3_endpoint")
            if endpoint_url:
                session_kwargs["endpoint_url"] = endpoint_url
            
            # Credentials
            access_key = self.config.get("s3_access_key")
            secret_key = self.config.get("s3_secret_key")
            if access_key and secret_key:
                session_kwargs["aws_access_key_id"] = access_key
                session_kwargs["aws_secret_access_key"] = secret_key
            
            session = boto3.session.Session()
            self._s3_client = session.client("s3", **session_kwargs)
            
            # Test connection
            self._s3_client.list_buckets()
            logger.info("Cloud storage (S3) initialized successfully")
            
        except NoCredentialsError:
            logger.warning("S3 credentials not found. Cloud sync disabled.")
            self._s3_client = None
        except Exception as e:
            logger.error(f"Error initializing cloud storage: {e}")
            self._s3_client = None
    
    def _compute_checksum(self, filepath: Path) -> str:
        """Compute MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _compress_checkpoint(self, source_path: Path, dest_path: Path) -> bool:
        """
        Compress a checkpoint file.
        
        Args:
            source_path: Path to the uncompressed file
            dest_path: Path for the compressed file
            
        Returns:
            True if successful
        """
        try:
            level = self.config.get("compression_level", 6)
            with open(source_path, 'rb') as f_in:
                with gzip.open(dest_path, 'wb', compresslevel=level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True
        except Exception as e:
            logger.error(f"Error compressing checkpoint: {e}")
            return False
    
    def _decompress_checkpoint(self, source_path: Path, dest_path: Path) -> bool:
        """
        Decompress a checkpoint file.
        
        Args:
            source_path: Path to the compressed file
            dest_path: Path for the decompressed file
            
        Returns:
            True if successful
        """
        try:
            with gzip.open(source_path, 'rb') as f_in:
                with open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True
        except Exception as e:
            logger.error(f"Error decompressing checkpoint: {e}")
            return False
    
    def _upload_to_cloud(self, local_path: Path, key: str) -> Optional[str]:
        """
        Upload a file to cloud storage.
        
        Args:
            local_path: Local file path
            key: S3 object key
            
        Returns:
            Cloud URL if successful, None otherwise
        """
        if not self._s3_client:
            return None
        
        bucket = self.config.get("s3_bucket")
        if not bucket:
            logger.warning("S3 bucket not configured")
            return None
        
        try:
            self._s3_client.upload_file(str(local_path), bucket, key)
            
            # Generate URL
            if self.config.get("s3_endpoint"):
                # MinIO-style URL
                url = f"{self.config['s3_endpoint']}/{bucket}/{key}"
            else:
                # AWS S3 URL
                url = f"s3://{bucket}/{key}"
            
            logger.info(f"Uploaded to cloud: {key}")
            return url
            
        except Exception as e:
            logger.error(f"Error uploading to cloud: {e}")
            return None
    
    def _download_from_cloud(self, key: str, local_path: Path) -> bool:
        """
        Download a file from cloud storage.
        
        Args:
            key: S3 object key
            local_path: Local destination path
            
        Returns:
            True if successful
        """
        if not self._s3_client:
            return False
        
        bucket = self.config.get("s3_bucket")
        if not bucket:
            return False
        
        try:
            self._s3_client.download_file(bucket, key, str(local_path))
            logger.info(f"Downloaded from cloud: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading from cloud: {e}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints based on retention policy."""
        max_checkpoints = self.config.get("max_checkpoints", 10)
        retention_days = self.config.get("backup_retention_days", 30)
        
        with self._lock:
            # Sort versions by creation time
            sorted_versions = sorted(
                self.versions.items(),
                key=lambda x: x[1].created_at,
                reverse=True
            )
            
            # Keep only max_checkpoints
            versions_to_remove = sorted_versions[max_checkpoints:]
            
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for version, info in versions_to_remove:
                # Don't delete if within retention period and it's a versioned checkpoint
                created = datetime.fromisoformat(info.created_at)
                if created > cutoff_date and version > self.current_version - self.config.get("max_versions", 5):
                    continue
                
                # Remove local file
                path = Path(info.path)
                if path.exists():
                    try:
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                        logger.info(f"Removed old checkpoint: {info.name}")
                    except Exception as e:
                        logger.error(f"Error removing checkpoint {info.name}: {e}")
                
                # Remove from versions
                del self.versions[version]
        
        self._save_versions()
    
    def register_pre_save_callback(self, callback: Callable):
        """Register a callback to run before saving a checkpoint."""
        self._pre_save_callbacks.append(callback)
    
    def register_post_save_callback(self, callback: Callable):
        """Register a callback to run after saving a checkpoint."""
        self._post_save_callbacks.append(callback)
    
    def save_checkpoint(
        self,
        system: Any,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compress: Optional[bool] = None,
        sync_to_cloud: Optional[bool] = None
    ) -> CheckpointInfo:
        """
        Save a checkpoint of the system state.
        
        Args:
            system: The system to checkpoint
            name: Checkpoint name (auto-generated if None)
            metadata: Additional metadata to store
            compress: Whether to compress (uses config default if None)
            sync_to_cloud: Whether to sync to cloud (uses config default if None)
            
        Returns:
            CheckpointInfo for the saved checkpoint
        """
        # Run pre-save callbacks
        for callback in self._pre_save_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in pre-save callback: {e}")
        
        with self._lock:
            self.current_version += 1
            version = self.current_version
            
            if name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"checkpoint_v{version}_{timestamp}"
            
            checkpoint_path = self.checkpoint_dir / name
            
            # Save system state
            if hasattr(system, 'save_checkpoint'):
                # Use the system's checkpoint method
                saved_path = system.save_checkpoint(str(checkpoint_path))
                checkpoint_path = Path(saved_path)
            elif hasattr(system, 'save_state'):
                # Use the system's state save method
                state_file = checkpoint_path / "state.pkl"
                state_file.parent.mkdir(parents=True, exist_ok=True)
                system.save_state(str(state_file))
            else:
                # Fallback: pickle the entire system
                state_file = checkpoint_path.with_suffix(".pkl")
                with open(state_file, 'wb') as f:
                    pickle.dump(system, f)
                checkpoint_path = state_file
            
            # Compute checksum
            checksum = None
            if checkpoint_path.is_file():
                checksum = self._compute_checksum(checkpoint_path)
            
            # Compress if enabled
            do_compress = compress if compress is not None else self.config.get("compression", True)
            compressed_path = None
            
            if do_compress and checkpoint_path.is_file():
                compressed_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".gz")
                if self._compress_checkpoint(checkpoint_path, compressed_path):
                    # Remove uncompressed file after successful compression
                    checkpoint_path.unlink()
                    checkpoint_path = compressed_path
                    compressed = True
                else:
                    compressed = False
            else:
                compressed = False
            
            # Get file size
            if checkpoint_path.is_dir():
                size_bytes = sum(f.stat().st_size for f in checkpoint_path.rglob('*') if f.is_file())
            else:
                size_bytes = checkpoint_path.stat().st_size
            
            # Create checkpoint info
            parent_version = max(self.versions.keys()) if self.versions else None
            info = CheckpointInfo(
                name=name,
                path=str(checkpoint_path),
                created_at=datetime.now().isoformat(),
                size_bytes=size_bytes,
                version=version,
                parent_version=parent_version,
                compressed=compressed,
                cloud_synced=False,
                cloud_url=None,
                checksum=checksum,
                metadata=metadata or {}
            )
            
            self.versions[version] = info
        
        # Save version index
        self._save_versions()
        
        # Sync to cloud if enabled
        do_sync = sync_to_cloud if sync_to_cloud is not None else self.config.get("sync_on_save", False)
        if do_sync and self._s3_client:
            key = f"checkpoints/{name}"
            if compressed:
                key += ".gz"
            cloud_url = self._upload_to_cloud(Path(info.path), key)
            if cloud_url:
                info.cloud_synced = True
                info.cloud_url = cloud_url
                self._save_versions()
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Run post-save callbacks
        for callback in self._post_save_callbacks:
            try:
                callback(info)
            except Exception as e:
                logger.error(f"Error in post-save callback: {e}")
        
        logger.info(f"Saved checkpoint v{version}: {name} ({size_bytes} bytes)")
        return info
    
    def load_checkpoint(
        self,
        version: Optional[int] = None,
        name: Optional[str] = None,
        system_class: Optional[type] = None
    ) -> Tuple[Any, CheckpointInfo]:
        """
        Load a checkpoint.
        
        Args:
            version: Version number to load (latest if None)
            name: Checkpoint name to load (takes precedence over version)
            system_class: Class to instantiate for loading
            
        Returns:
            Tuple of (loaded_system, checkpoint_info)
        """
        with self._lock:
            if name:
                # Find by name
                info = None
                for v in self.versions.values():
                    if v.name == name:
                        info = v
                        break
                if not info:
                    raise ValueError(f"Checkpoint not found: {name}")
            elif version:
                info = self.versions.get(version)
                if not info:
                    raise ValueError(f"Version not found: {version}")
            else:
                # Get latest version
                if not self.versions:
                    raise ValueError("No checkpoints available")
                version = max(self.versions.keys())
                info = self.versions[version]
        
        checkpoint_path = Path(info.path)
        
        # Download from cloud if needed and not available locally
        if not checkpoint_path.exists() and info.cloud_url and self._s3_client:
            key = f"checkpoints/{info.name}"
            if info.compressed:
                key += ".gz"
            self._download_from_cloud(key, checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Decompress if needed
        working_path = checkpoint_path
        if info.compressed:
            working_path = checkpoint_path.with_suffix('')
            self._decompress_checkpoint(checkpoint_path, working_path)
        
        # Load the checkpoint
        if system_class and hasattr(system_class, 'load_checkpoint'):
            system = system_class.load_checkpoint(str(working_path))
        else:
            # Fallback: try to unpickle
            state_file = working_path / "state.pkl" if working_path.is_dir() else working_path
            with open(state_file, 'rb') as f:
                system = pickle.load(f)
        
        # Cleanup decompressed file if it was temporary
        if info.compressed and working_path != checkpoint_path and working_path.exists():
            if working_path.is_dir():
                shutil.rmtree(working_path)
            else:
                working_path.unlink()
        
        logger.info(f"Loaded checkpoint v{info.version}: {info.name}")
        return system, info
    
    def rollback(self, steps: int = 1) -> CheckpointInfo:
        """
        Rollback to a previous checkpoint version.
        
        Args:
            steps: Number of versions to rollback
            
        Returns:
            CheckpointInfo for the rolled-back version
        """
        with self._lock:
            target_version = self.current_version - steps
            if target_version < 1:
                raise ValueError(f"Cannot rollback {steps} steps (current version: {self.current_version})")
            
            if target_version not in self.versions:
                raise ValueError(f"Version {target_version} not found")
            
            self.current_version = target_version
        
        self._save_versions()
        info = self.versions[target_version]
        logger.info(f"Rolled back to version {target_version}: {info.name}")
        return info
    
    def list_checkpoints(
        self,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Args:
            include_metadata: Whether to include full metadata
            
        Returns:
            List of checkpoint information
        """
        with self._lock:
            checkpoints = []
            for version in sorted(self.versions.keys(), reverse=True):
                info = self.versions[version]
                data = {
                    "version": info.version,
                    "name": info.name,
                    "created_at": info.created_at,
                    "size_bytes": info.size_bytes,
                    "compressed": info.compressed,
                    "cloud_synced": info.cloud_synced,
                }
                if include_metadata:
                    data["metadata"] = info.metadata
                    data["checksum"] = info.checksum
                    data["cloud_url"] = info.cloud_url
                    data["parent_version"] = info.parent_version
                checkpoints.append(data)
            return checkpoints
    
    def get_checkpoint_info(self, version: int) -> Optional[CheckpointInfo]:
        """Get information about a specific checkpoint version."""
        return self.versions.get(version)
    
    def delete_checkpoint(self, version: int) -> bool:
        """
        Delete a specific checkpoint version.
        
        Args:
            version: Version number to delete
            
        Returns:
            True if deleted successfully
        """
        with self._lock:
            if version not in self.versions:
                return False
            
            info = self.versions[version]
            path = Path(info.path)
            
            # Delete local file
            if path.exists():
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                except Exception as e:
                    logger.error(f"Error deleting checkpoint file: {e}")
                    return False
            
            # Delete from cloud if synced
            if info.cloud_synced and self._s3_client:
                try:
                    bucket = self.config.get("s3_bucket")
                    key = f"checkpoints/{info.name}"
                    if info.compressed:
                        key += ".gz"
                    self._s3_client.delete_object(Bucket=bucket, Key=key)
                except Exception as e:
                    logger.error(f"Error deleting cloud checkpoint: {e}")
            
            del self.versions[version]
        
        self._save_versions()
        logger.info(f"Deleted checkpoint v{version}")
        return True
    
    def sync_to_cloud(self, version: Optional[int] = None) -> List[CheckpointInfo]:
        """
        Sync checkpoints to cloud storage.
        
        Args:
            version: Specific version to sync (all unsynced if None)
            
        Returns:
            List of synced checkpoint infos
        """
        if not self._s3_client:
            logger.warning("Cloud storage not available")
            return []
        
        synced = []
        
        with self._lock:
            versions_to_sync = []
            if version:
                if version in self.versions:
                    versions_to_sync.append(version)
            else:
                # Sync all unsynced checkpoints
                versions_to_sync = [
                    v for v, info in self.versions.items() 
                    if not info.cloud_synced
                ]
            
            for v in versions_to_sync:
                info = self.versions[v]
                key = f"checkpoints/{info.name}"
                if info.compressed:
                    key += ".gz"
                
                cloud_url = self._upload_to_cloud(Path(info.path), key)
                if cloud_url:
                    info.cloud_synced = True
                    info.cloud_url = cloud_url
                    synced.append(info)
        
        if synced:
            self._save_versions()
        
        return synced
    
    def sync_from_cloud(self) -> List[CheckpointInfo]:
        """
        Sync checkpoints from cloud storage.
        
        Returns:
            List of downloaded checkpoint infos
        """
        if not self._s3_client:
            logger.warning("Cloud storage not available")
            return []
        
        bucket = self.config.get("s3_bucket")
        if not bucket:
            return []
        
        downloaded = []
        
        try:
            response = self._s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix="checkpoints/"
            )
            
            for obj in response.get("Contents", []):
                key = obj["Key"]
                filename = Path(key).name
                
                # Remove .gz suffix for name matching
                if filename.endswith(".gz"):
                    name = filename[:-3]
                else:
                    name = filename
                
                # Check if already have this checkpoint
                existing = any(info.name == name for info in self.versions.values())
                if existing:
                    continue
                
                # Download
                local_path = self.checkpoint_dir / filename
                if self._download_from_cloud(key, local_path):
                    # Create checkpoint info
                    self.current_version += 1
                    info = CheckpointInfo(
                        name=name,
                        path=str(local_path),
                        created_at=obj["LastModified"].isoformat(),
                        size_bytes=obj["Size"],
                        version=self.current_version,
                        compressed=filename.endswith(".gz"),
                        cloud_synced=True,
                        cloud_url=f"s3://{bucket}/{key}" if not self.config.get("s3_endpoint") else f"{self.config['s3_endpoint']}/{bucket}/{key}"
                    )
                    self.versions[self.current_version] = info
                    downloaded.append(info)
            
            if downloaded:
                self._save_versions()
                
        except Exception as e:
            logger.error(f"Error syncing from cloud: {e}")
        
        return downloaded
    
    def start_auto_checkpoint(
        self,
        system: Any,
        interval_minutes: Optional[int] = None
    ):
        """
        Start automatic periodic checkpointing.
        
        Args:
            system: The system to checkpoint
            interval_minutes: Override the configured interval
        """
        if self._auto_checkpoint_thread and self._auto_checkpoint_thread.is_alive():
            logger.warning("Auto-checkpoint already running")
            return
        
        self._system_reference = system
        self._stop_auto_checkpoint = False
        
        interval = interval_minutes or self.config.get("checkpoint_interval_minutes", 10)
        
        def checkpoint_loop():
            while not self._stop_auto_checkpoint:
                try:
                    time.sleep(interval * 60)
                    if not self._stop_auto_checkpoint and self._system_reference:
                        self.save_checkpoint(
                            self._system_reference,
                            name=f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        )
                except Exception as e:
                    logger.error(f"Error in auto-checkpoint: {e}")
        
        self._auto_checkpoint_thread = Thread(target=checkpoint_loop, daemon=True)
        self._auto_checkpoint_thread.start()
        logger.info(f"Auto-checkpoint started (interval: {interval} minutes)")
    
    def stop_auto_checkpoint(self):
        """Stop automatic checkpointing."""
        self._stop_auto_checkpoint = True
        if self._auto_checkpoint_thread:
            self._auto_checkpoint_thread.join(timeout=5)
        logger.info("Auto-checkpoint stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics."""
        with self._lock:
            total_size = sum(info.size_bytes for info in self.versions.values())
            cloud_synced = sum(1 for info in self.versions.values() if info.cloud_synced)
            compressed = sum(1 for info in self.versions.values() if info.compressed)
            
            return {
                "total_checkpoints": len(self.versions),
                "current_version": self.current_version,
                "total_size_bytes": total_size,
                "cloud_synced": cloud_synced,
                "compressed": compressed,
                "auto_checkpoint_running": (
                    self._auto_checkpoint_thread is not None and 
                    self._auto_checkpoint_thread.is_alive()
                ),
                "cloud_available": self._s3_client is not None,
            }


# Singleton instance for application-wide access
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager(config: Optional[Dict[str, Any]] = None) -> CheckpointManager:
    """Get or create the global checkpoint manager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager(config)
    return _checkpoint_manager


def reset_checkpoint_manager():
    """Reset the global checkpoint manager (for testing)."""
    global _checkpoint_manager
    _checkpoint_manager = None
