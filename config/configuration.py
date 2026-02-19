"""
Unified Configuration Management System for Atlas.

This module provides a comprehensive configuration system that supports:
- YAML config files with validation
- Environment variable overrides
- Default configurations for different modes (dev, prod, test)
- Hot-reloading of config changes
"""

import os
import yaml
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import copy

logger = logging.getLogger(__name__)


class ConfigMode(Enum):
    """Configuration modes for different environments."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    MINIMAL = "minimal"


@dataclass
class SystemConfig:
    """System-level configuration."""
    multimodal_size: int = 128
    learning_rate: float = 0.01
    learning_rule: str = "oja"
    prune_interval: int = 1000
    structural_plasticity_interval: int = 2000
    snapshot_interval: int = 1000
    enable_learning: bool = True
    enable_visualization: bool = True
    log_level: str = "INFO"
    max_iterations: int = 100000
    seed: Optional[int] = None


@dataclass
class VisualConfig:
    """Visual processor configuration."""
    input_width: int = 64
    input_height: int = 64
    use_grayscale: bool = True
    patch_size: int = 8
    stride: int = 4
    contrast_normalize: bool = True
    layer_sizes: List[int] = field(default_factory=lambda: [200, 100, 50])
    use_sparse_coding: bool = True
    learning_rate: float = 0.01


@dataclass
class AudioConfig:
    """Audio processor configuration."""
    sample_rate: int = 22050
    window_size: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    min_freq: int = 50
    max_freq: int = 8000
    normalize: bool = True
    layer_sizes: List[int] = field(default_factory=lambda: [150, 75, 40])
    use_sparse_coding: bool = True
    learning_rate: float = 0.01


@dataclass
class CaptureConfig:
    """AV capture configuration."""
    video_width: int = 640
    video_height: int = 480
    fps: int = 30
    audio_channels: int = 1
    chunk_size: int = 1024
    video_device_id: int = 0
    audio_device_id: int = 0


@dataclass
class MonitorConfig:
    """Monitoring and visualization configuration."""
    update_interval: float = 0.5
    save_snapshots: bool = False
    snapshot_interval: int = 1000
    snapshot_path: str = "snapshots"
    log_stats_interval: int = 10
    enable_dashboard: bool = True
    dashboard_port: int = 8080


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    enabled: bool = True
    checkpoint_interval: int = 5000
    checkpoint_dir: str = "checkpoints"
    max_checkpoints: int = 3
    load_latest: bool = True
    save_on_exit: bool = True
    compress_checkpoints: bool = False


@dataclass
class CloudConfig:
    """Cloud deployment configuration."""
    enabled: bool = False
    provider: str = "salad"  # salad, aws, gcp, azure
    api_key: Optional[str] = None
    region: str = "us-east"
    instance_type: str = "rtx3060"
    max_instances: int = 2
    auto_scale: bool = False
    min_instances: int = 1


@dataclass
class DatabaseConfig:
    """Database configuration."""
    enabled: bool = False
    url: Optional[str] = None
    host: str = "localhost"
    port: int = 5432
    name: str = "atlas"
    user: str = "atlas"
    password: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class AtlasConfig:
    """Main Atlas configuration container."""
    system: SystemConfig = field(default_factory=SystemConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Metadata
    config_version: str = "1.0.0"
    environment: str = "development"


class ConfigurationManager:
    """
    Unified configuration manager for Atlas.
    
    Features:
    - YAML config file loading with validation
    - Environment variable overrides
    - Preset configurations for different modes
    - Hot-reloading of configuration changes
    - Type-safe configuration access
    """
    
    # Environment variable prefix
    ENV_PREFIX = "ATLAS_"
    
    # Mapping of config sections to their dataclass types
    CONFIG_SCHEMA = {
        "system": SystemConfig,
        "visual": VisualConfig,
        "audio": AudioConfig,
        "capture": CaptureConfig,
        "monitor": MonitorConfig,
        "checkpointing": CheckpointConfig,
        "cloud": CloudConfig,
        "database": DatabaseConfig,
    }
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        mode: Optional[Union[str, ConfigMode]] = None,
        enable_hot_reload: bool = False,
        hot_reload_interval: float = 5.0
    ):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
            mode: Configuration mode (development, production, testing, minimal)
            enable_hot_reload: Whether to enable hot-reloading of config changes
            hot_reload_interval: Interval in seconds to check for config changes
        """
        self._config = AtlasConfig()
        self._config_path = config_path
        self._mode = ConfigMode(mode) if mode else ConfigMode.DEVELOPMENT
        self._enable_hot_reload = enable_hot_reload
        self._hot_reload_interval = hot_reload_interval
        self._last_modified_time: Optional[float] = None
        self._reload_callbacks: List[Callable[[], None]] = []
        self._lock = threading.RLock()
        self._reload_thread: Optional[threading.Thread] = None
        self._stop_reload = threading.Event()
        
        # Load configuration
        self._load_configuration()
        
        # Start hot-reload thread if enabled
        if self._enable_hot_reload and self._config_path:
            self._start_hot_reload()
    
    def _load_configuration(self) -> None:
        """Load configuration from all sources."""
        with self._lock:
            # 1. Start with defaults (already set in _config)
            
            # 2. Load preset based on mode
            self._load_preset(self._mode)
            
            # 3. Load from config file if specified
            if self._config_path:
                self._load_from_file(self._config_path)
            
            # 4. Apply environment variable overrides
            self._apply_env_overrides()
            
            # 5. Validate configuration
            self._validate_config()
            
            logger.info(f"Configuration loaded (mode: {self._mode.value})")
    
    def _load_preset(self, mode: ConfigMode) -> None:
        """Load preset configuration for the given mode."""
        preset_configs = {
            ConfigMode.DEVELOPMENT: self._get_development_defaults,
            ConfigMode.PRODUCTION: self._get_production_defaults,
            ConfigMode.TESTING: self._get_testing_defaults,
            ConfigMode.MINIMAL: self._get_minimal_defaults,
        }
        
        if mode in preset_configs:
            preset = preset_configs[mode]()
            self._deep_update_config(preset)
            logger.debug(f"Applied {mode.value} preset configuration")
    
    def _get_development_defaults(self) -> Dict[str, Any]:
        """Get development environment defaults."""
        return {
            "system": {
                "log_level": "DEBUG",
                "enable_visualization": True,
            },
            "monitor": {
                "save_snapshots": True,
                "enable_dashboard": True,
            },
            "checkpointing": {
                "enabled": True,
                "save_on_exit": True,
            },
        }
    
    def _get_production_defaults(self) -> Dict[str, Any]:
        """Get production environment defaults."""
        return {
            "system": {
                "log_level": "WARNING",
                "enable_visualization": False,
            },
            "monitor": {
                "save_snapshots": False,
                "enable_dashboard": False,
            },
            "checkpointing": {
                "enabled": True,
                "save_on_exit": True,
                "compress_checkpoints": True,
            },
            "cloud": {
                "enabled": True,
                "auto_scale": True,
            },
        }
    
    def _get_testing_defaults(self) -> Dict[str, Any]:
        """Get testing environment defaults."""
        return {
            "system": {
                "log_level": "ERROR",
                "enable_learning": False,
                "enable_visualization": False,
                "max_iterations": 1000,
            },
            "monitor": {
                "save_snapshots": False,
                "enable_dashboard": False,
            },
            "checkpointing": {
                "enabled": False,
            },
        }
    
    def _get_minimal_defaults(self) -> Dict[str, Any]:
        """Get minimal configuration for quick testing."""
        return {
            "system": {
                "multimodal_size": 32,
                "log_level": "WARNING",
                "enable_learning": True,
                "enable_visualization": False,
                "max_iterations": 100,
            },
            "visual": {
                "input_width": 32,
                "input_height": 32,
                "patch_size": 8,
                "stride": 8,
                "layer_sizes": [32, 16],
            },
            "audio": {
                "sample_rate": 16000,
                "window_size": 512,
                "hop_length": 256,
                "n_mels": 16,
                "layer_sizes": [32, 16],
            },
            "capture": {
                "video_width": 320,
                "video_height": 240,
                "fps": 15,
                "chunk_size": 512,
            },
            "monitor": {
                "save_snapshots": False,
                "enable_dashboard": False,
                "log_stats_interval": 100,
            },
            "checkpointing": {
                "enabled": False,
            },
        }
    
    def _load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Config file not found: {config_path}")
                return
            
            with open(path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                self._deep_update_config(file_config)
                self._last_modified_time = path.stat().st_mtime
                logger.info(f"Loaded configuration from {config_path}")
        
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        for key, value in os.environ.items():
            if key.startswith(self.ENV_PREFIX):
                # Convert ATLAS_SECTION_KEY to section.key path
                config_path = key[len(self.ENV_PREFIX):].lower().replace('_', '.', 1)
                self._set_config_value(config_path, self._parse_env_value(value))
                logger.debug(f"Applied environment override: {config_path} = {value}")
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as different types
        lower_value = value.lower()
        
        # Boolean
        if lower_value in ('true', 'yes', '1'):
            return True
        if lower_value in ('false', 'no', '0'):
            return False
        
        # None/Null
        if lower_value in ('none', 'null', ''):
            return None
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # List (comma-separated)
        if ',' in value:
            return [self._parse_env_value(v.strip()) for v in value.split(',')]
        
        # String (default)
        return value
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate system config
        if self._config.system.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self._config.system.multimodal_size <= 0:
            raise ValueError("multimodal_size must be positive")
        
        # Validate visual config
        if self._config.visual.input_width <= 0 or self._config.visual.input_height <= 0:
            raise ValueError("visual input dimensions must be positive")
        
        # Validate audio config
        if self._config.audio.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        
        # Convert log level string to int for logging module
        log_level = self._config.system.log_level.upper()
        numeric_level = getattr(logging, log_level, None)
        if not isinstance(numeric_level, int):
            logger.warning(f"Invalid log level: {log_level}, using INFO")
            self._config.system.log_level = "INFO"
    
    def _deep_update_config(self, updates: Dict[str, Any]) -> None:
        """Deep update configuration with dictionary values."""
        for section_name, section_values in updates.items():
            if section_name == "environment":
                self._config.environment = section_values
                continue
            if section_name == "config_version":
                self._config.config_version = section_values
                continue
                
            if hasattr(self._config, section_name):
                section = getattr(self._config, section_name)
                if isinstance(section_values, dict):
                    for key, value in section_values.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
    
    def _set_config_value(self, path: str, value: Any) -> None:
        """Set a configuration value by dot-separated path."""
        parts = path.split('.')
        if len(parts) != 2:
            logger.warning(f"Invalid config path: {path}")
            return
        
        section_name, key = parts
        if hasattr(self._config, section_name):
            section = getattr(self._config, section_name)
            if hasattr(section, key):
                setattr(section, key, value)
    
    def _start_hot_reload(self) -> None:
        """Start the hot-reload thread."""
        self._reload_thread = threading.Thread(target=self._reload_loop, daemon=True)
        self._reload_thread.start()
        logger.info(f"Hot-reload enabled (interval: {self._hot_reload_interval}s)")
    
    def _reload_loop(self) -> None:
        """Main loop for hot-reloading configuration."""
        while not self._stop_reload.wait(self._hot_reload_interval):
            self._check_and_reload()
    
    def _check_and_reload(self) -> None:
        """Check if config file has changed and reload if necessary."""
        if not self._config_path:
            return
        
        try:
            path = Path(self._config_path)
            if not path.exists():
                return
            
            current_mtime = path.stat().st_mtime
            if self._last_modified_time and current_mtime > self._last_modified_time:
                logger.info("Configuration file changed, reloading...")
                self.reload()
            
        except Exception as e:
            logger.error(f"Error checking config file: {e}")
    
    def reload(self) -> None:
        """Reload configuration from all sources."""
        with self._lock:
            # Reset to defaults
            self._config = AtlasConfig()
            # Reload everything
            self._load_configuration()
        
        # Notify callbacks
        for callback in self._reload_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in reload callback: {e}")
        
        logger.info("Configuration reloaded")
    
    def stop_hot_reload(self) -> None:
        """Stop the hot-reload thread."""
        self._stop_reload.set()
        if self._reload_thread and self._reload_thread.is_alive():
            self._reload_thread.join(timeout=1.0)
    
    def on_reload(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when configuration is reloaded."""
        self._reload_callbacks.append(callback)
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            path: Path to save configuration (defaults to loaded config path)
        """
        save_path = path or self._config_path
        if not save_path:
            raise ValueError("No config path specified")
        
        with self._lock:
            config_dict = self.to_dict()
            
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {save_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        with self._lock:
            return {
                "config_version": self._config.config_version,
                "environment": self._config.environment,
                "system": asdict(self._config.system),
                "visual": asdict(self._config.visual),
                "audio": asdict(self._config.audio),
                "capture": asdict(self._config.capture),
                "monitor": asdict(self._config.monitor),
                "checkpointing": asdict(self._config.checkpointing),
                "cloud": asdict(self._config.cloud),
                "database": asdict(self._config.database),
            }
    
    # Property accessors for configuration sections
    @property
    def system(self) -> SystemConfig:
        """Get system configuration."""
        with self._lock:
            return copy.deepcopy(self._config.system)
    
    @property
    def visual(self) -> VisualConfig:
        """Get visual configuration."""
        with self._lock:
            return copy.deepcopy(self._config.visual)
    
    @property
    def audio(self) -> AudioConfig:
        """Get audio configuration."""
        with self._lock:
            return copy.deepcopy(self._config.audio)
    
    @property
    def capture(self) -> CaptureConfig:
        """Get capture configuration."""
        with self._lock:
            return copy.deepcopy(self._config.capture)
    
    @property
    def monitor(self) -> MonitorConfig:
        """Get monitor configuration."""
        with self._lock:
            return copy.deepcopy(self._config.monitor)
    
    @property
    def checkpointing(self) -> CheckpointConfig:
        """Get checkpointing configuration."""
        with self._lock:
            return copy.deepcopy(self._config.checkpointing)
    
    @property
    def cloud(self) -> CloudConfig:
        """Get cloud configuration."""
        with self._lock:
            return copy.deepcopy(self._config.cloud)
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        with self._lock:
            return copy.deepcopy(self._config.database)
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-separated path.
        
        Args:
            path: Dot-separated path (e.g., "system.learning_rate")
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        with self._lock:
            parts = path.split('.')
            if len(parts) != 2:
                return default
            
            section_name, key = parts
            section = getattr(self._config, section_name, None)
            if section is None:
                return default
            
            return getattr(section, key, default)
    
    def set(self, path: str, value: Any) -> None:
        """
        Set a configuration value by dot-separated path.
        
        Args:
            path: Dot-separated path (e.g., "system.learning_rate")
            value: Value to set
        """
        with self._lock:
            self._set_config_value(path, value)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop hot-reload thread."""
        self.stop_hot_reload()


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def init_config(
    config_path: Optional[str] = None,
    mode: Optional[str] = None,
    enable_hot_reload: bool = False
) -> ConfigurationManager:
    """
    Initialize the global configuration manager.
    
    Args:
        config_path: Path to YAML configuration file
        mode: Configuration mode (development, production, testing, minimal)
        enable_hot_reload: Whether to enable hot-reloading
        
    Returns:
        ConfigurationManager instance
    """
    global _config_manager
    _config_manager = ConfigurationManager(
        config_path=config_path,
        mode=mode,
        enable_hot_reload=enable_hot_reload
    )
    return _config_manager


def get_config() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    if _config_manager is None:
        raise RuntimeError("Configuration not initialized. Call init_config() first.")
    return _config_manager


def get_system_config() -> SystemConfig:
    """Get system configuration."""
    return get_config().system


def get_visual_config() -> VisualConfig:
    """Get visual configuration."""
    return get_config().visual


def get_audio_config() -> AudioConfig:
    """Get audio configuration."""
    return get_config().audio


def get_capture_config() -> CaptureConfig:
    """Get capture configuration."""
    return get_config().capture


def get_monitor_config() -> MonitorConfig:
    """Get monitor configuration."""
    return get_config().monitor


def get_checkpointing_config() -> CheckpointConfig:
    """Get checkpointing configuration."""
    return get_config().checkpointing
