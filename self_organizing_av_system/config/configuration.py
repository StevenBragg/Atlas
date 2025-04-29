import yaml
import os
import logging
from typing import Dict, Any, Optional


class SystemConfig:
    """
    Configuration manager for the self-organizing AV system.
    
    This handles loading, validating, and providing configuration settings
    for all components of the system.
    """
    
    DEFAULT_CONFIG = {
        "system": {
            "multimodal_size": 100,
            "learning_rate": 0.01,
            "learning_rule": "oja",
            "prune_interval": 1000,
            "structural_plasticity_interval": 5000
        },
        "visual": {
            "input_width": 64,
            "input_height": 64,
            "use_grayscale": True,
            "patch_size": 8,
            "stride": 4,
            "contrast_normalize": True,
            "layer_sizes": [200, 100, 50]
        },
        "audio": {
            "sample_rate": 22050,
            "window_size": 1024,
            "hop_length": 512,
            "n_mels": 64,
            "min_freq": 50,
            "max_freq": 8000,
            "normalize": True,
            "layer_sizes": [150, 75, 40]
        },
        "capture": {
            "video_width": 640,
            "video_height": 480,
            "fps": 30,
            "audio_channels": 1,
            "chunk_size": 1024
        },
        "monitor": {
            "update_interval": 0.5,
            "save_snapshots": False,
            "snapshot_interval": 1000,
            "snapshot_path": "snapshots"
        },
        "checkpointing": {
            "enabled": True,
            "checkpoint_interval": 5000,
            "checkpoint_dir": "checkpoints",
            "max_checkpoints": 3,
            "load_latest": True,
            "save_on_exit": True
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to YAML configuration file (or None for defaults)
        """
        self.logger = logging.getLogger('Configuration')
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_file:
            self._load_from_file(config_file)
    
    def _load_from_file(self, config_file: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_file: Path to YAML configuration file
        """
        try:
            if not os.path.exists(config_file):
                self.logger.warning(f"Config file {config_file} not found, using defaults")
                return
            
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            # Update with loaded values
            self._update_config(loaded_config)
            self.logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading config from {config_file}: {e}")
            self.logger.info("Using default configuration")
    
    def _update_config(self, loaded_config: Dict[str, Any]) -> None:
        """
        Update configuration with loaded values.
        
        Args:
            loaded_config: Dictionary of configuration values
        """
        # Update each section if present
        for section in self.config:
            if section in loaded_config:
                # Update only keys that exist in the default config
                for key in self.config[section]:
                    if key in loaded_config[section]:
                        self.config[section][key] = loaded_config[section][key]
    
    def get_system_config(self) -> Dict[str, Any]:
        """
        Get system-level configuration.
        
        Returns:
            Dictionary with system configuration
        """
        return self.config["system"]
    
    def get_visual_config(self) -> Dict[str, Any]:
        """
        Get visual processor configuration.
        
        Returns:
            Dictionary with visual configuration
        """
        return self.config["visual"]
    
    def get_audio_config(self) -> Dict[str, Any]:
        """
        Get audio processor configuration.
        
        Returns:
            Dictionary with audio configuration
        """
        return self.config["audio"]
    
    def get_capture_config(self) -> Dict[str, Any]:
        """
        Get capture configuration.
        
        Returns:
            Dictionary with capture configuration
        """
        return self.config["capture"]
    
    def get_monitor_config(self) -> Dict[str, Any]:
        """
        Get monitor configuration.
        
        Returns:
            Dictionary with monitor configuration
        """
        return self.config["monitor"]
    
    def get_checkpointing_config(self) -> Dict[str, Any]:
        """
        Get checkpointing configuration.
        
        Returns:
            Dictionary with checkpointing configuration
        """
        return self.config["checkpointing"]
    
    def save_config(self, filename: str) -> bool:
        """
        Save current configuration to a file.
        
        Args:
            filename: Path to save configuration file
            
        Returns:
            Whether saving was successful
        """
        try:
            with open(filename, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"Saved configuration to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving config to {filename}: {e}")
            return False
    
    def __getitem__(self, key: str) -> Dict[str, Any]:
        """
        Get a configuration section.
        
        Args:
            key: Section name
            
        Returns:
            Configuration dictionary for the section
        """
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError(f"Configuration section {key} not found")
    
    def update(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: New value
        """
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
        else:
            raise KeyError(f"Configuration {section}.{key} not found") 