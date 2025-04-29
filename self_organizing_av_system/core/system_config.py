import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SystemConfig:
    """
    Configuration manager for the Self-Organizing Audio-Visual Learning System.
    
    This class handles loading, validating, and providing access to configuration
    settings for all components of the system.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        default_config_path: Optional[str] = None,
    ):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to user configuration file (optional)
            default_config_path: Path to default configuration file (optional)
        """
        self.config = {}
        
        # Try to load default configuration if specified
        if default_config_path and os.path.exists(default_config_path):
            self._load_config(default_config_path)
            logger.info(f"Loaded default configuration from {default_config_path}")
        else:
            # Look for default config in package directory
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_config = os.path.join(package_dir, "config", "default_config.yaml")
            
            if os.path.exists(default_config):
                self._load_config(default_config)
                logger.info(f"Loaded default configuration from {default_config}")
            else:
                # Initialize with minimal default configuration
                self._set_minimal_defaults()
                logger.warning("No default configuration found. Using minimal defaults.")
        
        # Load user configuration if specified
        if config_path and os.path.exists(config_path):
            # Update default config with user config (don't completely replace)
            user_config = self._load_yaml(config_path)
            self._deep_update(self.config, user_config)
            logger.info(f"Loaded user configuration from {config_path}")
        
        # Validate configuration
        self._validate_config()
    
    def _load_config(self, config_path: str):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_yaml(config_path)
    
    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """
        Load YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Dictionary with loaded configuration
        """
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            return {}
    
    def _set_minimal_defaults(self):
        """Set minimal default configuration."""
        self.config = {
            "system": {
                "multimodal_size": 128,
                "learning_rate": 0.01,
                "prune_interval": 100,
                "structural_plasticity_interval": 200,
                "snapshot_interval": 1000,
                "enable_learning": True,
                "enable_visualization": True,
                "log_level": "INFO"
            },
            "visual_processor": {
                "input_width": 128,
                "input_height": 128,
                "use_grayscale": True,
                "layer_sizes": [64, 128],
                "learning_rate": 0.01,
                "use_sparse_coding": True
            },
            "audio_processor": {
                "sample_rate": 16000,
                "window_size": 1024,
                "hop_length": 512,
                "n_mels": 64,
                "layer_sizes": [64, 128],
                "learning_rate": 0.01,
                "use_sparse_coding": True
            },
            "multimodal_association": {
                "association_mode": "hebbian",
                "normalization": "softmax",
                "lateral_inhibition": 0.2,
                "use_sparse_coding": True,
                "stability_threshold": 0.1,
                "enable_attention": True
            },
            "temporal_prediction": {
                "prediction_mode": "forward",
                "sequence_length": 5,
                "prediction_horizon": 3,
                "use_eligibility_trace": True,
                "enable_surprise_detection": True,
                "enable_recurrent_connections": True
            },
            "stability": {
                "inhibition_strategy": "KWA",
                "target_activity": 0.1,
                "homeostatic_rate": 0.01,
                "enable_adaptive_threshold": True
            },
            "structural_plasticity": {
                "plasticity_mode": "adaptive",
                "growth_rate": 0.05,
                "prune_threshold": 0.01,
                "novelty_threshold": 0.3,
                "enable_neuron_growth": True,
                "enable_connection_pruning": True,
                "enable_connection_sprouting": True
            },
            "av_capture": {
                "video_width": 640,
                "video_height": 480,
                "fps": 30,
                "sample_rate": 16000,
                "video_device_id": 0,
                "audio_device_id": 0
            },
            "monitoring": {
                "update_interval": 1.0,
                "save_snapshots": True,
                "snapshot_dir": "./snapshots",
                "log_stats_interval": 10
            }
        }
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """
        Recursively update a dictionary without removing keys not in update_dict.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _validate_config(self):
        """Validate configuration and set missing values to defaults."""
        # Ensure required top-level sections exist
        required_sections = [
            "system", "visual_processor", "audio_processor", 
            "multimodal_association", "temporal_prediction", 
            "stability", "structural_plasticity",
            "av_capture", "monitoring"
        ]
        
        for section in required_sections:
            if section not in self.config:
                self.config[section] = {}
                logger.warning(f"Missing configuration section: {section}. Using defaults.")
        
        # Validate system configuration
        system_config = self.config["system"]
        if "multimodal_size" not in system_config:
            system_config["multimodal_size"] = 128
            logger.warning("Missing multimodal_size. Using default: 128")
        
        if "learning_rate" not in system_config:
            system_config["learning_rate"] = 0.01
            logger.warning("Missing learning_rate. Using default: 0.01")
        
        # Parse log level string to int if needed
        if "log_level" in system_config and isinstance(system_config["log_level"], str):
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL
            }
            level_str = system_config["log_level"].upper()
            if level_str in level_map:
                system_config["log_level"] = level_map[level_str]
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get complete configuration.
        
        Returns:
            Dictionary with configuration
        """
        return self.config
    
    def get_system_config(self) -> Dict[str, Any]:
        """
        Get system-level configuration.
        
        Returns:
            Dictionary with system configuration
        """
        return self.config.get("system", {})
    
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific component.
        
        Args:
            component_name: Name of component
            
        Returns:
            Dictionary with component configuration
        """
        return self.config.get(component_name, {})
    
    def set_config_value(self, path: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            path: Dot-separated path to configuration item (e.g., "system.learning_rate")
            value: Value to set
        """
        parts = path.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        # Set the value
        config[parts[-1]] = value
    
    def save_config(self, file_path: str):
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save configuration to
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {e}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'SystemConfig':
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            SystemConfig instance
        """
        return cls(config_path=file_path)
    
    def __getitem__(self, key: str) -> Any:
        """
        Get configuration item.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
        """
        return self.config.get(key, {})
    
    def __contains__(self, key: str) -> bool:
        """
        Check if configuration contains key.
        
        Args:
            key: Configuration key
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self.config 