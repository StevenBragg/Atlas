"""
Configuration management for the self-organizing AV system.
"""

import os
import yaml
import logging

class Config:
    """
    Configuration manager for the self-organizing AV system.
    
    This class provides utilities for loading and accessing configuration settings.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default.
        """
        self.logger = logging.getLogger(__name__)
        self.config = {}
        
        if config_path is not None:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(config_path):
                self.logger.error(f"Configuration file not found: {config_path}")
                return False
                
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.logger.info(f"Loaded configuration from {config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False
    
    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value to return if key not found
            
        Returns:
            Configuration value or default
        """
        if '.' in key:
            # Handle nested keys
            parts = key.split('.')
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        else:
            # Simple key
            return self.config.get(key, default)
    
    def set(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            value: Value to set
        """
        if '.' in key:
            # Handle nested keys
            parts = key.split('.')
            config = self.config
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            config[parts[-1]] = value
        else:
            # Simple key
            self.config[key] = value
    
    def save(self, config_path):
        """
        Save configuration to a YAML file.
        
        Args:
            config_path: Path to save the configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            self.logger.info(f"Saved configuration to {config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False 