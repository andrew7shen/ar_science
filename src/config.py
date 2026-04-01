"""Configuration management for the Analogous Reasoning Agent."""

import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration singleton for the application."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def load(self, config_path: str = None):
        """Load configuration from YAML file."""
        if config_path is None:
            # Default to config.yaml in project root
            config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., "model.name")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if self._config is None:
            self.load()

        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_all(self) -> Dict:
        """Get the entire configuration dictionary."""
        if self._config is None:
            self.load()
        return self._config


# Global config instance
config = Config()


def get_config(key_path: str = None, default: Any = None) -> Any:
    """
    Convenience function to get config values.

    Args:
        key_path: Dot-separated path to config value (e.g., "model.name")
        default: Default value if key not found

    Returns:
        Configuration value, entire config dict if no key_path, or default
    """
    if key_path is None:
        return config.get_all()
    return config.get(key_path, default)
