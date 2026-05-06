"""
Configuration parsing and caching module.
"""

import json
import logging
import logging.config
import os
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any

from .version import validate_config_version

def get_config_dir() -> Path:
    """Returns the configuration directory, respecting environment overrides."""
    env_dir = os.environ.get("FREQ_EXTRACTOR_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).parent.parent.parent / "config"

def load_json_config(filename: str) -> Dict[str, Any]:
    """Loads a JSON configuration file."""
    filepath = get_config_dir() / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return data

@lru_cache(maxsize=1)
def get_setup() -> Dict[str, Any]:
    """Loads and validates setup.json."""
    data = load_json_config("setup.json")
    if "version" in data:
        validate_config_version(str(data["version"]))
    return data

@lru_cache(maxsize=1)
def get_rate_limits() -> Dict[str, Any]:
    """Loads rate_limits.json."""
    return load_json_config("rate_limits.json")

@lru_cache(maxsize=1)
def get_logging_config() -> Dict[str, Any]:
    """Loads logging_config.json."""
    return load_json_config("logging_config.json")

def setup_logging() -> None:
    """Configures system-wide logging using logging_config.json."""
    try:
        config = get_logging_config()
        # Ensure log directory exists
        for handler in config.get("handlers", {}).values():
            if "filename" in handler:
                log_file = Path(handler["filename"])
                # Resolve relative to project root
                if not log_file.is_absolute():
                    log_file = Path(__file__).parent.parent.parent / log_file
                    handler["filename"] = str(log_file)
                log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.config.dictConfig(config)
    except FileNotFoundError:
        logging.basicConfig(level=logging.INFO)
        logging.warning("logging_config.json not found, using basic logging.")
