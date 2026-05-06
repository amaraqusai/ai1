"""
Version validation and constants for the freq_extractor package.
"""

CODE_VERSION = "1.00"
MIN_CONFIG_VERSION = "1.00"
MAX_CONFIG_VERSION = "1.99"

def parse_version(version_str: str) -> float:
    """Parses a version string into a float for comparison."""
    try:
        return float(version_str)
    except ValueError:
        return 0.0

def validate_config_version(config_version: str) -> None:
    """Validates if the provided config version is compatible with the code."""
    c_ver = parse_version(config_version)
    min_ver = parse_version(MIN_CONFIG_VERSION)
    max_ver = parse_version(MAX_CONFIG_VERSION)
    if not (min_ver <= c_ver <= max_ver):
        raise ValueError(f"Config version {config_version} out of bounds [{MIN_CONFIG_VERSION}, {MAX_CONFIG_VERSION}].")
