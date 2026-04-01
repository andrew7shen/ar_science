"""
Utility functions for dataset creation pipeline.
"""

import json
from pathlib import Path
from datetime import datetime


def _init_colors(use_colors: bool = True) -> dict:
    """Initialize ANSI color codes for terminal output."""
    if use_colors:
        return {
            'C': '\033[96m',  # Cyan
            'B': '\033[94m',  # Blue
            'G': '\033[92m',  # Green
            'Y': '\033[93m',  # Yellow
            'R': '\033[0m'    # Reset
        }
    return {'C': '', 'B': '', 'G': '', 'Y': '', 'R': ''}


class TeeOutput:
    """Write to both terminal and file simultaneously."""
    def __init__(self, file_path, original_stdout):
        self.file = open(file_path, 'w')
        self.stdout = original_stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def _get_pricing(model: str) -> tuple[float, float]:
    """Get pricing (input, output) per million tokens for a given model.

    Returns:
        Tuple of (input_price, output_price) per million tokens
    """
    model_lower = model.lower()

    # Haiku pricing
    if "haiku" in model_lower:
        return (1.00, 5.00)

    # Sonnet pricing
    if "sonnet" in model_lower:
        return (3.00, 15.00)

    # Opus pricing
    if "opus" in model_lower:
        return (15.00, 75.00)

    # Perplexity Sonar/Sonar Pro (uses Sonnet pricing)
    if "sonar" in model_lower:
        return (3.00, 15.00)

    # Default to Sonnet
    return (3.00, 15.00)


def save_state(output_dir: Path, state: dict, stage: str):
    """Save complete state after each stage for resumption.

    Args:
        output_dir: Output directory path
        state: Complete state dictionary
        stage: Stage name (discovery, verification, extraction, assessment)
    """
    # Update state with current stage info
    state["last_completed_stage"] = stage
    state["last_updated"] = datetime.now().isoformat()

    # Save complete state
    state_file = output_dir / "state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    # Save stage-specific checkpoint
    stage_num = {
        "discovery": 1,
        "verification": 2,
        "extraction": 3,
        "assessment": 4
    }.get(stage, 0)

    if stage_num > 0:
        stage_file = output_dir / f"stage_{stage_num}_{stage}.json"
        stage_data = state.get(f"{stage}_data", [])
        with open(stage_file, 'w') as f:
            json.dump(stage_data, f, indent=2)


def load_state(output_dir: Path) -> dict:
    """Load state from previous run for resumption.

    Args:
        output_dir: Output directory path containing state.json

    Returns:
        Complete state dictionary

    Raises:
        FileNotFoundError: If state file doesn't exist
    """
    state_file = output_dir / "state.json"
    if not state_file.exists():
        raise FileNotFoundError(f"No state file found in {output_dir}")

    with open(state_file, 'r') as f:
        return json.load(f)


def load_config(config_path: str = "dataset_creation/config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Configuration dictionary
    """
    import yaml

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_config_value(config: dict, key_path: str, default=None):
    """Get nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "apis.perplexity.model")
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
