"""TOML configuration loading and persistence."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w

CONFIG_DIR_NAME = "local-whisper-transcribe"
CONFIG_FILE_NAME = "config.toml"

DEFAULT_CONFIG: dict[str, Any] = {
    "whisper": {
        "model": "small",
        "device": "auto",
        "compute_type": "auto",
    },
    "defaults": {
        "language": "auto",
        "format": "txt",
        "output_dir": "",
    },
    "ollama": {
        "url": "http://localhost:11434",
        "model": "llama3.2",
    },
    "diarization": {
        "enabled": False,
        "hf_token": "",
    },
    "meta": {
        "setup_complete": False,
    },
}


def get_config_path() -> Path:
    """Return the path to the user config file."""
    return Path.home() / ".config" / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def config_exists() -> bool:
    """Return True if the user config file exists on disk."""
    return get_config_path().exists()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict[str, Any]:
    """Load config from disk, creating defaults on first access."""
    path = get_config_path()
    if not path.exists():
        save_config(DEFAULT_CONFIG)
        return _deep_merge({}, DEFAULT_CONFIG)

    with path.open("rb") as f:
        data = tomllib.load(f)
    return _deep_merge(DEFAULT_CONFIG, data)


def save_config(config: dict[str, Any]) -> None:
    """Persist config to disk."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        tomli_w.dump(config, f)


def set_config_value(key: str, value: str) -> None:
    """Set a dotted config key (e.g. whisper.model) to a value."""
    config = load_config()
    parts = key.split(".")
    if len(parts) != 2:
        raise ValueError(f"Key must be in section.key format, got: {key}")

    section, option = parts
    if section not in config:
        raise ValueError(f"Unknown config section: {section}")
    if option not in config[section]:
        raise ValueError(f"Unknown config option: {option} in section [{section}]")

    if option in ("enabled", "setup_complete"):
        config[section][option] = value.lower() in ("true", "1", "yes", "on")
    else:
        config[section][option] = value
    save_config(config)


def reset_config() -> None:
    """Reset configuration to defaults."""
    save_config(_deep_merge({}, DEFAULT_CONFIG))


def is_setup_complete() -> bool:
    """Return True if the user has completed initial setup."""
    config = load_config()
    return bool(config.get("meta", {}).get("setup_complete", False))


def mask_config_value(section: str, key: str, value: Any) -> str:
    """Mask sensitive values for display."""
    if section == "diarization" and key == "hf_token" and value:
        token = str(value)
        if len(token) <= 4:
            return "***"
        return f"***{token[-4:]}"
    return str(value)


def get_hf_token(config: dict[str, Any], cli_token: str | None = None) -> str | None:
    """Resolve HuggingFace token from CLI flag, env var, or config."""
    if cli_token:
        return cli_token
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    config_token = config.get("diarization", {}).get("hf_token")
    if config_token:
        return str(config_token)
    return None
