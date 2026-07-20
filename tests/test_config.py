"""Tests for configuration helpers."""

from unittest.mock import patch

from local_whisper_transcribe.config import (
    DEFAULT_CONFIG,
    get_hf_token,
    is_setup_complete,
    load_config,
    mask_config_value,
    reset_config,
    set_config_value,
)


def test_mask_config_value_masks_token():
    assert mask_config_value("diarization", "hf_token", "hf_abcdefghij") == "***ghij"
    assert mask_config_value("diarization", "hf_token", "") == ""
    assert mask_config_value("whisper", "model", "small") == "small"


def test_set_config_value_boolean(tmp_path):
    config_file = tmp_path / "config.toml"
    with patch("local_whisper_transcribe.config.get_config_path", return_value=config_file):
        reset_config()
        set_config_value("diarization.enabled", "true")
        config = load_config()
        assert config["diarization"]["enabled"] is True
        set_config_value("meta.setup_complete", "yes")
        config = load_config()
        assert config["meta"]["setup_complete"] is True


def test_get_hf_token_priority():
    config = {"diarization": {"hf_token": "from_config"}}
    assert get_hf_token(config, "from_cli") == "from_cli"
    assert get_hf_token(config) == "from_config"

    with patch.dict("os.environ", {"HF_TOKEN": "from_env"}):
        assert get_hf_token({"diarization": {"hf_token": ""}}) == "from_env"


def test_reset_config(tmp_path):
    config_file = tmp_path / "config.toml"
    with patch("local_whisper_transcribe.config.get_config_path", return_value=config_file):
        set_config_value("whisper.model", "medium")
        reset_config()
        config = load_config()
        assert config["whisper"]["model"] == DEFAULT_CONFIG["whisper"]["model"]


def test_is_setup_complete_default_false(tmp_path):
    config_file = tmp_path / "config.toml"
    with patch("local_whisper_transcribe.config.get_config_path", return_value=config_file):
        reset_config()
        assert is_setup_complete() is False
        set_config_value("meta.setup_complete", "true")
        assert is_setup_complete() is True
