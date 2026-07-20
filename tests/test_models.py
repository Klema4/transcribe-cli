"""Tests for model download and cache helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_whisper_transcribe.models import (
    format_size,
    get_repo_id,
    is_model_cached,
    download_model,
)


def test_get_repo_id_known_model():
    assert get_repo_id("small") == "Systran/faster-whisper-small"


def test_get_repo_id_custom_repo():
    assert get_repo_id("user/custom-model") == "user/custom-model"


def test_get_repo_id_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_repo_id("not-a-model")


def test_format_size_bytes():
    assert format_size(512) == "0.5 KB"
    assert format_size(2 * 1024 * 1024) == "2.0 MB"
    assert format_size(3 * 1024 * 1024 * 1024) == "3.00 GB"
    assert format_size(None) == "—"


def test_is_model_cached_local_path(tmp_path):
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    assert is_model_cached(str(model_dir)) is True


@patch("local_whisper_transcribe.models.try_to_load_from_cache", return_value=None)
def test_is_model_cached_missing(_mock_cache):
    assert is_model_cached("small") is False


@patch("local_whisper_transcribe.models.try_to_load_from_cache")
def test_is_model_cached_hit(mock_cache, tmp_path):
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"x" * 10)
    mock_cache.return_value = str(model_file)
    assert is_model_cached("small") is True


@patch("local_whisper_transcribe.models.load_model")
@patch("local_whisper_transcribe.models.snapshot_download")
@patch("local_whisper_transcribe.models.is_model_cached", return_value=False)
@patch("local_whisper_transcribe.models.get_cached_model_path")
def test_download_model_fetches_from_hub(
    mock_cached_path,
    _mock_is_cached,
    mock_snapshot,
    mock_load_model,
    tmp_path,
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.bin").write_bytes(b"data")
    mock_snapshot.return_value = str(snapshot)
    mock_cached_path.return_value = snapshot

    messages: list[str] = []
    path = download_model("tiny", status_callback=messages.append)

    assert path == snapshot
    mock_snapshot.assert_called_once()
    mock_load_model.assert_called_once()
    assert any("Downloading" in message for message in messages)


@patch("local_whisper_transcribe.models.load_model")
@patch("local_whisper_transcribe.models.fw_download_model")
@patch("local_whisper_transcribe.models.is_model_cached", return_value=True)
@patch("local_whisper_transcribe.models.get_cached_model_path")
def test_download_model_uses_cache_when_present(
    mock_cached_path,
    _mock_is_cached,
    mock_fw_download,
    mock_load_model,
    tmp_path,
):
    cached = tmp_path / "cached"
    cached.mkdir()
    mock_cached_path.return_value = cached

    path = download_model("small")

    assert path == cached
    mock_fw_download.assert_called_once_with("small", local_files_only=True)
    mock_load_model.assert_called_once()
