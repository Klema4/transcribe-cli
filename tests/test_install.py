"""Tests for optional dependency checks."""

from unittest.mock import patch

from local_whisper_transcribe.install_extra import check_all_dependencies, is_diarization_installed


def test_is_diarization_installed_false():
    with patch("local_whisper_transcribe.install_extra.importlib.util.find_spec", return_value=None):
        assert is_diarization_installed() is False


def test_check_all_dependencies_includes_ffmpeg():
    with (
        patch(
            "local_whisper_transcribe.install_extra._check_ffmpeg",
            return_value=(True, "/usr/bin/ffmpeg"),
        ),
        patch(
            "local_whisper_transcribe.cuda_runtime.check_cuda_runtime",
            return_value=(False, "missing"),
        ),
        patch(
            "local_whisper_transcribe.install_extra.is_diarization_installed",
            return_value=False,
        ),
    ):
        deps = check_all_dependencies()
    names = [d.name for d in deps]
    assert "ffmpeg" in names
    assert "cuda 12 runtime (gpu)" in names
    assert "pyannote.audio (diarization)" in names
