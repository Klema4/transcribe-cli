"""Tests for CUDA runtime helpers."""

from pathlib import Path
from unittest.mock import patch

from local_whisper_transcribe.cuda_runtime import (
    CUBLAS_DLL,
    check_cuda_runtime,
    get_cuda_install_hint,
    is_cuda_runtime_installed,
)


def test_get_cuda_install_hint():
    assert get_cuda_install_hint() == "lwt install cuda"


def test_is_cuda_runtime_installed_true_when_cublas_dll_present(tmp_path):
    bin_dir = tmp_path / "cublas" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / CUBLAS_DLL).write_bytes(b"fake")

    with patch(
        "local_whisper_transcribe.cuda_runtime.get_cuda_dll_dirs",
        return_value=[bin_dir],
    ):
        assert is_cuda_runtime_installed() is True


def test_is_cuda_runtime_installed_false_when_missing(tmp_path):
    bin_dir = tmp_path / "empty"
    bin_dir.mkdir()

    with patch(
        "local_whisper_transcribe.cuda_runtime.get_cuda_dll_dirs",
        return_value=[bin_dir],
    ):
        assert is_cuda_runtime_installed() is False


@patch("local_whisper_transcribe.cuda_runtime.is_cuda_runtime_installed", return_value=True)
@patch("local_whisper_transcribe.cuda_runtime.configure_cuda_dll_paths", return_value=["/cuda/bin"])
@patch("ctranslate2.get_cuda_device_count", return_value=1)
def test_check_cuda_runtime_ready(_mock_count, _mock_paths, _mock_runtime):
    ok, detail = check_cuda_runtime()
    assert ok is True
    assert "ready" in detail


@patch("local_whisper_transcribe.cuda_runtime.is_cuda_runtime_installed", return_value=False)
@patch("local_whisper_transcribe.cuda_runtime.configure_cuda_dll_paths", return_value=[])
@patch("local_whisper_transcribe.cuda_runtime._windows_cuda_bin_dirs", return_value=[])
@patch("ctranslate2.get_cuda_device_count", return_value=1)
def test_check_cuda_runtime_missing_libs(_mock_count, _mock_bins, _mock_paths, _mock_runtime):
    ok, detail = check_cuda_runtime()
    assert ok is False
    assert "lwt install cuda" in detail


@patch("local_whisper_transcribe.cuda_runtime.is_cuda_runtime_installed", return_value=False)
@patch("local_whisper_transcribe.cuda_runtime.configure_cuda_dll_paths", return_value=[])
@patch(
    "local_whisper_transcribe.cuda_runtime._windows_cuda_bin_dirs",
    return_value=[Path("C:/CUDA/v12.8/bin")],
)
@patch("ctranslate2.get_cuda_device_count", return_value=1)
def test_check_cuda_runtime_toolkit_path(_mock_count, _mock_bins, _mock_paths, _mock_runtime, tmp_path):
    toolkit_bin = tmp_path / "bin"
    toolkit_bin.mkdir()
    (toolkit_bin / CUBLAS_DLL).write_bytes(b"fake")

    with patch(
        "local_whisper_transcribe.cuda_runtime._windows_cuda_bin_dirs",
        return_value=[toolkit_bin],
    ):
        ok, detail = check_cuda_runtime()
    assert ok is True
    assert "toolkit" in detail.lower()
