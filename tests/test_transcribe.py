"""Tests for transcription device fallback helpers."""

from local_whisper_transcribe.transcribe import _is_cuda_runtime_error, _resolve_compute_type


def test_is_cuda_runtime_error_matches_cublas():
    assert _is_cuda_runtime_error(RuntimeError("Library cublas64_12.dll is not found"))


def test_is_cuda_runtime_error_rejects_unrelated():
    assert not _is_cuda_runtime_error(ValueError("invalid language"))


def test_resolve_compute_type_cpu_fallback():
    assert _resolve_compute_type("float16", "cpu") == "int8"
