"""Tests for shared device resolution."""

from unittest.mock import patch

from local_whisper_transcribe.device import resolve_device


@patch("local_whisper_transcribe.device.torch_cuda_available", return_value=False)
@patch("local_whisper_transcribe.device.whisper_cuda_available", return_value=True)
def test_auto_without_diarization_uses_whisper_gpu(_w, _t):
    plan = resolve_device("auto", need_torch=False)
    assert plan.device == "cuda"


@patch("local_whisper_transcribe.device.torch_cuda_available", return_value=False)
@patch("local_whisper_transcribe.device.whisper_cuda_available", return_value=True)
def test_auto_with_diarization_forces_cpu_when_torch_cpu_only(_w, _t):
    notes: list[str] = []
    plan = resolve_device("auto", need_torch=True, on_note=notes.append)
    assert plan.device == "cpu"
    assert notes
    assert "CPU for both" in notes[0]


@patch("local_whisper_transcribe.device.torch_cuda_available", return_value=True)
@patch("local_whisper_transcribe.device.whisper_cuda_available", return_value=True)
def test_auto_with_diarization_uses_gpu_when_both_ready(_w, _t):
    plan = resolve_device("auto", need_torch=True)
    assert plan.device == "cuda"


@patch("local_whisper_transcribe.device.torch_cuda_available", return_value=True)
@patch("local_whisper_transcribe.device.whisper_cuda_available", return_value=True)
def test_explicit_cpu_wins(_w, _t):
    plan = resolve_device("cpu", need_torch=True)
    assert plan.device == "cpu"
