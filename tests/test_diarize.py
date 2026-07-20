"""Tests for speaker diarization helpers."""

import pytest

from local_whisper_transcribe.diarize import (
    DiarizationNotInstalledError,
    apply_speaker_names,
    diarize_audio,
    merge_transcription_with_diarization,
)
from local_whisper_transcribe.transcribe import Segment


def test_merge_transcription_with_diarization_assigns_best_overlap():
    whisper_segments = [
        Segment(start=0.0, end=2.0, text="Hello."),
        Segment(start=2.0, end=4.0, text="Hi there."),
        Segment(start=5.0, end=6.0, text="Thanks."),
    ]
    diarization_segments = [
        {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
        {"start": 2.5, "end": 5.0, "speaker": "SPEAKER_01"},
        {"start": 5.0, "end": 6.0, "speaker": "SPEAKER_00"},
    ]

    merged = merge_transcription_with_diarization(whisper_segments, diarization_segments)

    assert [seg.speaker for seg in merged] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_merge_transcription_with_diarization_handles_no_overlap():
    whisper_segments = [Segment(start=10.0, end=12.0, text="Late line.")]
    diarization_segments = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]

    merged = merge_transcription_with_diarization(whisper_segments, diarization_segments)

    assert merged[0].speaker is None


def test_apply_speaker_names():
    segments = [
        Segment(start=0.0, end=1.0, text="A", speaker="SPEAKER_00"),
        Segment(start=1.0, end=2.0, text="B", speaker="SPEAKER_01"),
    ]

    renamed = apply_speaker_names(segments, ["Jan", "Petra"])

    assert renamed[0].speaker == "Jan"
    assert renamed[1].speaker == "Petra"


def test_diarize_audio_requires_pyannote(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyannote.audio":
            raise ImportError("no pyannote")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(DiarizationNotInstalledError) as exc_info:
        diarize_audio("audio.wav", hf_token="test-token")

    assert "lwt install diarization" in str(exc_info.value)
