"""Speaker diarization via pyannote.audio (optional dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from local_whisper_transcribe.transcribe import Segment

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
HF_MODEL_URL = "https://huggingface.co/pyannote/speaker-diarization-3.1"


class DiarizationNotInstalledError(RuntimeError):
    """Raised when pyannote.audio is not installed."""


class DiarizationTokenError(RuntimeError):
    """Raised when a HuggingFace token is required but missing."""


def _require_pipeline():
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationNotInstalledError(
            "pyannote.audio is not installed.\n"
            "Install the optional diarization dependencies with:\n"
            "  lwt install diarization\n"
            f"Then accept the model license at {HF_MODEL_URL}"
        ) from exc
    return Pipeline


def _segment_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def diarize_audio(
    audio_path: Path,
    hf_token: str | None = None,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run speaker diarization and return segments with start, end, and speaker labels."""
    if not hf_token:
        raise DiarizationTokenError(
            "A HuggingFace token is required for speaker diarization.\n"
            f"1. Accept the model license at {HF_MODEL_URL}\n"
            "2. Create a token at https://huggingface.co/settings/tokens\n"
            "3. Set it via --hf-token, HF_TOKEN env var, or:\n"
            "     lwt config set diarization.hf_token <token>"
        )

    Pipeline = _require_pipeline()

    if progress_callback:
        progress_callback("Loading diarization model...")

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)

    kwargs: dict[str, int] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    if progress_callback:
        progress_callback("Running speaker diarization...")

    diarization = pipeline(str(audio_path), **kwargs)

    segments: list[dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )

    if progress_callback:
        progress_callback("Diarization complete.")

    return segments


def merge_transcription_with_diarization(
    whisper_segments: list[Segment],
    diarization_segments: list[dict[str, Any]],
) -> list[Segment]:
    """Assign each Whisper segment to the diarization speaker with maximum overlap."""
    merged: list[Segment] = []

    for whisper_seg in whisper_segments:
        best_speaker: str | None = None
        best_overlap = 0.0

        for dia_seg in diarization_segments:
            overlap = _segment_overlap(
                whisper_seg.start,
                whisper_seg.end,
                dia_seg["start"],
                dia_seg["end"],
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dia_seg["speaker"]

        merged.append(
            Segment(
                start=whisper_seg.start,
                end=whisper_seg.end,
                text=whisper_seg.text,
                speaker=best_speaker,
            )
        )

    return merged


def apply_speaker_names(segments: list[Segment], speaker_names: list[str]) -> list[Segment]:
    """Rename SPEAKER_00, SPEAKER_01, ... to user-provided display names."""
    if not speaker_names:
        return segments

    mapping = {f"SPEAKER_{index:02d}": name for index, name in enumerate(speaker_names)}
    renamed: list[Segment] = []
    for seg in segments:
        speaker = seg.speaker
        if speaker and speaker in mapping:
            speaker = mapping[speaker]
        renamed.append(
            Segment(start=seg.start, end=seg.end, text=seg.text, speaker=speaker)
        )
    return renamed
