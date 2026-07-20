"""faster-whisper transcription wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from faster_whisper import WhisperModel

KNOWN_MODELS = ("tiny", "base", "small", "medium", "large-v3")


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    language: str
    duration: float
    metadata: dict = field(default_factory=dict)


def _detect_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def load_model(
    model: str,
    device: str = "auto",
    compute_type: str = "auto",
) -> WhisperModel:
    """Load a faster-whisper model with automatic device/compute selection."""
    resolved_device = _detect_device(device)
    resolved_compute = _resolve_compute_type(compute_type, resolved_device)
    return WhisperModel(model, device=resolved_device, compute_type=resolved_compute)


def transcribe(
    audio_path: Path,
    model: WhisperModel | None = None,
    *,
    model_name: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
    task: str = "transcribe",
    initial_prompt: str | None = None,
    progress_callback: Callable[[float, float], None] | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file and return structured segments."""
    whisper = model or load_model(model_name, device=device, compute_type=compute_type)

    lang = None if language in (None, "", "auto") else language

    segments_iter, info = whisper.transcribe(
        str(audio_path),
        language=lang,
        task=task,
        vad_filter=True,
        initial_prompt=initial_prompt,
    )

    segments: list[Segment] = []
    duration = info.duration or 0.0

    for segment in segments_iter:
        segments.append(
            Segment(start=segment.start, end=segment.end, text=segment.text.strip())
        )
        if progress_callback and duration > 0:
            progress_callback(segment.end, duration)

    if progress_callback and duration > 0:
        progress_callback(duration, duration)

    return TranscriptionResult(
        segments=segments,
        language=info.language or (language or "unknown"),
        duration=duration,
        metadata={
            "language_probability": getattr(info, "language_probability", None),
            "duration": duration,
        },
    )
