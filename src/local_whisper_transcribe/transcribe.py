"""faster-whisper transcription wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Configure CUDA DLL search paths before ctranslate2 loads GPU libraries.
from local_whisper_transcribe.cuda_runtime import configure_cuda_dll_paths

configure_cuda_dll_paths()

from faster_whisper import WhisperModel

KNOWN_MODELS = ("tiny", "base", "small", "medium", "large-v3")

_CUDA_ERROR_MARKERS = (
    "cublas",
    "cudnn",
    "cudart",
    "cuda",
    "nvidia",
    "could not load",
    "cannot be loaded",
)


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


def _is_cuda_runtime_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _CUDA_ERROR_MARKERS)


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
        if device == "cpu" and compute_type in ("float16", "float32"):
            return "int8"
        return compute_type
    return "float16" if device == "cuda" else "int8"


def load_model(
    model: str,
    device: str = "auto",
    compute_type: str = "auto",
    *,
    on_device_fallback: Callable[[str], None] | None = None,
) -> WhisperModel:
    """Load a faster-whisper model with automatic device/compute selection."""
    configure_cuda_dll_paths()
    resolved_device = _detect_device(device)
    resolved_compute = _resolve_compute_type(compute_type, resolved_device)

    try:
        return WhisperModel(model, device=resolved_device, compute_type=resolved_compute)
    except Exception as exc:
        if resolved_device != "cuda" or not _is_cuda_runtime_error(exc):
            raise

        message = (
            "CUDA is available but required libraries are missing "
            f"({exc}). Falling back to CPU. "
            "Install GPU libraries with: lwt install cuda"
        )
        if on_device_fallback:
            on_device_fallback(message)

        cpu_compute = _resolve_compute_type(compute_type, "cpu")
        return WhisperModel(model, device="cpu", compute_type=cpu_compute)


def _transcribe_with_model(
    whisper: WhisperModel,
    audio_path: Path,
    *,
    language: str | None,
    task: str,
    initial_prompt: str | None,
    progress_callback: Callable[[float, float], None] | None,
) -> TranscriptionResult:
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
    on_device_fallback: Callable[[str], None] | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file and return structured segments."""
    whisper = model or load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        on_device_fallback=on_device_fallback,
    )

    try:
        return _transcribe_with_model(
            whisper,
            audio_path,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        if not _is_cuda_runtime_error(exc) or device not in ("auto", "cuda"):
            raise

        message = (
            "CUDA failed during transcription "
            f"({exc}). Retrying on CPU. "
            "Install GPU libraries with: lwt install cuda"
        )
        if on_device_fallback:
            on_device_fallback(message)

        whisper_cpu = load_model(
            model_name,
            device="cpu",
            compute_type=compute_type,
        )
        return _transcribe_with_model(
            whisper_cpu,
            audio_path,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            progress_callback=progress_callback,
        )
