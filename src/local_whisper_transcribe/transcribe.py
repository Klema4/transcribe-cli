"""faster-whisper transcription wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import platform
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
    """Legacy helper — prefer ``device.resolve_device`` for shared Whisper/diarization."""
    if device != "auto":
        return device
    from local_whisper_transcribe.device import whisper_cuda_available

    return "cuda" if whisper_cuda_available() else "cpu"


def _resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        if device == "cpu" and compute_type in ("float16", "float32"):
            return "int8"
        return compute_type
    return "float16" if device == "cuda" else "int8"


def _resolve_cpu_threads(device: str, cpu_threads: int | None) -> int | None:
    if cpu_threads is not None:
        return cpu_threads
    if device != "cpu":
        return None
    if platform.system() != "Darwin":
        return None

    # Apple Silicon is usually fast for Whisper on CPU with a few dedicated threads.
    # Keep one core for non-Whisper work and avoid over-subscribing tiny systems.
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1
    return min(8, max(1, cpu_count - 1))


def _build_model_kwargs(
    cpu_threads: int | None,
    num_workers: int | None,
) -> dict[str, int]:
    kwargs: dict[str, int] = {}
    if cpu_threads is not None:
        kwargs["cpu_threads"] = cpu_threads
    if num_workers is not None:
        kwargs["num_workers"] = num_workers
    return kwargs


def load_model(
    model: str,
    device: str = "auto",
    compute_type: str = "auto",
    cpu_threads: int | None = None,
    num_workers: int | None = None,
    *,
    on_device_fallback: Callable[[str], None] | None = None,
) -> WhisperModel:
    """Load a faster-whisper model with automatic device/compute selection."""
    configure_cuda_dll_paths()
    resolved_device = _detect_device(device)
    resolved_compute = _resolve_compute_type(compute_type, resolved_device)
    resolved_cpu_threads = _resolve_cpu_threads(resolved_device, cpu_threads)
    model_kwargs = {
        "device": resolved_device,
        "compute_type": resolved_compute,
    }
    model_kwargs.update(_build_model_kwargs(resolved_cpu_threads, num_workers))

    try:
        return WhisperModel(model, **model_kwargs)
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
        return WhisperModel(
            model,
            device="cpu",
            compute_type=cpu_compute,
            **_build_model_kwargs(_resolve_cpu_threads("cpu", cpu_threads), num_workers),
        )


def _transcribe_with_model(
    whisper: WhisperModel,
    audio_path: Path,
    *,
    language: str | None,
    task: str,
    initial_prompt: str | None,
    progress_callback: Callable[..., None] | None,
    beam_size: int = 5,
    condition_on_previous_text: bool = True,
    vad_filter: bool = True,
) -> TranscriptionResult:
    lang = None if language in (None, "", "auto") else language

    segments_iter, info = whisper.transcribe(
        str(audio_path),
        language=lang,
        task=task,
        beam_size=beam_size,
        condition_on_previous_text=condition_on_previous_text,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
    )

    segments: list[Segment] = []
    duration = info.duration or 0.0

    for segment in segments_iter:
        segments.append(
            Segment(start=segment.start, end=segment.end, text=segment.text.strip())
        )
        if progress_callback and duration > 0:
            progress_callback(segment.end, duration, segment.text.strip())

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
    cpu_threads: int | None = None,
    num_workers: int | None = None,
    language: str | None = None,
    task: str = "transcribe",
    initial_prompt: str | None = None,
    beam_size: int = 5,
    condition_on_previous_text: bool = True,
    vad_filter: bool = True,
    progress_callback: Callable[..., None] | None = None,
    on_device_fallback: Callable[[str], None] | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file and return structured segments."""
    whisper = model or load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        on_device_fallback=on_device_fallback,
    )

    try:
        return _transcribe_with_model(
            whisper,
            audio_path,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            vad_filter=vad_filter,
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
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        return _transcribe_with_model(
            whisper_cpu,
            audio_path,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            vad_filter=vad_filter,
            progress_callback=progress_callback,
        )
