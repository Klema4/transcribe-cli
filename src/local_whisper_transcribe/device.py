"""Shared CPU/CUDA device selection for Whisper and diarization."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import platform
from typing import Callable

from local_whisper_transcribe.cuda_runtime import configure_cuda_dll_paths


@dataclass(frozen=True)
class DevicePlan:
    """Resolved compute device shared by Whisper and (optionally) diarization."""

    device: str  # "cuda" | "mlx" | "cpu"
    whisper_cuda: bool
    torch_cuda: bool
    note: str | None = None


MLX_MODEL_REPOS: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


def is_apple_silicon() -> bool:
    """Return True when Python is running natively on an Apple M-series Mac."""
    return (
        platform.system() == "Darwin"
        and platform.machine().lower() in {"arm64", "arm64e", "aarch64"}
    )


def mlx_whisper_available() -> bool:
    """Return True when the optional Apple GPU backend can be imported."""
    return is_apple_silicon() and importlib.util.find_spec("mlx_whisper") is not None


def should_use_mlx(preferred: str = "auto", model: str | None = None) -> bool:
    """Return whether model management should use an MLX checkpoint."""
    preferred = (preferred or "auto").lower()
    if preferred == "mlx":
        return is_apple_silicon()
    if preferred != "auto" or not mlx_whisper_available():
        return False
    return model is None or model in MLX_MODEL_REPOS


def whisper_cuda_available() -> bool:
    """Return True if faster-whisper / CTranslate2 can see a CUDA GPU."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def torch_cuda_available() -> bool:
    """Return True if PyTorch can use CUDA (requires CUDA torch build)."""
    configure_cuda_dll_paths()
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(
    preferred: str = "auto",
    *,
    need_torch: bool = False,
    model: str | None = None,
    on_note: Callable[[str], None] | None = None,
) -> DevicePlan:
    """Resolve a single device for the whole job.

    When ``need_torch`` is True (diarization), CUDA is used only if *both*
    CTranslate2 and PyTorch see CUDA — never Whisper-on-GPU + diarization-on-CPU.
    """
    preferred = (preferred or "auto").lower()
    if preferred not in ("auto", "cuda", "mlx", "cpu"):
        preferred = "auto"

    whisper_ok = whisper_cuda_available()
    torch_ok = torch_cuda_available()

    if preferred == "cpu":
        return DevicePlan("cpu", whisper_ok, torch_ok, None)

    if preferred == "mlx":
        if need_torch:
            note = (
                "Apple GPU is available for Whisper, but diarization uses PyTorch. "
                "Using CPU for both because MLX and pyannote cannot share one device."
            )
        elif not is_apple_silicon():
            note = "Requested MLX, but this is not an Apple Silicon Mac. Using CPU."
        elif not mlx_whisper_available():
            note = (
                "Requested MLX, but mlx-whisper is not installed. "
                "Reinstall Transcribe CLI on Apple Silicon. Using CPU."
            )
        else:
            return DevicePlan("mlx", whisper_ok, torch_ok, None)
        if on_note:
            on_note(note)
        return DevicePlan("cpu", whisper_ok, torch_ok, note)

    if preferred == "cuda":
        if need_torch:
            if whisper_ok and torch_ok:
                return DevicePlan("cuda", whisper_ok, torch_ok, None)
            note = (
                "Requested CUDA, but Whisper and diarization need the same device. "
                f"Whisper GPU={'yes' if whisper_ok else 'no'}, "
                f"PyTorch GPU={'yes' if torch_ok else 'no'}. "
                "Falling back to CPU for both. "
                "Fix with: lwt install cuda"
            )
            if on_note:
                on_note(note)
            return DevicePlan("cpu", whisper_ok, torch_ok, note)
        if whisper_ok:
            return DevicePlan("cuda", whisper_ok, torch_ok, None)
        note = "Requested CUDA but no GPU for Whisper. Using CPU."
        if on_note:
            on_note(note)
        return DevicePlan("cpu", whisper_ok, torch_ok, note)

    # auto
    if need_torch:
        if whisper_ok and torch_ok:
            return DevicePlan("cuda", whisper_ok, torch_ok, None)
        if whisper_ok and not torch_ok:
            note = (
                "GPU is available for Whisper, but PyTorch is CPU-only "
                "(diarization needs CUDA PyTorch). "
                "Using CPU for both so they stay in sync. "
                "Install GPU stack with: lwt install cuda"
            )
            if on_note:
                on_note(note)
            return DevicePlan("cpu", whisper_ok, torch_ok, note)
        return DevicePlan("cpu", whisper_ok, torch_ok, None)

    if should_use_mlx("auto", model):
        return DevicePlan("mlx", whisper_ok, torch_ok, None)

    if whisper_ok:
        return DevicePlan("cuda", whisper_ok, torch_ok, None)
    return DevicePlan("cpu", whisper_ok, torch_ok, None)
