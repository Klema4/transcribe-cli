"""Whisper model metadata, download, and cache management."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from faster_whisper import download_model as fw_download_model
from faster_whisper.utils import _MODELS
from huggingface_hub import constants, snapshot_download, try_to_load_from_cache

from local_whisper_transcribe.transcribe import KNOWN_MODELS, load_model
from local_whisper_transcribe.device import MLX_MODEL_REPOS, should_use_mlx

# size, RAM, notes
MODEL_INFO: dict[str, tuple[str, str, str]] = {
    "tiny": ("~75 MB", "~1 GB", "Fastest, lowest accuracy"),
    "base": ("~150 MB", "~1 GB", "Fast, basic quality"),
    "small": ("~500 MB", "~2 GB", "Good balance (recommended)"),
    "medium": ("~1.5 GB", "~4 GB", "High quality, slower"),
    "large-v3": ("~3 GB", "~10 GB", "Best quality, needs GPU"),
}

_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]
_MLX_ALLOW_PATTERNS = [
    "config.json",
    "weights.npz",
    "weights.safetensors",
    "model.safetensors",
    "tokenizer.json",
    "*.tiktoken",
]


@dataclass
class CachedModelStatus:
    name: str
    cached: bool
    path: Path | None
    size_bytes: int | None
    repo_id: str


def get_repo_id(model_name: str, device: str = "auto") -> str:
    """Return the Hugging Face repo ID for a model name or path."""
    if "/" in model_name:
        return model_name
    if should_use_mlx(device, model_name):
        return MLX_MODEL_REPOS[model_name]
    repo_id = _MODELS.get(model_name)
    if repo_id is None:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Expected one of: {', '.join(KNOWN_MODELS)}"
        )
    return repo_id


def get_cache_dir() -> Path:
    """Return the Hugging Face Hub cache directory."""
    return Path(constants.HF_HUB_CACHE)


def _local_model_path(model_name: str) -> Path | None:
    path = Path(model_name)
    if path.exists():
        return path.resolve()
    return None


def _cached_snapshot_dir(model_name: str, device: str = "auto") -> Path | None:
    local_path = _local_model_path(model_name)
    if local_path is not None:
        return local_path

    try:
        repo_id = get_repo_id(model_name, device)
    except ValueError:
        return None

    filenames = _MLX_ALLOW_PATTERNS if should_use_mlx(device, model_name) else ["model.bin"]
    for filename in filenames:
        cached_file = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        if cached_file is not None:
            return Path(cached_file).parent
    return None


def is_model_cached(model_name: str, device: str = "auto") -> bool:
    """Return True if the model is available locally."""
    return _cached_snapshot_dir(model_name, device) is not None


def get_cached_model_path(model_name: str, device: str = "auto") -> Path | None:
    """Return the local path to a cached model, if present."""
    return _cached_snapshot_dir(model_name, device)


def _directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def format_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "—"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_model_status(model_name: str) -> CachedModelStatus:
    """Return cache status for a single model."""
    path = get_cached_model_path(model_name)
    try:
        repo_id = get_repo_id(model_name)
    except ValueError:
        repo_id = model_name
    size = _directory_size(path) if path else None
    return CachedModelStatus(
        name=model_name,
        cached=path is not None,
        path=path,
        size_bytes=size,
        repo_id=repo_id,
    )


def list_model_statuses() -> list[CachedModelStatus]:
    """Return cache status for all known models."""
    return [get_model_status(name) for name in KNOWN_MODELS]


def check_python_version() -> tuple[bool, str]:
    version = sys.version_info
    ok = version >= (3, 10)
    detail = f"Python {version.major}.{version.minor}.{version.micro}"
    if not ok:
        detail += " (requires >= 3.10)"
    return ok, detail


def check_cuda_available() -> tuple[bool, str]:
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        if count > 0:
            return True, f"CUDA available ({count} device(s))"
        return False, "No CUDA devices found (CPU will be used)"
    except Exception:
        return False, "CUDA not available (CPU will be used)"


def download_model(
    model_name: str,
    device: str = "auto",
    compute_type: str = "auto",
    *,
    force: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> Path:
    """Download and verify a Whisper model, returning its local path."""
    local_path = _local_model_path(model_name)
    if local_path is not None:
        if status_callback:
            status_callback(f"Using local model at {local_path}")
        load_model(str(local_path), device=device, compute_type=compute_type)
        return local_path

    use_mlx = should_use_mlx(device, model_name)
    repo_id = get_repo_id(model_name, device)
    already_cached = is_model_cached(model_name, device)

    if force and already_cached:
        snapshot_dir = _cached_snapshot_dir(model_name, device)
        if snapshot_dir is not None:
            hub_dir_name = "models--" + repo_id.replace("/", "--")
            cache_root = get_cache_dir()
            model_cache = cache_root / hub_dir_name
            if model_cache.exists():
                shutil.rmtree(model_cache)

    if status_callback:
        if already_cached and not force:
            status_callback(f"Model '{model_name}' already cached, verifying...")
        else:
            status_callback(
                f"Downloading '{model_name}' from Hugging Face ({repo_id})..."
            )
            status_callback(f"Cache directory: {get_cache_dir()}")

    if force or not already_cached:
        snapshot_download(
            repo_id,
            # MLX repositories can contain tokenizer and mel-filter files whose
            # names vary between model revisions, so keep the whole small repo.
            allow_patterns=None if use_mlx else _ALLOW_PATTERNS,
            force_download=force,
        )
    else:
        if not use_mlx:
            fw_download_model(model_name, local_files_only=True)

    if status_callback:
        status_callback("Loading model to verify download...")

    load_model(model_name, device=device, compute_type=compute_type)
    path = get_cached_model_path(model_name, device)
    if path is None:
        raise RuntimeError(f"Model '{model_name}' download finished but path not found.")
    return path
