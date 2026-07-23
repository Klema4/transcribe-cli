"""CUDA 12 runtime detection, DLL path setup, and installation."""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable

CUBLAS_PACKAGE = "nvidia-cublas-cu12"
CUDNN_SPEC = "nvidia-cudnn-cu12>=9,<10"
CUDA_TOOLKIT_WINGET_ID = "Nvidia.CUDA"
CUBLAS_DLL = "cublas64_12.dll"
# Match torchaudio wheels available for current Python (cu126 torch + cpu torchaudio).
# CUDA torchaudio DLLs are often blocked by Windows Application Control; CPU torchaudio
# is enough because we decode audio via ffmpeg and only need torchaudio for resample/fbank.
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
TORCH_CUDA_PACKAGES = ("torch==2.11.0+cu126",)
TORCHAUDIO_CPU_PACKAGES = ("torchaudio==2.11.0+cpu",)
TORCHAUDIO_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def get_cuda_install_hint() -> str:
    return "lwt install cuda"


def get_cuda_toolkit_install_command() -> str:
    system = platform.system()
    if system == "Windows":
        return f"winget install -e --id {CUDA_TOOLKIT_WINGET_ID}"
    if system == "Darwin":
        return "CUDA toolkit is not supported on macOS for this project"
    return "Install CUDA 12.x from https://developer.nvidia.com/cuda-downloads"


def _nvidia_bin_dir(package: str) -> Path | None:
    """Return nvidia.<package>.bin directory from pip wheels."""
    try:
        spec = importlib.util.find_spec(f"nvidia.{package}.bin")
    except ModuleNotFoundError:
        return None
    if spec and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            path = Path(location)
            if path.is_dir():
                return path
    return None


def _windows_cuda_bin_dirs() -> list[Path]:
    dirs: list[Path] = []
    toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit_root.is_dir():
        for version_dir in sorted(toolkit_root.glob("v12.*"), reverse=True):
            bin_dir = version_dir / "bin"
            if bin_dir.is_dir():
                dirs.append(bin_dir)

    cudnn_root = Path(r"C:\Program Files\NVIDIA\CUDNN")
    if cudnn_root.is_dir():
        for pattern in ("**/bin/12.*", "**/bin"):
            dirs.extend(path for path in cudnn_root.glob(pattern) if path.is_dir())

    return dirs


def get_cuda_dll_dirs() -> list[Path]:
    """Return directories that should contain CUDA/cuDNN DLLs."""
    dirs: list[Path] = []
    for package in ("cublas", "cudnn"):
        bin_dir = _nvidia_bin_dir(package)
        if bin_dir is not None:
            dirs.append(bin_dir)
    if platform.system() == "Windows":
        dirs.extend(_windows_cuda_bin_dirs())
    return dirs


def is_cuda_runtime_installed() -> bool:
    """Return True if CUDA 12 pip runtime packages are present."""
    for directory in get_cuda_dll_dirs():
        if (directory / CUBLAS_DLL).is_file():
            return True
    return False


def configure_cuda_dll_paths() -> list[str]:
    """Register CUDA/cuDNN DLL directories (required on Windows for Python 3.8+)."""
    configured: list[str] = []

    for directory in get_cuda_dll_dirs():
        path = str(directory)
        if path in configured:
            continue
        if platform.system() == "Windows":
            try:
                os.add_dll_directory(path)
            except OSError:
                continue
        configured.append(path)

    if configured and platform.system() == "Windows":
        os.environ["PATH"] = os.pathsep.join(configured) + os.pathsep + os.environ.get("PATH", "")

    return configured


def check_cuda_runtime() -> tuple[bool, str]:
    """Return whether GPU transcription libraries appear usable."""
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
    except Exception:
        return False, "CUDA not available"

    if count <= 0:
        return False, "No CUDA GPU detected"

    configure_cuda_dll_paths()

    whisper_ready = is_cuda_runtime_installed()
    if not whisper_ready:
        toolkit_bins = _windows_cuda_bin_dirs()
        whisper_ready = bool(
            toolkit_bins and any((path / CUBLAS_DLL).is_file() for path in toolkit_bins)
        )

    if not whisper_ready:
        return False, (
            f"GPU detected ({count}), but CUDA 12 libraries are missing "
            f"({CUBLAS_DLL}). Run: {get_cuda_install_hint()}"
        )

    torch_ok = is_torch_cuda_installed()
    if torch_ok:
        return True, f"CUDA ready for Whisper + diarization ({count} GPU)"

    toolkit_bins = _windows_cuda_bin_dirs()
    if toolkit_bins and any((path / CUBLAS_DLL).is_file() for path in toolkit_bins):
        return True, (
            f"CUDA toolkit found for Whisper ({count} GPU); "
            f"PyTorch is CPU-only — run `{get_cuda_install_hint()}` for GPU diarization"
        )

    return True, (
        f"CUDA runtime ready for Whisper ({count} GPU); "
        f"PyTorch is CPU-only — run `{get_cuda_install_hint()}` for GPU diarization"
    )


def _run_pip_install(
    packages: list[str],
    *,
    on_output: Callable[[str], None] | None = None,
    extra_args: list[str] | None = None,
) -> int:
    cmd = [sys.executable, "-m", "pip", "install", *(extra_args or []), *packages]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if on_output:
            on_output(line.rstrip())
    return proc.wait()


def is_torch_cuda_installed() -> bool:
    """Return True if the installed PyTorch build reports CUDA available."""
    configure_cuda_dll_paths()
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def install_torch_cuda(
    *,
    on_output: Callable[[str], None] | None = None,
) -> int:
    """Install CUDA PyTorch for diarization GPU (+ CPU torchaudio for resample/fbank)."""
    code = _run_pip_install(
        list(TORCH_CUDA_PACKAGES),
        on_output=on_output,
        extra_args=["--upgrade", "--force-reinstall", "--index-url", TORCH_CUDA_INDEX],
    )
    if code != 0:
        return code
    if on_output:
        on_output("Installing torchaudio (CPU build for resample/fbank)...")
    return _run_pip_install(
        list(TORCHAUDIO_CPU_PACKAGES),
        on_output=on_output,
        extra_args=[
            "--upgrade",
            "--force-reinstall",
            "--index-url",
            TORCHAUDIO_CPU_INDEX,
        ],
    )


def install_cuda_runtime(
    *,
    on_output: Callable[[str], None] | None = None,
    include_torch: bool = True,
) -> int:
    """Install CUDA 12 runtime for Whisper and (optionally) CUDA PyTorch for diarization."""
    code = _run_pip_install([CUBLAS_PACKAGE, CUDNN_SPEC], on_output=on_output)
    if code != 0:
        return code
    if include_torch:
        if on_output:
            on_output("Installing CUDA PyTorch (needed for GPU diarization)...")
        code = install_torch_cuda(on_output=on_output)
    return code


def install_cuda_toolkit(
    *,
    on_output: Callable[[str], None] | None = None,
) -> int:
    """Install the full NVIDIA CUDA Toolkit via winget (Windows only)."""
    if platform.system() != "Windows":
        if on_output:
            on_output("Full CUDA toolkit install is only automated on Windows.")
        return 1

    cmd = [
        "winget",
        "install",
        "-e",
        "--id",
        CUDA_TOOLKIT_WINGET_ID,
        "--accept-package-agreements",
        "--accept-source-agreements",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if on_output:
            on_output(line.rstrip())
    return proc.wait()
