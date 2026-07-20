"""System dependency checks for setup and diagnostics."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path


def check_python() -> tuple[bool, str]:
    """Return (ok, version_string)."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return sys.version_info >= (3, 10), version


def get_ffmpeg_install_command() -> str:
    """Platform-specific ffmpeg install command."""
    system = platform.system()
    if system == "Windows":
        return "winget install ffmpeg"
    if system == "Darwin":
        return "brew install ffmpeg"
    return "sudo apt install ffmpeg"


def _windows_ffmpeg_candidates() -> list[Path]:
    """Common ffmpeg locations on Windows (winget, manual installs)."""
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    candidates: list[Path] = []

    if local_app_data:
        candidates.append(Path(local_app_data) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe")
        packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if packages.is_dir():
            candidates.extend(packages.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"))
            candidates.extend(packages.glob("**/ffmpeg.exe"))

    for fixed in (
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
    ):
        candidates.append(fixed)

    return candidates


def find_ffmpeg() -> str | None:
    """Return ffmpeg executable path if found on PATH or common install locations."""
    path = shutil.which("ffmpeg")
    if path:
        return path

    if platform.system() == "Windows":
        for candidate in _windows_ffmpeg_candidates():
            if candidate.is_file():
                return str(candidate)

    return None


def check_cuda() -> tuple[bool, int]:
    """Return (cuda_available, device_count)."""
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        return count > 0, count
    except Exception:
        return False, 0


def check_ffmpeg_available() -> tuple[bool, str | None]:
    """Return (found, path_or_none)."""
    path = find_ffmpeg()
    return path is not None, path
