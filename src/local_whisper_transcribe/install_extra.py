"""Optional dependency installation helpers."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

DIARIZATION_PACKAGE = "pyannote.audio"
DIARIZATION_SPEC = "pyannote.audio>=3.1"

@dataclass
class DepStatus:
    name: str
    installed: bool
    optional: bool
    install_hint: str


def is_diarization_installed() -> bool:
    try:
        return importlib.util.find_spec("pyannote.audio") is not None
    except ModuleNotFoundError:
        return False


def install_diarization(
    *,
    on_output: Callable[[str], None] | None = None,
) -> int:
    """Install pyannote.audio via pip. Returns process exit code."""
    cmd = [sys.executable, "-m", "pip", "install", DIARIZATION_SPEC]
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


def check_all_dependencies() -> list[DepStatus]:
    """Verify core and optional dependencies."""
    from local_whisper_transcribe.cuda_runtime import check_cuda_runtime

    ffmpeg_ok, _ = _check_ffmpeg()
    cuda_ok, _cuda_detail = check_cuda_runtime()
    return [
        DepStatus("python", True, optional=False, install_hint=""),
        DepStatus(
            "ffmpeg",
            ffmpeg_ok,
            optional=False,
            install_hint="lwt setup",
        ),
        DepStatus(
            "cuda 12 runtime (gpu)",
            cuda_ok,
            optional=True,
            install_hint="lwt install cuda",
        ),
        DepStatus(
            "pyannote.audio (diarization)",
            is_diarization_installed(),
            optional=True,
            install_hint="lwt install diarization",
        ),
    ]


def _check_ffmpeg() -> tuple[bool, str | None]:
    from local_whisper_transcribe.system_checks import check_ffmpeg_available

    return check_ffmpeg_available()
