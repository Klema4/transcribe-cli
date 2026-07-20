"""Audio/video file handling and ffmpeg extraction."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv", ".m4v"}


class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg is not available on PATH."""


class UnsupportedMediaError(ValueError):
    """Raised when the file extension is not a supported audio/video format."""


def check_ffmpeg() -> str:
    """Return the ffmpeg executable path or raise with install instructions."""
    from local_whisper_transcribe.system_checks import find_ffmpeg, get_ffmpeg_install_command

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        install_cmd = get_ffmpeg_install_command()
        raise FFmpegNotFoundError(
            "ffmpeg is not installed or not found.\n"
            f"Install with: {install_cmd}\n"
            "If already installed via winget, restart your terminal so PATH updates.\n"
            "Or run: lwt setup"
        )
    return ffmpeg


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def detect_media_type(path: Path) -> str:
    """Return 'audio' or 'video' based on file extension."""
    suffix = path.suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise UnsupportedMediaError(
        f"Unsupported file extension '{suffix}'. "
        f"Supported audio: {', '.join(sorted(AUDIO_EXTENSIONS))}. "
        f"Supported video: {', '.join(sorted(VIDEO_EXTENSIONS))}."
    )


def extract_audio_to_wav(input_path: Path, output_path: Path) -> None:
    """Extract audio from a video file to 16 kHz mono PCM WAV."""
    ffmpeg = check_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract audio from {input_path}:\n{result.stderr}"
        )


def load_audio_waveform(audio_path: Path, sample_rate: int = 16000) -> dict:
    """Load audio as a pyannote-compatible in-memory waveform via ffmpeg."""
    import numpy as np
    import torch

    ffmpeg = check_ffmpeg()
    cmd = [
        ffmpeg,
        "-nostdin",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed to decode audio for diarization:\n{stderr}")

    if not result.stdout:
        raise RuntimeError(f"No audio data decoded from {audio_path}")

    waveform = np.frombuffer(result.stdout, dtype=np.float32).copy()
    return {
        "waveform": torch.from_numpy(waveform).unsqueeze(0),
        "sample_rate": sample_rate,
    }


@contextmanager
def prepare_audio(path: Path) -> Generator[Path, None, None]:
    """Yield a WAV path ready for transcription, extracting video audio if needed."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    media_type = detect_media_type(path)
    if media_type == "audio":
        yield path
        return

    check_ffmpeg()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        extract_audio_to_wav(path, tmp_path)
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)
