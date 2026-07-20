"""Tests for audio utilities."""

from pathlib import Path
from unittest.mock import patch

import pytest

from local_whisper_transcribe.audio import (
    FFmpegNotFoundError,
    UnsupportedMediaError,
    check_ffmpeg,
    detect_media_type,
    is_audio_file,
    is_video_file,
)


def test_is_audio_file():
    assert is_audio_file(Path("recording.wav"))
    assert is_audio_file(Path("song.MP3"))
    assert not is_audio_file(Path("video.mp4"))


def test_is_video_file():
    assert is_video_file(Path("meeting.mp4"))
    assert is_video_file(Path("clip.MKV"))
    assert not is_video_file(Path("audio.flac"))


def test_detect_media_type_audio():
    assert detect_media_type(Path("test.ogg")) == "audio"


def test_detect_media_type_video():
    assert detect_media_type(Path("test.webm")) == "video"


def test_detect_media_type_unsupported():
    with pytest.raises(UnsupportedMediaError):
        detect_media_type(Path("file.xyz"))


def test_check_ffmpeg_found():
    with patch("local_whisper_transcribe.system_checks.find_ffmpeg", return_value="/usr/bin/ffmpeg"):
        assert check_ffmpeg() == "/usr/bin/ffmpeg"


def test_check_ffmpeg_missing():
    with patch("local_whisper_transcribe.system_checks.find_ffmpeg", return_value=None):
        with pytest.raises(FFmpegNotFoundError) as exc_info:
            check_ffmpeg()
        assert "winget install ffmpeg" in str(exc_info.value)
