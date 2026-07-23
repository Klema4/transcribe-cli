"""Tests for progress UI helpers."""

from local_whisper_transcribe.progress_ui import format_timestamp, truncate_preview


def test_truncate_preview_short_text():
    assert truncate_preview("hello world") == "hello world"


def test_truncate_preview_collapses_whitespace():
    assert truncate_preview("hello   world") == "hello world"


def test_truncate_preview_long_text():
    text = "a" * 150
    result = truncate_preview(text)
    assert len(result) == 120
    assert result.endswith("...")


def test_format_timestamp():
    assert format_timestamp(125.0) == "02:05"
