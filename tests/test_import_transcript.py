"""Tests for loading existing transcript files."""

import json
from pathlib import Path

from local_whisper_transcribe.import_transcript import (
    default_clean_output_path,
    default_raw_backup_path,
    load_transcript,
    parse_json_transcript,
    parse_srt_transcript,
    parse_txt_transcript,
    parse_vtt_transcript,
)


def test_parse_txt_plain_lines():
    result = parse_txt_transcript("Hello world.\nSecond line.\n")
    assert len(result.segments) == 2
    assert result.segments[0].text == "Hello world."
    assert result.segments[1].text == "Second line."


def test_parse_txt_with_timestamp_range():
    content = "[00:00:00.000 -> 00:00:02.500] Hello world.\n"
    result = parse_txt_transcript(content)
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 2.5
    assert result.segments[0].text == "Hello world."


def test_parse_txt_with_speaker_timestamp():
    content = "[00:00:01.000] Jan: Hello there.\n"
    result = parse_txt_transcript(content)
    assert result.segments[0].speaker == "Jan"
    assert result.segments[0].text == "Hello there."


def test_parse_srt_with_speaker():
    content = (
        "1\n"
        "00:00:00,000 --> 00:00:02,500\n"
        "[Jan] Hello world.\n\n"
        "2\n"
        "00:00:02,500 --> 00:00:05,000\n"
        "Second line.\n"
    )
    result = parse_srt_transcript(content)
    assert len(result.segments) == 2
    assert result.segments[0].speaker == "Jan"
    assert result.segments[0].text == "Hello world."


def test_parse_vtt():
    content = "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello world.\n"
    result = parse_vtt_transcript(content)
    assert result.segments[0].text == "Hello world."


def test_parse_json_transcript():
    payload = {
        "language": "cs",
        "duration": 5.0,
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Ahoj", "speaker": "Jan"},
        ],
    }
    result = parse_json_transcript(json.dumps(payload))
    assert result.language == "cs"
    assert result.segments[0].speaker == "Jan"


def test_load_transcript_from_file(tmp_path: Path):
    path = tmp_path / "meeting.txt"
    path.write_text("First line.\nSecond line.\n", encoding="utf-8")
    result = load_transcript(path)
    assert len(result.segments) == 2


def test_default_clean_output_path_from_raw():
    assert default_clean_output_path(Path("meeting.raw.txt"), None) == Path("meeting.txt")


def test_default_raw_backup_path():
    assert default_raw_backup_path(Path("meeting.txt")) == Path("meeting.raw.txt")
