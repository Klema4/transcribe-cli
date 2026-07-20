"""Tests for output formatting."""

from local_whisper_transcribe.output import format_json, format_srt, format_txt, format_vtt
from local_whisper_transcribe.transcribe import Segment, TranscriptionResult


def _sample_segments() -> list[Segment]:
    return [
        Segment(start=0.0, end=2.5, text="Hello world."),
        Segment(start=2.5, end=5.0, text="Second line."),
    ]


def test_format_txt_plain():
    text = format_txt(_sample_segments())
    assert text == "Hello world.\nSecond line."


def test_format_txt_with_timestamps():
    text = format_txt(_sample_segments(), with_timestamps=True)
    assert "[00:00:00.000 -> 00:00:02.500] Hello world." in text
    assert "[00:00:02.500 -> 00:00:05.000] Second line." in text


def test_format_srt():
    srt = format_srt(_sample_segments())
    assert "1\n" in srt
    assert "00:00:00,000 --> 00:00:02,500" in srt
    assert "Hello world." in srt
    assert "2\n" in srt
    assert "00:00:02,500 --> 00:00:05,000" in srt


def test_format_vtt():
    vtt = format_vtt(_sample_segments())
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in vtt
    assert "Hello world." in vtt


def test_format_json():
    result = TranscriptionResult(
        segments=_sample_segments(),
        language="en",
        duration=5.0,
        metadata={"duration": 5.0},
    )
    payload = format_json(result, source="test.wav")
    assert '"language": "en"' in payload
    assert '"source": "test.wav"' in payload
    assert '"start": 0.0' in payload
    assert '"speaker": null' in payload
    assert "Hello world." in payload


def test_format_txt_with_speakers_groups_consecutive_segments():
    segments = [
        Segment(start=0.0, end=1.0, text="Hello.", speaker="Jan"),
        Segment(start=1.0, end=2.0, text="How are you?", speaker="Jan"),
        Segment(start=2.0, end=3.0, text="Fine.", speaker="Petra"),
    ]
    text = format_txt(segments, with_timestamps=True, with_speakers=True)
    assert text == "[00:00:00.000] Jan: Hello. How are you?\n[00:00:02.000] Petra: Fine."


def test_format_srt_with_speaker_prefix():
    segments = [Segment(start=0.0, end=2.5, text="Hello world.", speaker="Jan")]
    srt = format_srt(segments)
    assert "[Jan] Hello world." in srt


def test_format_vtt_with_speaker_prefix():
    segments = [Segment(start=0.0, end=2.5, text="Hello world.", speaker="Jan")]
    vtt = format_vtt(segments)
    assert "[Jan] Hello world." in vtt
