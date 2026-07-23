"""Tests for Ollama post-processing helpers."""

from unittest.mock import patch

from local_whisper_transcribe.postprocess import (
    _build_clean_prompt,
    clean_transcript_segments,
    parse_cleaned_lines,
)
from local_whisper_transcribe.transcribe import Segment


def test_parse_cleaned_lines():
    response = "[0] Hello world.\n[1] Second line.\n"
    assert parse_cleaned_lines(response) == {0: "Hello world.", 1: "Second line."}


def test_parse_cleaned_lines_ignores_blank_lines():
    response = "\n[2] Fixed text.\n\n"
    assert parse_cleaned_lines(response) == {2: "Fixed text."}


def test_build_clean_prompt_includes_language():
    prompt = _build_clean_prompt([(0, "hm jo jo test")], "cs")
    assert "language is cs" in prompt
    assert "[0] hm jo jo test" in prompt
    assert "Do NOT summarize" in prompt


def test_clean_transcript_segments_updates_text():
    segments = [
        Segment(start=0.0, end=1.0, text="hm jo jo červený"),
        Segment(start=1.0, end=2.0, text="eh test"),
    ]

    def fake_batch(batch, *, language, model, url):
        return {idx: f"cleaned {idx}" for idx, _ in batch}

    with patch(
        "local_whisper_transcribe.postprocess._clean_segment_batch",
        side_effect=fake_batch,
    ):
        cleaned = clean_transcript_segments(
            segments,
            language="cs",
            model="llama3.2",
            url="http://localhost:11434",
            batch_size=10,
        )

    assert cleaned[0].text == "cleaned 0"
    assert cleaned[1].text == "cleaned 1"
    assert cleaned[0].start == 0.0
    assert cleaned[1].speaker is None


def test_clean_transcript_segments_keeps_original_on_missing_line():
    segments = [Segment(start=0.0, end=1.0, text="original")]

    with patch(
        "local_whisper_transcribe.postprocess._clean_segment_batch",
        return_value={},
    ):
        cleaned = clean_transcript_segments(segments)

    assert cleaned[0].text == "original"


def test_clean_transcript_segments_calls_preview():
    segments = [Segment(start=0.0, end=1.0, text="hm test")]

    with patch(
        "local_whisper_transcribe.postprocess._clean_segment_batch",
        return_value={0: "test"},
    ):
        previews: list[str] = []
        clean_transcript_segments(segments, on_preview=previews.append)

    assert any("hm test" in line for line in previews)
    assert any("test" in line for line in previews)


def test_clean_transcript_segments_calls_progress():
    segments = [
        Segment(start=0.0, end=1.0, text="a"),
        Segment(start=1.0, end=2.0, text="b"),
        Segment(start=3.0, end=4.0, text="c"),
    ]
    progress_calls: list[tuple[int, int]] = []

    with patch(
        "local_whisper_transcribe.postprocess._clean_segment_batch",
        return_value={},
    ):
        clean_transcript_segments(
            segments,
            batch_size=2,
            on_progress=lambda current, total: progress_calls.append((current, total)),
        )

    assert progress_calls == [(1, 2), (2, 2)]


def test_clean_transcript_segments_calls_batch_complete():
    segments = [
        Segment(start=0.0, end=1.0, text="a"),
        Segment(start=1.0, end=2.0, text="b"),
        Segment(start=2.0, end=3.0, text="c"),
    ]
    snapshots: list[list[str]] = []

    def fake_batch(batch, *, language, model, url):
        return {idx: f"cleaned-{idx}" for idx, _ in batch}

    with patch(
        "local_whisper_transcribe.postprocess._clean_segment_batch",
        side_effect=fake_batch,
    ):
        clean_transcript_segments(
            segments,
            batch_size=2,
            on_batch_complete=lambda cleaned, _batch, _total: snapshots.append(
                [seg.text for seg in cleaned]
            ),
        )

    assert snapshots[0] == ["cleaned-0", "cleaned-1", "c"]
    assert snapshots[1] == ["cleaned-0", "cleaned-1", "cleaned-2"]
