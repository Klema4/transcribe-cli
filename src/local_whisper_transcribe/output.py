"""Export transcription results to various formats."""

from __future__ import annotations

import json
from pathlib import Path

from local_whisper_transcribe.transcribe import Segment, TranscriptionResult


def _format_timestamp_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _has_speakers(segments: list[Segment]) -> bool:
    return any(seg.speaker for seg in segments)


def _group_speaker_segments(segments: list[Segment]) -> list[list[Segment]]:
    if not segments:
        return []

    groups: list[list[Segment]] = [[segments[0]]]
    for seg in segments[1:]:
        if seg.speaker == groups[-1][-1].speaker:
            groups[-1].append(seg)
        else:
            groups.append([seg])
    return groups


def _format_segment_text(seg: Segment) -> str:
    if seg.speaker:
        return f"[{seg.speaker}] {seg.text}"
    return seg.text


def format_txt(
    segments: list[Segment],
    *,
    with_timestamps: bool = False,
    with_speakers: bool = False,
) -> str:
    lines: list[str] = []
    use_speakers = with_speakers and _has_speakers(segments)

    if use_speakers:
        for group in _group_speaker_segments(segments):
            text = " ".join(seg.text for seg in group)
            speaker = group[0].speaker
            if with_timestamps:
                timestamp = _format_timestamp_vtt(group[0].start)
                if speaker:
                    lines.append(f"[{timestamp}] {speaker}: {text}")
                else:
                    lines.append(f"[{timestamp}] {text}")
            elif speaker:
                lines.append(f"{speaker}: {text}")
            else:
                lines.append(text)
        return "\n".join(lines)

    for seg in segments:
        if with_timestamps:
            lines.append(
                f"[{_format_timestamp_vtt(seg.start)} -> {_format_timestamp_vtt(seg.end)}] {seg.text}"
            )
        else:
            lines.append(seg.text)
    return "\n".join(lines)


def format_srt(segments: list[Segment]) -> str:
    blocks: list[str] = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n"
            f"{_format_timestamp_srt(seg.start)} --> {_format_timestamp_srt(seg.end)}\n"
            f"{_format_segment_text(seg)}\n"
        )
    return "\n".join(blocks)


def format_vtt(segments: list[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(
            f"{_format_timestamp_vtt(seg.start)} --> {_format_timestamp_vtt(seg.end)}\n"
            f"{_format_segment_text(seg)}\n"
        )
    return "\n".join(lines)


def format_json(result: TranscriptionResult, *, source: str | None = None) -> str:
    payload = {
        "language": result.language,
        "duration": result.duration,
        "metadata": result.metadata,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "speaker": s.speaker,
            }
            for s in result.segments
        ],
    }
    if source:
        payload["source"] = source
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_result(
    result: TranscriptionResult,
    output_path: Path,
    fmt: str,
    *,
    with_timestamps: bool = False,
    source: str | None = None,
) -> Path:
    """Write transcription result to disk in the requested format."""
    fmt = fmt.lower()
    has_speakers = _has_speakers(result.segments)
    if fmt == "txt":
        content = format_txt(
            result.segments,
            with_timestamps=with_timestamps or has_speakers,
            with_speakers=has_speakers,
        )
    elif fmt == "srt":
        content = format_srt(result.segments)
    elif fmt == "vtt":
        content = format_vtt(result.segments)
    elif fmt == "json":
        content = format_json(result, source=source)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def default_output_path(
    input_path: Path,
    output_dir: Path | None,
    fmt: str,
) -> Path:
    """Build a default output path from the input filename."""
    stem = input_path.stem
    directory = output_dir or input_path.parent
    return directory / f"{stem}.{fmt.lower()}"
