"""Load existing transcript files back into structured segments."""

from __future__ import annotations

import json
import re
from pathlib import Path

from local_whisper_transcribe.transcribe import Segment, TranscriptionResult

_SPEAKER_BRACKET_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$", re.DOTALL)
_TXT_RANGE_RE = re.compile(
    r"^\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.+)$"
)
_TXT_SPEAKER_TS_RE = re.compile(
    r"^\[(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*([^:]+):\s*(.+)$"
)
_SRT_BLOCK_RE = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
    r"((?:.+\n?)+?)(?=\n\d+\s*\n|\Z)",
    re.MULTILINE,
)
_VTT_CUE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n"
    r"((?:.+\n?)+?)(?=\n\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*|\Z)",
    re.MULTILINE,
)


def _parse_vtt_timestamp(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(".")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


def _parse_srt_timestamp(value: str) -> float:
    return _parse_vtt_timestamp(value.replace(",", "."))


def _split_speaker_text(text: str) -> tuple[str | None, str]:
    stripped = text.strip()
    match = _SPEAKER_BRACKET_RE.match(stripped)
    if match:
        return match.group(1), match.group(2).strip()
    return None, stripped


def _detect_format(path: Path, fmt: str | None) -> str:
    if fmt:
        return fmt.lower()
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("txt", "srt", "vtt", "json"):
        return suffix
    raise ValueError(
        f"Cannot detect format for '{path.name}'. Use --format txt|srt|vtt|json."
    )


def parse_json_transcript(content: str) -> TranscriptionResult:
    payload = json.loads(content)
    segments = [
        Segment(
            start=float(item["start"]),
            end=float(item["end"]),
            text=str(item.get("text", "")),
            speaker=item.get("speaker"),
        )
        for item in payload.get("segments", [])
    ]
    return TranscriptionResult(
        segments=segments,
        language=str(payload.get("language", "auto")),
        duration=float(payload.get("duration", 0.0)),
        metadata=dict(payload.get("metadata", {})),
    )


def parse_srt_transcript(content: str) -> TranscriptionResult:
    segments: list[Segment] = []
    for match in _SRT_BLOCK_RE.finditer(content.strip() + "\n"):
        start = _parse_srt_timestamp(match.group(2))
        end = _parse_srt_timestamp(match.group(3))
        speaker, text = _split_speaker_text(match.group(4))
        segments.append(Segment(start=start, end=end, text=text, speaker=speaker))
    if not segments:
        raise ValueError("No SRT cues found in file.")
    duration = max(seg.end for seg in segments)
    return TranscriptionResult(
        segments=segments,
        language="auto",
        duration=duration,
        metadata={},
    )


def parse_vtt_transcript(content: str) -> TranscriptionResult:
    body = content.strip()
    if body.upper().startswith("WEBVTT"):
        body = body.split("\n", 1)[1] if "\n" in body else ""

    segments: list[Segment] = []
    for match in _VTT_CUE_RE.finditer(body.strip() + "\n"):
        start = _parse_vtt_timestamp(match.group(1))
        end = _parse_vtt_timestamp(match.group(2))
        speaker, text = _split_speaker_text(match.group(3))
        segments.append(Segment(start=start, end=end, text=text, speaker=speaker))
    if not segments:
        raise ValueError("No VTT cues found in file.")
    duration = max(seg.end for seg in segments)
    return TranscriptionResult(
        segments=segments,
        language="auto",
        duration=duration,
        metadata={},
    )


def parse_txt_transcript(content: str) -> TranscriptionResult:
    segments: list[Segment] = []
    lines = [line for line in content.splitlines() if line.strip()]

    for index, line in enumerate(lines):
        range_match = _TXT_RANGE_RE.match(line.strip())
        if range_match:
            start = _parse_vtt_timestamp(range_match.group(1))
            end = _parse_vtt_timestamp(range_match.group(2))
            segments.append(Segment(start=start, end=end, text=range_match.group(3).strip()))
            continue

        speaker_ts_match = _TXT_SPEAKER_TS_RE.match(line.strip())
        if speaker_ts_match:
            start = _parse_vtt_timestamp(speaker_ts_match.group(1))
            end = start + 1.0
            segments.append(
                Segment(
                    start=start,
                    end=end,
                    text=speaker_ts_match.group(3).strip(),
                    speaker=speaker_ts_match.group(2).strip(),
                )
            )
            continue

        if ": " in line and not line.strip().startswith("["):
            speaker, text = line.split(":", 1)
            if speaker and not speaker.startswith("http"):
                segments.append(
                    Segment(
                        start=float(index),
                        end=float(index + 1),
                        text=text.strip(),
                        speaker=speaker.strip(),
                    )
                )
                continue

        segments.append(
            Segment(start=float(index), end=float(index + 1), text=line.strip())
        )

    if not segments:
        raise ValueError("TXT file is empty.")
    duration = max(seg.end for seg in segments)
    return TranscriptionResult(
        segments=segments,
        language="auto",
        duration=duration,
        metadata={},
    )


def load_transcript(path: Path, *, fmt: str | None = None) -> TranscriptionResult:
    """Load a transcript file produced by lwt (or compatible formats)."""
    resolved_fmt = _detect_format(path, fmt)
    content = path.read_text(encoding="utf-8")

    if resolved_fmt == "json":
        return parse_json_transcript(content)
    if resolved_fmt == "srt":
        return parse_srt_transcript(content)
    if resolved_fmt == "vtt":
        return parse_vtt_transcript(content)
    if resolved_fmt == "txt":
        return parse_txt_transcript(content)
    raise ValueError(f"Unsupported format: {resolved_fmt}")


def default_clean_output_path(input_path: Path, output: Path | None) -> Path:
    """Pick output path for ``lwt clean``."""
    if output:
        return output
    stem = input_path.stem
    if stem.endswith(".raw"):
        stem = stem[: -len(".raw")]
    return input_path.with_name(f"{stem}{input_path.suffix}")


def default_raw_backup_path(output_path: Path) -> Path:
    """Path for the untouched transcript backup."""
    return output_path.with_stem(f"{output_path.stem}.raw")
