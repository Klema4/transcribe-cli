"""Ollama-based translation, summarization, and transcript cleaning."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import replace
from typing import Callable

import httpx

from local_whisper_transcribe.transcribe import Segment

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
CLEAN_BATCH_SIZE = 15

_NUMBERED_LINE_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")


def check_ollama_available(url: str = DEFAULT_OLLAMA_URL, timeout: float = 5.0) -> bool:
    """Return True if Ollama is reachable."""
    try:
        response = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _ollama_generate(
    prompt: str,
    *,
    model: str,
    url: str,
    stream: bool = False,
    on_chunk: Callable[[str], None] | None = None,
    temperature: float | None = None,
) -> str:
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    endpoint = f"{url.rstrip('/')}/api/generate"

    if not stream:
        response = httpx.post(endpoint, json=payload, timeout=300.0)
        response.raise_for_status()
        return response.json().get("response", "")

    parts: list[str] = []
    with httpx.stream("POST", endpoint, json=payload, timeout=300.0) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            import json

            data = json.loads(line)
            chunk = data.get("response", "")
            if chunk:
                parts.append(chunk)
                if on_chunk:
                    on_chunk(chunk)
            if data.get("done"):
                break
    return "".join(parts)


def translate_text(
    text: str,
    target_lang: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
    *,
    stream: bool = False,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    """Translate text to the target language using Ollama."""
    prompt = (
        f"Translate the following text into {target_lang}. "
        f"Return only the translation, without comments or explanations.\n\n"
        f"Text:\n{text}"
    )
    return _ollama_generate(prompt, model=model, url=url, stream=stream, on_chunk=on_chunk)


def summarize_meeting(
    text: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
    *,
    stream: bool = False,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    """Generate structured meeting notes from a transcript."""
    prompt = (
        "Create structured meeting notes from the following transcript. "
        "Use these sections:\n"
        "1. Summary (2-3 sentences)\n"
        "2. Key points\n"
        "3. Decisions\n"
        "4. Action items (who/what/when, if determinable)\n"
        "5. Open questions\n\n"
        f"Transcript:\n{text}"
    )
    return _ollama_generate(prompt, model=model, url=url, stream=stream, on_chunk=on_chunk)


def stream_translate(
    text: str,
    target_lang: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
) -> Iterator[str]:
    """Yield translation chunks from Ollama."""
    collected: list[str] = []

    def on_chunk(chunk: str) -> None:
        collected.append(chunk)

    result = translate_text(
        text,
        target_lang,
        model=model,
        url=url,
        stream=True,
        on_chunk=lambda c: collected.append(c),
    )
    if result:
        yield result
    for chunk in collected:
        yield chunk


def _build_clean_prompt(batch: list[tuple[int, str]], language: str | None) -> str:
    if language and language != "auto":
        lang_note = f"The transcript language is {language}."
    else:
        lang_note = "Keep the original language of each line."

    lines_text = "\n".join(f"[{idx}] {text}" for idx, text in batch)
    return (
        "You are cleaning a speech-to-text transcript. "
        f"{lang_note}\n\n"
        "For each numbered line below:\n"
        "- Remove filler words and hesitations (e.g. um, uh, hm, eh, repeated words).\n"
        "- Fix obvious transcription errors using context (garbled nonsense words).\n"
        "- Do NOT summarize, reorder, merge, or split lines.\n"
        "- Do NOT change meaning, tone, or add new content.\n"
        "- Preserve proper nouns unless they are clearly garbled.\n"
        "- Return EXACTLY one output line per input line, keeping the same [N] prefix.\n\n"
        f"Input:\n{lines_text}\n\n"
        "Output:"
    )


def parse_cleaned_lines(response: str) -> dict[int, str]:
    """Parse numbered lines like ``[3] cleaned text`` from model output."""
    parsed: dict[int, str] = {}
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _NUMBERED_LINE_RE.match(line)
        if match:
            parsed[int(match.group(1))] = match.group(2).strip()
    return parsed


def _clean_segment_batch(
    batch: list[tuple[int, str]],
    *,
    language: str | None,
    model: str,
    url: str,
) -> dict[int, str]:
    prompt = _build_clean_prompt(batch, language)
    response = _ollama_generate(
        prompt,
        model=model,
        url=url,
        stream=False,
        temperature=0.1,
    )
    return parse_cleaned_lines(response)


def clean_transcript_segments(
    segments: list[Segment],
    *,
    language: str | None = None,
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
    batch_size: int = CLEAN_BATCH_SIZE,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Segment]:
    """Clean transcript segments via Ollama while preserving segment boundaries."""
    if not segments:
        return []

    cleaned = list(segments)
    total_batches = (len(segments) + batch_size - 1) // batch_size

    for batch_num, start in enumerate(range(0, len(segments), batch_size), start=1):
        if on_progress:
            on_progress(batch_num, total_batches)

        end = min(start + batch_size, len(segments))
        batch = [(idx, segments[idx].text) for idx in range(start, end)]
        updates = _clean_segment_batch(
            batch,
            language=language,
            model=model,
            url=url,
        )

        for idx in range(start, end):
            if idx in updates and updates[idx]:
                cleaned[idx] = replace(cleaned[idx], text=updates[idx])

    return cleaned
