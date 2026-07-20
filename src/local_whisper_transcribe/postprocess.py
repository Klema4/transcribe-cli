"""Ollama-based translation and summarization."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Callable

import httpx

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"


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
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }
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
        f"Přelož následující text do jazyka {target_lang}. "
        f"Vrať pouze překlad, bez komentářů a vysvětlení.\n\n"
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
        "Vytvoř strukturované zápisy ze schůzky na základě následujícího přepisu. "
        "Použij tyto sekce:\n"
        "1. Shrnutí (2-3 věty)\n"
        "2. Klíčové body\n"
        "3. Rozhodnutí\n"
        "4. Akční body (kdo/co/kdy, pokud lze určit)\n"
        "5. Otevřené otázky\n\n"
        f"Přepis:\n{text}"
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
