"""Ollama API helpers for CLI management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import httpx

from local_whisper_transcribe.postprocess import check_ollama_available


@dataclass
class OllamaStatus:
    available: bool
    url: str
    models: list[str] = field(default_factory=list)
    error: str | None = None


def get_ollama_status(url: str, timeout: float = 5.0) -> OllamaStatus:
    """Check whether Ollama is running and list installed models."""
    base = url.rstrip("/")
    if not check_ollama_available(base, timeout=timeout):
        return OllamaStatus(
            available=False,
            url=base,
            error="Ollama není dostupná. Spusťte Ollama nebo zkontrolujte URL.",
        )
    try:
        response = httpx.get(f"{base}/api/tags", timeout=timeout)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        return OllamaStatus(available=True, url=base, models=models)
    except httpx.HTTPError as exc:
        return OllamaStatus(available=False, url=base, error=str(exc))


def list_ollama_models(url: str) -> list[str]:
    return get_ollama_status(url).models


def pull_ollama_model(
    model: str,
    url: str,
    *,
    on_progress: Callable[[str, int | None], None] | None = None,
) -> None:
    """Pull a model from Ollama with streaming progress."""
    endpoint = f"{url.rstrip('/')}/api/pull"
    payload = {"name": model, "stream": True}

    with httpx.stream("POST", endpoint, json=payload, timeout=None) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            status = data.get("status", "")
            total = data.get("total")
            completed = data.get("completed")
            percent: int | None = None
            if total and completed is not None:
                percent = int(completed / total * 100)
            if on_progress:
                on_progress(status, percent)
            if data.get("error"):
                raise RuntimeError(data["error"])
