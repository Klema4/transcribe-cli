# Ollama

Whisper natively translates only to **English** (`--task translate`). For other languages or structured meeting summaries, use local [Ollama](https://ollama.com).

## Setup

1. Install and start Ollama
2. `lwt ollama pull llama3.2`
3. Optional: `lwt config set ollama.model llama3.2`

## Usage

```bash
lwt transcribe meeting.mp4 --translate-to Czech
lwt transcribe meeting.mp4 --summarize
lwt transcribe standup.mp4 --summarize --ollama-model llama3.2
```

## Commands

```bash
lwt ollama status
lwt ollama list
lwt ollama pull llama3.2
```

Default URL: `http://localhost:11434` — change via `lwt config set ollama.url ...`
