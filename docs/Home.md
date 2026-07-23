# Transcribe CLI

Offline meeting transcription (`lwt`) from audio and video using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with automatic MLX/Metal acceleration on Apple Silicon. Runs **100% on your machine**.

**Everything goes through `lwt`** — no manual pip extras, env vars, or `huggingface-cli`.

## Wiki

| Page | Description |
|------|-------------|
| [[Installation]] | Install package, ffmpeg, PATH fixes |
| [[First-Time Setup]] | `lwt setup` wizard and model download |
| [[Usage]] | Basic transcription examples |
| [[Commands]] | Full CLI reference |
| [[HuggingFace Diarization]] | Required for `--diarize` — account, licenses, token |
| [[Ollama]] | Cleaning, translation, summarization |
| [[GPU and CUDA]] | `lwt install cuda`, GPU acceleration |
| [[Configuration]] | `config.toml` and `lwt config` |
| [[Troubleshooting]] | Common errors and fixes |
| [[Development]] | Tests and project structure |

## Quick start

```bash
git clone https://github.com/Klema4/transcribe-cli.git
cd transcribe-cli
pip install -e .

lwt setup
lwt transcribe meeting.mp4
```

## Features

- Audio (WAV, MP3, FLAC, …) and video (MP4, MKV, AVI, …)
- Export: TXT, SRT, VTT, JSON
- GPU via MLX/Metal on Apple Silicon or CUDA on NVIDIA (automatic CPU fallback)
- Optional speaker diarization (`--diarize`) — see [[HuggingFace Diarization]]
- Optional clean / translate / summarize via Ollama — see [[Ollama]]
- Rich terminal UI

## License

MIT — see [repository](https://github.com/Klema4/transcribe-cli)
