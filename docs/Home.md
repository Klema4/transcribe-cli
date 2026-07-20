# local-whisper-transcribe

Local CLI tool (`lwt`) for transcribing meetings from audio and video files using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). All transcription runs **100% offline** on your machine.

**Everything is controlled through the `lwt` command** — no manual pip extras, environment variables, or `huggingface-cli` required.

## Wiki

| Page | Description |
|------|-------------|
| [[Installation]] | Install Python package, ffmpeg, PATH fixes |
| [[First-Time Setup]] | `lwt setup` wizard and model download |
| [[Usage]] | Basic transcription examples |
| [[Commands]] | Full CLI reference |
| [[HuggingFace Diarization]] | **Required** for `--diarize` — account, licenses, token |
| [[Ollama]] | Translation and meeting summarization |
| [[GPU and CUDA]] | `lwt install cuda`, GPU acceleration |
| [[Configuration]] | `config.toml` and `lwt config` |
| [[Troubleshooting]] | Common errors and fixes |
| [[Development]] | Tests and project structure |

## Quick start

```bash
git clone https://github.com/Klema4/local-whisper-transcribe.git
cd local-whisper-transcribe
pip install -e .

lwt setup
lwt transcribe meeting.mp4
```

## Features

- Audio (WAV, MP3, FLAC, …) and video (MP4, MKV, AVI, …)
- Export: TXT, SRT, VTT, JSON
- GPU via CUDA (auto fallback to CPU)
- Optional speaker diarization (`--diarize`) — see [[HuggingFace Diarization]]
- Optional translation/summary via Ollama — see [[Ollama]]
- Rich terminal UI

## License

MIT — see [repository](https://github.com/Klema4/local-whisper-transcribe)
