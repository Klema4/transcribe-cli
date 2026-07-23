# Development

```bash
git clone https://github.com/Klema4/transcribe-cli.git
cd transcribe-cli
pip install -e ".[dev]"
pytest
```

Python import path remains `local_whisper_transcribe`; the product name is **Transcribe CLI** (`lwt`).

## Structure

```
src/local_whisper_transcribe/
├── cli.py            # Typer CLI + Rich UI
├── transcribe.py     # faster-whisper + MLX Apple Silicon wrapper
├── audio.py          # ffmpeg extraction
├── output.py         # TXT/SRT/VTT/JSON
├── diarize.py        # pyannote diarization
├── postprocess.py    # Ollama
├── import_transcript.py
├── progress_ui.py
├── models.py         # Whisper model cache
├── cuda_runtime.py   # CUDA 12 install & DLL paths
├── config.py         # TOML config
├── setup_wizard.py   # lwt setup
├── ollama_ops.py     # Ollama API
├── install_extra.py  # optional deps
├── prompts.py        # Rich prompts
└── system_checks.py  # ffmpeg/CUDA checks
```

## Tests

```bash
pytest
pytest -q tests/test_diarize.py
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for issues, pull requests, and local setup.
