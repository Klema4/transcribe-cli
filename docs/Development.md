# Development

```bash
git clone https://github.com/Klema4/local-whisper-transcribe.git
cd local-whisper-transcribe
pip install -e ".[dev]"
pytest
```

## Structure

```
src/local_whisper_transcribe/
├── cli.py            # Typer CLI + Rich UI
├── transcribe.py     # faster-whisper wrapper
├── audio.py          # ffmpeg extraction
├── output.py         # TXT/SRT/VTT/JSON
├── diarize.py        # pyannote diarization
├── postprocess.py    # Ollama
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

44 tests covering output formats, models, diarization, CUDA, config.
