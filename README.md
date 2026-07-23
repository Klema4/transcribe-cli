# Transcribe CLI

Offline meeting transcription from audio or video. Command: **`lwt`**.

Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with automatic [MLX](https://github.com/ml-explore/mlx) GPU acceleration on Apple Silicon. Everything runs on your machine — optional AI cleaning, translation, summarization, and speaker labels included.

Full docs: [Wiki](https://github.com/Klema4/transcribe-cli/wiki) · [docs/](docs/Home.md)

---

## Quick start

### 1. Requirements

| Need | Required? | Install |
|------|-----------|---------|
| Python 3.10+ | yes | — |
| [ffmpeg](https://ffmpeg.org) | yes | see below |
| NVIDIA GPU | no | faster runs |
| Apple Silicon GPU | no | enabled automatically on native arm64 macOS |
| [Ollama](https://ollama.com) | no | `--clean`, `--translate-to`, `--summarize` |
| HuggingFace token | no | only for `--diarize` |

**ffmpeg**

```powershell
# Windows
winget install ffmpeg
```

```bash
# macOS
brew install ffmpeg

# Linux (Debian/Ubuntu)
sudo apt install ffmpeg
```

### 2. Install

```bash
git clone https://github.com/Klema4/transcribe-cli.git
cd transcribe-cli
pip install -e .
lwt --help
```

<details>
<summary>Windows: <code>lwt</code> not found?</summary>

Python Scripts may be outside PATH. Use either:

```powershell
# Works immediately
python -m local_whisper_transcribe.cli setup
python -m local_whisper_transcribe.cli transcribe meeting.mp4
```

Or add Scripts to PATH permanently:

```powershell
python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
# Settings → System → About → Advanced → Environment Variables → Path → New
```

</details>

### 3. Setup (once)

```bash
lwt setup
```

Picks a Whisper model, checks ffmpeg and the available acceleration backend (MLX on Apple Silicon or CUDA on NVIDIA), optionally configures Ollama and diarization.

```bash
lwt setup --quick              # model only
lwt setup --model medium       # choose model up front
```

### 4. Transcribe

```bash
lwt transcribe meeting.mp4
lwt t meeting.mp4                        # short alias
lwt transcribe call.wav -o notes.txt
```

---

## Everyday commands

```bash
# Language + context
lwt transcribe meeting.wav --language cs --prompt "Technical API meeting"

# Subtitles
lwt transcribe video.mp4 --format srt -o subtitles.srt
lwt transcribe video.mp4 --format vtt -o subtitles.vtt

# JSON
lwt transcribe meeting.mp4 --format json -o output.json

# AI clean / translate / summarize (needs Ollama)
lwt ollama pull llama3.2
lwt transcribe meeting.mp4 --clean
lwt transcribe meeting.en.mp3 --translate-to Czech
lwt transcribe standup.mp4 --summarize

# Clean an existing transcript file
lwt clean meeting.txt

# Speakers (needs HuggingFace — see below)
lwt install diarization
lwt config set diarization.hf_token hf_YOUR_TOKEN
lwt transcribe panel.mp3 --diarize --speaker-names "Alice,Bob"
```

**Full pipeline example**

```bash
lwt transcribe meeting.mp4 \
  --language en \
  --diarize --speaker-names "Alice,Bob" \
  --clean --summarize \
  -o meeting-notes.txt
```

| Output | When |
|--------|------|
| `meeting-notes.txt` | always |
| `meeting-notes.raw.txt` | with `--clean` (original Whisper text) |
| `meeting-notes.summary.md` | with `--summarize` |

---

## Command cheat sheet

| Command | Purpose |
|---------|---------|
| `lwt setup` | First-time wizard |
| `lwt transcribe` / `lwt t` | Transcribe audio/video |
| `lwt clean` | Clean an existing transcript via Ollama |
| `lwt models list` / `download` / `status` | Whisper models |
| `lwt install cuda` | CUDA 12 GPU libraries |
| `lwt install diarization` | pyannote speaker labels |
| `lwt install check` | Verify dependencies |
| `lwt ollama status` / `pull` / `list` | Local LLM |
| `lwt config show` / `set` / `path` / `reset` | Settings |

**Useful `transcribe` flags**

| Flag | What it does |
|------|----------------|
| `-l` / `--language` | Force language (`cs`, `en`, …) |
| `-m` / `--model` | Whisper model |
| `-f` / `--format` | `txt` · `srt` · `vtt` · `json` |
| `-o` / `--output` | Output path |
| `--clean` | Remove fillers, fix ASR errors (Ollama) |
| `--translate-to` | Translate via Ollama |
| `--summarize` | Meeting notes via Ollama |
| `--diarize` | Who spoke when |
| `--no-diarize` | Disable diarization for this run, even if enabled in config |
| `--speaker-names` | `"Alice,Bob,Carol"` |

→ Full flag list: [`lwt --help`](docs/Commands.md) · [Commands wiki](https://github.com/Klema4/transcribe-cli/wiki/Commands)

---

## Optional: Ollama

Used for `--clean`, `--translate-to`, `--summarize`, and `lwt clean`.

1. Install [Ollama](https://ollama.com) and start it  
2. `lwt ollama pull llama3.2`  
3. `lwt config set ollama.model llama3.2`  

| Flag | Result |
|------|--------|
| `--clean` | Cleaner transcript + `*.raw.*` backup |
| `--translate-to LANG` | Translated file |
| `--summarize` | `*.summary.md` |

Details: [Ollama docs](docs/Ollama.md)

---

## Optional: speaker diarization

Transcription works without this. Only needed for `--diarize`.

1. Create a free [HuggingFace](https://huggingface.co/join) account  
2. Accept **all four** model licenses (easy to miss the last one):
   - [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
   - [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) ← required by pyannote 4.x  
3. Create a **Read** token at [settings/tokens](https://huggingface.co/settings/tokens)  
4. Save and install:

```bash
lwt config set diarization.hf_token hf_YOUR_TOKEN
lwt install diarization
lwt install verify-diarization
lwt transcribe meeting.wav --diarize
```

Details: [HuggingFace diarization](docs/HuggingFace-Diarization.md)

---

## Models

| Model | Size | RAM | Notes |
|-------|------|-----|-------|
| tiny | ~75 MB | ~1 GB | Fastest |
| base | ~150 MB | ~1 GB | Basic |
| **small** | ~500 MB | ~2 GB | **Default — good balance** |
| medium | ~1.5 GB | ~4 GB | Higher quality |
| large-v3 | ~3 GB | ~10 GB | Best (GPU recommended) |

```bash
lwt models list
lwt models download medium
lwt config set whisper.model medium
```

---

## Configuration

```bash
lwt config show
lwt config path
lwt config set whisper.model medium
lwt config set defaults.language cs
lwt config set defaults.format srt
lwt config set ollama.model llama3.2
lwt config set diarization.hf_token hf_xxx
lwt config set whisper.device cpu          # force CPU
lwt config set whisper.device mlx          # force MLX on Apple Silicon
```

Config file (created on first run):

- **Windows:** `%USERPROFILE%\.config\transcribe-cli\config.toml`
- **Linux/macOS:** `~/.config/transcribe-cli/config.toml`

(Legacy installs under `~/.config/local-whisper-transcribe/` are still read.)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `lwt` not found (Windows) | `python -m local_whisper_transcribe.cli …` or add Scripts to PATH |
| ffmpeg missing after winget | Restart the terminal; or `$env:Path += ";$env:LOCALAPPDATA\Microsoft\WinGet\Links"` |
| No model | `lwt setup` or `lwt models download small` |
| Ollama flags fail | `lwt ollama status` → start Ollama → `lwt ollama pull llama3.2` |
| Diarization 403 / gated | Accept all 4 HF licenses + save token → `lwt install verify-diarization` |
| `cublas64_12.dll` missing | `lwt install cuda` (or `lwt config set whisper.device cpu`) |
| Apple GPU not used | Reinstall with `pip install -e .` from a native arm64 Python environment |
| Slow | On Apple Silicon, reinstall with `pip install -e .` to enable MLX; on NVIDIA, install CUDA via `lwt install cuda` |

More: [Troubleshooting](docs/Troubleshooting.md) · [GPU & CUDA](docs/GPU-and-CUDA.md)

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

Python package path remains `local_whisper_transcribe` (CLI entry point: `lwt`).

```
src/local_whisper_transcribe/
├── cli.py            # Typer CLI + Rich UI
├── transcribe.py     # faster-whisper + MLX Apple Silicon backend
├── audio.py          # ffmpeg
├── output.py         # TXT / SRT / VTT / JSON
├── diarize.py        # pyannote
├── postprocess.py    # Ollama clean / translate / summarize
├── import_transcript.py
├── progress_ui.py
├── models.py
├── config.py
├── setup_wizard.py
├── ollama_ops.py
└── install_extra.py
```

---

## Contributing

Bug reports, ideas, and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT
