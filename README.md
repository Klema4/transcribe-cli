# local-whisper-transcribe

Local CLI tool (`lwt`) for transcribing meetings from audio and video files using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). All processing runs **100% offline** on your machine. Optional translation, summarization, and speaker diarization also run locally.

**Everything is controlled exclusively through the `lwt` command** — no manual pip extras, environment variables, or `huggingface-cli` required.

---

## Table of contents

- [Features](#features)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [First-time setup](#first-time-setup)
- [Basic usage](#basic-usage)
- [Examples](#examples)
- [All commands](#all-commands)
- [Speaker diarization](#speaker-diarization)
- [Ollama (translation & summarization)](#ollama-translation--summarization)
- [Model recommendations](#model-recommendations)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## Features

- Transcribe audio (WAV, MP3, FLAC, OGG, M4A, …) and video (MP4, MKV, AVI, MOV, …)
- Export to **TXT**, **SRT**, **VTT**, or **JSON**
- Automatic language detection or manual selection (`--language cs`)
- GPU acceleration via CUDA (automatic fallback to CPU)
- Voice Activity Detection (VAD) for cleaner segments
- Optional translation and structured meeting summary via local [Ollama](https://ollama.com)
- Optional speaker diarization via [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- Rich terminal UI with progress bars, tables, and colored output

---

## System requirements

| Requirement | Required | Notes |
|-------------|----------|-------|
| **Python 3.10+** | yes | `python --version` |
| **ffmpeg** | yes | Extracts audio from video files |
| **CUDA / NVIDIA GPU** | no | Speeds up transcription |
| **Ollama** | no | Translation and summarization |
| **HuggingFace token** | no | Only for speaker diarization |

### Installing ffmpeg

**Windows:**
```powershell
winget install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt install ffmpeg
```

---

## Installation

```bash
git clone https://github.com/Klema4/local-whisper-transcribe.git
cd local-whisper-transcribe
pip install -e .
```

Verify the command works:

```bash
lwt --help
```

### Windows: `lwt` not recognized

On Windows, Python often installs scripts outside your PATH. Use one of these options:

**Option A — run via Python module (works immediately):**

```powershell
python -m local_whisper_transcribe.cli setup
python -m local_whisper_transcribe.cli transcribe meeting.mp4
```

**Option B — add Python Scripts to PATH (permanent fix):**

```powershell
# Check where lwt.exe was installed
python -c "import sysconfig; print(sysconfig.get_path('scripts'))"

# Example output:
# C:\Users\YOU\AppData\Local\Python\pythoncore-3.14-64\Scripts

# Add to PATH for current session:
$env:Path += ";C:\Users\YOU\AppData\Local\Python\pythoncore-3.14-64\Scripts"

# Then verify:
lwt --help
```

To add permanently: **Settings → System → About → Advanced system settings → Environment Variables → Path → New** → paste the Scripts folder path → restart your terminal.

---

## First-time setup

After installation, run the interactive setup wizard:

```bash
lwt setup
```

The wizard will:
1. Check Python, ffmpeg, and CUDA availability
2. Show a table of Whisper models (size, RAM, recommendations)
3. Let you pick and download a model
4. Optionally configure diarization (HuggingFace token)
5. Optionally check Ollama and pull an LLM model

Quick variants (model download only):

```bash
lwt setup --quick
lwt setup --model small
lwt setup --model medium --force   # re-download model
```

---

## Basic usage

```bash
# Transcribe a video file
lwt transcribe meeting.mp4

# Short alias
lwt t meeting.mp4

# Transcribe with output file
lwt transcribe call.wav -o transcript.txt
```

On first run without prior setup, `lwt` will interactively guide you through downloading a model.

---

## Examples

### Transcribe in a specific language

```bash
lwt transcribe meeting.wav --language cs --prompt "Technical API meeting"
```

### Subtitles (SRT / VTT)

```bash
lwt transcribe video.mp4 --format srt -o subtitles.srt
lwt transcribe video.mp4 --format vtt -o subtitles.vtt
```

### Structured JSON output

```bash
lwt transcribe meeting.mp4 --format json -o output.json
```

### Translate to another language (via Ollama)

```bash
lwt ollama pull llama3.2
lwt transcribe meeting.en.mp3 --translate-to Czech
```

### Summarize a meeting (via Ollama)

```bash
lwt transcribe standup.mp4 --summarize
lwt transcribe standup.mp4 --summarize --ollama-model llama3.2
```

### Speaker diarization

```bash
lwt install diarization
lwt config set diarization.hf_token hf_YOUR_TOKEN
lwt transcribe panel.mp3 --diarize --speaker-names "Alice,Bob,Carol"
```

Example output:

```
[00:01:23] Alice: Good morning, let's start the meeting.
[00:01:45] Bob: Sounds good, I have the presentation ready.
```

### Combine everything

```bash
lwt transcribe meeting.mp4 \
  --language en \
  --diarize \
  --speaker-names "Alice,Bob" \
  --summarize \
  --format txt \
  -o meeting-notes.txt
```

---

## All commands

### Transcription

```bash
lwt transcribe <file>              # basic transcription
lwt t <file>                       # alias

# Flags
  --language, -l       Source language (cs, en, de, …) or auto-detect
  --prompt, -p         Context prompt for better accuracy
  --model, -m          Whisper model (tiny, base, small, medium, large-v3)
  --format, -f         Output format: txt, srt, vtt, json
  --output, -o         Output file path
  --task               transcribe (default) or translate (English only)
  --diarize            Identify speakers (requires pyannote)
  --num-speakers       Number of speakers (hint for diarization)
  --speaker-names      Comma-separated speaker names
  --hf-token           HuggingFace token for diarization
  --translate-to       Translate to language via Ollama
  --summarize          Structured summary via Ollama
  --ollama-model       Ollama model for post-processing
```

### Setup & onboarding

```bash
lwt setup                            # full interactive wizard
lwt onboard                          # alias for setup
lwt setup --model medium             # pre-select model
lwt setup --quick                    # model download only
lwt setup --model small --force      # re-download model
```

### Models

```bash
lwt models list                      # available models with sizes
lwt models status                    # cached models on disk
lwt models download small            # download a specific model
lwt models download medium --force   # re-download
```

### Install optional dependencies

```bash
lwt install diarization              # install pyannote.audio
lwt install check                    # verify all dependencies
```

### Ollama

```bash
lwt ollama status                    # is Ollama running? which models?
lwt ollama list                      # list installed models
lwt ollama pull llama3.2             # download a model
```

### Configuration

```bash
lwt config show                      # display all settings
lwt config set whisper.model medium
lwt config set defaults.language cs
lwt config set defaults.format srt
lwt config set diarization.hf_token hf_xxx
lwt config set ollama.model llama3.2
lwt config set ollama.url http://localhost:11434
lwt config path                      # show config file location
lwt config reset                     # reset to defaults
```

---

## Speaker diarization

Diarization identifies who said what. It requires:

1. Install the dependency: `lwt install diarization`
2. Accept the model license at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Create a HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Save the token: `lwt config set diarization.hf_token hf_xxx`

Or configure everything via `lwt setup`.

```bash
lwt transcribe meeting.wav --diarize
lwt transcribe meeting.wav --diarize --speaker-names "Alice,Bob,Carol"
lwt transcribe meeting.wav --diarize --num-speakers 3
```

If you use `--diarize` without a token, `lwt` will prompt you interactively and save it to your config.

---

## Ollama (translation & summarization)

Whisper natively translates only to English (`--task translate`). For translation to any language or meeting summarization, use local Ollama:

1. Install and start [Ollama](https://ollama.com)
2. Pull a model: `lwt ollama pull llama3.2`
3. Transcribe with flags:

```bash
lwt transcribe meeting.mp4 --translate-to Czech
lwt transcribe meeting.mp4 --summarize
```

Set the default Ollama model via `lwt config set ollama.model llama3.2` or during `lwt setup`.

---

## Model recommendations

| Model    | Size     | RAM      | Recommendation                          |
|----------|----------|----------|-----------------------------------------|
| tiny     | ~75 MB   | ~1 GB    | Fastest, lowest accuracy                |
| base     | ~150 MB  | ~1 GB    | Fast, basic quality                     |
| small    | ~500 MB  | ~2 GB    | **Good balance (default)**              |
| medium   | ~1.5 GB  | ~4 GB    | High quality, slower                    |
| large-v3 | ~3 GB    | ~10 GB   | Best quality, ideally with GPU          |

Models are cached at:
- **Windows:** `%USERPROFILE%\.cache\huggingface\hub\`
- **Linux/macOS:** `~/.cache/huggingface/hub/`

---

## Configuration

The config file is created automatically on first run:

- **Windows:** `%USERPROFILE%\.config\local-whisper-transcribe\config.toml`
- **Linux/macOS:** `~/.config/local-whisper-transcribe/config.toml`

Default contents:

```toml
[whisper]
model = "small"
device = "auto"          # auto | cuda | cpu
compute_type = "auto"

[defaults]
language = "auto"        # auto = automatic detection
format = "txt"
output_dir = ""

[ollama]
url = "http://localhost:11434"
model = "llama3.2"

[diarization]
enabled = false
hf_token = ""

[meta]
setup_complete = false
```

Change everything via `lwt config set` — no manual file editing required.

---

## Troubleshooting

### `lwt` command not found (Windows)

`lwt.exe` is installed but the Python Scripts folder is not on your PATH. Quick fix:

```powershell
python -m local_whisper_transcribe.cli setup
python -m local_whisper_transcribe.cli transcribe meeting.mp4
```

Permanent fix — add the Scripts folder to PATH (see [Windows: `lwt` not recognized](#windows-lwt-not-recognized) above).

### ffmpeg not detected (but already installed via winget)

winget installs ffmpeg correctly, but your **current terminal session** may not have the updated PATH yet.

**Fix — restart your terminal** (close and reopen PowerShell/CMD), then verify:

```powershell
ffmpeg -version
```

If it still fails, add WinGet Links manually for this session:

```powershell
$env:Path += ";$env:LOCALAPPDATA\Microsoft\WinGet\Links"
ffmpeg -version
```

`lwt` also auto-detects ffmpeg in common winget install locations even without PATH.

### Model not downloaded

```bash
lwt setup
# or
lwt models download small
```

### Ollama not running

```bash
lwt ollama status
# Start the Ollama app, then:
lwt ollama pull llama3.2
```

### Diarization failing

```bash
lwt install check
lwt install diarization
lwt config set diarization.hf_token hf_YOUR_TOKEN
```

Make sure you accepted the license at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).

### Slow transcription

- Use a smaller model: `lwt config set whisper.model small`
- If you have an NVIDIA GPU, install CUDA — `lwt setup` detects it automatically

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

Project structure:

```
src/local_whisper_transcribe/
├── cli.py            # Typer CLI + Rich UI
├── transcribe.py     # faster-whisper wrapper
├── audio.py          # ffmpeg audio extraction
├── output.py         # TXT/SRT/VTT/JSON export
├── diarize.py        # pyannote diarization
├── postprocess.py    # Ollama translation & summarization
├── models.py         # Whisper model management
├── config.py         # TOML configuration
├── setup_wizard.py   # interactive onboarding
├── ollama_ops.py     # Ollama API
└── install_extra.py  # optional dependency installation
```

---

## License

MIT
