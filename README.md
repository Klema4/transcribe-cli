# Transcribe CLI

Local CLI tool (`lwt`) for transcribing meetings from audio and video files using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). All processing runs **100% offline** on your machine. Optional AI transcript cleaning, translation, summarization, and speaker diarization also run locally.

**Everything is controlled exclusively through the `lwt` command** — no manual pip extras, environment variables, or `huggingface-cli` required.

📖 **Full documentation:** [GitHub Wiki](https://github.com/Klema4/local-whisper-transcribe/wiki) · [docs/](docs/Home.md) folder in this repo

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
  - [HuggingFace setup for diarization](#huggingface-setup-for-diarization)
- [Ollama (translation, summarization & cleaning)](#ollama-translation-summarization--cleaning)
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
- Optional **AI transcript cleaning** via Ollama (`--clean`) — removes filler words and fixes obvious speech-to-text errors
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
| **Ollama** | no | Transcript cleaning (`--clean`), translation (`--translate-to`), and summarization (`--summarize`) |
| **HuggingFace account + token** | no | Required for speaker diarization only (see [HuggingFace setup](#huggingface-setup-for-diarization)) |

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
git clone https://github.com/Klema4/transcribe-cli.git
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
4. Optionally configure diarization (requires [HuggingFace setup](#huggingface-setup-for-diarization))
5. Optionally check Ollama and pull an LLM model (used for `--clean`, `--translate-to`, and `--summarize`)

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

### Clean transcript with AI (via Ollama)

Use `--clean` to post-process the transcript with your configured Ollama model (set during `lwt setup` or via `lwt config set ollama.model`):

```bash
lwt ollama pull llama3.2
lwt transcribe meeting.mp4 --clean
```

What it does:

- Removes filler words and hesitations (`hm`, `eh`, `jo jo jo`, repeated words, …)
- Fixes obvious speech-to-text errors using context (e.g. garbled nonsense → the intended word)
- Preserves segment boundaries and timestamps (SRT/VTT stay in sync)
- Saves the **original** transcript as `meeting.raw.txt` (or `.raw.srt`, `.raw.json`, …) next to the cleaned output

The default Ollama model is read from your config (`ollama.model`). Override per run with `--ollama-model`:

```bash
lwt transcribe meeting.mp4 --clean --ollama-model mistral
```

Requires a running Ollama instance (`lwt ollama status`).

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

> **Important:** `--diarize` requires a free HuggingFace account, accepting model licenses, and saving an access token. See [HuggingFace setup for diarization](#huggingface-setup-for-diarization) below.

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
  --clean \
  --summarize \
  --format txt \
  -o meeting-notes.txt
```

This produces:

- `meeting-notes.txt` — cleaned transcript
- `meeting-notes.raw.txt` — original Whisper output (because of `--clean`)
- `meeting-notes.summary.md` — meeting summary (because of `--summarize`)

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
  --clean              Clean transcript via Ollama (fillers + ASR fixes; uses ollama.model from config)
  --translate-to       Translate to language via Ollama
  --summarize          Structured summary via Ollama
  --ollama-model       Ollama model for --clean, --translate-to, and --summarize
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
lwt install cuda                     # CUDA 12 GPU libraries (cuBLAS + cuDNN)
lwt install cuda --toolkit           # + full NVIDIA CUDA Toolkit via winget (Windows)
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

Diarization identifies who said what (Speaker 1, Speaker 2, …). It uses [pyannote.audio](https://github.com/pyannote/pyannote-audio) and downloads models from HuggingFace on first use.

**Transcription works without this.** You only need HuggingFace if you use `--diarize`.

### HuggingFace setup for diarization

This is a **one-time setup**. Without it you will get a `403` / `gated repo` error.

#### Step 1 — Create a HuggingFace account

Sign up for free at [huggingface.co/join](https://huggingface.co/join).

#### Step 2 — Accept model licenses (all 4 required with pyannote 4.x)

Log in, open each link below, and click **Agree and access repository** (or similar):

| Model | Link |
|-------|------|
| Speaker diarization pipeline | [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) |
| Segmentation model | [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) |
| Speaker embedding model | [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) |
| **Community PLDA (pyannote 4.x)** | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) |

You must accept **all four**. The 4th (`community-1`) is easy to miss but required by pyannote.audio 4.x.

Verify before transcribing:

```bash
lwt install verify-diarization
```

#### Step 3 — Create an access token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **Create new token**
3. Name it e.g. `lwt-diarization`
4. Type: **Read** (sufficient for downloading models)
5. Copy the token — it starts with `hf_`

#### Step 4 — Save the token in `lwt`

```bash
lwt config set diarization.hf_token hf_your_token_here
```

Or enter it during `lwt setup` when prompted.

Verify:

```bash
lwt config show
```

The token is stored locally in your config file (`lwt config path`). It is only used to download pyannote models from HuggingFace.

#### Step 5 — Install pyannote and transcribe

```bash
lwt install diarization
lwt transcribe meeting.wav --diarize
lwt transcribe meeting.wav --diarize --speaker-names "Alice,Bob,Carol"
lwt transcribe meeting.wav --diarize --num-speakers 3
```

On first run, pyannote models are downloaded and cached locally. After that, diarization runs offline.

### Quick reference

```bash
lwt install diarization
lwt config set diarization.hf_token hf_xxx
lwt transcribe meeting.wav --diarize
```

If you use `--diarize` without a saved token, `lwt` will prompt you interactively and save it to your config.

You can also pass the token per command (not saved):

```bash
lwt transcribe meeting.wav --diarize --hf-token hf_xxx
```

---

## Ollama (translation, summarization & cleaning)

Whisper natively translates only to English (`--task translate`). For AI transcript cleaning, translation to any language, or meeting summarization, use local Ollama:

1. Install and start [Ollama](https://ollama.com)
2. Pull a model: `lwt ollama pull llama3.2`
3. Set it as default (or choose during `lwt setup`):

```bash
lwt config set ollama.model llama3.2
```

4. Transcribe with flags:

```bash
lwt transcribe meeting.mp4 --clean
lwt transcribe meeting.mp4 --translate-to Czech
lwt transcribe meeting.mp4 --summarize
lwt transcribe meeting.mp4 --clean --summarize
```

| Flag | What it does | Extra output |
|------|----------------|--------------|
| `--clean` | Remove fillers, fix obvious ASR errors | `*.raw.*` backup of original transcript |
| `--translate-to` | Translate full transcript | `*.<lang>.txt` |
| `--summarize` | Structured meeting notes | `*.summary.md` |

All three use the same Ollama model: `ollama.model` from config, unless you pass `--ollama-model`.

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

Required when using `--clean`, `--translate-to`, or `--summarize`:

```bash
lwt ollama status
# Start the Ollama app, then:
lwt ollama pull llama3.2
```

If you only need raw transcription, omit those flags:

```bash
lwt transcribe meeting.mp4
```

### Diarization failing (403 / gated repo)

**Symptom:** `Cannot access gated repo`, `403 Client Error`, or `you are not in the authorized list`.

**Fix:**

1. Make sure you accepted licenses for **all 4 models** (see [HuggingFace setup](#huggingface-setup-for-diarization)):
   - [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
   - **[speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)** (often the missing one)
2. Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Save it and verify:

```bash
lwt install diarization
lwt config set diarization.hf_token hf_YOUR_TOKEN
lwt install verify-diarization
```

4. Retry transcription.

If you only need transcription or summarization, omit `--diarize`:

```bash
lwt transcribe meeting.mp4 --summarize
```

### CUDA / `cublas64_12.dll` not found

Your GPU is detected, but CUDA 12 libraries are missing. Install them via `lwt`:

```bash
lwt install cuda
```

For the full NVIDIA CUDA Toolkit on Windows (optional):

```bash
lwt install cuda --toolkit
```

`lwt` will automatically fall back to CPU if GPU libraries are still missing.

To force CPU and skip GPU attempts:

```bash
lwt config set whisper.device cpu
```

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
├── postprocess.py    # Ollama cleaning, translation & summarization
├── models.py         # Whisper model management
├── config.py         # TOML configuration
├── setup_wizard.py   # interactive onboarding
├── ollama_ops.py     # Ollama API
└── install_extra.py  # optional dependency installation
```

---

## License

MIT
