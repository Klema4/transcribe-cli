# Installation

## Requirements

| Requirement | Required | Notes |
|-------------|----------|-------|
| Python 3.10+ | yes | `python --version` |
| ffmpeg | yes | Video → audio extraction |
| NVIDIA GPU + CUDA | no | Faster transcription |
| Ollama | no | Translation & summarization |
| HuggingFace account | no | Only for `--diarize` — see [[HuggingFace Diarization]] |

## Install the package

```bash
git clone https://github.com/Klema4/local-whisper-transcribe.git
cd local-whisper-transcribe
pip install -e .
lwt --help
```

## Install ffmpeg

**Windows:**
```powershell
winget install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

## Windows: `lwt` not recognized

Python installs `lwt.exe` outside PATH on many setups.

**Quick fix:**
```powershell
python -m local_whisper_transcribe.cli setup
python -m local_whisper_transcribe.cli transcribe meeting.mp4
```

**Permanent fix:** Add Python Scripts to PATH:
```powershell
python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
$env:Path += ";C:\Users\YOU\AppData\Local\Python\pythoncore-3.14-64\Scripts"
```

Restart terminal after adding to system Environment Variables.

Next: [[First-Time Setup]]
