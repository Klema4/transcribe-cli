# First-Time Setup

```bash
lwt setup
```

The wizard will:
1. Check Python, ffmpeg, and the available acceleration backend (MLX on Apple Silicon or CUDA on NVIDIA)
2. Show Whisper models (size, RAM)
3. Download selected model (skips if already cached)
4. Optionally configure diarization — [[HuggingFace Diarization]]
5. Optionally configure Ollama — [[Ollama]]

## Quick variants

```bash
lwt setup --quick
lwt setup --model small
lwt setup --model medium --force
```

## Model cache

Models download to:
- **Windows:** `%USERPROFILE%\.cache\huggingface\hub\`
- **Linux/macOS:** `~/.cache/huggingface/hub/`

If a model is already downloaded, setup asks **re-download?** with default **No**.

Next: [[Usage]]
