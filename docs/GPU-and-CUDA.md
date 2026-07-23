# GPU and CUDA

Whisper uses faster-whisper/CUDA on NVIDIA and MLX/Metal on Apple Silicon. Diarization uses pyannote/PyTorch; when diarization is enabled on Apple Silicon, Whisper and diarization intentionally use CPU because MLX and pyannote cannot share this device plan.

## Install CUDA stack (recommended)

```bash
lwt install cuda
```

This installs:

1. **CUDA 12 runtime** (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) — Whisper GPU
2. **CUDA PyTorch** (`torch+cu126`) — diarization neural nets on GPU
3. **CPU torchaudio** — resample/fbank helpers (CUDA torchaudio is often blocked on Windows)

Optional full toolkit (Windows):

```bash
lwt install cuda --toolkit
```

## Check

```bash
lwt install check
```

Look for CUDA ready for Whisper + diarization.

## Behaviour

| Mode | Device |
|------|--------|
| Transcribe only | NVIDIA CUDA, Apple MLX/Metal, or CPU fallback |
| Transcribe + `--diarize` | GPU only if **both** CTranslate2 and PyTorch see CUDA; otherwise **CPU for both** |

Force CPU:

```bash
lwt config set whisper.device cpu
```

On a native arm64 Python installation, `device = "auto"` selects MLX automatically on Apple Silicon. Force it with:

```bash
lwt config set whisper.device mlx
```
