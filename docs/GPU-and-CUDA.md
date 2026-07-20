# GPU and CUDA

Whisper (faster-whisper / CTranslate2) and diarization (pyannote / PyTorch) share **one** device per job: both GPU or both CPU — never mixed.

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
| Transcribe only | GPU if CTranslate2 sees CUDA |
| Transcribe + `--diarize` | GPU only if **both** CTranslate2 and PyTorch see CUDA; otherwise **CPU for both** |

Force CPU:

```bash
lwt config set whisper.device cpu
```
