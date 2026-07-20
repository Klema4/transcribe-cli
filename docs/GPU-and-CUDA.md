# GPU and CUDA

## Install CUDA 12 runtime (recommended)

Fixes `cublas64_12.dll is not found`:

```bash
lwt install cuda
```

Installs `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` via pip.

Optional full toolkit (Windows):
```bash
lwt install cuda --toolkit
```

Restart terminal after install.

## Verify

```bash
lwt install check
```

Should show: `CUDA runtime ready (1 GPU)`

## CPU fallback

If GPU libraries are missing, `lwt` automatically retries on CPU with a warning.

Force CPU only:
```bash
lwt config set whisper.device cpu
```

## Model recommendations

| Model | Size | RAM | Notes |
|-------|------|-----|-------|
| tiny | ~75 MB | ~1 GB | Fastest |
| base | ~150 MB | ~1 GB | Basic |
| small | ~500 MB | ~2 GB | **Default** |
| medium | ~1.5 GB | ~4 GB | High quality |
| large-v3 | ~3 GB | ~10 GB | Best, use GPU |
