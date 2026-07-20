# Troubleshooting

## `lwt` not found (Windows)

```powershell
python -m local_whisper_transcribe.cli transcribe meeting.mp4
```

Add Python Scripts to PATH — see [[Installation]].

## ffmpeg not detected

Restart terminal after `winget install ffmpeg`.

```powershell
$env:Path += ";$env:LOCALAPPDATA\Microsoft\WinGet\Links"
ffmpeg -version
```

`lwt` also auto-detects winget ffmpeg paths.

## Model not downloaded

```bash
lwt setup
lwt models download small
```

## Diarization 403

**Symptoms:** `403`, `gated repo`, `not authorized`

**Fix:**
1. Accept licenses for **all 3 models** — [[HuggingFace Diarization]]
2. Create Read token at https://huggingface.co/settings/tokens
3. `lwt config set diarization.hf_token hf_YOUR_TOKEN`
4. Retry

Skip diarization:
```bash
lwt transcribe meeting.mp4 --summarize
```

## CUDA / cublas64_12.dll

```bash
lwt install cuda
```

Or force CPU:
```bash
lwt config set whisper.device cpu
```

See [[GPU and CUDA]].

## Ollama not running

```bash
lwt ollama status
lwt ollama pull llama3.2
```

## Slow transcription

- Use smaller model: `lwt config set whisper.model small`
- Install GPU libs: `lwt install cuda`

## Re-running after diarization failure

Whisper runs from scratch each time. To test diarization quickly:
```bash
lwt transcribe meeting.mp4 --diarize --model small
```

Use `large-v3` only after HF setup is verified.
