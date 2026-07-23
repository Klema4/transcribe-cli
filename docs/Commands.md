# Commands

## Transcription

```bash
lwt transcribe <file>
lwt t <file>                         # alias
```

| Flag | Description |
|------|-------------|
| `--language`, `-l` | Source language or auto-detect |
| `--prompt`, `-p` | Context prompt |
| `--model`, `-m` | tiny, base, small, medium, large-v3 |
| `--format`, `-f` | txt, srt, vtt, json |
| `--output`, `-o` | Output file |
| `--task` | transcribe (default) or translate (English only) |
| `--diarize` | Speaker identification — [[HuggingFace Diarization]] |
| `--no-diarize` | Disable diarization for this run, overriding the config |
| `--num-speakers` | Hint for diarization |
| `--speaker-names` | Comma-separated names |
| `--hf-token` | HuggingFace token |
| `--translate-to` | Translate via Ollama |
| `--clean` | Clean transcript via Ollama |
| `--summarize` | Meeting summary via Ollama |
| `--ollama-model` | Ollama model for post-processing |

## Clean existing transcript

```bash
lwt clean <transcript>
lwt clean meeting.txt -o meeting.clean.txt
```

## Setup

```bash
lwt setup
lwt onboard                            # alias
lwt setup --model medium
lwt setup --quick
lwt setup --model small --force
```

## Models

```bash
lwt models list
lwt models status
lwt models download small
lwt models download medium --force
```

## Install

```bash
lwt install cuda                     # CUDA 12 cuBLAS + cuDNN
lwt install cuda --toolkit           # + NVIDIA CUDA Toolkit (Windows)
lwt install diarization              # pyannote.audio
lwt install check
```

## Ollama

```bash
lwt ollama status
lwt ollama list
lwt ollama pull llama3.2
```

## Config

```bash
lwt config show
lwt config set whisper.model medium
lwt config set diarization.hf_token hf_xxx
lwt config path
lwt config reset
```

See [[Configuration]] for all keys.
