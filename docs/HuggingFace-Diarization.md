# HuggingFace Diarization

Diarization (`--diarize`) identifies **who said what**. It uses [pyannote.audio](https://github.com/pyannote/pyannote-audio) and downloads models from HuggingFace on first use.

**Regular transcription works without this.** You only need HuggingFace if you use `--diarize`.

Without setup you get: `403 Client Error`, `gated repo`, or `not in the authorized list`.

---

## Step 1 — Create account

Sign up free: [huggingface.co/join](https://huggingface.co/join)

## Step 2 — Accept ALL 4 model licenses

Log in, open each link, click **Agree and access repository**:

| Model | Link |
|-------|------|
| Diarization pipeline | https://huggingface.co/pyannote/speaker-diarization-3.1 |
| Segmentation | https://huggingface.co/pyannote/segmentation-3.0 |
| Speaker embedding | https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM |
| **Community PLDA (pyannote 4.x)** | https://huggingface.co/pyannote/speaker-diarization-community-1 |

**All four are required** with pyannote.audio 4.x. Missing one (especially `community-1`) = access error.

Verify before transcribing:

```bash
lwt install verify-diarization
```

## Step 3 — Create access token

1. https://huggingface.co/settings/tokens
2. **Create new token**
3. Name: `lwt-diarization`
4. Type: **Read**
5. Copy token (`hf_...`)

## Step 4 — Save in lwt

```bash
lwt config set diarization.hf_token hf_your_token_here
lwt config show
```

Or enter during `lwt setup`.

## Step 5 — Install & run

```bash
lwt install diarization
lwt transcribe meeting.wav --diarize
lwt transcribe meeting.wav --diarize --speaker-names "Alice,Bob"
```

Models cache locally after first download — then diarization runs offline.

## Per-command token (not saved)

```bash
lwt transcribe meeting.wav --diarize --hf-token hf_xxx
```

## Troubleshooting

See [[Troubleshooting#Diarization 403]]
