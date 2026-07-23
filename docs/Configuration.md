# Configuration

Config file location:
```bash
lwt config path
```

- **Windows:** `%USERPROFILE%\.config\transcribe-cli\config.toml`
- **Linux/macOS:** `~/.config/transcribe-cli/config.toml`

Legacy path `~/.config/local-whisper-transcribe/` is still loaded if present.

## Default

```toml
[whisper]
model = "small"
device = "auto"          # auto | cuda | cpu
compute_type = "auto"
cpu_threads = "auto"     # auto | integer (Apple Silicon auto => CPU-bound threads tuned for M1/M2/M3/M4)
num_workers = "auto"     # auto | integer (decode workers, auto keeps default)
beam_size = 5            # decoding beam, lower = faster
condition_on_previous_text = true
vad_filter = true

[defaults]
language = "auto"
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

## Common settings

```bash
lwt config set whisper.model medium
lwt config set whisper.device cpu
lwt config set defaults.language cs
lwt config set diarization.hf_token hf_xxx
lwt config set ollama.model llama3.2
lwt config reset
```

Use `lwt config set` — no manual file editing needed.
