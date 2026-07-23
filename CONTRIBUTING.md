# Contributing to Transcribe CLI

Thanks for helping improve Transcribe CLI. Contributions of all sizes are welcome — bug reports, docs fixes, feature ideas, and pull requests.

## Ways to contribute

- **Report bugs** — [open an issue](https://github.com/Klema4/transcribe-cli/issues/new)
- **Suggest features** — describe the problem and your idea in an issue first
- **Improve docs** — README, `docs/`, or CLI help text
- **Fix bugs / add features** — open a pull request

If you are unsure where to start, look for issues labeled `good first issue` or `help wanted`.

## Before you start

1. Search [existing issues](https://github.com/Klema4/transcribe-cli/issues) and [pull requests](https://github.com/Klema4/transcribe-cli/pulls) to avoid duplicates.
2. For larger changes, open an issue first so we can align on the approach.
3. Keep changes focused. Prefer small, reviewable PRs over large mixed ones.

## Development setup

```bash
git clone https://github.com/Klema4/transcribe-cli.git
cd transcribe-cli
pip install -e ".[dev]"
lwt --help
pytest
```

Requirements: **Python 3.10+** and **ffmpeg**.

Notes:

- Product name: **Transcribe CLI**
- CLI command: **`lwt`**
- Python package path: `local_whisper_transcribe` (keep this import path unless a rename is explicitly planned)

More detail: [docs/Development.md](docs/Development.md)

## Issues

### Bug reports

Please include:

- Transcribe CLI / `lwt` version (`lwt --version` if available)
- OS (Windows / macOS / Linux)
- Python version
- Exact command you ran
- What you expected vs what happened
- Full error output / traceback
- Whether CUDA, Ollama, or diarization was involved

### Feature requests

Please include:

- The problem you are trying to solve
- Why the current CLI is not enough
- A concrete example of the desired command / behavior

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Make your change with a clear purpose.
3. Add or update tests when behavior changes.
4. Update docs/README if user-facing behavior changes.
5. Run the test suite:

```bash
pytest
```

6. Open a PR against `main` with:
   - a short summary of **why** the change is needed
   - what you changed
   - how you tested it
   - linked issue number, if any (`Fixes #123`)

### PR guidelines

- Match the existing code style and keep diffs small.
- Prefer clear names and straightforward logic over clever abstractions.
- Do not commit secrets, tokens, or large binary/model files.
- User-facing CLI strings should stay in English.
- Branding should say **Transcribe CLI** / `lwt` in docs and help text.

## Code of conduct

Be respectful and constructive. Assume good intent. Focus feedback on the code and the problem, not the person.

## Questions?

Open an issue and tag it as a question, or comment on a related issue/PR. Maintainers and other contributors will help when they can.

Thanks again for contributing.
