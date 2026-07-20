"""Interactive onboarding wizard for lwt setup."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from local_whisper_transcribe.prompts import ask_confirm, ask_prompt

from local_whisper_transcribe.config import get_config_path, get_hf_token, load_config, save_config
from local_whisper_transcribe.diarize import HF_MODEL_URL
from local_whisper_transcribe.install_extra import install_diarization, is_diarization_installed
from local_whisper_transcribe.ollama_ops import get_ollama_status, pull_ollama_model
from local_whisper_transcribe.cuda_runtime import (
    check_cuda_runtime,
    install_cuda_runtime,
    is_cuda_runtime_installed,
)
from local_whisper_transcribe.system_checks import check_python, get_ffmpeg_install_command
from local_whisper_transcribe.models import (
    KNOWN_MODELS,
    MODEL_INFO,
    download_model,
    format_size,
    get_model_status,
    is_model_cached,
)


def run_setup_wizard(
    console: Console,
    *,
    model: str | None = None,
    force: bool = False,
) -> None:
    """Run the full interactive setup wizard."""
    console.print("[bold]Welcome to the lwt setup wizard[/bold]")
    console.print("This wizard checks dependencies and saves your configuration.\n")

    config = load_config()

    # Python
    py_ok, py_version = check_python()
    if py_ok:
        console.print(f"[green]OK[/green] Python {py_version}")
    else:
        console.print(f"[red]FAIL[/red] Python {py_version} — Python 3.10+ required")
        raise typer.Exit(1)

    # ffmpeg
    from local_whisper_transcribe.audio import FFmpegNotFoundError, check_ffmpeg

    try:
        ffmpeg_path = check_ffmpeg()
        console.print(f"[green]OK[/green] ffmpeg found: [cyan]{ffmpeg_path}[/cyan]")
    except FFmpegNotFoundError:
        install_cmd = get_ffmpeg_install_command()
        console.print("[red]FAIL[/red] ffmpeg is not installed or not on PATH.")
        console.print(f"  Install with: [bold cyan]{install_cmd}[/bold cyan]")
        if not ask_confirm(
            "Continue without ffmpeg? (video files will not work)",
            default=False,
            console=console,
        ):
            raise typer.Exit(1)

    # CUDA
    cuda_ok, cuda_detail = check_cuda_runtime()
    if cuda_ok:
        console.print(f"[green]OK[/green] {cuda_detail}")
    else:
        console.print(f"[yellow]![/yellow] {cuda_detail}")
        if "lwt install cuda" in cuda_detail and ask_confirm(
            "Install CUDA 12 runtime libraries now? (lwt install cuda)",
            default=True,
            console=console,
        ):
            with console.status("[bold green]Installing CUDA 12 runtime..."):
                code = install_cuda_runtime()
            if code == 0:
                console.print("[green]OK[/green] CUDA 12 runtime installed")
                runtime_ok, runtime_detail = check_cuda_runtime()
                if runtime_ok:
                    console.print(f"[green]OK[/green] {runtime_detail}")
            else:
                console.print("[red]Installation failed.[/red] Try: [cyan]lwt install cuda[/cyan]")

    # Whisper model
    console.print("\n[bold]Whisper model[/bold]")
    table = Table(show_header=True, header_style="bold green")
    table.add_column("Model", style="cyan")
    table.add_column("Size")
    table.add_column("Notes")
    for name in KNOWN_MODELS:
        size, _ram, notes = MODEL_INFO.get(name, ("?", "?", ""))
        table.add_row(name, size, notes)
    console.print(table)

    default_model = model or config["whisper"]["model"]
    if model:
        selected = model
    else:
        selected = ask_prompt("Choose a model", default=default_model, console=console)
    config["whisper"]["model"] = selected

    device = config["whisper"]["device"]
    compute_type = config["whisper"]["compute_type"]
    already_cached = is_model_cached(selected)

    if already_cached and not force:
        status = get_model_status(selected)
        console.print(
            f"[green]OK[/green] Model [cyan]{selected}[/cyan] is already downloaded "
            f"([dim]{format_size(status.size_bytes)}[/dim])"
        )
        should_download = ask_confirm(
            f"Re-download model [cyan]{selected}[/cyan]?",
            default=False,
            console=console,
        )
        download_force = should_download
    elif model:
        should_download = True
        download_force = force
    else:
        should_download = ask_confirm(
            f"Download model [cyan]{selected}[/cyan] now?",
            default=not already_cached,
            console=console,
        )
        download_force = force

    if should_download:
        messages: list[str] = []

        def on_status(message: str) -> None:
            messages.append(message)

        status_label = (
            f"Verifying model {selected}..."
            if already_cached and not download_force
            else f"Downloading model {selected}..."
        )
        with console.status(f"[bold green]{status_label}"):
            download_model(
                selected,
                device=device,
                compute_type=compute_type,
                force=download_force,
                status_callback=on_status,
            )
        for message in messages:
            console.print(f"  [dim]•[/dim] {message}")
        console.print(f"[green]OK[/green] Model [cyan]{selected}[/cyan] ready")
    elif already_cached:
        console.print(f"[green]OK[/green] Model [cyan]{selected}[/cyan] ready")

    language = ask_prompt(
        "Default language (auto = automatic detection)",
        default=config["defaults"]["language"],
        console=console,
    )
    config["defaults"]["language"] = language

    out_fmt = ask_prompt(
        "Default output format (txt/srt/vtt/json)",
        default=config["defaults"]["format"],
        console=console,
    )
    config["defaults"]["format"] = out_fmt

    # Diarization
    console.print("\n[bold]Optional: Speaker diarization[/bold]")
    console.print(
        f"Requires accepting the license at [link={HF_MODEL_URL}]{HF_MODEL_URL}[/link]\n"
        "and a HuggingFace token."
    )
    enable_diarization = ask_confirm(
        "Set up speaker diarization?",
        default=config["diarization"]["enabled"],
        console=console,
    )
    config["diarization"]["enabled"] = enable_diarization

    if enable_diarization:
        if not is_diarization_installed():
            console.print("pyannote.audio is not installed.")
            if ask_confirm("Install now? (lwt install diarization)", default=True, console=console):
                console.print("[bold]Installing pyannote.audio...[/bold]")
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
                )
                with progress:
                    task_id = progress.add_task("pip install pyannote.audio", total=None)

                    def on_line(line: str) -> None:
                        progress.update(task_id, description=line[:60])

                    code = install_diarization(on_output=on_line)
                if code != 0:
                    console.print(
                        "[red]Installation failed.[/red] Try: [cyan]lwt install diarization[/cyan]"
                    )
                else:
                    console.print("[green]OK[/green] pyannote.audio installed")

        existing_token = get_hf_token(config) or ""
        token_default = existing_token
        token = ask_prompt(
            "HuggingFace token (leave empty to skip)",
            default=token_default,
            password=bool(token_default),
            console=console,
        )
        config["diarization"]["hf_token"] = token
        if token:
            console.print("[green]OK[/green] HF token saved to configuration")
        else:
            console.print(
                "[yellow]![/yellow] Token not saved. Set it later:\n"
                "  [cyan]lwt config set diarization.hf_token <token>[/cyan]"
            )

    # Ollama
    console.print("\n[bold]Optional: Ollama (translation and summarization)[/bold]")
    ollama_url = config["ollama"]["url"]
    status = get_ollama_status(ollama_url)
    if status.available:
        console.print(f"[green]OK[/green] Ollama is running at {ollama_url}")
        if status.models:
            console.print(f"  Installed models: {', '.join(status.models)}")
        else:
            console.print("  [yellow]No models found[/yellow]")
    else:
        console.print(f"[yellow]![/yellow] Ollama is not available at {ollama_url}")
        console.print("  Start Ollama or check: [cyan]lwt ollama status[/cyan]")

    if status.available and ask_confirm("Set default Ollama model?", default=True, console=console):
        default_ollama = config["ollama"]["model"]
        ollama_model = ask_prompt("Ollama model", default=default_ollama, console=console)
        config["ollama"]["model"] = ollama_model

        if ollama_model not in status.models:
            if ask_confirm(
                f"Download model [cyan]{ollama_model}[/cyan]?",
                default=True,
                console=console,
            ):
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=console,
                )
                with progress:
                    task_id = progress.add_task(f"Downloading {ollama_model}...", total=100)

                    def on_pull(msg: str, percent: int | None) -> None:
                        if percent is not None:
                            progress.update(task_id, completed=percent, description=msg)
                        else:
                            progress.update(task_id, description=msg)

                    try:
                        pull_ollama_model(ollama_model, ollama_url, on_progress=on_pull)
                        progress.update(task_id, completed=100)
                        console.print(f"[green]OK[/green] Model [cyan]{ollama_model}[/cyan] downloaded")
                    except Exception as exc:
                        console.print(f"[red]Download failed:[/red] {exc}")
                        console.print(f"Try: [cyan]lwt ollama pull {ollama_model}[/cyan]")

    config.setdefault("meta", {})["setup_complete"] = True
    save_config(config)

    console.print("\n[bold green]Setup complete![/bold green]")
    console.print(f"Configuration: [cyan]{get_config_path()}[/cyan]")
    console.print("\nExamples:")
    console.print("  [bold cyan]lwt transcribe meeting.mp4[/bold cyan]")
    console.print("  [bold cyan]lwt transcribe meeting.mp4 --diarize --summarize[/bold cyan]")
