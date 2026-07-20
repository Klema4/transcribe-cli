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
    console.print("[bold]Vítejte v průvodci nastavením lwt[/bold]")
    console.print("Tento průvodce zkontroluje závislosti a uloží konfiguraci.\n")

    config = load_config()

    # Python
    py_ok, py_version = check_python()
    if py_ok:
        console.print(f"[green]✓[/green] Python {py_version}")
    else:
        console.print(f"[red]✗[/red] Python {py_version} — vyžadováno Python 3.10+")
        raise typer.Exit(1)

    # ffmpeg
    from local_whisper_transcribe.audio import FFmpegNotFoundError, check_ffmpeg

    try:
        ffmpeg_path = check_ffmpeg()
        console.print(f"[green]✓[/green] ffmpeg nalezen: [cyan]{ffmpeg_path}[/cyan]")
    except FFmpegNotFoundError:
        install_cmd = get_ffmpeg_install_command()
        console.print("[red]✗[/red] ffmpeg není nainstalován nebo není v PATH.")
        console.print(f"  Nainstalujte příkazem: [bold cyan]{install_cmd}[/bold cyan]")
        if not ask_confirm(
            "Pokračovat bez ffmpeg? (video soubory nebudou fungovat)",
            default=False,
            console=console,
        ):
            raise typer.Exit(1)

    # CUDA
    cuda_ok, cuda_detail = check_cuda_runtime()
    if cuda_ok:
        console.print(f"[green]✓[/green] {cuda_detail}")
    else:
        console.print(f"[yellow]![/yellow] {cuda_detail}")
        if "lwt install cuda" in cuda_detail and ask_confirm(
            "Nainstalovat CUDA 12 runtime knihovny nyní? (lwt install cuda)",
            default=True,
            console=console,
        ):
            with console.status("[bold green]Instaluji CUDA 12 runtime..."):
                code = install_cuda_runtime()
            if code == 0:
                console.print("[green]✓[/green] CUDA 12 runtime nainstalován")
                runtime_ok, runtime_detail = check_cuda_runtime()
                if runtime_ok:
                    console.print(f"[green]✓[/green] {runtime_detail}")
            else:
                console.print("[red]Instalace selhala.[/red] Zkuste: [cyan]lwt install cuda[/cyan]")

    # Whisper model
    console.print("\n[bold]Whisper model[/bold]")
    table = Table(show_header=True, header_style="bold green")
    table.add_column("Model", style="cyan")
    table.add_column("Velikost")
    table.add_column("Poznámka")
    for name in KNOWN_MODELS:
        size, _ram, notes = MODEL_INFO.get(name, ("?", "?", ""))
        table.add_row(name, size, notes)
    console.print(table)

    default_model = model or config["whisper"]["model"]
    if model:
        selected = model
    else:
        selected = ask_prompt("Vyberte model", default=default_model, console=console)
    config["whisper"]["model"] = selected

    device = config["whisper"]["device"]
    compute_type = config["whisper"]["compute_type"]
    already_cached = is_model_cached(selected)

    if already_cached and not force:
        status = get_model_status(selected)
        console.print(
            f"[green]✓[/green] Model [cyan]{selected}[/cyan] už je stažený "
            f"([dim]{format_size(status.size_bytes)}[/dim])"
        )
        should_download = ask_confirm(
            f"Přestáhnout model [cyan]{selected}[/cyan]?",
            default=False,
            console=console,
        )
        download_force = should_download
    elif model:
        should_download = True
        download_force = force
    else:
        should_download = ask_confirm(
            f"Stáhnout model [cyan]{selected}[/cyan] nyní?",
            default=not already_cached,
            console=console,
        )
        download_force = force

    if should_download:
        messages: list[str] = []

        def on_status(message: str) -> None:
            messages.append(message)

        status_label = (
            f"Ověřuji model {selected}..."
            if already_cached and not download_force
            else f"Stahuji model {selected}..."
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
        console.print(f"[green]✓[/green] Model [cyan]{selected}[/cyan] připraven")
    elif already_cached:
        console.print(f"[green]✓[/green] Model [cyan]{selected}[/cyan] připraven")

    language = ask_prompt(
        "Výchozí jazyk (auto = automatická detekce)",
        default=config["defaults"]["language"],
        console=console,
    )
    config["defaults"]["language"] = language

    out_fmt = ask_prompt(
        "Výchozí výstupní formát (txt/srt/vtt/json)",
        default=config["defaults"]["format"],
        console=console,
    )
    config["defaults"]["format"] = out_fmt

    # Diarization
    console.print("\n[bold]Volitelně: Rozpoznání mluvčích (diarization)[/bold]")
    console.print(
        f"Vyžaduje licenci na [link={HF_MODEL_URL}]{HF_MODEL_URL}[/link]\n"
        "a HuggingFace token."
    )
    enable_diarization = ask_confirm(
        "Nastavit rozpoznání mluvčích?",
        default=config["diarization"]["enabled"],
        console=console,
    )
    config["diarization"]["enabled"] = enable_diarization

    if enable_diarization:
        if not is_diarization_installed():
            console.print("Balíček pyannote.audio není nainstalován.")
            if ask_confirm("Nainstalovat nyní? (lwt install diarization)", default=True, console=console):
                console.print("[bold]Instaluji pyannote.audio...[/bold]")
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
                    console.print("[red]Instalace selhala.[/red] Zkuste: [cyan]lwt install diarization[/cyan]")
                else:
                    console.print("[green]✓[/green] pyannote.audio nainstalován")

        existing_token = get_hf_token(config) or ""
        token_default = existing_token
        token = ask_prompt(
            "HuggingFace token (nechte prázdné pro přeskočení)",
            default=token_default,
            password=bool(token_default),
            console=console,
        )
        config["diarization"]["hf_token"] = token
        if token:
            console.print("[green]✓[/green] HF token uložen do konfigurace")
        else:
            console.print(
                "[yellow]![/yellow] Token neuložen. Nastavte později:\n"
                "  [cyan]lwt config set diarization.hf_token <token>[/cyan]"
            )

    # Ollama
    console.print("\n[bold]Volitelně: Ollama (překlad a shrnutí)[/bold]")
    ollama_url = config["ollama"]["url"]
    status = get_ollama_status(ollama_url)
    if status.available:
        console.print(f"[green]✓[/green] Ollama běží na {ollama_url}")
        if status.models:
            console.print(f"  Nainstalované modely: {', '.join(status.models)}")
        else:
            console.print("  [yellow]Žádné modely nenalezeny[/yellow]")
    else:
        console.print(f"[yellow]![/yellow] Ollama není dostupná na {ollama_url}")
        console.print("  Spusťte Ollama nebo zkontrolujte: [cyan]lwt ollama status[/cyan]")

    if status.available and ask_confirm("Nastavit výchozí Ollama model?", default=True, console=console):
        default_ollama = config["ollama"]["model"]
        ollama_model = ask_prompt("Ollama model", default=default_ollama, console=console)
        config["ollama"]["model"] = ollama_model

        if ollama_model not in status.models:
            if ask_confirm(
                f"Stáhnout model [cyan]{ollama_model}[/cyan]?",
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
                    task_id = progress.add_task(f"Stahuji {ollama_model}...", total=100)

                    def on_pull(msg: str, percent: int | None) -> None:
                        if percent is not None:
                            progress.update(task_id, completed=percent, description=msg)
                        else:
                            progress.update(task_id, description=msg)

                    try:
                        pull_ollama_model(ollama_model, ollama_url, on_progress=on_pull)
                        progress.update(task_id, completed=100)
                        console.print(f"[green]✓[/green] Model [cyan]{ollama_model}[/cyan] stažen")
                    except Exception as exc:
                        console.print(f"[red]Stažení selhalo:[/red] {exc}")
                        console.print(f"Zkuste: [cyan]lwt ollama pull {ollama_model}[/cyan]")

    config.setdefault("meta", {})["setup_complete"] = True
    save_config(config)

    console.print("\n[bold green]✓ Nastavení dokončeno![/bold green]")
    console.print(f"Konfigurace: [cyan]{get_config_path()}[/cyan]")
    console.print("\nPříklad použití:")
    console.print("  [bold cyan]lwt transcribe meeting.mp4[/bold cyan]")
    console.print("  [bold cyan]lwt transcribe schuzka.mp4 --diarize --summarize[/bold cyan]")
