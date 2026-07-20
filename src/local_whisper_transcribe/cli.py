"""Typer CLI entry point with Rich UI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from local_whisper_transcribe.prompts import ask_confirm, ask_prompt

from local_whisper_transcribe import __version__
from local_whisper_transcribe.audio import FFmpegNotFoundError, check_ffmpeg, prepare_audio
from local_whisper_transcribe.config import (
    get_config_path,
    get_hf_token,
    is_setup_complete,
    load_config,
    mask_config_value,
    reset_config,
    save_config,
    set_config_value,
)
from local_whisper_transcribe.diarize import (
    HF_MODEL_URL,
    HF_MODELS_TO_ACCEPT,
    DiarizationAccessError,
    DiarizationNotInstalledError,
    DiarizationTokenError,
    apply_speaker_names,
    diarize_audio,
    merge_transcription_with_diarization,
    verify_diarization_access,
)
from local_whisper_transcribe.install_extra import (
    check_all_dependencies,
    install_diarization,
    is_diarization_installed,
)
from local_whisper_transcribe.cuda_runtime import (
    check_cuda_runtime,
    install_cuda_runtime,
    install_cuda_toolkit,
    install_torch_cuda,
    is_cuda_runtime_installed,
    is_torch_cuda_installed,
)
from local_whisper_transcribe.device import resolve_device
from local_whisper_transcribe.models import (
    MODEL_INFO,
    check_cuda_available,
    check_python_version,
    download_model,
    format_size,
    get_cache_dir,
    is_model_cached,
    list_model_statuses,
)
from local_whisper_transcribe.ollama_ops import get_ollama_status, list_ollama_models, pull_ollama_model
from local_whisper_transcribe.output import default_output_path, export_result, format_txt
from local_whisper_transcribe.postprocess import (
    check_ollama_available,
    summarize_meeting,
    translate_text,
)
from local_whisper_transcribe.setup_wizard import run_setup_wizard
from local_whisper_transcribe.transcribe import KNOWN_MODELS, load_model, transcribe

app = typer.Typer(
    name="lwt",
    help="Local Whisper Transcribe — offline meeting transcription.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
models_app = typer.Typer(help="Whisper model management.")
config_app = typer.Typer(help="Configuration management.")
install_app = typer.Typer(help="Install optional dependencies.")
ollama_app = typer.Typer(help="Manage Ollama models and status.")
app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")
app.add_typer(install_app, name="install")
app.add_typer(ollama_app, name="ollama")

console = Console(legacy_windows=False)

BANNER = r"""
 _               _ _            __        __           _             
| |   _ __ _   _| | | __ _ _ __\ \      / /___ _ __ __| | ___ _ __   
| |  | '__| | | | | |/ _` | '_ \\ \ /\ / // _ \ '__/ _` |/ _ \ '__|  
| |__| |  | |_| | | | (_| | | | |\ V  V /|  __/ | | (_| |  __/ |     
|_____|_|   \__,_|_|_|\__,_|_| |_| \_/\_/  \___|_|  \__,_|\___|_|     
  Transcribe meetings locally with faster-whisper
"""


def _print_banner() -> None:
    console.print(Text(BANNER, style="bold cyan"))


def _resolve_output_path(
    input_path: Path,
    output: Path | None,
    output_dir: str,
    fmt: str,
) -> Path:
    if output:
        return output
    dir_path = Path(output_dir) if output_dir else None
    return default_output_path(input_path, dir_path, fmt)


def _models_table(*, show_ram: bool = False, show_cached: bool = False) -> Table:
    table = Table(title="Whisper Models", show_header=True, header_style="bold green")
    table.add_column("Model", style="cyan")
    table.add_column("Size")
    if show_ram:
        table.add_column("RAM")
    table.add_column("Notes")
    if show_cached:
        table.add_column("Cached")

    for name in KNOWN_MODELS:
        size, ram, notes = MODEL_INFO.get(name, ("?", "?", ""))
        style = "bold" if name == "small" else ""
        row = [name, size]
        if show_ram:
            row.append(ram)
        row.append(notes)
        if show_cached:
            row.append("[green]yes[/green]" if is_model_cached(name) else "no")
        table.add_row(*row, style=style)
    return table


def _check_dependencies() -> list[tuple[str, bool, str]]:
    py_ok, py_detail = check_python_version()
    checks: list[tuple[str, bool, str]] = [("Python", py_ok, py_detail)]

    try:
        ffmpeg_path = check_ffmpeg()
        checks.append(("ffmpeg", True, ffmpeg_path))
    except FFmpegNotFoundError as exc:
        checks.append(("ffmpeg", False, str(exc).split("\n")[0]))

    cuda_ok, cuda_detail = check_cuda_available()
    checks.append(("CUDA (optional)", cuda_ok, cuda_detail))
    runtime_ok, runtime_detail = check_cuda_runtime()
    checks.append(("CUDA 12 runtime", runtime_ok, runtime_detail))
    return checks


def _run_quick_setup(
    model: str | None = None,
    *,
    force: bool = False,
    interactive: bool = True,
) -> str:
    """Quick setup: dependencies + model download only."""
    _print_banner()
    console.print(Panel("Quick setup — model download", style="bold green"))

    checks = _check_dependencies()
    dep_table = Table(title="System Dependencies", show_header=True, header_style="bold blue")
    dep_table.add_column("Component")
    dep_table.add_column("Status")
    dep_table.add_column("Details")
    for name, ok, detail in checks:
        status = "[green]OK[/green]" if ok else "[yellow]Missing[/yellow]"
        if name == "CUDA (optional)":
            status = "[green]OK[/green]" if ok else "[dim]Not available[/dim]"
        dep_table.add_row(name, status, detail)
    console.print(dep_table)

    if not checks[0][1]:
        console.print("[red]Error:[/red] Python 3.10+ is required.")
        raise typer.Exit(1)

    console.print()
    console.print(_models_table(show_ram=True))

    selected = model
    if not selected:
        if interactive:
            selected = Prompt.ask(
                "Choose a model",
                choices=list(KNOWN_MODELS),
                default="small",
                console=console,
            )
        else:
            selected = "small"

    if selected not in KNOWN_MODELS:
        console.print(f"[red]Error:[/red] Unknown model '{selected}'.")
        raise typer.Exit(1)

    config = load_config()
    device = config["whisper"]["device"]
    compute_type = config["whisper"]["compute_type"]
    messages: list[str] = []

    with console.status("[bold green]Preparing model download..."):
        try:
            model_path = download_model(
                selected,
                device=device,
                compute_type=compute_type,
                force=force,
                status_callback=messages.append,
            )
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc

    for message in messages:
        console.print(f"  [dim]•[/dim] {message}")

    config["whisper"]["model"] = selected
    config.setdefault("meta", {})["setup_complete"] = True
    save_config(config)

    summary = Table(title="Ready to transcribe!", show_header=False, box=None)
    summary.add_row("Model", f"[cyan]{selected}[/cyan]")
    summary.add_row("Path", str(model_path))
    summary.add_row("Cache", str(get_cache_dir()))
    summary.add_row("Config", str(get_config_path()))
    console.print()
    console.print(Panel(summary, border_style="green"))
    console.print("\nRun [bold]lwt transcribe <file>[/bold] to transcribe your first meeting.")
    console.print("For full setup (diarization, Ollama): [bold]lwt setup[/bold]")
    return selected


def _ensure_model_ready(model_name: str) -> None:
    """Prompt for setup when model cache is missing."""
    if is_model_cached(model_name):
        return

    console.print()
    console.print(
        "[yellow]Model is not downloaded locally.[/yellow] "
        "Run [bold]lwt setup[/bold] or [bold]lwt models download "
        f"{model_name}[/bold]."
    )

    if ask_confirm("Download model now?", default=True, console=console):
        config = load_config()
        download_model(
            model_name,
            device=config["whisper"]["device"],
            compute_type=config["whisper"]["compute_type"],
        )
        return

    raise typer.Exit(1)


def _print_diarization_hf_check(hf_token: str) -> bool:
    """Verify HuggingFace access before transcription. Returns True if all models OK."""
    results = verify_diarization_access(hf_token)
    failed = [(model_id, detail) for model_id, ok, detail in results if not ok]
    if not failed:
        return True

    table = Table(title="HuggingFace diarization access", show_header=True)
    table.add_column("Model")
    table.add_column("Status")
    for model_id, ok, detail in results:
        status = "[green]OK[/green]" if ok else f"[red]{detail}[/red]"
        table.add_row(model_id, status)
    console.print(table)
    console.print(
        "\n[red]Diarization will fail[/red] until you accept licenses for ALL models "
        "(including [bold]speaker-diarization-community-1[/bold] — required by pyannote 4.x).\n"
        "Links:\n"
        + "\n".join(f"  [cyan]https://huggingface.co/{m}[/cyan]" for m in HF_MODELS_TO_ACCEPT)
        + "\n\nThen run: [cyan]lwt install verify-diarization[/cyan]"
    )
    return False


def _resolve_hf_token(
    config: dict,
    cli_token: str | None,
    *,
    interactive: bool = True,
) -> str | None:
    """Resolve HF token, prompting interactively if needed."""
    token = get_hf_token(config, cli_token)
    if token:
        return token

    if not interactive:
        return None

    console.print(
        "\n[yellow]Speaker diarization requires a HuggingFace token.[/yellow]\n"
        f"Accept the license at {HF_MODEL_URL}"
    )
    if ask_confirm("Enter token now?", default=True, console=console):
        token = ask_prompt("HuggingFace token", password=True, console=console)
        if token:
            set_config_value("diarization.hf_token", token)
            console.print("[green]OK[/green] Token saved to configuration")
            return token
    return None


def _run_transcribe(
    input_file: Path,
    *,
    language: str | None,
    prompt: str | None,
    model: str | None,
    fmt: str,
    output: Path | None,
    translate_to: str | None,
    summarize: bool,
    ollama_model: str | None,
    task: str,
    diarize: bool = False,
    num_speakers: int | None = None,
    speaker_names: str | None = None,
    hf_token: str | None = None,
) -> None:
    _print_banner()
    config = load_config()

    model_name = model or config["whisper"]["model"]
    device = config["whisper"]["device"]
    compute_type = config["whisper"]["compute_type"]
    lang = language or config["defaults"]["language"]
    out_fmt = (fmt or config["defaults"]["format"]).lower()
    output_dir = config["defaults"]["output_dir"]
    ollama_url = config["ollama"]["url"]
    ollama_llm = ollama_model or config["ollama"]["model"]
    use_diarization = diarize or config["diarization"]["enabled"]

    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)

    if task not in ("transcribe", "translate"):
        console.print("[red]Error:[/red] --task must be 'transcribe' or 'translate'.")
        raise typer.Exit(1)

    if out_fmt not in ("txt", "srt", "vtt", "json"):
        console.print(f"[red]Error:[/red] Unsupported format: {out_fmt}")
        raise typer.Exit(1)

    needs_ollama = bool(translate_to) or summarize
    if needs_ollama and not check_ollama_available(ollama_url):
        console.print(
            f"[red]Error:[/red] Ollama is not available at {ollama_url}.\n"
            "Start Ollama or run: [bold cyan]lwt ollama status[/bold cyan]\n"
            "To download a model: [bold cyan]lwt ollama pull llama3.2[/bold cyan]"
        )
        raise typer.Exit(1)

    if use_diarization:
        resolved_hf_token = _resolve_hf_token(config, hf_token)
        if not resolved_hf_token:
            console.print(
                "[red]Error:[/red] Diarization requires a HuggingFace token.\n"
                "Set: [bold cyan]lwt config set diarization.hf_token <token>[/bold cyan]\n"
                "Or run: [bold cyan]lwt setup[/bold cyan]"
            )
            raise typer.Exit(1)
        if not is_diarization_installed():
            console.print(
                "[red]Error:[/red] pyannote.audio is not installed.\n"
                "Install: [bold cyan]lwt install diarization[/bold cyan]"
            )
            if ask_confirm("Install now?", default=True, console=console):
                with console.status("[bold green]Installing pyannote.audio..."):
                    code = install_diarization(on_output=lambda _: None)
                if code != 0:
                    raise typer.Exit(1)
                console.print("[green]OK[/green] pyannote.audio installed")
            else:
                raise typer.Exit(1)
        if not _print_diarization_hf_check(resolved_hf_token):
            raise typer.Exit(1)
    else:
        resolved_hf_token = None

    def on_device_note(message: str) -> None:
        console.print(f"[yellow]![/yellow] {message}")

    # If diarization wants GPU but PyTorch is CPU-only, offer to fix once.
    if use_diarization and device in ("auto", "cuda") and not is_torch_cuda_installed():
        from local_whisper_transcribe.device import whisper_cuda_available

        if whisper_cuda_available():
            console.print(
                "[yellow]Whisper can use GPU, but PyTorch is CPU-only "
                "(diarization would otherwise force both to CPU).[/yellow]"
            )
            if ask_confirm("Install CUDA PyTorch now? (lwt install cuda)", default=True, console=console):
                with console.status("[bold green]Installing CUDA PyTorch..."):
                    code = install_torch_cuda(on_output=lambda _: None)
                if code == 0 and is_torch_cuda_installed():
                    console.print("[green]OK[/green] CUDA PyTorch ready.")
                else:
                    console.print(
                        "[yellow]CUDA PyTorch install failed or not detected yet. "
                        "Using CPU for both.[/yellow]"
                    )

    device_plan = resolve_device(
        device,
        need_torch=use_diarization,
        on_note=on_device_note,
    )
    device = device_plan.device
    console.print(
        f"[dim]Compute device: [bold]{device}[/bold]"
        + (" (Whisper + diarization)" if use_diarization else " (Whisper)")
        + "[/dim]"
    )

    if not is_setup_complete() and not is_model_cached(model_name):
        console.print(
            "[yellow]First run?[/yellow] We recommend: [bold cyan]lwt setup[/bold cyan]"
        )
        if ask_confirm("Continue with quick model download?", default=True, console=console):
            _run_quick_setup(model=model_name, interactive=False)
        else:
            raise typer.Exit(1)
    else:
        _ensure_model_ready(model_name)

    try:
        with prepare_audio(input_file) as audio_path:
            def on_device_fallback(message: str) -> None:
                console.print(f"[yellow]![/yellow] {message}")

            with console.status("[bold green]Loading Whisper model..."):
                whisper_model = load_model(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    on_device_fallback=on_device_fallback,
                )

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            )

            result_holder: list = []

            def on_progress(current: float, total: float) -> None:
                if progress.tasks:
                    progress.update(progress.tasks[0].id, completed=current, total=total)

            with progress:
                task_id = progress.add_task("Transcribing...", total=100)
                result_holder.append(
                    transcribe(
                        audio_path,
                        model=whisper_model,
                        model_name=model_name,
                        device=device,
                        compute_type=compute_type,
                        language=lang,
                        task=task,
                        initial_prompt=prompt,
                        progress_callback=on_progress,
                        on_device_fallback=on_device_fallback,
                    )
                )
                progress.update(task_id, completed=100)

            result = result_holder[0]

            if use_diarization and resolved_hf_token:
                diarize_progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=console,
                )
                with diarize_progress:
                    diarize_task = diarize_progress.add_task("Diarization...", total=100)

                    def on_diarize_progress(
                        completed: float, total: float, description: str
                    ) -> None:
                        diarize_progress.update(
                            diarize_task,
                            completed=completed,
                            total=total,
                            description=f"Diarization: {description}",
                        )

                    diarization_segments = diarize_audio(
                        audio_path,
                        resolved_hf_token,
                        device=device,
                        num_speakers=num_speakers,
                        progress_callback=on_diarize_progress,
                    )

                result.segments = merge_transcription_with_diarization(
                    result.segments,
                    diarization_segments,
                )
                if speaker_names:
                    names = [name.strip() for name in speaker_names.split(",") if name.strip()]
                    result.segments = apply_speaker_names(result.segments, names)
                result.metadata["diarization"] = {
                    "enabled": True,
                    "num_speakers": len({seg.speaker for seg in result.segments if seg.speaker}),
                }

    except DiarizationNotInstalledError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except DiarizationTokenError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except DiarizationAccessError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except FFmpegNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    out_path = _resolve_output_path(input_file, output, output_dir, out_fmt)
    export_result(
        result,
        out_path,
        out_fmt,
        with_timestamps=(out_fmt == "txt"),
        source=str(input_file),
    )

    full_text = format_txt(result.segments, with_speakers=use_diarization)
    extras: dict[str, Path] = {}

    if translate_to:
        console.print(f"\n[bold]Translating to {translate_to} via Ollama...[/bold]")
        translation_path = out_path.with_suffix(f".{translate_to}.txt")
        live_text = Text()

        def on_chunk(chunk: str) -> None:
            live_text.append(chunk)

        with Live(Panel(live_text, title="Translation"), console=console, refresh_per_second=8):
            translated = translate_text(
                full_text,
                translate_to,
                model=ollama_llm,
                url=ollama_url,
                stream=True,
                on_chunk=on_chunk,
            )
        translation_path.write_text(translated, encoding="utf-8")
        extras["translation"] = translation_path

    if summarize:
        console.print("\n[bold]Generating meeting summary via Ollama...[/bold]")
        summary_path = out_path.with_suffix(".summary.md")
        live_text = Text()

        def on_summary_chunk(chunk: str) -> None:
            live_text.append(chunk)

        with Live(Panel(live_text, title="Summary"), console=console, refresh_per_second=8):
            summary = summarize_meeting(
                full_text,
                model=ollama_llm,
                url=ollama_url,
                stream=True,
                on_chunk=on_summary_chunk,
            )
        summary_path.write_text(summary, encoding="utf-8")
        extras["summary"] = summary_path

    table = Table(title="Transcription complete", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    table.add_row("Input", str(input_file))
    table.add_row("Output", str(out_path))
    table.add_row("Language", result.language)
    table.add_row("Duration", f"{result.duration:.1f}s")
    table.add_row("Segments", str(len(result.segments)))
    table.add_row("Model", model_name)
    if use_diarization:
        speakers = sorted({seg.speaker for seg in result.segments if seg.speaker})
        table.add_row("Speakers", ", ".join(speakers) if speakers else "unknown")
    for label, path in extras.items():
        table.add_row(label.capitalize(), str(path))
    console.print()
    console.print(table)

    if summarize and "summary" in extras:
        console.print()
        console.print(Panel(Markdown(extras["summary"].read_text(encoding="utf-8")), title="Summary preview"))


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        Optional[bool], typer.Option("--version", "-V", help="Show version and exit.")
    ] = None,
) -> None:
    if version:
        console.print(f"lwt version {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _print_banner()
        if not is_setup_complete():
            console.print(
                "[yellow]First run?[/yellow] Run [bold cyan]lwt setup[/bold cyan] "
                "to get started."
            )
        console.print("Run [bold]lwt --help[/bold] for available commands.")


@app.command("transcribe")
def transcribe_cmd(
    input_file: Annotated[Path, typer.Argument(help="Audio or video file to transcribe.")],
    language: Annotated[
        Optional[str], typer.Option("--language", "-l", help="Source language (auto-detect if omitted).")
    ] = None,
    prompt: Annotated[
        Optional[str], typer.Option("--prompt", "-p", help="Initial prompt for Whisper.")
    ] = None,
    model: Annotated[
        Optional[str], typer.Option("--model", "-m", help="Whisper model name or local path.")
    ] = None,
    fmt: Annotated[
        str, typer.Option("--format", "-f", help="Output format: txt, srt, vtt, json.")
    ] = "",
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Output file path.")
    ] = None,
    translate_to: Annotated[
        Optional[str], typer.Option("--translate-to", help="Translate transcript via Ollama.")
    ] = None,
    summarize: Annotated[
        bool, typer.Option("--summarize", help="Generate meeting summary via Ollama.")
    ] = False,
    ollama_model: Annotated[
        Optional[str], typer.Option("--ollama-model", help="Ollama model for post-processing.")
    ] = None,
    task: Annotated[
        str, typer.Option("--task", help="Whisper task: transcribe or translate (to English).")
    ] = "transcribe",
    diarize: Annotated[
        bool, typer.Option("--diarize", help="Identify speakers with pyannote.audio.")
    ] = False,
    num_speakers: Annotated[
        Optional[int], typer.Option("--num-speakers", help="Expected number of speakers.")
    ] = None,
    speaker_names: Annotated[
        Optional[str],
        typer.Option("--speaker-names", help='Rename speakers, e.g. "Jan,Petra,Martin".'),
    ] = None,
    hf_token: Annotated[
        Optional[str], typer.Option("--hf-token", help="HuggingFace token for pyannote models.")
    ] = None,
) -> None:
    """Transcribe an audio or video file locally."""
    _run_transcribe(
        input_file,
        language=language,
        prompt=prompt,
        model=model,
        fmt=fmt,
        output=output,
        translate_to=translate_to,
        summarize=summarize,
        ollama_model=ollama_model,
        task=task,
        diarize=diarize,
        num_speakers=num_speakers,
        speaker_names=speaker_names,
        hf_token=hf_token,
    )


@app.command("t")
def transcribe_short(
    input_file: Annotated[Path, typer.Argument(help="Audio or video file to transcribe.")],
    language: Annotated[Optional[str], typer.Option("--language", "-l")] = None,
    prompt: Annotated[Optional[str], typer.Option("--prompt", "-p")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
    fmt: Annotated[str, typer.Option("--format", "-f")] = "",
    output: Annotated[Optional[Path], typer.Option("--output", "-o")] = None,
    translate_to: Annotated[Optional[str], typer.Option("--translate-to")] = None,
    summarize: Annotated[bool, typer.Option("--summarize")] = False,
    ollama_model: Annotated[Optional[str], typer.Option("--ollama-model")] = None,
    task: Annotated[str, typer.Option("--task")] = "transcribe",
    diarize: Annotated[bool, typer.Option("--diarize")] = False,
    num_speakers: Annotated[Optional[int], typer.Option("--num-speakers")] = None,
    speaker_names: Annotated[Optional[str], typer.Option("--speaker-names")] = None,
    hf_token: Annotated[Optional[str], typer.Option("--hf-token")] = None,
) -> None:
    """Alias for transcribe."""
    _run_transcribe(
        input_file,
        language=language,
        prompt=prompt,
        model=model,
        fmt=fmt,
        output=output,
        translate_to=translate_to,
        summarize=summarize,
        ollama_model=ollama_model,
        task=task,
        diarize=diarize,
        num_speakers=num_speakers,
        speaker_names=speaker_names,
        hf_token=hf_token,
    )


@app.command("setup")
def setup_cmd(
    model: Annotated[
        Optional[str], typer.Option("--model", "-m", help="Whisper model to download.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-download even if the model is cached.")
    ] = False,
    quick: Annotated[
        bool, typer.Option("--quick", help="Skip diarization/Ollama steps (model only).")
    ] = False,
) -> None:
    """Interactive onboarding: check dependencies, download model, configure options."""
    _print_banner()
    if quick:
        _run_quick_setup(model=model, force=force, interactive=model is None)
    else:
        run_setup_wizard(console, model=model, force=force)


@app.command("onboard", hidden=True)
def onboard_cmd(
    model: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Alias for setup."""
    setup_cmd(model=model, force=force, quick=False)


@install_app.command("cuda")
def install_cuda_cmd(
    toolkit: Annotated[
        bool,
        typer.Option(
            "--toolkit",
            help="Also install the full NVIDIA CUDA Toolkit via winget (Windows).",
        ),
    ] = False,
) -> None:
    """Install CUDA stack for Whisper (CTranslate2) and diarization (PyTorch)."""
    _print_banner()

    if is_cuda_runtime_installed():
        console.print("[green]OK[/green] CUDA 12 runtime libraries are already installed.")
    else:
        console.print("[bold]Installing CUDA 12 runtime libraries...[/bold]")
        console.print("Packages: [cyan]nvidia-cublas-cu12[/cyan], [cyan]nvidia-cudnn-cu12[/cyan]")
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        )
        with progress:
            task_id = progress.add_task("pip install CUDA runtime", total=None)

            def on_line(line: str) -> None:
                progress.update(task_id, description=line[:70])

            code = install_cuda_runtime(on_output=on_line, include_torch=False)

        if code != 0:
            console.print("[red]CUDA runtime installation failed.[/red]")
            raise typer.Exit(1)

        console.print("[green]OK[/green] CUDA 12 runtime libraries installed.")

    if is_torch_cuda_installed():
        console.print("[green]OK[/green] CUDA PyTorch is already available.")
    else:
        console.print("\n[bold]Installing CUDA PyTorch (required for GPU diarization)...[/bold]")
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        )
        with progress:
            task_id = progress.add_task("pip install torch+cu126", total=None)

            def on_torch_line(line: str) -> None:
                progress.update(task_id, description=line[:70])

            torch_code = install_torch_cuda(on_output=on_torch_line)

        if torch_code != 0:
            console.print(
                "[red]CUDA PyTorch installation failed.[/red]\n"
                "Whisper may still use GPU; diarization would stay on CPU.\n"
                "Retry: [cyan]lwt install cuda[/cyan]"
            )
            raise typer.Exit(1)
        console.print("[green]OK[/green] CUDA PyTorch installed.")

    if toolkit:
        console.print("\n[bold]Installing NVIDIA CUDA Toolkit via winget...[/bold]")
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        )
        with progress:
            task_id = progress.add_task("winget install Nvidia.CUDA", total=None)

            def on_toolkit_line(line: str) -> None:
                progress.update(task_id, description=line[:70])

            toolkit_code = install_cuda_toolkit(on_output=on_toolkit_line)

        if toolkit_code != 0:
            console.print("[yellow]CUDA Toolkit install did not complete.[/yellow]")
        else:
            console.print("[green]OK[/green] CUDA Toolkit installed.")

    runtime_ok, runtime_detail = check_cuda_runtime()
    console.print(f"\nStatus: {runtime_detail}")
    console.print("[dim]Restart your terminal, then run transcription again.[/dim]")


@install_app.command("diarization")
def install_diarization_cmd() -> None:
    """Install pyannote.audio for speaker diarization."""
    _print_banner()
    if is_diarization_installed():
        console.print("[green]OK[/green] pyannote.audio is already installed.")
        return

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
            progress.update(task_id, description=line[:70])

        code = install_diarization(on_output=on_line)

    if code != 0:
        console.print("[red]Installation failed.[/red]")
        raise typer.Exit(1)

    console.print("[green]OK[/green] pyannote.audio installed.")
    console.print("Accept licenses for ALL 4 models (see [cyan]lwt install verify-diarization[/cyan]):")
    for model_id in HF_MODELS_TO_ACCEPT:
        console.print(f"  [cyan]https://huggingface.co/{model_id}[/cyan]")
    console.print("Token: [cyan]lwt config set diarization.hf_token <token>[/cyan]")


@install_app.command("verify-diarization")
def verify_diarization_cmd(
    hf_token: Annotated[
        Optional[str], typer.Option("--hf-token", help="HuggingFace token (or use config).")
    ] = None,
) -> None:
    """Verify HuggingFace access to all diarization models before transcribing."""
    _print_banner()
    config = load_config()
    token = get_hf_token(config, hf_token)
    if not token:
        console.print(
            "[red]Missing HuggingFace token.[/red]\n"
            "Set: [cyan]lwt config set diarization.hf_token <token>[/cyan]"
        )
        raise typer.Exit(1)

    if not is_diarization_installed():
        console.print("[yellow]pyannote.audio is not installed.[/yellow] Run: lwt install diarization")
        raise typer.Exit(1)

    console.print(
        "[dim]torchcodec warnings when importing pyannote are normal — we handle audio via ffmpeg.[/dim]\n"
    )

    results = verify_diarization_access(token)
    table = Table(title="HuggingFace model access", show_header=True)
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Detail")

    all_ok = True
    for model_id, ok, detail in results:
        if ok:
            table.add_row(model_id, "[green]OK[/green]", detail)
        else:
            all_ok = False
            table.add_row(model_id, "[red]FAIL[/red]", detail)

    console.print(table)

    if all_ok:
        console.print("\n[green]OK[/green] All models accessible. Diarization should work.")
        with console.status("[bold green]Testing pipeline load..."):
            try:
                from local_whisper_transcribe.diarize import _load_pipeline

                _load_pipeline(token)
            except DiarizationAccessError as exc:
                console.print(f"\n[red]Pipeline failed:[/red] {exc}")
                raise typer.Exit(1) from exc
            except Exception as exc:
                console.print(f"\n[red]Pipeline failed:[/red] {exc}")
                raise typer.Exit(1) from exc
        console.print("[green]OK[/green] Pipeline loaded.")
        return

    console.print(
        "\n[red]Missing access to one or more models.[/red]\n"
        "For each FAIL link, click [bold]Agree and access repository[/bold].\n"
        "Often missing: 4th model [cyan]pyannote/speaker-diarization-community-1[/cyan]"
    )
    raise typer.Exit(1)


@install_app.command("check")
def install_check_cmd() -> None:
    """Verify all optional dependencies are installed."""
    _print_banner()
    table = Table(title="Dependencies", show_header=True, header_style="bold blue")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Install hint")

    for dep in check_all_dependencies():
        status = "[green]OK[/green]" if dep.installed else "[yellow]Missing[/yellow]"
        hint = dep.install_hint if not dep.installed else "—"
        table.add_row(dep.name, status, hint)

    console.print(table)


@ollama_app.command("status")
def ollama_status_cmd() -> None:
    """Check if Ollama is running and list available models."""
    _print_banner()
    config = load_config()
    status = get_ollama_status(config["ollama"]["url"])

    if status.available:
        console.print(f"[green]OK[/green] Ollama is running at [cyan]{status.url}[/cyan]")
        if status.models:
            console.print(f"Models: {', '.join(status.models)}")
        else:
            console.print("[yellow]No models found.[/yellow] Pull one: [cyan]lwt ollama pull llama3.2[/cyan]")
    else:
        console.print(f"[red]FAIL[/red] Ollama is not available at {status.url}")
        if status.error:
            console.print(f"  {status.error}")
        console.print("Start Ollama and try again.")


@ollama_app.command("list")
def ollama_list_cmd() -> None:
    """List installed Ollama models."""
    _print_banner()
    config = load_config()
    models = list_ollama_models(config["ollama"]["url"])
    if not models:
        console.print("[yellow]No models found.[/yellow]")
        console.print("Pull a model: [cyan]lwt ollama pull llama3.2[/cyan]")
        return

    table = Table(title="Ollama Models", show_header=True, header_style="bold green")
    table.add_column("Model", style="cyan")
    for name in models:
        table.add_row(name)
    console.print(table)


@ollama_app.command("pull")
def ollama_pull_cmd(
    model: Annotated[str, typer.Argument(help="Model name to pull (e.g. llama3.2).")],
) -> None:
    """Pull an Ollama model."""
    _print_banner()
    config = load_config()
    url = config["ollama"]["url"]

    if not check_ollama_available(url):
        console.print(
            f"[red]Error:[/red] Ollama is not available at {url}.\n"
            "Start Ollama and try: [cyan]lwt ollama status[/cyan]"
        )
        raise typer.Exit(1)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task_id = progress.add_task(f"Pulling {model}...", total=100)

        def on_pull(msg: str, percent: int | None) -> None:
            if percent is not None:
                progress.update(task_id, completed=percent, description=msg)
            else:
                progress.update(task_id, description=msg)

        try:
            pull_ollama_model(model, url, on_progress=on_pull)
            progress.update(task_id, completed=100)
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc

    console.print(f"[green]OK[/green] Model [cyan]{model}[/cyan] pulled.")


@models_app.command("list")
def models_list() -> None:
    """List available Whisper models with size and recommendations."""
    _print_banner()
    console.print(_models_table(show_ram=True))


@models_app.command("download")
def models_download(
    name: Annotated[str, typer.Argument(help="Model name (e.g. small, medium).")],
    force: Annotated[
        bool, typer.Option("--force", help="Re-download even if the model is cached.")
    ] = False,
) -> None:
    """Download a specific Whisper model."""
    if name not in KNOWN_MODELS:
        console.print(f"[red]Error:[/red] Unknown model '{name}'.")
        raise typer.Exit(1)

    config = load_config()
    messages: list[str] = []

    with console.status(f"[bold green]Downloading {name}..."):
        try:
            path = download_model(
                name,
                device=config["whisper"]["device"],
                compute_type=config["whisper"]["compute_type"],
                force=force,
                status_callback=messages.append,
            )
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc

    for message in messages:
        console.print(f"  [dim]•[/dim] {message}")

    console.print(f"[green]OK[/green] Model [bold]{name}[/bold] ready at [cyan]{path}[/cyan]")


@models_app.command("status")
def models_status() -> None:
    """Show which Whisper models are cached locally."""
    _print_banner()
    table = Table(title="Cached Models", show_header=True, header_style="bold green")
    table.add_column("Model", style="cyan")
    table.add_column("Cached")
    table.add_column("Size")
    table.add_column("Path")

    for status in list_model_statuses():
        cached = "[green]yes[/green]" if status.cached else "no"
        path = str(status.path) if status.path else "—"
        table.add_row(status.name, cached, format_size(status.size_bytes), path)

    console.print(table)
    console.print(f"\nHugging Face cache: [cyan]{get_cache_dir()}[/cyan]")


@config_app.command("show")
def config_show() -> None:
    """Display current configuration."""
    _print_banner()
    config = load_config()
    table = Table(title="Configuration", show_header=True, header_style="bold blue")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for section, values in config.items():
        for key, value in values.items():
            display = mask_config_value(section, key, value)
            table.add_row(f"{section}.{key}", display)
    console.print(table)


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Config key in section.option format.")],
    value: Annotated[str, typer.Argument(help="New value.")],
) -> None:
    """Update a configuration value."""
    try:
        set_config_value(key, value)
        display = value
        if key == "diarization.hf_token":
            display = mask_config_value("diarization", "hf_token", value)
        console.print(f"[green]OK[/green] Set [bold]{key}[/bold] = [cyan]{display}[/cyan]")
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


@config_app.command("path")
def config_path_cmd() -> None:
    """Show config file location."""
    console.print(str(get_config_path()))


@config_app.command("reset")
def config_reset_cmd(
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")
    ] = False,
) -> None:
    """Reset configuration to defaults."""
    if not yes and not ask_confirm(
        "Reset configuration to default values?",
        default=False,
        console=console,
    ):
        raise typer.Exit()
    reset_config()
    console.print("[green]OK[/green] Configuration reset.")
    console.print(f"File: [cyan]{get_config_path()}[/cyan]")


if __name__ == "__main__":
    app()
