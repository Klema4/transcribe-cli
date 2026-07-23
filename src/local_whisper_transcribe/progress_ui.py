"""Rich progress bar with a live preview line underneath."""

from __future__ import annotations

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

PREVIEW_MAX_LEN = 120


def truncate_preview(text: str, max_len: int = PREVIEW_MAX_LEN) -> str:
    """Collapse whitespace and trim long preview lines for the terminal."""
    one_line = " ".join(text.split())
    if not one_line:
        return "..."
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 3] + "..."


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


class ProgressWithPreview:
    """Progress bar with a second line showing the current segment or stage."""

    def __init__(
        self,
        console: Console,
        *,
        initial_preview: str = "",
        preview_style: str = "dim",
    ) -> None:
        self._console = console
        self._preview_style = preview_style
        self._preview = Text(
            truncate_preview(initial_preview) if initial_preview else "...",
            style=preview_style,
        )
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        self._live: Live | None = None

    def __enter__(self) -> ProgressWithPreview:
        self._live = Live(
            Group(self.progress, self._preview),
            console=self._console,
            refresh_per_second=12,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._live:
            self._live.__exit__(*args)

    def add_task(self, description: str, *, total: float = 100) -> int:
        return self.progress.add_task(description, total=total)

    def update_task(self, task_id: int, **kwargs: object) -> None:
        self.progress.update(task_id, **kwargs)

    def set_preview(self, text: str, *, style: str | None = None) -> None:
        self._preview = Text(truncate_preview(text), style=style or self._preview_style)
        if self._live:
            self._live.update(Group(self.progress, self._preview))
