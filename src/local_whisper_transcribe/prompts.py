"""Rich-based interactive prompts (typer.prompt does not render markup)."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.text import Text


def ask_prompt(
    message: str,
    *,
    default: str = "",
    console: Console | None = None,
    password: bool = False,
) -> str:
    """Prompt for text input with optional Rich markup in the message."""
    con = console or Console()
    return Prompt.ask(Text.from_markup(message), default=default, console=con, password=password)


def ask_confirm(
    message: str,
    *,
    default: bool = True,
    console: Console | None = None,
) -> bool:
    """Prompt for yes/no with Rich markup support."""
    con = console or Console()
    return Confirm.ask(Text.from_markup(message), default=default, console=con)
