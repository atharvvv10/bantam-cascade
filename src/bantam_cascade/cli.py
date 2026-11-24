"""Typer command-line interface for Bantam Cascade."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import typer
from rich.console import Console

from bantam_cascade import __version__

app = typer.Typer(
    name="bantam-cascade",
    help="Certify the cheapest small-CPU-model + big-LLM fallback cascade.",
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"bantam-cascade {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Bantam Cascade CLI."""


@app.command()
def init(
    config: Path = typer.Option(
        Path("configs/smoke.yaml"),
        "--config",
        "-c",
        help="Config file to copy into the run directory.",
    ),
    run_dir: Path = typer.Option(
        Path("runs/smoke"),
        "--run-dir",
        "-o",
        help="Run directory to create.",
    ),
) -> None:
    """Create a run directory containing a copy of the config."""
    if not config.is_file():
        raise typer.BadParameter(f"Config file does not exist: {config}")
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "config.yaml"
    copyfile(config, target)
    console.print(f"Created [bold]{target}[/bold].")
    console.print(
        "Next: fill in any placeholders, then run the pipeline (commands arrive in later phases)."
    )
