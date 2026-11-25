"""Typer command-line interface for Bantam Cascade."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import typer
from rich.console import Console
from rich.table import Table

from bantam_cascade import __version__
from bantam_cascade.config import Config, load_config
from bantam_cascade.data import load_dataset_for_config
from bantam_cascade.splits import split_by_hash
from bantam_cascade.teacher.cache import LabelCache
from bantam_cascade.teacher.client import GroqTeacherClient
from bantam_cascade.teacher.select import (
    evaluate_candidate,
    recommend,
    teacher_sanity_gate,
)

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


def _calibration_data(
    config: Config,
    limit: int | None,
) -> tuple[tuple[str, ...], dict[str, str], tuple[str, ...]]:
    """Return calibration texts, their gold labels, and the in-scope label list."""
    if limit is not None and limit <= 0:
        raise typer.BadParameter("--limit must be positive")
    dataset = load_dataset_for_config(config)
    gold_by_text = {example.text: example.label for example in dataset.train}
    official_test = tuple(example.text for example in dataset.test)
    split = split_by_hash(
        list(gold_by_text),
        config=config.splits,
        official_test=official_test,
    )
    chosen = split.calibration if limit is None else split.calibration[:limit]
    return chosen, gold_by_text, dataset.labels


def _cache_path(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "teacher_cache.sqlite3"


@app.command()
def label(
    config: Path = typer.Option(Path("configs/smoke.yaml"), "--config", "-c"),
    run_dir: Path = typer.Option(Path("runs/phase1"), "--run-dir", "-o"),
    limit: int | None = typer.Option(None, "--limit", help="Examples to process."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate without API calls."),
    model: str | None = typer.Option(None, "--model", help="Override the pinned model id."),
) -> None:
    """Label calibration examples with the pinned teacher, or estimate a dry run."""
    parsed = load_config(config)
    sample_size = parsed.splits.calibration if limit is None else limit
    texts, gold_by_text, labels = _calibration_data(parsed, sample_size)
    cache = LabelCache(_cache_path(run_dir))
    client = GroqTeacherClient(parsed.teacher)

    if dry_run:
        estimate = client.dry_run_estimate(
            texts,
            labels,
            model=model,
            cache=cache,
            limit=sample_size,
        )
        console.print("[bold]Dry run — no API calls made.[/bold]")
        console.print(f"Model: {model or parsed.teacher.model}")
        console.print(
            f"Texts: {estimate.texts} | to call: {estimate.requests} | cached: {estimate.cached}"
        )
        console.print(
            f"Estimated tokens: {estimate.input_tokens} in / {estimate.output_tokens} out"
        )
        console.print(
            f"Estimated days at {parsed.teacher.rate_limits.requests_per_day} RPD: "
            f"{estimate.estimated_days}"
        )
        console.print(f"Estimated cost from configured prices: ${estimate.estimated_cost:.6f}")
        return

    if parsed.dataset == "smoke-tiny":
        raise typer.BadParameter(
            "The smoke config is offline-only. Use --dry-run or a real dataset config."
        )

    try:
        batch = client.label_many(
            texts,
            labels,
            cache=cache,
            model=model,
            limit=sample_size,
        )
    finally:
        client.close()

    results = batch.results
    correct = sum(1 for result in results if result.label == gold_by_text[result.text])
    output_path = run_dir / "teacher_labels.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "text": result.text,
                        "gold": gold_by_text[result.text],
                        "predicted": result.label,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "latency_ms": result.latency_ms,
                        "cached": result.cached,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    console.print(f"Labeled {len(results)} examples; wrote [bold]{output_path}[/bold].")
    console.print(
        f"Live calls: {batch.totals.requests} | cache hits: {batch.totals.cached} | "
        f"invalid outputs: {batch.totals.invalid}"
    )
    if results:
        console.print(f"Calibration accuracy vs gold: {correct / len(results):.3f}")


@app.command(name="select-teacher")
def select_teacher(
    config: Path = typer.Option(Path("configs/smoke.yaml"), "--config", "-c"),
    run_dir: Path = typer.Option(Path("runs/phase1"), "--run-dir", "-o"),
    models: str | None = typer.Option(
        None,
        "--models",
        help="Comma-separated candidate ids (up to 3). Defaults to the configured model.",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Examples per candidate."),
    best_student_accuracy: float | None = typer.Option(
        None,
        "--best-student-accuracy",
        help="If given, check that the recommended teacher is at least this accurate.",
    ),
) -> None:
    """Compare up to three teacher candidates on gold-labeled calibration data."""
    parsed = load_config(config)
    if parsed.dataset == "smoke-tiny":
        raise typer.BadParameter("The smoke config is offline-only. Use `label --dry-run` instead.")
    if models is None:
        candidate_ids = [parsed.teacher.model]
    else:
        candidate_ids = list(
            dict.fromkeys(item.strip() for item in models.split(",") if item.strip())
        )
    if not candidate_ids or len(candidate_ids) > 3:
        raise typer.BadParameter("Provide between one and three candidate model ids.")
    if best_student_accuracy is not None and not 0.0 <= best_student_accuracy <= 1.0:
        raise typer.BadParameter("--best-student-accuracy must be in [0, 1]")

    sample_size = parsed.splits.calibration if limit is None else limit
    texts, gold_by_text, labels = _calibration_data(parsed, sample_size)
    gold = [gold_by_text[text] for text in texts]
    cache = LabelCache(_cache_path(run_dir))
    client = GroqTeacherClient(parsed.teacher)

    table = Table(title="Teacher candidates")
    table.add_column("Model")
    table.add_column("Accuracy")
    table.add_column("Invalid")
    table.add_column("Tokens")
    table.add_column("Est. cost")
    collected = []
    try:
        for model in candidate_ids:
            metrics = evaluate_candidate(
                client=client,
                cache=cache,
                texts=texts,
                gold_labels=gold,
                labels=labels,
                model=model,
                price_per_million_input_tokens=parsed.teacher.price_per_million_input_tokens,
                price_per_million_output_tokens=parsed.teacher.price_per_million_output_tokens,
            )
            collected.append(metrics)
            table.add_row(
                model,
                f"{metrics.accuracy:.3f}",
                f"{metrics.invalid_rate:.3f}",
                str(metrics.input_tokens + metrics.output_tokens),
                f"${metrics.estimated_cost:.6f}",
            )
    finally:
        client.close()

    console.print(table)
    best = recommend(collected)
    if best is None:
        console.print("[yellow]No candidates evaluated.[/yellow]")
        return
    console.print(f"Recommended teacher: [bold]{best.model}[/bold] (accuracy {best.accuracy:.3f}).")

    if best_student_accuracy is not None:
        passed = teacher_sanity_gate(
            teacher_accuracy=best.accuracy,
            best_student_accuracy=best_student_accuracy,
        )
        if passed:
            console.print("[green]Sanity gate passed: teacher meets the student baseline.[/green]")
        else:
            console.print(
                "[red]Sanity gate failed: teacher is worse than the student baseline.[/red]"
            )
            raise typer.Exit(code=1)
