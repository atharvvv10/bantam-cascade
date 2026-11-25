"""Typer command-line interface for Bantam Cascade."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from bantam_cascade import __version__
from bantam_cascade.calibration import select_threshold
from bantam_cascade.cascade import TeacherPrediction, evaluate_cascade
from bantam_cascade.config import Config, load_config
from bantam_cascade.data import load_dataset_for_config
from bantam_cascade.splits import SplitResult, split_by_hash
from bantam_cascade.students import (
    StudentModel,
    StudentReport,
    build_student,
    restore_student,
    save_student,
)
from bantam_cascade.students.base import confidence_scores
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


def _split_data(config: Config) -> tuple[SplitResult, dict[str, str], tuple[str, ...]]:
    """Return the disjoint splits, gold labels by text, and in-scope labels."""
    dataset = load_dataset_for_config(config)
    gold_by_text = {example.text: example.label for example in dataset.train}
    official_test = tuple(example.text for example in dataset.test)
    split = split_by_hash(
        list(gold_by_text),
        config=config.splits,
        official_test=official_test,
    )
    return split, gold_by_text, dataset.labels


def _read_teacher_rows(path: Path) -> dict[str, TeacherPrediction]:
    """Read saved teacher rows into a text-keyed prediction map."""
    if not path.is_file():
        return {}
    predictions: dict[str, TeacherPrediction] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            predictions[row["text"]] = TeacherPrediction(
                label=row["predicted"],
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                latency_ms=float(row["latency_ms"]),
            )
    return predictions


def _append_teacher_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Append teacher rows to a JSONL file, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _predict_with_latency(
    student: StudentModel, texts: Sequence[str]
) -> tuple[np.ndarray, list[float]]:
    """Predict one text at a time and record per-example CPU latency."""
    rows: list[np.ndarray] = []
    latencies: list[float] = []
    for text in texts:
        start = perf_counter()
        proba = np.asarray(student.predict_proba([text]), dtype=np.float64)[0]
        latencies.append((perf_counter() - start) * 1000.0)
        rows.append(proba)
    return np.asarray(rows), latencies


@app.command(name="train-students")
def train_students(
    config: Path = typer.Option(Path("configs/smoke.yaml"), "--config", "-c"),
    run_dir: Path = typer.Option(Path("runs/phase2"), "--run-dir", "-o"),
    limit: int | None = typer.Option(
        None, "--limit", help="Training examples per student (default: full split)."
    ),
) -> None:
    """Train configured CPU-only students on gold-labeled training texts."""
    parsed = load_config(config)
    if limit is not None and limit <= 0:
        raise typer.BadParameter("--limit must be positive")
    split, gold_by_text, _labels = _split_data(parsed)
    train_texts = split.student_train if limit is None else split.student_train[:limit]

    table = Table(title="Student training")
    table.add_column("Student")
    table.add_column("Gold train texts")
    table.add_column("Calibration accuracy")
    table.add_column("Mean confidence")
    for name in parsed.students:
        student = build_student(name)
        student.fit(train_texts, [gold_by_text[text] for text in train_texts])
        save_student(student, run_dir / "students" / f"{name}.joblib")

        cal_proba, _latencies = _predict_with_latency(student, split.calibration)
        gold = [gold_by_text[text] for text in split.calibration]
        predicted = np.asarray(student.classes)[cal_proba.argmax(axis=1)]
        report = StudentReport(
            name=name,
            examples=len(gold),
            accuracy=float(np.mean(predicted == np.asarray(gold))),
            mean_confidence=float(np.mean(confidence_scores(cal_proba))),
        )
        table.add_row(
            report.name,
            str(len(train_texts)),
            f"{report.accuracy:.3f}",
            f"{report.mean_confidence:.3f}",
        )
    console.print(table)
    console.print(f"Saved students under [bold]{run_dir / 'students'}[/bold].")


@app.command(name="build-cascade")
def build_cascade(
    config: Path = typer.Option(Path("configs/smoke.yaml"), "--config", "-c"),
    run_dir: Path = typer.Option(Path("runs/phase2"), "--run-dir", "-o"),
    limit: int | None = typer.Option(
        None, "--limit", help="Certification examples (default: full split)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate teacher calls only."),
    calibration_labels: Path | None = typer.Option(
        None,
        "--calibration-labels",
        help="Phase 1 teacher_labels.jsonl (default: <run-dir>/teacher_labels.jsonl).",
    ),
) -> None:
    """Calibrate students and evaluate cascades on gold-labeled certification data."""
    parsed = load_config(config)
    if parsed.dataset == "smoke-tiny":
        raise typer.BadParameter("The smoke config is offline-only. Use a real dataset config.")
    if limit is not None and limit <= 0:
        raise typer.BadParameter("--limit must be positive")
    split, gold_by_text, labels = _split_data(parsed)
    cert_texts = split.certification if limit is None else split.certification[:limit]

    cache = LabelCache(_cache_path(run_dir))
    client = GroqTeacherClient(parsed.teacher)
    if dry_run:
        estimate = client.dry_run_estimate(
            cert_texts,
            labels,
            cache=cache,
            limit=len(cert_texts),
        )
        client.close()
        console.print("[bold]Dry run — no API calls made.[/bold]")
        console.print(
            f"Certification texts: {estimate.texts} | to call: {estimate.requests} | "
            f"cached: {estimate.cached}"
        )
        console.print(
            f"Estimated days at {parsed.teacher.rate_limits.requests_per_day} RPD: "
            f"{estimate.estimated_days}"
        )
        return

    cal_path = calibration_labels or run_dir / "teacher_labels.jsonl"
    cal_predictions = _read_teacher_rows(cal_path)
    missing_calibration = [text for text in split.calibration if text not in cal_predictions]
    if missing_calibration:
        client.close()
        raise typer.BadParameter(
            f"Missing {len(missing_calibration)} calibration teacher labels. "
            "Run `bantam-cascade label --config ... --run-dir ...` first."
        )

    cert_path = run_dir / "certification_teacher_labels.jsonl"
    cert_predictions = _read_teacher_rows(cert_path)
    missing = [text for text in cert_texts if text not in cert_predictions]
    try:
        if missing:
            batch = client.label_many(missing, labels, cache=cache, limit=len(missing))
            new_rows: list[dict[str, object]] = [
                {
                    "text": result.text,
                    "gold": gold_by_text[result.text],
                    "predicted": result.label,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "latency_ms": result.latency_ms,
                    "cached": result.cached,
                }
                for result in batch.results
            ]
            _append_teacher_rows(cert_path, new_rows)
            for teacher_result in batch.results:
                cert_predictions[teacher_result.text] = TeacherPrediction(
                    label=teacher_result.label,
                    input_tokens=teacher_result.input_tokens,
                    output_tokens=teacher_result.output_tokens,
                    latency_ms=teacher_result.latency_ms,
                )
    finally:
        client.close()

    table = Table(title="Cascade candidates")
    table.add_column("Student")
    table.add_column("Threshold")
    table.add_column("Accuracy")
    table.add_column("Fallback")
    table.add_column("Teacher calls")
    table.add_column("Est. cost")
    table.add_column("p95 ms")
    output_path = run_dir / "cascade_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for name in parsed.students:
            student_path = run_dir / "students" / f"{name}.joblib"
            if not student_path.is_file():
                raise typer.BadParameter(
                    f"Missing trained student {student_path}. Run `train-students` first."
                )
            student = restore_student(student_path)
            cal_proba, _cal_latency = _predict_with_latency(student, split.calibration)
            threshold_result = select_threshold(
                proba=cal_proba,
                classes=student.classes,
                gold=[gold_by_text[text] for text in split.calibration],
                teacher_predictions=[cal_predictions[text].label for text in split.calibration],
                grid_size=parsed.threshold_grid_size,
            )
            cert_proba, cert_latency = _predict_with_latency(student, cert_texts)
            result = evaluate_cascade(
                student=name,
                threshold=threshold_result.threshold,
                proba=cert_proba,
                classes=student.classes,
                gold=[gold_by_text[text] for text in cert_texts],
                teacher_predictions=[cert_predictions[text] for text in cert_texts],
                student_latency_ms=cert_latency,
                price_per_million_input_tokens=parsed.teacher.price_per_million_input_tokens,
                price_per_million_output_tokens=parsed.teacher.price_per_million_output_tokens,
            )
            handle.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
            table.add_row(
                name,
                f"{result.threshold:.3f}",
                f"{result.accuracy:.3f}",
                f"{result.fallback_rate:.3f}",
                str(result.teacher_calls),
                f"${result.estimated_cost:.6f}",
                f"{result.p95_latency_ms:.1f}",
            )
    console.print(table)
    console.print(f"Wrote [bold]{output_path}[/bold]. These are measurements, not certificates.")
