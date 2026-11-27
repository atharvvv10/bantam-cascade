"""Typer command-line interface for Bantam Cascade."""

from __future__ import annotations

import hashlib
import json
import math
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
from bantam_cascade.cascade import CascadeResult, TeacherPrediction, evaluate_cascade
from bantam_cascade.certification import (
    CERTIFICATION_ASSUMPTIONS,
    Candidate,
    certify_candidates,
    certify_candidates_with_recall,
    delta_from_confidence_level,
    required_examples_for_accuracy,
)
from bantam_cascade.config import Config, load_config
from bantam_cascade.data import load_dataset_for_config
from bantam_cascade.ledger import (
    CertificationLedger,
    CertificationReuseError,
    corrected_delta,
)
from bantam_cascade.splits import SplitResult, certification_split_hash, split_by_hash
from bantam_cascade.students import (
    StudentModel,
    StudentReport,
    build_student,
    restore_student,
    save_student,
)
from bantam_cascade.students.base import confidence_scores, labels_from_proba
from bantam_cascade.teacher.cache import LabelCache
from bantam_cascade.teacher.factory import build_teacher_client
from bantam_cascade.teacher.select import (
    evaluate_candidate,
    recommend,
    teacher_sanity_gate,
)
from bantam_cascade.validity import ValidityScenario, run_validity_experiment
from bantam_cascade.verification import verify_frozen_certificate

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
    console.print("Next: label, train-students, build-cascade, then certify using this --run-dir.")


@app.command()
def plan(
    target_accuracy: float = typer.Option(0.95, "--target-accuracy"),
    confidence_level: float = typer.Option(0.95, "--confidence-level"),
    assumed_accuracy: float = typer.Option(0.98, "--assumed-accuracy"),
) -> None:
    """Size a certification split before collecting any test labels."""
    try:
        examples = required_examples_for_accuracy(
            target_accuracy=target_accuracy,
            confidence_level=confidence_level,
            assumed_accuracy=assumed_accuracy,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title="Certification split plan")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Target accuracy", f"{target_accuracy:.3f}")
    table.add_row("Confidence", f"{confidence_level:.3f}")
    table.add_row("Assumed true accuracy", f"{assumed_accuracy:.3f}")
    table.add_row("Minimum certification examples", str(examples))
    table.add_row(
        "Expected errors at assumed accuracy", str(math.ceil((1.0 - assumed_accuracy) * examples))
    )
    console.print(table)
    console.print(
        "This uses the Hoeffding-Bentkus bound with a conservative rounded-up error count. "
        "It is a planning number, not a certificate."
    )


@app.command()
def demo(
    text: str = typer.Argument(..., help="Text to classify with the saved student."),
    run_dir: Path = typer.Option(
        Path("runs/sms-groq"),
        "--run-dir",
        "-o",
        help="Completed run directory containing config.yaml and a saved student.",
    ),
) -> None:
    """Classify one text offline; no teacher API calls are made."""
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise typer.BadParameter(f"Missing {config_path}. Run `bantam-cascade init` first.")
    parsed = load_config(config_path)
    if len(parsed.students) != 1:
        raise typer.BadParameter("demo expects a run with exactly one configured student")
    name = parsed.students[0]
    student_path = run_dir / "students" / f"{name}.joblib"
    if not student_path.is_file():
        raise typer.BadParameter(
            f"Missing trained student {student_path}. Run `train-students` first."
        )

    student = restore_student(student_path)
    proba = np.asarray(student.predict_proba([text]), dtype=np.float64)[0]
    label = student.classes[int(proba.argmax())]
    confidence = float(proba.max())

    threshold = 0.0
    preregistration = run_dir / "preregistration.json"
    if preregistration.is_file():
        payload = json.loads(preregistration.read_text(encoding="utf-8"))
        value = payload.get("selected_threshold")
        if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
            threshold = float(value)

    table = Table(title="Bantam Cascade demo")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Student", name)
    table.add_row("Predicted label", label)
    table.add_row("Confidence", f"{confidence:.3f}")
    table.add_row("Pre-registered threshold", f"{threshold:.3f}")
    table.add_row(
        "Cascade action",
        "student answers" if confidence >= threshold else "teacher fallback would be used",
    )
    console.print(table)
    if confidence < threshold:
        console.print("[yellow]Teacher fallback is not called in demo mode.[/yellow]")
    console.print("Demo uses saved artifacts only; it does not re-run certification.")


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
    client = build_teacher_client(parsed.teacher)

    if dry_run:
        estimate = client.dry_run_estimate(
            texts,
            labels,
            model=model,
            cache=cache,
            limit=sample_size,
        )
        console.print("[bold]Dry run — no API calls made.[/bold]")
        console.print(f"Provider: {parsed.teacher.provider}")
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
                        "confidence": result.confidence,
                        "probabilities": result.probabilities,
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
    client = build_teacher_client(parsed.teacher)

    table = Table(title="Teacher candidates")
    table.add_column("Provider")
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
                client.provider,
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


def _read_teacher_gold(path: Path) -> dict[str, str]:
    """Read the saved gold labels so certification can verify their provenance."""
    if not path.is_file():
        return {}
    gold_by_text: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            gold = row.get("gold")
            if not isinstance(gold, str):
                raise typer.BadParameter(f"Missing gold label in {path}.")
            gold_by_text[str(row["text"])] = gold
    return gold_by_text


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
    client = build_teacher_client(parsed.teacher)
    if dry_run:
        estimate = client.dry_run_estimate(
            cert_texts,
            labels,
            cache=cache,
            limit=len(cert_texts),
        )
        client.close()
        console.print("[bold]Dry run — no API calls made.[/bold]")
        console.print(f"Provider: {parsed.teacher.provider} | Model: {parsed.teacher.model}")
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
                    "confidence": result.confidence,
                    "probabilities": result.probabilities,
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


def _cascade_results(path: Path) -> dict[str, CascadeResult]:
    """Load measured cascade rows keyed by student name."""
    if not path.is_file():
        raise typer.BadParameter(f"Missing {path}. Run `bantam-cascade build-cascade ...` first.")
    results: dict[str, CascadeResult] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                result = CascadeResult(**row)
                if result.student in results:
                    raise typer.BadParameter(
                        f"Duplicate cascade result for {result.student!r} on line {line_number}."
                    )
                results[result.student] = result
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise typer.BadParameter(f"Invalid cascade results in {path}: {exc}") from exc
    return results


def _split_hash(texts: Sequence[str]) -> str:
    """Hash the certification texts so the ledger can detect repeated looks."""
    return certification_split_hash(texts)


def _synthetic_result(*, student: str, accuracy: float, p95_latency_ms: float) -> CascadeResult:
    """Build a synthetic measured cascade for the status demonstration."""
    examples = 300
    errors = round((1.0 - accuracy) * examples)
    measured = 1.0 - errors / examples
    return CascadeResult(
        student=student,
        threshold=0.0,
        examples=examples,
        accuracy=measured,
        student_only_accuracy=measured,
        teacher_only_accuracy=measured,
        fallback_rate=0.0,
        teacher_calls=0,
        estimated_cost=0.0,
        p95_latency_ms=p95_latency_ms,
    )


def _synthetic_candidate(*, student: str, accuracy: float, p95_latency_ms: float) -> Candidate:
    """Build a matching calibration/certification pair for a synthetic case."""
    result = _synthetic_result(student=student, accuracy=accuracy, p95_latency_ms=p95_latency_ms)
    return Candidate(student=student, threshold=0.0, calibration=result, certification=result)


def _status_demo_cases() -> tuple[tuple[str, list[Candidate]], ...]:
    """Return synthetic inputs that exercise each certification status."""
    return (
        (
            "CERTIFIED",
            [_synthetic_candidate(student="fast", accuracy=0.987, p95_latency_ms=1.0)],
        ),
        (
            "INSUFFICIENT_EVIDENCE",
            [_synthetic_candidate(student="weak", accuracy=0.92, p95_latency_ms=1.0)],
        ),
        (
            "NO_FEASIBLE_CONFIG",
            [_synthetic_candidate(student="slow", accuracy=0.987, p95_latency_ms=500.0)],
        ),
    )


def _status_demo_payload() -> dict[str, object]:
    """Build the JSON payload for the synthetic status demonstration."""
    cases_payload: list[dict[str, object]] = []
    for expected, candidates in _status_demo_cases():
        outcome = certify_candidates(
            candidates,
            minimum_accuracy=0.95,
            delta=0.05,
            max_p95_latency_ms=200.0,
        )
        cases_payload.append({"expected_status": expected, "outcome": outcome.to_dict()})
    return {
        "synthetic": True,
        "not_a_certificate": True,
        "note": "Status-logic demonstration only; no real dataset or deployment claim.",
        "target_accuracy": 0.95,
        "confidence_level": 0.95,
        "max_p95_latency_ms": 200.0,
        "cases": cases_payload,
    }


@app.command(name="status-demo")
def status_demo(
    output_dir: Path = typer.Option(Path("runs/status-demo"), "--output-dir", "-o"),
) -> None:
    """Demonstrate all three certification statuses with synthetic evidence."""
    payload = _status_demo_payload()
    cases = _status_demo_cases()
    cases_payload = payload["cases"]
    assert isinstance(cases_payload, list)
    table = Table(title="Certification status demonstration (synthetic)")
    table.add_column("Expected")
    table.add_column("Actual")
    table.add_column("Errors")
    for (expected, _candidates), case in zip(cases, cases_payload, strict=True):
        assert isinstance(case, dict)
        outcome = case["outcome"]
        assert isinstance(outcome, dict)
        candidates = outcome["candidates"]
        assert isinstance(candidates, list)
        errors = candidates[0]["errors"] if candidates else 0
        table.add_row(expected, str(outcome["status"]), str(errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "status_demo.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    console.print(table)
    console.print("[yellow]Synthetic demonstration only; this is not a certificate.[/yellow]")
    console.print(f"Wrote [bold]{output_path}[/bold].")


def _validity_scenarios(*, trials: int, examples: int) -> tuple[ValidityScenario, ...]:
    """Return the two pre-registered validity scenarios."""
    return (
        ValidityScenario(
            name="boundary_below_target",
            student_intercept=0.9495,
            student_slope=0.0,
            teacher_accuracy=0.9495,
            target_accuracy=0.95,
            confidence_level=0.95,
            examples=examples,
            calibration_examples=300,
            trials=trials,
            grid_size=200,
            population_size=20_000,
            seed=11,
        ),
        ValidityScenario(
            name="above_target",
            student_intercept=0.96,
            student_slope=0.03,
            teacher_accuracy=0.97,
            target_accuracy=0.95,
            confidence_level=0.95,
            examples=examples,
            calibration_examples=300,
            trials=trials,
            grid_size=200,
            population_size=20_000,
            seed=12,
        ),
    )


@app.command()
def validity(
    output_dir: Path = typer.Option(Path("runs/validity"), "--output-dir", "-o"),
    trials: int = typer.Option(200, "--trials"),
    examples: int = typer.Option(600, "--examples"),
) -> None:
    """Run a Monte Carlo validity check with a naive-threshold control."""
    if trials <= 0 or examples <= 0:
        raise typer.BadParameter("--trials and --examples must be positive")
    scenarios = _validity_scenarios(trials=trials, examples=examples)

    output_dir.mkdir(parents=True, exist_ok=True)
    table = Table(title="Certification validity experiment")
    table.add_column("Scenario")
    table.add_column("Valid false certs")
    table.add_column("Naive false certs")
    table.add_column("Valid cert rate")
    table.add_column("Naive cert rate")
    table.add_column("Allowed")
    for scenario in scenarios:
        report = run_validity_experiment(scenario)
        path = output_dir / f"{scenario.name}.json"
        path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        table.add_row(
            scenario.name,
            f"{report.false_certificate_rate:.3f}",
            f"{report.naive_false_certificate_rate:.3f}",
            f"{report.certification_rate:.3f}",
            f"{report.naive_certification_rate:.3f}",
            f"{report.max_false_certificate_rate_allowed:.3f}",
        )
    console.print(table)
    console.print(f"Wrote reports under [bold]{output_dir}[/bold].")


@app.command()
def certify(
    config: Path = typer.Option(Path("configs/smoke.yaml"), "--config", "-c"),
    run_dir: Path = typer.Option(Path("runs/phase2"), "--run-dir", "-o"),
    calibration_labels: Path | None = typer.Option(
        None,
        "--calibration-labels",
        help="Phase 1 teacher_labels.jsonl (default: <run-dir>/teacher_labels.jsonl).",
    ),
    ledger: Path | None = typer.Option(
        None,
        "--ledger",
        help="Certification ledger (default: <run-dir>/certification_ledger.sqlite3).",
    ),
    allow_reuse: bool = typer.Option(
        False,
        "--allow-reuse",
        help="Permit another look at this split and apply the multiple-looks correction.",
    ),
) -> None:
    """Certify pre-registered cascades with Learn-then-Test on gold labels."""
    parsed = load_config(config)
    split, gold_by_text, _labels = _split_data(parsed)
    calibration_texts = split.calibration
    certification_texts = split.certification
    calibration_gold = [gold_by_text[text] for text in calibration_texts]

    calibration_path = calibration_labels or run_dir / "teacher_labels.jsonl"
    certification_path = run_dir / "certification_teacher_labels.jsonl"
    calibration_predictions = _read_teacher_rows(calibration_path)
    certification_predictions = _read_teacher_rows(certification_path)
    missing_calibration = [
        text for text in calibration_texts if text not in calibration_predictions
    ]
    missing_certification = [
        text for text in certification_texts if text not in certification_predictions
    ]
    if missing_calibration:
        raise typer.BadParameter(
            f"Missing {len(missing_calibration)} calibration teacher labels. "
            "Run `bantam-cascade label ...` first."
        )
    if missing_certification:
        raise typer.BadParameter(
            f"Missing {len(missing_certification)} certification teacher labels. "
            "Run `bantam-cascade build-cascade ...` first."
        )

    calibration_gold_rows = _read_teacher_gold(calibration_path)
    certification_gold_rows = _read_teacher_gold(certification_path)
    wrong_calibration_gold = [
        text for text in calibration_texts if calibration_gold_rows.get(text) != gold_by_text[text]
    ]
    wrong_certification_gold = [
        text
        for text in certification_texts
        if certification_gold_rows.get(text) != gold_by_text[text]
    ]
    if wrong_calibration_gold or wrong_certification_gold:
        raise typer.BadParameter(
            "Saved gold labels do not match the dataset for "
            f"{len(wrong_calibration_gold)} calibration and "
            f"{len(wrong_certification_gold)} certification examples."
        )

    measured = _cascade_results(run_dir / "cascade_results.jsonl")
    missing_students = [name for name in parsed.students if name not in measured]
    extra_students = [name for name in measured if name not in parsed.students]
    if missing_students or extra_students:
        raise typer.BadParameter(
            "Cascade results do not match configured students. "
            f"Missing: {missing_students or 'none'}; unexpected: {extra_students or 'none'}."
        )
    for name, result in measured.items():
        if result.examples != len(certification_texts):
            raise typer.BadParameter(
                f"{name} measured {result.examples} examples but the certification split has "
                f"{len(certification_texts)}. Re-run build-cascade with the full split."
            )

    split_hash = _split_hash(certification_texts)
    ledger_path = ledger or run_dir / "certification_ledger.sqlite3"
    certification_ledger = CertificationLedger(ledger_path)
    try:
        prior_looks = certification_ledger.check(split_hash, allow_reuse=allow_reuse)
    except CertificationReuseError as exc:
        raise typer.BadParameter(str(exc)) from exc

    base_delta = delta_from_confidence_level(parsed.constraints.confidence_level)
    test_delta = corrected_delta(base_delta, prior_looks)

    restored: dict[str, StudentModel] = {}
    candidates: list[Candidate] = []
    for name in parsed.students:
        student_path = run_dir / "students" / f"{name}.joblib"
        if not student_path.is_file():
            raise typer.BadParameter(
                f"Missing trained student {student_path}. Run `train-students` first."
            )
        if name not in restored:
            restored[name] = restore_student(student_path)
        student = restored[name]
        result = measured[name]
        calibration_proba, calibration_latency = _predict_with_latency(student, calibration_texts)
        selected = select_threshold(
            proba=calibration_proba,
            classes=student.classes,
            gold=calibration_gold,
            teacher_predictions=[calibration_predictions[text].label for text in calibration_texts],
            grid_size=parsed.threshold_grid_size,
        )
        if abs(selected.threshold - result.threshold) > 1e-9:
            raise typer.BadParameter(
                f"{name} used threshold {result.threshold:.6f}, but calibration data select "
                f"{selected.threshold:.6f}. Re-run build-cascade with matching labels."
            )
        calibration_result = evaluate_cascade(
            student=name,
            threshold=result.threshold,
            proba=calibration_proba,
            classes=student.classes,
            gold=calibration_gold,
            teacher_predictions=[calibration_predictions[text] for text in calibration_texts],
            student_latency_ms=calibration_latency,
            price_per_million_input_tokens=parsed.teacher.price_per_million_input_tokens,
            price_per_million_output_tokens=parsed.teacher.price_per_million_output_tokens,
        )
        candidates.append(
            Candidate(
                student=name,
                threshold=result.threshold,
                calibration=calibration_result,
                certification=result,
            )
        )

    # The order uses only calibration-split measurements. After this point no
    # certification data is used to change the sequence of hypotheses.
    candidates.sort(
        key=lambda item: (
            -item.calibration.accuracy,
            item.calibration.estimated_cost,
            item.calibration.fallback_rate,
            item.student,
        )
    )
    outcome = certify_candidates(
        candidates,
        minimum_accuracy=parsed.constraints.minimum_accuracy,
        delta=test_delta,
        max_p95_latency_ms=parsed.constraints.max_p95_latency_ms,
    )
    certification_ledger.record(
        split_hash,
        [f"{item.student}:{item.threshold:.6f}" for item in candidates],
        delta=test_delta,
        allow_reuse=allow_reuse,
        prior_looks=prior_looks,
    )

    output_path = run_dir / "certification_result.json"
    payload = outcome.to_dict()
    payload.update(
        {
            "split_hash": split_hash,
            "base_delta": base_delta,
            "prior_looks": prior_looks,
            "provider": parsed.teacher.provider,
            "model": parsed.teacher.model,
            "prompt_version": parsed.teacher.prompt_version,
            "candidate_ordering": "calibration accuracy desc, then cost, fallback, name",
            "ledger": str(ledger_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    table = Table(title="Learn-then-Test certification")
    table.add_column("Student")
    table.add_column("Threshold")
    table.add_column("Errors")
    table.add_column("Accuracy")
    table.add_column("HB p")
    table.add_column("Exact binom p")
    table.add_column(f"delta={test_delta:.4f}")
    table.add_column("Rejected")
    table.add_column("p95 ms")
    for item in outcome.candidates:
        table.add_row(
            item.student,
            f"{item.threshold:.3f}",
            str(item.errors),
            f"{item.measured_accuracy:.3f}",
            f"{item.p_value:.4g}",
            f"{item.exact_binomial_p_value:.4g}",
            f"{item.delta:.4f}",
            "yes" if item.certified else "no",
            f"{item.p95_latency_ms:.1f}",
        )
    console.print(table)
    console.print(
        f"Status: [bold]{outcome.status}[/bold] | target accuracy: "
        f"{parsed.constraints.minimum_accuracy:.3f} | confidence: "
        f"{parsed.constraints.confidence_level:.3f}"
    )
    if outcome.chosen is not None:
        console.print(
            f"Cheapest certified cascade: [bold]{outcome.chosen.student}[/bold] "
            f"(threshold {outcome.chosen.threshold:.3f}, "
            f"estimated cost ${outcome.chosen.estimated_cost:.6f}, "
            f"p95 {outcome.chosen.p95_latency_ms:.1f} ms)."
        )
    else:
        console.print(
            "[yellow]No certified configuration satisfies accuracy, confidence, and "
            "latency constraints.[/yellow]"
        )
    console.print("Assumptions attached to any certified claim:")
    for assumption in CERTIFICATION_ASSUMPTIONS:
        console.print(f"  - {assumption}")
    console.print(f"Wrote [bold]{output_path}[/bold].")
    if outcome.status != "CERTIFIED":
        raise typer.Exit(code=1)


@app.command(name="certify-recall")
def certify_recall(
    config: Path = typer.Option(Path("configs/smoke.yaml"), "--config", "-c"),
    run_dir: Path = typer.Option(Path("runs/phase2"), "--run-dir", "-o"),
    calibration_labels: Path | None = typer.Option(
        None,
        "--calibration-labels",
        help="Phase 1 teacher_labels.jsonl (default: <run-dir>/teacher_labels.jsonl).",
    ),
    preregistration: Path | None = typer.Option(
        None,
        "--preregistration",
        help="Pre-registration JSON (default: <run-dir>/preregistration.json).",
    ),
    ledger: Path | None = typer.Option(
        None,
        "--ledger",
        help="Certification ledger (default: <run-dir>/certification_ledger.sqlite3).",
    ),
) -> None:
    """Certify co-primary accuracy and positive-class recall on a fresh split."""
    parsed = load_config(config)
    prereg_path = preregistration or run_dir / "preregistration.json"
    if not prereg_path.is_file():
        raise typer.BadParameter(f"Pre-registration file not found: {prereg_path}")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))

    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    if prereg.get("config_sha256") != config_hash:
        raise typer.BadParameter("Config hash does not match the pre-registration.")
    split, gold_by_text, labels = _split_data(parsed)
    split_hash = _split_hash(split.certification)
    if prereg.get("split_hash") != split_hash:
        raise typer.BadParameter("Certification split hash does not match the pre-registration.")

    positive_label = str(prereg.get("positive_label", ""))
    if positive_label not in labels:
        raise typer.BadParameter(
            f"Pre-registered positive label {positive_label!r} is not in the dataset labels."
        )
    base_delta = delta_from_confidence_level(parsed.constraints.confidence_level)
    per_endpoint_delta = float(prereg.get("per_endpoint_delta", 0.0))
    if not math.isclose(per_endpoint_delta, base_delta / 2.0, rel_tol=0.0, abs_tol=1e-12):
        raise typer.BadParameter(
            "Pre-registration must split the family-wise delta evenly across the two endpoints."
        )

    calibration_texts = split.calibration
    certification_texts = split.certification
    certification_gold = [gold_by_text[text] for text in certification_texts]
    calibration_path = calibration_labels or run_dir / "teacher_labels.jsonl"
    certification_path = run_dir / "certification_teacher_labels.jsonl"
    calibration_predictions = _read_teacher_rows(calibration_path)
    certification_predictions = _read_teacher_rows(certification_path)
    missing_calibration = [
        text for text in calibration_texts if text not in calibration_predictions
    ]
    missing_certification = [
        text for text in certification_texts if text not in certification_predictions
    ]
    if missing_calibration or missing_certification:
        raise typer.BadParameter(
            f"Missing {len(missing_calibration)} calibration and "
            f"{len(missing_certification)} certification teacher labels."
        )
    calibration_gold_rows = _read_teacher_gold(calibration_path)
    certification_gold_rows = _read_teacher_gold(certification_path)
    wrong_gold = [
        text for text in calibration_texts if calibration_gold_rows.get(text) != gold_by_text[text]
    ] + [
        text
        for text in certification_texts
        if certification_gold_rows.get(text) != gold_by_text[text]
    ]
    if wrong_gold:
        raise typer.BadParameter(
            f"Saved gold labels do not match the dataset for {len(wrong_gold)} examples."
        )

    ledger_path = ledger or run_dir / "certification_ledger.sqlite3"
    certification_ledger = CertificationLedger(ledger_path)
    try:
        prior_looks = certification_ledger.check(split_hash, allow_reuse=False)
    except CertificationReuseError as exc:
        raise typer.BadParameter(str(exc)) from exc

    measured = _cascade_results(run_dir / "cascade_results.jsonl")
    missing_students = [name for name in parsed.students if name not in measured]
    extra_students = [name for name in measured if name not in parsed.students]
    if missing_students or extra_students:
        raise typer.BadParameter(
            "Cascade results do not match configured students. "
            f"Missing: {missing_students or 'none'}; unexpected: {extra_students or 'none'}."
        )
    for name, result in measured.items():
        if result.examples != len(certification_texts):
            raise typer.BadParameter(
                f"{name} measured {result.examples} examples but the certification split has "
                f"{len(certification_texts)}. Re-run build-cascade with the full split."
            )

    restored: dict[str, StudentModel] = {}
    candidates: list[Candidate] = []
    for name in parsed.students:
        student_path = run_dir / "students" / f"{name}.joblib"
        if not student_path.is_file():
            raise typer.BadParameter(
                f"Missing trained student {student_path}. Run `train-students` first."
            )
        restored[name] = restore_student(student_path)
        student = restored[name]
        result = measured[name]
        calibration_proba, calibration_latency = _predict_with_latency(student, calibration_texts)
        selected = select_threshold(
            proba=calibration_proba,
            classes=student.classes,
            gold=[gold_by_text[text] for text in calibration_texts],
            teacher_predictions=[calibration_predictions[text].label for text in calibration_texts],
            grid_size=parsed.threshold_grid_size,
        )
        if abs(selected.threshold - result.threshold) > 1e-9:
            raise typer.BadParameter(
                f"{name} used threshold {result.threshold:.6f}, but calibration selects "
                f"{selected.threshold:.6f}. Re-run build-cascade with matching labels."
            )
        calibration_result = evaluate_cascade(
            student=name,
            threshold=result.threshold,
            proba=calibration_proba,
            classes=student.classes,
            gold=[gold_by_text[text] for text in calibration_texts],
            teacher_predictions=[calibration_predictions[text] for text in calibration_texts],
            student_latency_ms=calibration_latency,
            price_per_million_input_tokens=parsed.teacher.price_per_million_input_tokens,
            price_per_million_output_tokens=parsed.teacher.price_per_million_output_tokens,
        )
        candidates.append(
            Candidate(
                student=name,
                threshold=result.threshold,
                calibration=calibration_result,
                certification=result,
            )
        )
    candidates.sort(
        key=lambda item: (
            -item.calibration.accuracy,
            item.calibration.estimated_cost,
            item.calibration.fallback_rate,
            item.student,
        )
    )

    recall_counts: list[tuple[int, int]] = []
    for candidate in candidates:
        student = restored[candidate.student]
        proba, _certification_latency = _predict_with_latency(student, certification_texts)
        predicted = labels_from_proba(proba, student.classes)
        confidences = confidence_scores(proba)
        cascade_labels = [
            predicted[index]
            if confidences[index] >= candidate.threshold
            else certification_predictions[text].label
            for index, text in enumerate(certification_texts)
        ]
        positive_indices = [
            index for index, gold in enumerate(certification_gold) if gold == positive_label
        ]
        if not positive_indices:
            raise typer.BadParameter(
                f"Certification split has no {positive_label!r} examples; recall is not testable."
            )
        recall_errors = sum(
            1 for index in positive_indices if cascade_labels[index] != positive_label
        )
        recall_counts.append((recall_errors, len(positive_indices)))

    test_delta = corrected_delta(base_delta, prior_looks)
    outcome = certify_candidates_with_recall(
        candidates,
        minimum_accuracy=parsed.constraints.minimum_accuracy,
        minimum_recall=parsed.constraints.minimum_accuracy,
        delta=test_delta,
        max_p95_latency_ms=parsed.constraints.max_p95_latency_ms,
        recall_counts=recall_counts,
    )
    certification_ledger.record(
        split_hash,
        [f"{item.student}:{item.threshold:.6f}" for item in candidates],
        delta=test_delta,
        allow_reuse=False,
        prior_looks=prior_looks,
    )

    payload = dict(outcome)
    payload.update(
        {
            "split_hash": split_hash,
            "config_sha256": config_hash,
            "base_delta": base_delta,
            "prior_looks": prior_looks,
            "positive_label": positive_label,
            "preregistration": str(prereg_path),
            "provider": parsed.teacher.provider,
            "model": parsed.teacher.model,
            "prompt_version": parsed.teacher.prompt_version,
            "candidate_ordering": "calibration accuracy desc, then cost, fallback, name",
            "ledger": str(ledger_path),
        }
    )
    output_path = run_dir / "recall_certification_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    table = Table(title="Co-primary accuracy and recall certification")
    table.add_column("Student")
    table.add_column("Acc err")
    table.add_column("Acc HB p")
    table.add_column("Recall err/n")
    table.add_column("Recall HB p")
    table.add_column(f"delta={per_endpoint_delta:.4f}")
    table.add_column("Rejected")
    table.add_column("p95 ms")
    candidate_rows = outcome["candidates"]
    assert isinstance(candidate_rows, list)
    for row in candidate_rows:
        assert isinstance(row, dict)
        table.add_row(
            str(row["student"]),
            str(row["accuracy_errors"]),
            f"{float(row['accuracy_p_value']):.4g}",
            f"{row['recall_errors']}/{row['recall_examples']}",
            f"{float(row['recall_p_value']):.4g}",
            f"{float(row['per_endpoint_delta']):.4f}",
            "yes" if row["certified"] else "no",
            f"{float(row['p95_latency_ms']):.1f}",
        )
    console.print(table)
    console.print(
        f"Status: [bold]{outcome['status']}[/bold] | accuracy and {positive_label} recall target: "
        f"{parsed.constraints.minimum_accuracy:.3f} | family-wise delta: {test_delta:.4f}"
    )
    console.print("Assumptions attached to any certified claim:")
    assumptions = outcome["assumptions"]
    assert isinstance(assumptions, list)
    for assumption in assumptions:
        assert isinstance(assumption, str)
        console.print(f"  - {assumption}")
    console.print(f"Wrote [bold]{output_path}[/bold].")
    if outcome["status"] != "CERTIFIED":
        raise typer.Exit(code=1)


@app.command(name="verify-certificate")
def verify_certificate(
    config: Path = typer.Option(
        Path("configs/sms-spam-groq.yaml"),
        "--config",
        "-c",
        help="Frozen config whose hash was preregistered.",
    ),
    run_dir: Path = typer.Option(
        Path("runs/sms-groq"), "--run-dir", "-o", help="Run directory with the frozen artifacts."
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Where to write the verification report."
    ),
) -> None:
    """Verify a frozen certificate's hashes and arithmetic without a new test look."""
    try:
        report = verify_frozen_certificate(config_path=config, run_dir=run_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    output_path = output or run_dir.parent / "certificate-verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    table = Table(title="Frozen certificate verification (no new test look)")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Detail")
    checks = report["checks"]
    assert isinstance(checks, list)
    for check in checks:
        assert isinstance(check, dict)
        table.add_row(
            str(check["name"]),
            "pass" if check["passed"] else "FAIL",
            str(check["detail"]),
        )
    console.print(table)
    verified = bool(report["verified"])
    console.print(f"Verified: [bold]{verified}[/bold] | wrote [bold]{output_path}[/bold].")
    if not verified:
        raise typer.Exit(code=1)


@app.command()
def reproduce(
    output_dir: Path = typer.Option(Path("runs/reproduction"), "--output-dir", "-o"),
    trials: int = typer.Option(200, "--trials"),
    examples: int = typer.Option(600, "--examples"),
    certificate_config: Path = typer.Option(
        Path("configs/sms-spam-groq.yaml"), "--certificate-config"
    ),
    certificate_run_dir: Path = typer.Option(Path("runs/sms-groq"), "--certificate-run-dir"),
) -> None:
    """Regenerate all offline evidence and verify the frozen certificate."""
    if trials <= 0 or examples <= 0:
        raise typer.BadParameter("--trials and --examples must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        verification = verify_frozen_certificate(
            config_path=certificate_config, run_dir=certificate_run_dir
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    (output_dir / "certificate_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    plan = {
        "target_accuracy": 0.95,
        "confidence_level": 0.95,
        "sizing": {
            f"{assumed:.3f}": required_examples_for_accuracy(
                target_accuracy=0.95,
                confidence_level=0.95,
                assumed_accuracy=assumed,
            )
            for assumed in (0.970, 0.975, 0.980, 0.985, 0.990)
        },
    }
    (output_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    validity_reports: list[dict[str, object]] = []
    for scenario in _validity_scenarios(trials=trials, examples=examples):
        report = run_validity_experiment(scenario).to_dict()
        validity_reports.append(report)
        (output_dir / f"validity_{scenario.name}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    status_payload = _status_demo_payload()
    (output_dir / "status_demo.json").write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "note": (
            "One-command offline reproduction of the pre-test size plan, validity "
            "simulation, status demonstration, and frozen-certificate verification. "
            "No certification split was re-labelled or tested again. Live teacher "
            "labels require user-supplied keys and are not regenerated here."
        ),
        "version": __version__,
        "certificate_verified": verification["verified"],
        "plan": plan,
        "validity": validity_reports,
        "status_demo_synthetic": True,
        "commands_for_live_runs": [
            "bantam-cascade label "
            "--config configs/sms-spam-recall-groq.yaml --run-dir runs/sms-recall-groq",
            "bantam-cascade train-students "
            "--config configs/sms-spam-recall-groq.yaml --run-dir runs/sms-recall-groq",
            "bantam-cascade build-cascade "
            "--config configs/sms-spam-recall-groq.yaml --run-dir runs/sms-recall-groq",
            "bantam-cascade certify-recall "
            "--config configs/sms-spam-recall-groq.yaml --run-dir runs/sms-recall-groq",
        ],
    }
    manifest_path = output_dir / "reproduction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    console.print(
        f"Offline evidence written under [bold]{output_dir}[/bold]. "
        f"Frozen certificate verified: [bold]{verification['verified']}[/bold]."
    )
    if not verification["verified"]:
        raise typer.Exit(code=1)
