import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from bantam_cascade.calibration import select_threshold
from bantam_cascade.cascade import TeacherPrediction, evaluate_cascade
from bantam_cascade.cli import _split_data, _split_hash, app
from bantam_cascade.config import load_config
from bantam_cascade.data import make_smoke_dataset
from bantam_cascade.splits import split_by_hash
from bantam_cascade.students import StudentModel, restore_student

RUNNER = CliRunner()
ROOT = Path(__file__).resolve().parents[1]


def _write_config(tmp_path: Path, *, minimum_accuracy: float = 0.8) -> Path:
    raw = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    raw["constraints"]["minimum_accuracy"] = minimum_accuracy
    raw["students"] = ["tfidf_baseline"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _write_teacher_rows(path: Path, texts: tuple[str, ...], gold: dict[str, str]) -> None:
    rows = [
        {
            "text": text,
            "gold": gold[text],
            "predicted": gold[text],
            "input_tokens": 100,
            "output_tokens": 0,
            "latency_ms": 1.0,
            "cached": False,
        }
        for text in texts
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def offline_phase3(tmp_path: Path) -> tuple[Path, Path]:
    """Train a student and write offline teacher/cascade artifacts."""
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    run_dir = tmp_path / "run"
    trained = RUNNER.invoke(
        app,
        [
            "train-students",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
        ],
    )
    assert trained.exit_code == 0, trained.output

    dataset = make_smoke_dataset(seed=config.seed)
    gold = {example.text: example.label for example in dataset.train}
    split = split_by_hash(
        list(gold),
        config=config.splits,
        official_test=tuple(example.text for example in dataset.test),
    )
    _write_teacher_rows(run_dir / "teacher_labels.jsonl", split.calibration, gold)
    _write_teacher_rows(run_dir / "certification_teacher_labels.jsonl", split.certification, gold)

    student = restore_student(run_dir / "students" / "tfidf_baseline.joblib")
    calibration_proba, _ = _predict(student, split.calibration)
    threshold = select_threshold(
        proba=calibration_proba,
        classes=student.classes,
        gold=[gold[text] for text in split.calibration],
        teacher_predictions=[gold[text] for text in split.calibration],
        grid_size=config.threshold_grid_size,
    ).threshold
    certification_proba, latency = _predict(student, split.certification)
    result = evaluate_cascade(
        student="tfidf_baseline",
        threshold=threshold,
        proba=certification_proba,
        classes=student.classes,
        gold=[gold[text] for text in split.certification],
        teacher_predictions=[
            TeacherPrediction(
                label=gold[text],
                input_tokens=100,
                output_tokens=0,
                latency_ms=1.0,
            )
            for text in split.certification
        ],
        student_latency_ms=latency,
        price_per_million_input_tokens=0.0,
        price_per_million_output_tokens=0.0,
    )
    results_path = run_dir / "cascade_results.jsonl"
    results_path.write_text(json.dumps(result.__dict__) + "\n", encoding="utf-8")
    return config_path, run_dir


def _predict(student: StudentModel, texts: tuple[str, ...]) -> tuple[np.ndarray, list[float]]:
    rows = []
    latencies: list[float] = []
    for text in texts:
        rows.append(np.asarray(student.predict_proba([text]), dtype=np.float64)[0])
        latencies.append(1.0)
    return np.asarray(rows), latencies


def test_help_lists_certify_command() -> None:
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "certify" in result.stdout


def test_certify_certifies_offline(offline_phase3: tuple[Path, Path]) -> None:
    config_path, run_dir = offline_phase3
    result = RUNNER.invoke(
        app,
        ["certify", "--config", str(config_path), "--run-dir", str(run_dir)],
    )
    assert result.exit_code == 0, result.output
    assert "CERTIFIED" in result.stdout
    assert "Assumptions attached" in result.stdout
    payload = json.loads((run_dir / "certification_result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "CERTIFIED"
    assert payload["chosen"]["student"] == "tfidf_baseline"
    assert payload["prior_looks"] == 0
    assert payload["delta"] == pytest.approx(0.1)


def test_certify_refuses_reuse_unless_opted_in(offline_phase3: tuple[Path, Path]) -> None:
    config_path, run_dir = offline_phase3
    args = ["certify", "--config", str(config_path), "--run-dir", str(run_dir)]
    first = RUNNER.invoke(app, args)
    assert first.exit_code == 0, first.output

    second = RUNNER.invoke(app, args)
    assert second.exit_code != 0
    assert "already been used" in second.output

    with pytest.warns(UserWarning, match="multiple-looks correction"):
        third = RUNNER.invoke(app, args + ["--allow-reuse"])
    assert third.exit_code == 0, third.output
    payload = json.loads((run_dir / "certification_result.json").read_text(encoding="utf-8"))
    assert payload["prior_looks"] == 1
    assert payload["delta"] == pytest.approx(0.05)


def test_certify_reports_insufficient_evidence(offline_phase3: tuple[Path, Path]) -> None:
    config_path, run_dir = offline_phase3
    results_path = run_dir / "cascade_results.jsonl"
    row = json.loads(results_path.read_text(encoding="utf-8"))
    row["accuracy"] = 6 / row["examples"]
    row["student_only_accuracy"] = 6 / row["examples"]
    results_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = RUNNER.invoke(
        app,
        ["certify", "--config", str(config_path), "--run-dir", str(run_dir)],
    )
    assert result.exit_code == 1
    assert "INSUFFICIENT_EVIDENCE" in result.stdout
    payload = json.loads((run_dir / "certification_result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "INSUFFICIENT_EVIDENCE"
    assert payload["chosen"] is None


def test_certify_requires_cascade_results(offline_phase3: tuple[Path, Path]) -> None:
    config_path, run_dir = offline_phase3
    (run_dir / "cascade_results.jsonl").unlink()
    result = RUNNER.invoke(
        app,
        ["certify", "--config", str(config_path), "--run-dir", str(run_dir)],
    )
    assert result.exit_code != 0
    assert "build-cascade" in result.output


def _write_recall_prereg(config_path: Path, run_dir: Path) -> str:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["constraints"]["minimum_accuracy"] = 0.6
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)
    split, _gold, labels = _split_data(config)
    positive_label = labels[0]
    payload = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "split_hash": _split_hash(split.certification),
        "positive_label": positive_label,
        "per_endpoint_delta": 0.05,
    }
    (run_dir / "preregistration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return positive_label


def test_certify_recall_certifies_offline(offline_phase3: tuple[Path, Path]) -> None:
    config_path, run_dir = offline_phase3
    positive_label = _write_recall_prereg(config_path, run_dir)
    result = RUNNER.invoke(
        app,
        ["certify-recall", "--config", str(config_path), "--run-dir", str(run_dir)],
    )
    assert result.exit_code == 0, result.output
    assert "CERTIFIED" in result.stdout
    payload = json.loads((run_dir / "recall_certification_result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "CERTIFIED"
    assert payload["positive_label"] == positive_label
    assert payload["per_endpoint_delta"] == pytest.approx(0.05)
    assert payload["prior_looks"] == 0


def test_certify_recall_requires_matching_config(offline_phase3: tuple[Path, Path]) -> None:
    config_path, run_dir = offline_phase3
    _write_recall_prereg(config_path, run_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["seed"] = 43
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    result = RUNNER.invoke(
        app,
        ["certify-recall", "--config", str(config_path), "--run-dir", str(run_dir)],
    )
    assert result.exit_code != 0
    assert "Config hash" in result.output
