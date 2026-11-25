import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import bantam_cascade.cli as cli_module
from bantam_cascade.cli import app
from bantam_cascade.config import load_config
from bantam_cascade.data import make_smoke_dataset
from bantam_cascade.splits import split_by_hash

RUNNER = CliRunner()
ROOT = Path(__file__).resolve().parents[1]


def _write_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load((ROOT / "configs" / "banking77.yaml").read_text(encoding="utf-8"))
    raw["students"] = ["static_embed"]
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
            "output_tokens": 10,
            "latency_ms": 5.0,
            "cached": False,
        }
        for text in texts
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class _StubTeacherClient:
    """Teacher client stub that fails if a live call is attempted."""

    def __init__(self, config: object) -> None:
        """Accept the teacher config but make no network calls."""
        self.config = config

    def dry_run_estimate(
        self,
        texts: object,
        labels: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        """Return a canned estimate without a key or network access."""
        return SimpleNamespace(
            texts=len(texts),
            requests=len(texts),
            cached=0,
            input_tokens=1000,
            output_tokens=100,
            estimated_days=1,
            estimated_cost=0.0,
        )

    def label_many(self, *args: object, **kwargs: object) -> object:
        """Fail loudly: all teacher labels should already be cached on disk."""
        raise AssertionError("Unexpected live teacher call in offline CLI test")

    def close(self) -> None:
        """Close the stub client."""


@pytest.fixture
def offline_phase2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Set up a config, dataset, and pre-saved teacher labels for offline tests."""
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    dataset = make_smoke_dataset(n_per_label=3000, seed=config.seed)
    monkeypatch.setattr(cli_module, "load_dataset_for_config", lambda _config: dataset)
    monkeypatch.setattr(cli_module, "GroqTeacherClient", _StubTeacherClient)
    gold = {example.text: example.label for example in dataset.train}
    split = split_by_hash(
        list(gold),
        config=config.splits,
        official_test=tuple(example.text for example in dataset.test),
    )
    run_dir = tmp_path / "run"
    _write_teacher_rows(run_dir / "teacher_labels.jsonl", split.calibration, gold)
    _write_teacher_rows(run_dir / "certification_teacher_labels.jsonl", split.certification, gold)
    return config_path, run_dir


def test_help_lists_phase2_commands() -> None:
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train-students" in result.stdout
    assert "build-cascade" in result.stdout


def test_train_students_offline(offline_phase2: tuple[Path, Path]) -> None:
    config_path, run_dir = offline_phase2
    result = RUNNER.invoke(
        app,
        [
            "train-students",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--limit",
            "200",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Student training" in result.stdout
    assert (run_dir / "students" / "static_embed.joblib").is_file()


def test_build_cascade_dry_run_is_offline(offline_phase2: tuple[Path, Path]) -> None:
    config_path, run_dir = offline_phase2
    result = RUNNER.invoke(
        app,
        [
            "build-cascade",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--limit",
            "20",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.stdout
    assert "no API calls" in result.stdout


def test_train_then_build_cascade_offline(offline_phase2: tuple[Path, Path]) -> None:
    config_path, run_dir = offline_phase2
    trained = RUNNER.invoke(
        app,
        [
            "train-students",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--limit",
            "200",
        ],
    )
    assert trained.exit_code == 0, trained.output
    built = RUNNER.invoke(
        app,
        [
            "build-cascade",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--limit",
            "20",
        ],
    )
    assert built.exit_code == 0, built.output
    assert "Cascade candidates" in built.stdout
    results_path = run_dir / "cascade_results.jsonl"
    assert results_path.is_file()
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["student"] == "static_embed"
    assert 0.0 <= rows[0]["accuracy"] <= 1.0


def test_build_cascade_requires_training(offline_phase2: tuple[Path, Path]) -> None:
    config_path, run_dir = offline_phase2
    result = RUNNER.invoke(
        app,
        [
            "build-cascade",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--limit",
            "5",
        ],
    )
    assert result.exit_code != 0
    assert "train-students" in result.output
