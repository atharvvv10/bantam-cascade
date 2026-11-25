from pathlib import Path

import pytest
from typer.testing import CliRunner

import bantam_cascade.cli as cli_module
import bantam_cascade.teacher.client as client_module
from bantam_cascade.cli import app
from bantam_cascade.data import make_smoke_dataset
from bantam_cascade.teacher.select import CandidateMetrics

RUNNER = CliRunner()
ROOT = Path(__file__).resolve().parents[1]


def test_help_lists_phase1_commands() -> None:
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "label" in result.stdout
    assert "select-teacher" in result.stdout


def test_label_dry_run_is_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(client_module, "_env_file_paths", lambda: ())
    result = RUNNER.invoke(
        app,
        [
            "label",
            "--config",
            str(ROOT / "configs" / "smoke.yaml"),
            "--run-dir",
            str(tmp_path / "run"),
            "--limit",
            "10",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Dry run" in result.stdout
    assert "no API calls" in result.stdout
    assert "Estimated days" in result.stdout


def test_label_refuses_live_smoke_run(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "label",
            "--config",
            str(ROOT / "configs" / "smoke.yaml"),
            "--run-dir",
            str(tmp_path / "run"),
            "--limit",
            "1",
        ],
    )
    assert result.exit_code != 0
    assert "offline-only" in result.output


def test_select_teacher_with_mocked_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli_module,
        "load_dataset_for_config",
        lambda config: make_smoke_dataset(n_per_label=3000, seed=config.seed),
    )

    def fake_evaluate(**kwargs: object) -> CandidateMetrics:
        return CandidateMetrics(
            model=str(kwargs["model"]),
            examples=10,
            correct=9,
            invalid=0,
            input_tokens=100,
            output_tokens=20,
            total_latency_ms=50.0,
            estimated_cost=0.0,
        )

    monkeypatch.setattr(cli_module, "evaluate_candidate", fake_evaluate)
    result = RUNNER.invoke(
        app,
        [
            "select-teacher",
            "--config",
            str(ROOT / "configs" / "banking77.yaml"),
            "--run-dir",
            str(tmp_path / "run"),
            "--models",
            "m1,m2",
            "--limit",
            "10",
            "--best-student-accuracy",
            "0.8",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Recommended teacher" in result.stdout
    assert "Sanity gate passed" in result.stdout
