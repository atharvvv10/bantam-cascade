from pathlib import Path

from typer.testing import CliRunner

from bantam_cascade.cli import app

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]


def test_help_and_version() -> None:
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "bantam-cascade" in help_result.stdout
    assert "init" in help_result.stdout

    version_result = runner.invoke(app, ["--version"])
    assert version_result.exit_code == 0
    assert "bantam-cascade" in version_result.stdout


def test_init_copies_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "init",
            "--config",
            str(ROOT / "configs" / "smoke.yaml"),
            "--run-dir",
            str(run_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (run_dir / "config.yaml").is_file()


def test_demo_classifies_offline(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = ROOT / "configs" / "smoke.yaml"
    assert (
        runner.invoke(app, ["init", "--config", str(config), "--run-dir", str(run_dir)]).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app,
            ["train-students", "--config", str(config), "--run-dir", str(run_dir)],
        ).exit_code
        == 0
    )
    result = runner.invoke(app, ["demo", "billing invoice help", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.stdout
    assert "Predicted label" in result.stdout
    assert "static_embed" in result.stdout
    assert "does not re-run certification" in result.stdout


def test_init_rejects_missing_config(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["init", "--config", str(tmp_path / "missing.yaml"), "--run-dir", str(tmp_path / "run")],
    )
    assert result.exit_code != 0
