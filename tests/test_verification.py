import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from bantam_cascade.certification import (
    alpha_from_minimum_accuracy,
    delta_from_confidence_level,
    hb_p_value,
)
from bantam_cascade.cli import _split_data, _split_hash, app
from bantam_cascade.config import load_config
from bantam_cascade.ledger import CertificationLedger
from bantam_cascade.splits import splits_fingerprint
from bantam_cascade.verification import verify_frozen_certificate

RUNNER = CliRunner()
ROOT = Path(__file__).resolve().parents[1]


def _artifacts(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    config_path = tmp_path / "config.yaml"
    config_path.write_bytes((ROOT / "configs" / "smoke.yaml").read_bytes())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = load_config(config_path)
    split, _gold, _labels = _split_data(config)
    threshold = 0.5
    examples = len(split.certification)
    errors = 0
    alpha = alpha_from_minimum_accuracy(config.constraints.minimum_accuracy)
    delta = delta_from_confidence_level(config.constraints.confidence_level)
    chosen = {
        "student": "tfidf_baseline",
        "threshold": threshold,
        "p_value": hb_p_value(errors=errors, examples=examples, alpha=alpha),
        "certified": True,
        "latency_ok": True,
        "examples": examples,
        "errors": errors,
        "measured_accuracy": 1.0 - errors / examples,
        "calibration_accuracy": 1.0,
        "fallback_rate": 0.0,
        "estimated_cost": 0.0,
        "p95_latency_ms": 1.0,
    }
    prereg = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "splits_fingerprint": splits_fingerprint(split, seed=config.splits.seed),
        "selected_threshold": threshold,
        "minimum_accuracy": config.constraints.minimum_accuracy,
        "confidence_level": config.constraints.confidence_level,
        "max_p95_latency_ms": config.constraints.max_p95_latency_ms,
    }
    ledger_path = run_dir / "certification_ledger.sqlite3"
    ledger = CertificationLedger(ledger_path)
    split_hash = _split_hash(split.certification)
    ledger.record(split_hash, ["tfidf_baseline:0.500000"], delta=delta, allow_reuse=False)
    result = {
        "status": "CERTIFIED",
        "alpha": alpha,
        "delta": delta,
        "chosen": chosen,
        "certified": [chosen],
        "candidates": [chosen],
        "split_hash": split_hash,
        "prior_looks": 0,
        "ledger": str(ledger_path),
    }
    (run_dir / "preregistration.json").write_text(
        json.dumps(prereg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "certification_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config_path, run_dir, result


def test_verify_frozen_certificate_detects_tampering(tmp_path: Path) -> None:
    config_path, run_dir, result = _artifacts(tmp_path)
    report = verify_frozen_certificate(config_path=config_path, run_dir=run_dir)
    assert report["verified"] is True

    result["chosen"]["p_value"] = 0.5  # type: ignore[index]
    (run_dir / "certification_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = verify_frozen_certificate(config_path=config_path, run_dir=run_dir)
    assert report["verified"] is False


def test_verify_certificate_cli_writes_report(tmp_path: Path) -> None:
    config_path, run_dir, _result = _artifacts(tmp_path)
    output = tmp_path / "verification.json"
    invocation = RUNNER.invoke(
        app,
        [
            "verify-certificate",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["verified"] is True


def test_reproduce_cli_regenerates_offline_evidence(tmp_path: Path) -> None:
    config_path, run_dir, _result = _artifacts(tmp_path)
    output = tmp_path / "reproduction"
    invocation = RUNNER.invoke(
        app,
        [
            "reproduce",
            "--output-dir",
            str(output),
            "--trials",
            "2",
            "--examples",
            "40",
            "--certificate-config",
            str(config_path),
            "--certificate-run-dir",
            str(run_dir),
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    payload = json.loads((output / "reproduction_manifest.json").read_text(encoding="utf-8"))
    assert payload["certificate_verified"] is True
    assert payload["plan"]["sizing"]["0.980"] == 240
    assert (output / "validity_boundary_below_target.json").is_file()
    status = json.loads((output / "status_demo.json").read_text(encoding="utf-8"))
    assert status["synthetic"] is True
