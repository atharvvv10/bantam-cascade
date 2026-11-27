"""Offline verification of a frozen certification artifact.

This module checks hashes and arithmetic in an already-recorded result. It never
re-labels the certification split, re-runs a model, or makes another statistical
look at the data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from bantam_cascade.certification import (
    alpha_from_minimum_accuracy,
    delta_from_confidence_level,
    exact_binomial_p_value,
    hb_p_value,
)
from bantam_cascade.config import load_config
from bantam_cascade.data import load_dataset_for_config
from bantam_cascade.splits import certification_split_hash, split_by_hash, splits_fingerprint


@dataclass(frozen=True)
class VerificationCheck:
    """One independent consistency check on a frozen result."""

    name: str
    passed: bool
    detail: str


def _check(checks: list[VerificationCheck], name: str, passed: bool, detail: str = "") -> None:
    checks.append(VerificationCheck(name=name, passed=passed, detail=detail))


def verify_frozen_certificate(*, config_path: Path, run_dir: Path) -> dict[str, object]:
    """Recheck hashes, p-values, and status recorded in a frozen certificate.

    Args:
        config_path: Config whose bytes were hashed in the pre-registration.
        run_dir: Directory holding ``preregistration.json`` and
            ``certification_result.json``.

    Returns:
        A JSON-serializable report with individual pass/fail checks and an
        overall ``verified`` flag.

    Raises:
        FileNotFoundError: If a required artifact is missing.
        ValueError: If the artifacts cannot be parsed.
    """
    prereg_path = run_dir / "preregistration.json"
    result_path = run_dir / "certification_result.json"
    for path in (config_path, prereg_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing verification input: {path}")

    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(prereg, dict) or not isinstance(result, dict):
        raise ValueError("Pre-registration and result must be JSON objects")

    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config = load_config(config_path)
    dataset = load_dataset_for_config(config)
    gold_by_text = {example.text: example.label for example in dataset.train}
    split = split_by_hash(
        list(gold_by_text),
        config=config.splits,
        official_test=tuple(example.text for example in dataset.test),
    )
    split_hash = certification_split_hash(split.certification)
    fingerprint = splits_fingerprint(split, seed=config.splits.seed)

    checks: list[VerificationCheck] = []
    _check(
        checks,
        "config_hash_matches_preregistration",
        prereg.get("config_sha256") == config_hash,
        config_hash,
    )
    _check(
        checks,
        "splits_fingerprint_matches_preregistration",
        prereg.get("splits_fingerprint") == fingerprint,
        fingerprint,
    )
    _check(
        checks,
        "split_hash_matches_result",
        result.get("split_hash") == split_hash,
        split_hash,
    )
    _check(checks, "status_is_certified", result.get("status") == "CERTIFIED")
    _check(checks, "no_prior_looks", result.get("prior_looks") == 0)

    chosen = result.get("chosen")
    if not isinstance(chosen, dict):
        chosen = {}
    threshold = float(prereg.get("selected_threshold", -1.0))
    _check(
        checks,
        "threshold_matches_preregistration",
        abs(float(chosen.get("threshold", -2.0)) - threshold) <= 1e-12,
        f"pre-registered {threshold}",
    )

    examples = int(chosen.get("examples", 0))
    errors = int(chosen.get("errors", -1))
    measured_accuracy = float(chosen.get("measured_accuracy", -1.0))
    expected_errors = round((1.0 - measured_accuracy) * examples)
    _check(
        checks,
        "error_count_matches_measured_accuracy",
        examples > 0 and errors == expected_errors,
        f"{errors} / {examples}",
    )

    minimum_accuracy = float(prereg.get("minimum_accuracy", config.constraints.minimum_accuracy))
    confidence_level = float(prereg.get("confidence_level", config.constraints.confidence_level))
    alpha = alpha_from_minimum_accuracy(minimum_accuracy)
    delta = delta_from_confidence_level(confidence_level)
    recomputed = hb_p_value(errors=errors, examples=examples, alpha=alpha)
    stored = float(chosen.get("p_value", -1.0))
    _check(
        checks,
        "hoeffding_bentkus_p_value_recomputed",
        abs(recomputed - stored) <= 1e-12,
        f"stored {stored:.6g}, recomputed {recomputed:.6g}",
    )
    if "exact_binomial_p_value" in chosen:
        exact = exact_binomial_p_value(errors=errors, examples=examples, alpha=alpha)
        stored_exact = float(chosen["exact_binomial_p_value"])
        _check(
            checks,
            "exact_binomial_p_value_recomputed",
            abs(exact - stored_exact) <= 1e-12,
            f"stored {stored_exact:.6g}, recomputed {exact:.6g}",
        )
    _check(
        checks,
        "p_value_below_delta",
        stored <= delta,
        f"p={stored:.6g}, delta={delta:.6g}",
    )
    _check(
        checks,
        "result_delta_matches_confidence_level",
        abs(float(result.get("delta", -1.0)) - delta) <= 1e-12,
        f"delta={delta:.6g}",
    )
    budget = float(prereg.get("max_p95_latency_ms", config.constraints.max_p95_latency_ms))
    p95 = float(chosen.get("p95_latency_ms", float("inf")))
    _check(
        checks,
        "p95_latency_within_budget",
        p95 <= budget,
        f"p95={p95:.3f} ms, budget={budget:.3f} ms",
    )

    ledger_value = result.get("ledger")
    ledger_path = (
        Path(str(ledger_value)) if ledger_value else run_dir / "certification_ledger.sqlite3"
    )
    if not ledger_path.is_absolute() and not ledger_path.is_file():
        ledger_path = run_dir.parent.parent / ledger_path
    if ledger_path.is_file():
        from bantam_cascade.ledger import CertificationLedger

        looks = CertificationLedger(ledger_path).prior_looks(split_hash)
        _check(checks, "ledger_has_single_look", looks == 1, f"{looks} recorded look(s)")
    else:
        _check(checks, "ledger_has_single_look", False, f"missing ledger: {ledger_path}")

    verified = all(check.passed for check in checks)
    return {
        "verified": verified,
        "note": (
            "Offline verification of stored hashes, counts, p-values, and ledger only. "
            "The certification split was not re-labelled or tested again."
        ),
        "status": result.get("status"),
        "config_sha256": config_hash,
        "split_hash": split_hash,
        "checks": [asdict(check) for check in checks],
    }
