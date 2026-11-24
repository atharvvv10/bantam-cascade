import sqlite3

import pytest

from bantam_cascade.ledger import CertificationLedger, CertificationReuseError, corrected_delta


def test_first_certification_is_recorded(tmp_path) -> None:
    ledger = CertificationLedger(tmp_path / "ledger.db")
    assert ledger.prior_looks("split-1") == 0
    assert ledger.record("split-1", ["candidate-a", "candidate-b"], delta=0.05) == 0
    assert ledger.prior_looks("split-1") == 1
    assert ledger.prior_looks("split-2") == 0

    with sqlite3.connect(tmp_path / "ledger.db") as conn:
        row = conn.execute(
            "SELECT configs_json, delta, prior_looks, created_at FROM certifications"
        ).fetchone()
    assert row is not None
    assert '"candidate-a"' in row[0]
    assert row[1] == pytest.approx(0.05)
    assert row[2] == 0
    assert row[3]


def test_reuse_requires_opt_in_and_warns(tmp_path) -> None:
    ledger = CertificationLedger(tmp_path / "ledger.db")
    ledger.record("split-1", ["c"], delta=0.05)
    with pytest.raises(CertificationReuseError, match="already been used"):
        ledger.record("split-1", ["c"], delta=0.05)

    with pytest.warns(UserWarning, match="multiple-looks correction"):
        assert ledger.record("split-1", ["c"], delta=0.05, allow_reuse=True) == 1
    assert ledger.prior_looks("split-1") == 2


def test_corrected_delta() -> None:
    assert corrected_delta(0.1, 0) == pytest.approx(0.1)
    assert corrected_delta(0.1, 1) == pytest.approx(0.05)
    assert corrected_delta(0.1, 2) == pytest.approx(0.1 / 3)
    with pytest.raises(ValueError):
        corrected_delta(1.0, 0)
    with pytest.raises(ValueError):
        corrected_delta(0.1, -1)
