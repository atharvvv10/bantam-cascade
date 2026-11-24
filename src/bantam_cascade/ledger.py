"""Single-use certification ledger backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console


class CertificationReuseError(RuntimeError):
    """Raised when the certification split would be used twice without opt-in."""


def corrected_delta(delta: float, prior_looks: int) -> float:
    """Apply the multiple-looks correction: divide delta by total looks."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if prior_looks < 0:
        raise ValueError(f"prior_looks must be non-negative, got {prior_looks}")
    return delta / (prior_looks + 1)


class CertificationLedger:
    """Record certification runs and refuse silent reuse of the certification split.

    ``--allow-reuse`` is represented by ``allow_reuse=True``; it is *not* a free
    pass. Callers must apply :func:`corrected_delta` to the delta they use.
    """

    def __init__(self, path: str | Path) -> None:
        """Open (or create) the SQLite ledger at ``path``."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS certifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    split_hash TEXT NOT NULL,
                    configs_json TEXT NOT NULL,
                    delta REAL NOT NULL,
                    prior_looks INTEGER NOT NULL,
                    allow_reuse INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def prior_looks(self, split_hash: str) -> int:
        """Return how many certification runs have already used this split."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM certifications WHERE split_hash = ?", (split_hash,)
            ).fetchone()
        return int(row[0]) if row else 0

    def check(self, split_hash: str, *, allow_reuse: bool = False) -> int:
        """Check whether certification may proceed and return the prior-looks count.

        Raises:
            CertificationReuseError: If the split was used before and ``allow_reuse`` is False.
        """
        looks = self.prior_looks(split_hash)
        if looks == 0:
            return 0
        if not allow_reuse:
            raise CertificationReuseError(
                "The certification split has already been used. Refusing to certify again "
                "unless --allow-reuse is passed (which applies a multiple-looks correction)."
            )
        warning = (
            f"Certification split reused {looks} time(s). Applying multiple-looks correction: "
            f"delta is divided by {looks + 1}."
        )
        warnings.warn(warning, stacklevel=2)
        Console(stderr=True).print(f"[yellow]WARNING:[/yellow] {warning}")
        return looks

    def record(
        self,
        split_hash: str,
        configs: Sequence[str],
        *,
        delta: float,
        allow_reuse: bool = False,
    ) -> int:
        """Record a certification run and return the number of prior looks."""
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        looks = self.check(split_hash, allow_reuse=allow_reuse)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO certifications
                    (split_hash, configs_json, delta, prior_looks, allow_reuse, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    split_hash,
                    json.dumps(list(configs)),
                    delta,
                    looks,
                    int(allow_reuse),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return looks
