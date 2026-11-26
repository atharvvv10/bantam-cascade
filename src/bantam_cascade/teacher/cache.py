"""Resumable SQLite cache for teacher labels."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CachedLabel:
    """A previously cached teacher label and its measured usage."""

    label: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    confidence: float | None = None
    probabilities: dict[str, float] | None = None


def cache_key(text: str, *, model: str, prompt_version: str) -> str:
    """Return a stable key for a text/model/prompt-version triple."""
    digest = hashlib.sha256()
    for part in (prompt_version, model, text):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class LabelCache:
    """Store teacher labels so interrupted runs can resume without re-labeled calls."""

    def __init__(self, path: str | Path) -> None:
        """Open or create the SQLite cache at ``path``, migrating older schemas."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS labels (
                    key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    label TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    confidence REAL,
                    probabilities TEXT
                )
                """
            )
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(labels)")}
            if "confidence" not in columns:
                conn.execute("ALTER TABLE labels ADD COLUMN confidence REAL")
            if "probabilities" not in columns:
                conn.execute("ALTER TABLE labels ADD COLUMN probabilities TEXT")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get(self, text: str, *, model: str, prompt_version: str) -> CachedLabel | None:
        """Return a cached label, or ``None`` if this triple was never labeled."""
        key = cache_key(text, model=model, prompt_version=prompt_version)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT label, input_tokens, output_tokens, latency_ms, confidence, probabilities
                FROM labels WHERE key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None

        confidence = float(row[4]) if row[4] is not None else None
        probabilities: dict[str, float] | None = None
        raw_probabilities = row[5]
        if isinstance(raw_probabilities, str) and raw_probabilities:
            decoded: Any = None
            try:
                decoded = json.loads(raw_probabilities)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                probabilities = {
                    str(name): float(value)
                    for name, value in decoded.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
        return CachedLabel(
            label=str(row[0]),
            input_tokens=int(row[1]),
            output_tokens=int(row[2]),
            latency_ms=float(row[3]),
            confidence=confidence,
            probabilities=probabilities,
        )

    def put(
        self,
        text: str,
        *,
        model: str,
        prompt_version: str,
        label: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        confidence: float | None = None,
        probabilities: dict[str, float] | None = None,
    ) -> None:
        """Store a teacher label and replace any prior value for the same key."""
        key = cache_key(text, model=model, prompt_version=prompt_version)
        probability_json = (
            json.dumps(probabilities, sort_keys=True) if probabilities is not None else None
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO labels
                    (key, model, prompt_version, label, input_tokens, output_tokens, latency_ms,
                     confidence, probabilities)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    label = excluded.label,
                    input_tokens = excluded.input_tokens,
                    output_tokens = excluded.output_tokens,
                    latency_ms = excluded.latency_ms,
                    confidence = excluded.confidence,
                    probabilities = excluded.probabilities
                """,
                (
                    key,
                    model,
                    prompt_version,
                    label,
                    input_tokens,
                    output_tokens,
                    latency_ms,
                    confidence,
                    probability_json,
                ),
            )

    def count(self, *, model: str, prompt_version: str) -> int:
        """Count cached entries for one model and prompt version."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM labels WHERE model = ? AND prompt_version = ?",
                (model, prompt_version),
            ).fetchone()
        return int(row[0]) if row is not None else 0
