import sqlite3
from pathlib import Path

from bantam_cascade.teacher.cache import LabelCache


def test_put_get_and_scope_isolation(tmp_path: Path) -> None:
    cache = LabelCache(tmp_path / "cache.sqlite3")
    assert cache.get("hello", model="m1", prompt_version="v1") is None

    cache.put(
        "hello",
        model="m1",
        prompt_version="v1",
        label="billing",
        input_tokens=10,
        output_tokens=2,
        latency_ms=12.5,
    )
    hit = cache.get("hello", model="m1", prompt_version="v1")
    assert hit is not None
    assert hit.label == "billing"
    assert (hit.input_tokens, hit.output_tokens, hit.latency_ms) == (10, 2, 12.5)
    assert cache.count(model="m1", prompt_version="v1") == 1
    assert cache.count(model="m2", prompt_version="v1") == 0
    assert cache.get("hello", model="m2", prompt_version="v1") is None
    assert cache.get("hello", model="m1", prompt_version="v2") is None


def test_cache_supports_resume_after_partial_run(tmp_path: Path) -> None:
    cache = LabelCache(tmp_path / "cache.sqlite3")
    for text in ("one", "two"):
        cache.put(
            text,
            model="m1",
            prompt_version="v1",
            label="billing",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1.0,
        )
    assert cache.get("one", model="m1", prompt_version="v1") is not None
    assert cache.get("two", model="m1", prompt_version="v1") is not None
    assert cache.get("three", model="m1", prompt_version="v1") is None
    assert cache.count(model="m1", prompt_version="v1") == 2


def test_cache_round_trips_decision_confidence(tmp_path: Path) -> None:
    cache = LabelCache(tmp_path / "cache.sqlite3")
    cache.put(
        "hello",
        model="gemini-decision-1",
        prompt_version="v1",
        label="billing",
        input_tokens=10,
        output_tokens=0,
        latency_ms=12.5,
        confidence=0.87,
        probabilities={"billing": 0.87, "shipping": 0.13},
    )
    hit = cache.get("hello", model="gemini-decision-1", prompt_version="v1")
    assert hit is not None
    assert hit.confidence == 0.87
    assert hit.probabilities == {"billing": 0.87, "shipping": 0.13}


def test_cache_migrates_legacy_schema(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE labels (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                label TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                latency_ms REAL NOT NULL
            )
            """
        )
    cache = LabelCache(path)
    cache.put(
        "old",
        model="m1",
        prompt_version="v1",
        label="billing",
        input_tokens=1,
        output_tokens=2,
        latency_ms=3.0,
        confidence=0.5,
        probabilities={"billing": 0.5},
    )
    hit = cache.get("old", model="m1", prompt_version="v1")
    assert hit is not None
    assert hit.confidence == 0.5
    assert hit.probabilities == {"billing": 0.5}
