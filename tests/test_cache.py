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
