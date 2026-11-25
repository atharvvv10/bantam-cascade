from pathlib import Path

import httpx
import pytest

import bantam_cascade.teacher.client as client_module
from bantam_cascade.config import load_config
from bantam_cascade.teacher.cache import LabelCache
from bantam_cascade.teacher.client import (
    DailyLimitExceeded,
    GroqTeacherClient,
    MissingAPIKeyError,
    TeacherRateLimitError,
)

ROOT = Path(__file__).resolve().parents[1]


class _Clock:
    """Manual clock whose sleep advances time without really waiting."""

    def __init__(self) -> None:
        """Start at time zero."""
        self.value = 0.0

    def __call__(self) -> float:
        """Return the current fake time."""
        return self.value

    def sleep(self, seconds: float) -> None:
        """Advance fake time by ``seconds``."""
        self.value += seconds


def _teacher_config() -> object:
    return load_config(ROOT / "configs" / "smoke.yaml").teacher


def _success_response(label: str = "billing") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": label}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    )


def test_label_many_uses_cache_and_parses_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _success_response()

    clock = _Clock()
    client = GroqTeacherClient(
        _teacher_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=clock.sleep,
        now=clock,
    )
    cache = LabelCache(tmp_path / "cache.sqlite3")
    try:
        first = client.label_many(["hello"], ["billing"], cache=cache)
        second = client.label_many(["hello"], ["billing"], cache=cache)
    finally:
        client.close()

    assert first.results[0].label == "billing"
    assert first.totals.requests == 1
    assert first.totals.input_tokens == 10
    assert second.totals.requests == 0
    assert second.totals.cached == 1
    assert len(requests) == 1
    assert requests[0].headers["authorization"].startswith("Bearer ")
    assert cache.count(model="smoke-fake-model", prompt_version="v1") == 1


def test_429_respects_retry_after(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    responses = [
        httpx.Response(429, headers={"retry-after": "2"}),
        _success_response(),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    clock = _Clock()
    client = GroqTeacherClient(
        _teacher_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=clock.sleep,
        now=clock,
    )
    cache = LabelCache(tmp_path / "cache.sqlite3")
    try:
        batch = client.label_many(["hello"], ["billing"], cache=cache)
    finally:
        client.close()

    assert batch.results[0].label == "billing"
    assert clock.value >= 2.0


def test_429_raises_after_max_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "0"})

    clock = _Clock()
    client = GroqTeacherClient(
        _teacher_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=clock.sleep,
        now=clock,
        max_attempts=2,
    )
    cache = LabelCache(tmp_path / "cache.sqlite3")
    try:
        with pytest.raises(TeacherRateLimitError):
            client.label_many(["hello"], ["billing"], cache=cache)
    finally:
        client.close()


def test_daily_limit_stops_with_resume_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    teacher = load_config(ROOT / "configs" / "smoke.yaml").teacher
    limited = teacher.model_copy(
        update={"rate_limits": teacher.rate_limits.model_copy(update={"requests_per_day": 1})}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _success_response()

    clock = _Clock()
    client = GroqTeacherClient(
        limited,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=clock.sleep,
        now=clock,
    )
    cache = LabelCache(tmp_path / "cache.sqlite3")
    try:
        client.label_many(["one"], ["billing"], cache=cache)
        with pytest.raises(DailyLimitExceeded, match="rerun later"):
            client.label_many(["two"], ["billing"], cache=cache)
    finally:
        client.close()


def test_missing_key_and_env_file_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(client_module, "_env_file_paths", lambda: ())
    client = GroqTeacherClient(_teacher_config())
    try:
        with pytest.raises(MissingAPIKeyError, match="GROQ_API_KEY is not set"):
            client._api_key()
    finally:
        client.close()

    env_file = tmp_path / ".env"
    env_file.write_text('GROQ_API_KEY="from-file-key"\n', encoding="utf-8")
    monkeypatch.setattr(client_module, "_env_file_paths", lambda: (env_file,))
    assert client._api_key() == "from-file-key"


def test_dry_run_does_not_need_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(client_module, "_env_file_paths", lambda: ())
    client = GroqTeacherClient(_teacher_config())
    try:
        estimate = client.dry_run_estimate(["x" * 100, "y" * 100], ["billing"], limit=2)
    finally:
        client.close()
    assert estimate.texts == 2
    assert estimate.requests == 2
    assert estimate.input_tokens > 0
    assert estimate.estimated_cost == 0.0
