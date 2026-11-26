import json
from pathlib import Path

import httpx
import pytest

import bantam_cascade.teacher.client as client_module
from bantam_cascade.config import load_config
from bantam_cascade.teacher.cache import LabelCache
from bantam_cascade.teacher.client import GroqTeacherClient, MissingAPIKeyError
from bantam_cascade.teacher.factory import build_teacher_client
from bantam_cascade.teacher.gemini import GeminiDecisionClient

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["billing", "shipping"]


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


def _gemini_config() -> object:
    return load_config(ROOT / "configs" / "banking77-gemini.yaml").teacher


def _success_response(choice: str = "billing") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": {"billing": 0.8, "shipping": 0.2},
                    "confidence": 0.8,
                }
            },
            "usage": {
                "total_input_tokens": 123,
                "total_output_tokens": 0,
                "total_tokens": 123,
            },
        },
    )


def test_label_many_posts_choice_question_and_caches_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _success_response()

    clock = _Clock()
    client = GeminiDecisionClient(
        _gemini_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=clock.sleep,
        now=clock,
    )
    cache = LabelCache(tmp_path / "cache.sqlite3")
    try:
        first = client.label_many(["hello"], LABELS, cache=cache)
        second = client.label_many(["hello"], LABELS, cache=cache)
    finally:
        client.close()

    assert first.results[0].label == "billing"
    assert first.results[0].confidence == 0.8
    assert first.results[0].probabilities == {"billing": 0.8, "shipping": 0.2}
    assert first.totals.output_tokens == 0
    assert second.totals.requests == 0
    assert second.totals.cached == 1
    assert len(requests) == 1
    assert requests[0].headers["x-goog-api-key"] == "test-key-not-real"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "gemini-decision-1"
    assert payload["input"] == "hello"
    question = payload["questions"]["intent"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(LABELS)
    hit = cache.get("hello", model="gemini-decision-1", prompt_version="v1")
    assert hit is not None
    assert hit.confidence == 0.8
    assert hit.probabilities == {"billing": 0.8, "shipping": 0.2}


def test_invalid_choice_becomes_invalid_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

    def handler(request: httpx.Request) -> httpx.Response:
        return _success_response("not-an-allowed-label")

    client = GeminiDecisionClient(
        _gemini_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=_Clock().sleep,
        now=_Clock(),
    )
    cache = LabelCache(tmp_path / "cache.sqlite3")
    try:
        batch = client.label_many(["hello"], LABELS, cache=cache)
    finally:
        client.close()
    assert batch.results[0].label == "invalid"
    assert batch.totals.invalid == 1


def test_429_respects_retry_after(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    responses = [httpx.Response(429, headers={"retry-after": "2"}), _success_response()]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    clock = _Clock()
    client = GeminiDecisionClient(
        _gemini_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=clock.sleep,
        now=clock,
    )
    cache = LabelCache(tmp_path / "cache.sqlite3")
    try:
        batch = client.label_many(["hello"], LABELS, cache=cache)
    finally:
        client.close()
    assert batch.results[0].label == "billing"
    assert clock.value >= 2.0


def test_missing_gemini_key_has_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(client_module, "_env_file_paths", lambda: ())
    client = GeminiDecisionClient(_gemini_config())
    try:
        with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY is not set"):
            client._api_key()
    finally:
        client.close()


def test_dry_run_uses_zero_output_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(client_module, "_env_file_paths", lambda: ())
    client = GeminiDecisionClient(_gemini_config())
    try:
        estimate = client.dry_run_estimate(["x" * 100], LABELS, limit=1)
    finally:
        client.close()
    assert estimate.requests == 1
    assert estimate.input_tokens > 0
    assert estimate.output_tokens == 0
    assert estimate.estimated_cost == 0.0


def test_choice_requires_two_labels(tmp_path: Path) -> None:
    client = GeminiDecisionClient(_gemini_config())
    cache = LabelCache(tmp_path / "cache.sqlite3")
    try:
        with pytest.raises(ValueError, match="2 to 255"):
            client.label_many(["hello"], ["billing"], cache=cache)
    finally:
        client.close()


def test_factory_selects_provider_from_config() -> None:
    groq_config = load_config(ROOT / "configs" / "banking77.yaml").teacher
    groq_client = build_teacher_client(groq_config)
    gemini_client = build_teacher_client(_gemini_config())
    try:
        assert isinstance(groq_client, GroqTeacherClient)
        assert isinstance(gemini_client, GeminiDecisionClient)
    finally:
        groq_client.close()
        gemini_client.close()
