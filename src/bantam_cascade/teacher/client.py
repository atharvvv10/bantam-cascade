"""Groq teacher client with proactive limits, retries, caching, and dry-run estimates."""

from __future__ import annotations

import os
import random
import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx

from bantam_cascade.config import RateLimits, TeacherConfig
from bantam_cascade.teacher.cache import LabelCache
from bantam_cascade.teacher.parse import INVALID_LABEL, parse_label
from bantam_cascade.teacher.prompt import build_prompt

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
_DEFAULT_MAX_ATTEMPTS = 5
_MAX_COMPLETION_TOKENS = 64


class TeacherError(RuntimeError):
    """Base error for teacher API failures."""


class MissingAPIKeyError(TeacherError):
    """Raised when ``GROQ_API_KEY`` is absent from the environment."""


class TeacherRateLimitError(TeacherError):
    """Raised after retries are exhausted on a 429 response."""


class DailyLimitExceeded(TeacherError):
    """Raised when the configured requests-per-day budget is exhausted."""


@dataclass(frozen=True)
class LabelResult:
    """One teacher label, whether it came from cache or a live call."""

    text: str
    label: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool


@dataclass(frozen=True)
class TeacherTotals:
    """Measured totals for live calls in one batch."""

    requests: int
    cached: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    invalid: int


@dataclass(frozen=True)
class LabelBatch:
    """Results and live-call totals for one labeling batch."""

    results: tuple[LabelResult, ...]
    totals: TeacherTotals


@dataclass(frozen=True)
class DryRunEstimate:
    """Call and token estimates computed without contacting Groq."""

    texts: int
    requests: int
    cached: int
    input_tokens: int
    output_tokens: int
    estimated_days: int
    estimated_cost: float


def estimate_tokens(text: str) -> int:
    """Estimate token count using a conservative four-characters-per-token rule."""
    return max(1, (len(text) + 3) // 4)


def output_token_estimate(labels: Sequence[str]) -> int:
    """Estimate completion tokens from the longest allowed label."""
    if not labels:
        raise ValueError("labels must not be empty")
    return estimate_tokens(max(labels, key=len))


def _parse_retry_after(value: str | None) -> float | None:
    """Parse Groq's ``retry-after`` header into seconds."""
    if not value:
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s)?\s*", value, flags=re.IGNORECASE)
    if match is None:
        return None
    amount = float(match.group(1))
    return amount / 1000.0 if match.group(2) and match.group(2).lower() == "ms" else amount


def _env_file_paths() -> tuple[Path, ...]:
    """Return candidate .env locations: current directory, then project root."""
    candidates = [Path.cwd() / ".env"]
    project_root = Path(__file__).resolve().parents[3]
    if project_root != Path.cwd():
        candidates.append(project_root / ".env")
    return tuple(dict.fromkeys(candidates))


def _read_env_file(path: Path) -> dict[str, str]:
    """Read simple ``KEY=value`` lines from a .env file without extra dependencies."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        name = key.strip()
        if name:
            values[name] = value.strip().strip('"').strip("'")
    return values


class _RateLimiter:
    """Proactive RPM/RPD/TPM limiter with injectable clock and sleep."""

    def __init__(
        self,
        limits: RateLimits,
        *,
        sleep: Callable[[float], None],
        now: Callable[[], float],
    ) -> None:
        self._rpm = limits.requests_per_minute
        self._rpd = limits.requests_per_day
        self._tpm = limits.tokens_per_minute
        self._sleep = sleep
        self._now = now
        self._events: deque[tuple[float, int]] = deque()
        self._day_requests = 0

    def _prune(self) -> None:
        current = self._now()
        while self._events and current - self._events[0][0] >= 60.0:
            self._events.popleft()

    def _sleep_until_minute_reset(self) -> None:
        if not self._events:
            return
        remaining = 60.0 - (self._now() - self._events[0][0])
        if remaining > 0:
            self._sleep(remaining + 0.01)

    def acquire(self, estimated_tokens: int) -> None:
        """Wait until a request fits within all configured limits."""
        if estimated_tokens > self._tpm:
            raise TeacherRateLimitError(
                f"Estimated prompt of {estimated_tokens} tokens exceeds TPM limit {self._tpm}."
            )
        while True:
            self._prune()
            if self._day_requests >= self._rpd:
                raise DailyLimitExceeded(
                    f"Requests-per-day limit ({self._rpd}) reached. Cached labels are safe; "
                    "rerun later to resume from the cache."
                )
            if len(self._events) >= self._rpm:
                self._sleep_until_minute_reset()
                continue
            minute_tokens = sum(tokens for _, tokens in self._events)
            if minute_tokens + estimated_tokens > self._tpm:
                self._sleep_until_minute_reset()
                continue
            return

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """Record a completed request's actual token usage."""
        self._events.append((self._now(), input_tokens + output_tokens))
        self._day_requests += 1


class GroqTeacherClient:
    """Label texts with a pinned Groq model, limited by user-supplied budgets."""

    def __init__(
        self,
        config: TeacherConfig,
        *,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        """Create a client; the API key is read only when a live call is made."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._config = config
        self._owns_client = http_client is None
        self._http = http_client if http_client is not None else httpx.Client(timeout=30.0)
        self._limiter = _RateLimiter(config.rate_limits, sleep=sleep, now=now)
        self._sleep = sleep
        self._now = now
        self._max_attempts = max_attempts

    def close(self) -> None:
        """Close the underlying HTTP client when this instance owns it."""
        if self._owns_client:
            self._http.close()

    @staticmethod
    def _api_key() -> str:
        key = os.environ.get("GROQ_API_KEY")
        if not key or not key.strip():
            for env_path in _env_file_paths():
                key = _read_env_file(env_path).get("GROQ_API_KEY")
                if key and key.strip():
                    break
        if not key or not key.strip():
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not set. Put it in a .env file at the project root or set the "
                "environment variable before a live run; never commit or print it."
            )
        cleaned = key.strip()
        if cleaned.lower() in {"your-key-here", "changeme", "tbd"} or "<" in cleaned:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is still a placeholder; replace it with your key."
            )
        return cleaned

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(8.0, 0.5 * (2.0**attempt))

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }
        for attempt in range(self._max_attempts):
            try:
                response = self._http.post(GROQ_CHAT_COMPLETIONS_URL, json=payload, headers=headers)
            except httpx.TransportError as exc:
                if attempt + 1 == self._max_attempts:
                    raise TeacherError(
                        f"Groq request failed after {self._max_attempts} attempts "
                        f"({type(exc).__name__})."
                    ) from exc
                self._sleep(self._backoff(attempt))
                continue

            if response.status_code == 429:
                if attempt + 1 == self._max_attempts:
                    raise TeacherRateLimitError(
                        f"Groq returned HTTP 429 after {self._max_attempts} attempts."
                    )
                retry_after = _parse_retry_after(response.headers.get("retry-after"))
                delay = self._backoff(attempt)
                if retry_after is not None:
                    delay = max(delay, retry_after)
                self._sleep(delay + random.uniform(0.0, 0.2))
                continue

            if response.status_code >= 400:
                raise TeacherError(f"Groq API returned HTTP {response.status_code}.")
            try:
                data: Any = response.json()
            except ValueError as exc:
                raise TeacherError("Groq returned a non-JSON response.") from exc
            if not isinstance(data, dict):
                raise TeacherError("Groq returned an unexpected response shape.")
            return cast(dict[str, object], data)
        raise TeacherError("Exhausted Groq retries without a response.")

    @staticmethod
    def _extract(data: dict[str, object]) -> tuple[str, int, int]:
        choices = cast(list[object], data.get("choices", []))
        if not choices:
            raise TeacherError("Groq response contained no choices.")
        choice = cast(dict[str, object], choices[0])
        message = cast(dict[str, object], choice.get("message", {}))
        content = message.get("content")
        if not isinstance(content, str):
            raise TeacherError("Groq response contained no message content.")
        usage = cast(dict[str, object], data.get("usage", {}))
        input_tokens = int(cast(int, usage.get("prompt_tokens", 0)))
        output_tokens = int(cast(int, usage.get("completion_tokens", 0)))
        if input_tokens < 0 or output_tokens < 0:
            raise TeacherError("Groq reported negative token usage.")
        return content, input_tokens, output_tokens

    def _payload(self, prompt: str, *, model: str) -> dict[str, object]:
        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You classify user messages into exactly one allowed label.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": _MAX_COMPLETION_TOKENS,
        }

    def label_many(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        cache: LabelCache,
        model: str | None = None,
        prompt_version: str | None = None,
        limit: int | None = None,
    ) -> LabelBatch:
        """Label texts, skipping cache hits and respecting proactive rate limits.

        Args:
            texts: User messages to label.
            labels: Allowed label names.
            cache: Resumable label cache.
            model: Override the configured model (used by teacher selection).
            prompt_version: Override the configured prompt version.
            limit: Maximum number of texts to process.

        Returns:
            A :class:`LabelBatch` with all labels and live-call totals.
        """
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        selected = list(texts[:limit]) if limit is not None else list(texts)
        if not selected:
            return LabelBatch(
                (),
                TeacherTotals(
                    requests=0, cached=0, input_tokens=0, output_tokens=0, latency_ms=0.0, invalid=0
                ),
            )

        chosen_model = model or self._config.model
        version = prompt_version or self._config.prompt_version
        expected_output = output_token_estimate(labels)

        results: list[LabelResult] = []
        requests = cached_count = input_tokens_total = output_tokens_total = invalid = 0
        latency_total = 0.0

        for text in selected:
            hit = cache.get(text, model=chosen_model, prompt_version=version)
            if hit is not None:
                results.append(
                    LabelResult(
                        text=text,
                        label=hit.label,
                        input_tokens=hit.input_tokens,
                        output_tokens=hit.output_tokens,
                        latency_ms=hit.latency_ms,
                        cached=True,
                    )
                )
                cached_count += 1
                if hit.label == INVALID_LABEL:
                    invalid += 1
                continue

            prompt = build_prompt(labels, text, version=version)
            self._limiter.acquire(estimate_tokens(prompt) + expected_output)
            started = self._now()
            data = self._post(self._payload(prompt, model=chosen_model))
            latency_ms = (self._now() - started) * 1000.0
            raw_label, input_tokens, output_tokens = self._extract(data)
            label = parse_label(raw_label, labels)
            self._limiter.record(input_tokens, output_tokens)
            cache.put(
                text,
                model=chosen_model,
                prompt_version=version,
                label=label,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )
            results.append(
                LabelResult(
                    text=text,
                    label=label,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    cached=False,
                )
            )
            requests += 1
            input_tokens_total += input_tokens
            output_tokens_total += output_tokens
            latency_total += latency_ms
            if label == INVALID_LABEL:
                invalid += 1

        return LabelBatch(
            results=tuple(results),
            totals=TeacherTotals(
                requests=requests,
                cached=cached_count,
                input_tokens=input_tokens_total,
                output_tokens=output_tokens_total,
                latency_ms=latency_total,
                invalid=invalid,
            ),
        )

    def dry_run_estimate(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        model: str | None = None,
        prompt_version: str | None = None,
        cache: LabelCache | None = None,
        limit: int = 200,
    ) -> DryRunEstimate:
        """Estimate calls, tokens, days, and cost without reading the API key.

        Args:
            texts: Candidate texts to label.
            labels: Allowed label names.
            model: Override the configured model.
            prompt_version: Override the configured prompt version.
            cache: Optional cache to exclude already-labeled texts.
            limit: Number of texts to estimate.

        Returns:
            A :class:`DryRunEstimate` derived only from configuration and text lengths.
        """
        if limit < 0:
            raise ValueError("limit must be non-negative")
        chosen_model = model or self._config.model
        version = prompt_version or self._config.prompt_version
        expected_output = output_token_estimate(labels)
        requests = cached_count = input_tokens = output_tokens = 0

        for text in texts[:limit]:
            if cache is not None and cache.get(text, model=chosen_model, prompt_version=version):
                cached_count += 1
                continue
            prompt = build_prompt(labels, text, version=version)
            input_tokens += estimate_tokens(prompt)
            output_tokens += expected_output
            requests += 1

        rpd = self._config.rate_limits.requests_per_day
        estimated_days = (requests + rpd - 1) // rpd if requests else 0
        estimated_cost = (
            input_tokens * self._config.price_per_million_input_tokens
            + output_tokens * self._config.price_per_million_output_tokens
        ) / 1_000_000.0
        return DryRunEstimate(
            texts=min(limit, len(texts)),
            requests=requests,
            cached=cached_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_days=estimated_days,
            estimated_cost=estimated_cost,
        )
