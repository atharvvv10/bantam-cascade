"""Select a teacher client from validated configuration."""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx

from bantam_cascade.config import TeacherConfig
from bantam_cascade.teacher.client import (
    DEFAULT_MAX_ATTEMPTS,
    GroqTeacherClient,
    TeacherClient,
)
from bantam_cascade.teacher.gemini import GeminiDecisionClient


def build_teacher_client(
    config: TeacherConfig,
    *,
    http_client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> TeacherClient:
    """Build the client for the configured provider without reading any API key.

    Args:
        config: Validated teacher settings whose provider is pinned by the user.
        http_client: Optional injected HTTP client for tests.
        sleep: Injectable sleep function for rate-limit tests.
        now: Injectable monotonic clock for rate-limit tests.
        max_attempts: Maximum retries for transient provider errors.

    Returns:
        A provider client implementing :class:`TeacherClient`.

    Raises:
        ValueError: If the provider is not supported.
    """
    if config.provider == "groq":
        return GroqTeacherClient(
            config,
            http_client=http_client,
            sleep=sleep,
            now=now,
            max_attempts=max_attempts,
        )
    if config.provider == "gemini":
        return GeminiDecisionClient(
            config,
            http_client=http_client,
            sleep=sleep,
            now=now,
            max_attempts=max_attempts,
        )
    raise ValueError(f"Unsupported teacher provider: {config.provider}")
