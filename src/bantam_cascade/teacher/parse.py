"""Strict parsing of teacher outputs into gold label names."""

from __future__ import annotations

import re
from collections.abc import Sequence

INVALID_LABEL = "invalid"
_PREFIXES = ("label:", "intent:", "answer:", "class:")


def _clean_output(output: str) -> str:
    """Remove common wrappers and return the first meaningful line."""
    text = output.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    lowered = text.lower()
    for prefix in _PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    text = text.strip("\"'` ")
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def parse_label(output: str, labels: Sequence[str]) -> str:
    """Map a raw teacher output to one allowed label or ``invalid``.

    Exact matches win; a case-insensitive match is accepted when unambiguous;
    otherwise a label is accepted only if it appears exactly once as a whole
    token. Anything else becomes :data:`INVALID_LABEL`.

    Args:
        output: Raw text from the teacher model.
        labels: Allowed label names.

    Returns:
        A valid label from ``labels``, or ``"invalid"``.
    """
    allowed = tuple(labels)
    if not allowed:
        raise ValueError("labels must not be empty")

    cleaned = _clean_output(output)
    if cleaned in allowed:
        return cleaned

    by_lower: dict[str, str] = {}
    for label in allowed:
        by_lower.setdefault(label.lower(), label)
    lowered = cleaned.lower()
    if lowered in by_lower:
        return by_lower[lowered]

    matches: set[str] = set()
    for label in allowed:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])"
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            matches.add(label)
    if len(matches) == 1:
        return next(iter(matches))
    return INVALID_LABEL
