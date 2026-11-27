"""Deterministic, hash-based disjoint text splits."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from bantam_cascade.config import SplitsConfig


class SplitError(ValueError):
    """Raised when requested splits overlap or contain duplicate texts."""


@dataclass(frozen=True)
class SplitResult:
    """Texts assigned to the three internal splits and, optionally, the official test split."""

    student_train: tuple[str, ...]
    calibration: tuple[str, ...]
    certification: tuple[str, ...]
    official_test: tuple[str, ...] = ()


def text_hash(text: str, seed: int) -> str:
    """Return a deterministic SHA-256 digest for a text and seed."""
    return hashlib.sha256(f"{seed}\x00{text}".encode()).hexdigest()


def verify_no_overlap(*groups: tuple[str, ...]) -> None:
    """Assert that no exact text appears in two groups.

    Args:
        *groups: Named-free groups of texts.

    Raises:
        SplitError: If any text occurs in more than one group.
    """
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1 :]:
            overlap = set(left) & set(right)
            if overlap:
                example = sorted(overlap)[0]
                raise SplitError(f"Splits overlap on exact text: {example!r}")


def split_by_hash(
    texts: list[str] | tuple[str, ...],
    *,
    config: SplitsConfig,
    official_test: tuple[str, ...] = (),
) -> SplitResult:
    """Assign texts to disjoint splits by sorting on a seeded text hash.

    Exact duplicate texts in the input are an error, and any text shared with the
    official test split is also an error.

    Args:
        texts: Texts from the official training split.
        config: Split sizes and seed.
        official_test: Texts from the official test split, used only for final reporting.

    Returns:
        A :class:`SplitResult` with deterministic assignments.

    Raises:
        SplitError: On duplicates, overlap, or insufficient data.
    """
    duplicate_counts = {text: count for text, count in Counter(texts).items() if count > 1}
    if duplicate_counts:
        example = sorted(duplicate_counts)[0]
        raise SplitError(f"Duplicate text in training input: {example!r}")

    required = config.student_train + config.calibration + config.certification
    if len(texts) < required:
        raise SplitError(
            f"Need at least {required} unique training texts for the requested splits; "
            f"got {len(texts)}"
        )

    shared_with_test = set(texts) & set(official_test)
    if shared_with_test:
        example = sorted(shared_with_test)[0]
        raise SplitError(f"Text appears in both training pool and official test split: {example!r}")

    ordered = sorted(texts, key=lambda text: text_hash(text, config.seed))
    train_end = config.student_train
    cal_end = train_end + config.calibration
    result = SplitResult(
        student_train=tuple(ordered[:train_end]),
        calibration=tuple(ordered[train_end:cal_end]),
        certification=tuple(ordered[cal_end:required]),
        official_test=tuple(official_test),
    )
    verify_no_overlap(
        result.student_train,
        result.calibration,
        result.certification,
        result.official_test,
    )
    return result


def certification_split_hash(texts: Sequence[str]) -> str:
    """Hash the certification texts so a ledger can detect repeated looks."""
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def splits_fingerprint(result: SplitResult, *, seed: int) -> str:
    """Return a stable fingerprint of split membership for the certification ledger."""
    digest = hashlib.sha256()
    digest.update(str(seed).encode("utf-8"))
    for group in (
        result.student_train,
        result.calibration,
        result.certification,
        result.official_test,
    ):
        digest.update(b"\x00")
        for text in sorted(group):
            digest.update(text.encode("utf-8"))
            digest.update(b"\x01")
    return digest.hexdigest()
