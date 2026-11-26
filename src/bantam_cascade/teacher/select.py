"""Teacher candidate evaluation, recommendation, and sanity gating."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bantam_cascade.teacher.cache import LabelCache
from bantam_cascade.teacher.client import TeacherClient
from bantam_cascade.teacher.parse import INVALID_LABEL


@dataclass(frozen=True)
class CandidateMetrics:
    """Accuracy and resource usage for one teacher candidate."""

    model: str
    examples: int
    correct: int
    invalid: int
    input_tokens: int
    output_tokens: int
    total_latency_ms: float
    estimated_cost: float

    def __post_init__(self) -> None:
        """Reject impossible metric combinations."""
        if self.examples <= 0:
            raise ValueError("examples must be positive")
        if not 0 <= self.correct <= self.examples:
            raise ValueError("correct must be between 0 and examples")
        if not 0 <= self.invalid <= self.examples:
            raise ValueError("invalid must be between 0 and examples")
        if min(self.input_tokens, self.output_tokens, self.total_latency_ms) < 0:
            raise ValueError("token and latency totals must be non-negative")
        if self.estimated_cost < 0:
            raise ValueError("estimated_cost must be non-negative")

    @property
    def accuracy(self) -> float:
        """Fraction of examples whose returned label matched the gold label."""
        return self.correct / self.examples

    @property
    def invalid_rate(self) -> float:
        """Fraction of examples where the teacher returned an unparseable label."""
        return self.invalid / self.examples

    @property
    def mean_latency_ms(self) -> float:
        """Mean latency across live calls in this evaluation."""
        return self.total_latency_ms / self.examples


def evaluate_candidate(
    *,
    client: TeacherClient,
    cache: LabelCache,
    texts: Sequence[str],
    gold_labels: Sequence[str],
    labels: Sequence[str],
    model: str,
    price_per_million_input_tokens: float,
    price_per_million_output_tokens: float,
    prompt_version: str | None = None,
) -> CandidateMetrics:
    """Label a calibration sample with one candidate and score against gold labels."""
    if len(texts) != len(gold_labels):
        raise ValueError("texts and gold_labels must have the same length")
    if not texts:
        raise ValueError("at least one example is required")

    batch = client.label_many(
        texts,
        labels,
        cache=cache,
        model=model,
        prompt_version=prompt_version,
    )
    correct = sum(
        1 for result, gold in zip(batch.results, gold_labels, strict=True) if result.label == gold
    )
    invalid = sum(1 for result in batch.results if result.label == INVALID_LABEL)
    estimated_cost = (
        batch.totals.input_tokens * price_per_million_input_tokens
        + batch.totals.output_tokens * price_per_million_output_tokens
    ) / 1_000_000.0
    return CandidateMetrics(
        model=model,
        examples=len(batch.results),
        correct=correct,
        invalid=invalid,
        input_tokens=batch.totals.input_tokens,
        output_tokens=batch.totals.output_tokens,
        total_latency_ms=batch.totals.latency_ms,
        estimated_cost=estimated_cost,
    )


def recommend(metrics: Sequence[CandidateMetrics]) -> CandidateMetrics | None:
    """Pick the highest-accuracy candidate, breaking ties by cost then tokens."""
    if not metrics:
        return None
    return max(
        metrics,
        key=lambda item: (
            item.accuracy,
            -item.estimated_cost,
            -(item.input_tokens + item.output_tokens),
        ),
    )


def teacher_sanity_gate(
    *,
    teacher_accuracy: float,
    best_student_accuracy: float,
    margin: float = 0.0,
) -> bool:
    """Return whether the teacher is at least as accurate as the best student.

    Args:
        teacher_accuracy: Measured teacher accuracy on the calibration sample.
        best_student_accuracy: Best measured student accuracy on the same sample.
        margin: Required accuracy advantage for the teacher.

    Returns:
        ``True`` when the sanity condition holds.
    """
    for name, value in (
        ("teacher_accuracy", teacher_accuracy),
        ("best_student_accuracy", best_student_accuracy),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    return teacher_accuracy + 1e-12 >= best_student_accuracy + margin
