"""Cascade evaluation on held-out, gold-labeled certification data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bantam_cascade.students.base import confidence_scores, labels_from_proba


@dataclass(frozen=True)
class TeacherPrediction:
    """A teacher label with its measured token usage and latency."""

    label: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    def __post_init__(self) -> None:
        """Reject negative resource measurements."""
        if self.input_tokens < 0 or self.output_tokens < 0 or self.latency_ms < 0.0:
            raise ValueError("token and latency measurements must be non-negative")


@dataclass(frozen=True)
class CascadeResult:
    """Measured behavior of one student-plus-teacher cascade."""

    student: str
    threshold: float
    examples: int
    accuracy: float
    student_only_accuracy: float
    teacher_only_accuracy: float
    fallback_rate: float
    teacher_calls: int
    estimated_cost: float
    p95_latency_ms: float

    def __post_init__(self) -> None:
        """Reject impossible cascade statistics."""
        if self.examples <= 0:
            raise ValueError("examples must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        for name in (
            "accuracy",
            "student_only_accuracy",
            "teacher_only_accuracy",
            "fallback_rate",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not 0 <= self.teacher_calls <= self.examples:
            raise ValueError("teacher_calls must be between 0 and examples")
        if self.estimated_cost < 0.0 or self.p95_latency_ms < 0.0:
            raise ValueError("cost and latency must be non-negative")


def evaluate_cascade(
    *,
    student: str,
    threshold: float,
    proba: np.ndarray,
    classes: Sequence[str],
    gold: Sequence[str],
    teacher_predictions: Sequence[TeacherPrediction],
    student_latency_ms: Sequence[float],
    price_per_million_input_tokens: float,
    price_per_million_output_tokens: float,
) -> CascadeResult:
    """Score a fixed-threshold cascade against gold labels.

    Costs are estimates computed only from the configured prices and the
    measured token usage of the teacher calls actually taken.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if price_per_million_input_tokens < 0.0 or price_per_million_output_tokens < 0.0:
        raise ValueError("prices must be non-negative")
    confidences = confidence_scores(proba)
    predicted = labels_from_proba(proba, classes)
    count = len(gold)
    if count == 0:
        raise ValueError("at least one certification example is required")
    if (
        len(predicted) != count
        or len(teacher_predictions) != count
        or len(student_latency_ms) != count
    ):
        raise ValueError("all certification inputs must have equal lengths")

    use_student = confidences >= threshold
    cascade_labels = [
        predicted[index] if use_student[index] else teacher_predictions[index].label
        for index in range(count)
    ]
    accuracy = sum(1 for p, g in zip(cascade_labels, gold, strict=True) if p == g) / count
    student_only_accuracy = sum(1 for p, g in zip(predicted, gold, strict=True) if p == g) / count
    teacher_only_accuracy = (
        sum(1 for item, g in zip(teacher_predictions, gold, strict=True) if item.label == g) / count
    )
    fallback_indices = [index for index, keep in enumerate(use_student) if not keep]
    input_tokens = sum(teacher_predictions[index].input_tokens for index in fallback_indices)
    output_tokens = sum(teacher_predictions[index].output_tokens for index in fallback_indices)
    estimated_cost = (
        input_tokens * price_per_million_input_tokens
        + output_tokens * price_per_million_output_tokens
    ) / 1_000_000.0
    per_example_latency = [
        float(student_latency_ms[index])
        + (0.0 if use_student[index] else teacher_predictions[index].latency_ms)
        for index in range(count)
    ]
    p95 = float(np.percentile(per_example_latency, 95))
    return CascadeResult(
        student=student,
        threshold=threshold,
        examples=count,
        accuracy=accuracy,
        student_only_accuracy=student_only_accuracy,
        teacher_only_accuracy=teacher_only_accuracy,
        fallback_rate=len(fallback_indices) / count,
        teacher_calls=len(fallback_indices),
        estimated_cost=estimated_cost,
        p95_latency_ms=p95,
    )
