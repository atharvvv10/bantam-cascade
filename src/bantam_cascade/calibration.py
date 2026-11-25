"""Confidence calibration and cascade threshold selection on gold-labeled data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bantam_cascade.students.base import confidence_scores, labels_from_proba


@dataclass(frozen=True)
class ThresholdResult:
    """Best confidence threshold found on the calibration split."""

    threshold: float
    calibration_accuracy: float
    fallback_rate: float
    student_accuracy: float
    teacher_accuracy: float
    examples: int

    def __post_init__(self) -> None:
        """Reject impossible threshold statistics."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.examples <= 0:
            raise ValueError("examples must be positive")
        for name in (
            "calibration_accuracy",
            "fallback_rate",
            "student_accuracy",
            "teacher_accuracy",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


def select_threshold(
    *,
    proba: np.ndarray,
    classes: Sequence[str],
    gold: Sequence[str],
    teacher_predictions: Sequence[str],
    grid_size: int,
) -> ThresholdResult:
    """Pick the threshold maximizing cascade accuracy on calibration data.

    Ties are broken toward a lower teacher fallback rate, which is cheaper.

    Args:
        proba: Student probabilities on the calibration split.
        classes: Class names in probability-column order.
        gold: Gold human labels for the calibration split.
        teacher_predictions: Teacher labels for the same calibration texts.
        grid_size: Number of thresholds evenly spaced over ``[0, 1]``.

    Returns:
        The selected :class:`ThresholdResult`.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    confidences = confidence_scores(proba)
    predicted = labels_from_proba(proba, classes)
    count = len(gold)
    if count == 0:
        raise ValueError("at least one calibration example is required")
    if len(teacher_predictions) != count or len(predicted) != count:
        raise ValueError("gold, student, and teacher inputs must have equal lengths")

    student_accuracy = sum(1 for p, g in zip(predicted, gold, strict=True) if p == g) / count
    teacher_accuracy = (
        sum(1 for p, g in zip(teacher_predictions, gold, strict=True) if p == g) / count
    )
    best_key: tuple[float, float] | None = None
    best: ThresholdResult | None = None
    for threshold in np.linspace(0.0, 1.0, grid_size):
        value = float(threshold)
        use_student = confidences >= value
        cascade = [
            predicted[index] if use_student[index] else teacher_predictions[index]
            for index in range(count)
        ]
        accuracy = sum(1 for p, g in zip(cascade, gold, strict=True) if p == g) / count
        fallback_rate = 1.0 - float(np.count_nonzero(use_student)) / count
        key = (accuracy, -fallback_rate)
        if best_key is None or key > best_key:
            best_key = key
            best = ThresholdResult(
                threshold=value,
                calibration_accuracy=accuracy,
                fallback_rate=fallback_rate,
                student_accuracy=student_accuracy,
                teacher_accuracy=teacher_accuracy,
                examples=count,
            )
    if best is None:  # pragma: no cover - grid_size >= 2 guarantees a candidate
        raise RuntimeError("threshold selection produced no candidate")
    return best
