import numpy as np
import pytest

from bantam_cascade.calibration import ThresholdResult, select_threshold


def test_select_threshold_prefers_perfect_student_at_zero() -> None:
    proba = np.array([[0.95, 0.05], [0.9, 0.1], [0.2, 0.8]])
    result = select_threshold(
        proba=proba,
        classes=("yes", "no"),
        gold=["yes", "yes", "no"],
        teacher_predictions=["no", "no", "no"],
        grid_size=11,
    )
    assert result.threshold == 0.0
    assert result.calibration_accuracy == 1.0
    assert result.fallback_rate == 0.0
    assert result.student_accuracy == 1.0
    assert result.teacher_accuracy == 1 / 3


def test_select_threshold_falls_back_to_teacher_when_student_is_weak() -> None:
    proba = np.array([[0.55, 0.45], [0.55, 0.45]])
    result = select_threshold(
        proba=proba,
        classes=("no", "yes"),
        gold=["yes", "yes"],
        teacher_predictions=["yes", "yes"],
        grid_size=11,
    )
    assert result.threshold > 0.5
    assert result.calibration_accuracy == 1.0
    assert result.fallback_rate == 1.0


def test_select_threshold_tie_breaks_to_lower_fallback() -> None:
    proba = np.array([[0.9, 0.1], [0.9, 0.1]])
    result = select_threshold(
        proba=proba,
        classes=("yes", "no"),
        gold=["yes", "yes"],
        teacher_predictions=["yes", "yes"],
        grid_size=6,
    )
    assert result.threshold == 0.0


def test_select_threshold_validation() -> None:
    with pytest.raises(ValueError, match="grid_size"):
        select_threshold(
            proba=np.array([[0.9, 0.1]]),
            classes=("yes", "no"),
            gold=["yes"],
            teacher_predictions=["yes"],
            grid_size=1,
        )
    with pytest.raises(ValueError, match="equal lengths"):
        select_threshold(
            proba=np.array([[0.9, 0.1]]),
            classes=("yes", "no"),
            gold=["yes"],
            teacher_predictions=["yes", "no"],
            grid_size=5,
        )
    with pytest.raises(ValueError, match="threshold"):
        ThresholdResult(1.2, 1.0, 0.0, 1.0, 1.0, 1)
