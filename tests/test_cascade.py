import numpy as np
import pytest

from bantam_cascade.cascade import CascadeResult, TeacherPrediction, evaluate_cascade


def _teacher(label: str = "yes") -> TeacherPrediction:
    return TeacherPrediction(label=label, input_tokens=100, output_tokens=10, latency_ms=50.0)


def test_evaluate_cascade_measures_accuracy_cost_and_latency() -> None:
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.8, 0.2]])
    result = evaluate_cascade(
        student="static_embed",
        threshold=0.7,
        proba=proba,
        classes=("no", "yes"),
        gold=["no", "yes", "yes"],
        teacher_predictions=[_teacher("no"), _teacher("yes"), _teacher("no")],
        student_latency_ms=[2.0, 3.0, 4.0],
        price_per_million_input_tokens=2.0,
        price_per_million_output_tokens=1.0,
    )
    assert result.examples == 3
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.student_only_accuracy == pytest.approx(2 / 3)
    assert result.teacher_only_accuracy == pytest.approx(2 / 3)
    assert result.teacher_calls == 1
    assert result.fallback_rate == pytest.approx(1 / 3)
    assert result.estimated_cost == pytest.approx((100 * 2.0 + 10 * 1.0) / 1_000_000)
    assert result.p95_latency_ms == pytest.approx(48.1)


def test_threshold_one_uses_only_teacher() -> None:
    result = evaluate_cascade(
        student="m",
        threshold=1.0,
        proba=np.array([[0.9, 0.1]]),
        classes=("yes", "no"),
        gold=["yes"],
        teacher_predictions=[_teacher("yes")],
        student_latency_ms=[1.0],
        price_per_million_input_tokens=0.0,
        price_per_million_output_tokens=0.0,
    )
    assert result.accuracy == 1.0
    assert result.teacher_calls == 1
    assert result.estimated_cost == 0.0


def test_validation() -> None:
    with pytest.raises(ValueError, match="negative"):
        TeacherPrediction("yes", -1, 0, 0.0)
    with pytest.raises(ValueError, match="prices"):
        evaluate_cascade(
            student="m",
            threshold=0.5,
            proba=np.array([[0.6, 0.4]]),
            classes=("yes", "no"),
            gold=["yes"],
            teacher_predictions=[_teacher()],
            student_latency_ms=[1.0],
            price_per_million_input_tokens=-1.0,
            price_per_million_output_tokens=0.0,
        )
    with pytest.raises(ValueError, match="examples"):
        CascadeResult("m", 0.5, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
