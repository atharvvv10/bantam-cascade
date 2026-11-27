import math

import pytest

from bantam_cascade.cascade import CascadeResult
from bantam_cascade.certification import (
    Candidate,
    alpha_from_minimum_accuracy,
    binomial_cdf,
    certify_candidates,
    certify_candidates_with_recall,
    delta_from_confidence_level,
    errors_from_accuracy,
    exact_binomial_p_value,
    fixed_sequence_rejections,
    hb_p_value,
    kl_divergence,
    required_examples_for_accuracy,
)


def _result(
    student: str,
    *,
    accuracy: float,
    examples: int = 300,
    threshold: float = 0.5,
    cost: float = 0.0,
    p95: float = 1.0,
    fallback_rate: float = 0.0,
) -> CascadeResult:
    errors = round((1.0 - accuracy) * examples)
    measured_accuracy = 1.0 - errors / examples
    return CascadeResult(
        student=student,
        threshold=threshold,
        examples=examples,
        accuracy=measured_accuracy,
        student_only_accuracy=measured_accuracy,
        teacher_only_accuracy=measured_accuracy,
        fallback_rate=fallback_rate,
        teacher_calls=round(fallback_rate * examples),
        estimated_cost=cost,
        p95_latency_ms=p95,
    )


def _candidate(
    student: str,
    *,
    accuracy: float,
    cost: float = 0.0,
    p95: float = 1.0,
    threshold: float = 0.5,
) -> Candidate:
    certification = _result(
        student,
        accuracy=accuracy,
        cost=cost,
        p95=p95,
        threshold=threshold,
    )
    calibration = _result(
        student,
        accuracy=accuracy,
        cost=cost,
        p95=p95,
        threshold=threshold,
    )
    return Candidate(
        student=student,
        threshold=threshold,
        calibration=calibration,
        certification=certification,
    )


def test_alpha_and_delta_conversions() -> None:
    assert alpha_from_minimum_accuracy(0.95) == pytest.approx(0.05)
    assert delta_from_confidence_level(0.95) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        alpha_from_minimum_accuracy(1.0)
    with pytest.raises(ValueError):
        delta_from_confidence_level(0.0)


def test_errors_from_accuracy_requires_integer_count() -> None:
    assert errors_from_accuracy(accuracy=0.92, examples=300) == 24
    with pytest.raises(ValueError, match="integer error count"):
        errors_from_accuracy(accuracy=0.923, examples=300)


def test_kl_and_binomial_helpers() -> None:
    assert kl_divergence(0.2, 0.2) == pytest.approx(0.0)
    assert kl_divergence(0.0, 0.1) == pytest.approx(-math.log(0.9))
    with pytest.raises(ValueError):
        kl_divergence(0.1, 0.0)
    assert binomial_cdf(-1, 10, 0.5) == pytest.approx(0.0)
    assert binomial_cdf(0, 10, 0.5) == pytest.approx(2**-10)
    assert binomial_cdf(10, 10, 0.5) == pytest.approx(1.0)


def test_exact_binomial_and_hb_p_values() -> None:
    exact = exact_binomial_p_value(errors=0, examples=100, alpha=0.1)
    assert exact == pytest.approx(0.9**100)
    hb = hb_p_value(errors=0, examples=100, alpha=0.1)
    assert hb == pytest.approx(exact)
    assert exact_binomial_p_value(errors=10, examples=100, alpha=0.1) > exact
    with pytest.raises(ValueError, match="between 0 and examples"):
        exact_binomial_p_value(errors=-1, examples=100, alpha=0.1)


def test_hb_p_value_is_small_only_when_risk_is_below_alpha() -> None:
    assert hb_p_value(errors=0, examples=100, alpha=0.1) == pytest.approx(0.9**100)
    assert hb_p_value(errors=10, examples=100, alpha=0.1) == pytest.approx(1.0)
    assert hb_p_value(errors=20, examples=100, alpha=0.1) == pytest.approx(1.0)


def test_required_examples_uses_conservative_error_count() -> None:
    assert (
        required_examples_for_accuracy(
            target_accuracy=0.95,
            confidence_level=0.95,
            assumed_accuracy=0.98,
        )
        == 240
    )
    with pytest.raises(ValueError, match="target_accuracy"):
        required_examples_for_accuracy(
            target_accuracy=0.98,
            confidence_level=0.95,
            assumed_accuracy=0.95,
        )


def test_fixed_sequence_stops_at_first_failure() -> None:
    assert fixed_sequence_rejections([0.01, 0.02, 0.2, 0.001], 0.05) == (0, 1)
    assert fixed_sequence_rejections([0.2, 0.001], 0.05) == ()


def test_certification_selects_cheapest_rejected_candidate() -> None:
    expensive = _candidate("minilm", accuracy=0.97, cost=1.0)
    cheap = _candidate("static", accuracy=0.95, cost=0.0)
    outcome = certify_candidates(
        [expensive, cheap],
        minimum_accuracy=0.90,
        delta=0.05,
        max_p95_latency_ms=10.0,
    )
    assert outcome.status == "CERTIFIED"
    assert len(outcome.certified) == 2
    assert [item.certified for item in outcome.candidates] == [True, True]
    assert outcome.chosen is not None
    assert outcome.chosen.student == "static"
    assert outcome.to_dict()["status"] == "CERTIFIED"


def test_first_failure_prevents_later_rejections() -> None:
    failing = _candidate("weak", accuracy=0.80)
    strong = _candidate("strong", accuracy=1.0)
    outcome = certify_candidates(
        [failing, strong],
        minimum_accuracy=0.90,
        delta=0.05,
        max_p95_latency_ms=10.0,
    )
    assert outcome.status == "INSUFFICIENT_EVIDENCE"
    assert outcome.certified == ()
    assert [item.certified for item in outcome.candidates] == [False, False]
    assert outcome.chosen is None


def test_certified_but_too_slow_is_no_feasible_config() -> None:
    slow = _candidate("slow", accuracy=1.0, p95=100.0)
    outcome = certify_candidates(
        [slow],
        minimum_accuracy=0.90,
        delta=0.05,
        max_p95_latency_ms=10.0,
    )
    assert outcome.status == "NO_FEASIBLE_CONFIG"
    assert len(outcome.certified) == 1
    assert outcome.chosen is None


def test_empty_candidate_list_is_explicit() -> None:
    outcome = certify_candidates([], minimum_accuracy=0.90, delta=0.05, max_p95_latency_ms=10.0)
    assert outcome.status == "NO_FEASIBLE_CONFIG"
    assert outcome.candidates == ()


def test_candidate_rejects_mismatched_identity() -> None:
    calibration = _result("one", accuracy=0.95, threshold=0.5)
    certification = _result("two", accuracy=0.95, threshold=0.5)
    with pytest.raises(ValueError, match="must match"):
        Candidate(
            student="one",
            threshold=0.5,
            calibration=calibration,
            certification=certification,
        )


def test_recall_certification_requires_both_endpoints() -> None:
    strong = _candidate("strong", accuracy=0.987)
    weak_recall = _candidate("weak-recall", accuracy=0.987)
    outcome = certify_candidates_with_recall(
        [strong, weak_recall],
        minimum_accuracy=0.95,
        minimum_recall=0.95,
        delta=0.05,
        max_p95_latency_ms=10.0,
        recall_counts=[(0, 100), (10, 100)],
    )
    assert outcome["status"] == "CERTIFIED"
    candidates = outcome["candidates"]
    assert isinstance(candidates, list)
    assert [row["certified"] for row in candidates] == [True, False]
    assert outcome["per_endpoint_delta"] == pytest.approx(0.025)


def test_recall_certification_reports_no_feasible_config() -> None:
    slow = _candidate("slow", accuracy=0.987, p95=500.0)
    outcome = certify_candidates_with_recall(
        [slow],
        minimum_accuracy=0.95,
        minimum_recall=0.95,
        delta=0.05,
        max_p95_latency_ms=10.0,
        recall_counts=[(0, 100)],
    )
    assert outcome["status"] == "NO_FEASIBLE_CONFIG"
    assert outcome["chosen"] is None


def test_recall_certification_rejects_mismatched_counts() -> None:
    candidate = _candidate("one", accuracy=0.987)
    with pytest.raises(ValueError, match="one entry per candidate"):
        certify_candidates_with_recall(
            [candidate],
            minimum_accuracy=0.95,
            minimum_recall=0.95,
            delta=0.05,
            max_p95_latency_ms=10.0,
            recall_counts=[],
        )
