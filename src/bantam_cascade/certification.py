"""Learn-then-Test certification for cascades on a single-use test split."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal

from bantam_cascade.cascade import CascadeResult

CertificationStatus = Literal["CERTIFIED", "INSUFFICIENT_EVIDENCE", "NO_FEASIBLE_CONFIG"]

CERTIFICATION_ASSUMPTIONS: tuple[str, ...] = (
    "Certification examples are i.i.d. draws from the deployment distribution.",
    "Deployment distribution matches the calibration and certification data used here.",
    "Teacher model id, date, and prompt version are pinned by config and cached.",
    "API determinism is not guaranteed; results are relative to the cached teacher labels.",
    "Accuracy is measured only against gold human labels.",
)


@dataclass(frozen=True)
class Candidate:
    """One pre-registered cascade configuration with learn- and test-split results."""

    student: str
    threshold: float
    calibration: CascadeResult
    certification: CascadeResult

    def __post_init__(self) -> None:
        """Reject candidates whose learn and test results disagree on identity."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.calibration.student != self.student or self.certification.student != self.student:
            raise ValueError("calibration, certification, and candidate student must match")
        if (
            self.calibration.threshold != self.threshold
            or self.certification.threshold != self.threshold
        ):
            raise ValueError("calibration, certification, and candidate threshold must match")


@dataclass(frozen=True)
class CandidateEvaluation:
    """Certification test result for one candidate configuration."""

    student: str
    threshold: float
    p_value: float
    delta: float
    certified: bool
    latency_ok: bool
    examples: int
    errors: int
    measured_accuracy: float
    calibration_accuracy: float
    fallback_rate: float
    estimated_cost: float
    p95_latency_ms: float
    exact_binomial_p_value: float = 0.0


@dataclass(frozen=True)
class RecallCandidateEvaluation:
    """Co-primary accuracy-and-recall test result for one candidate configuration."""

    student: str
    threshold: float
    accuracy_errors: int
    accuracy_examples: int
    accuracy_p_value: float
    accuracy_exact_p_value: float
    recall_errors: int
    recall_examples: int
    recall_p_value: float
    recall_exact_p_value: float
    per_endpoint_delta: float
    certified: bool
    latency_ok: bool
    measured_accuracy: float
    measured_recall: float
    calibration_accuracy: float
    fallback_rate: float
    estimated_cost: float
    p95_latency_ms: float


@dataclass(frozen=True)
class CertificationOutcome:
    """Final outcome of a Learn-then-Test certification run."""

    status: CertificationStatus
    alpha: float
    delta: float
    candidates: tuple[CandidateEvaluation, ...]
    certified: tuple[CandidateEvaluation, ...]
    chosen: CandidateEvaluation | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary of the certification outcome."""
        return {
            "status": self.status,
            "alpha": self.alpha,
            "delta": self.delta,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "certified": [asdict(candidate) for candidate in self.certified],
            "chosen": None if self.chosen is None else asdict(self.chosen),
            "assumptions": list(CERTIFICATION_ASSUMPTIONS),
        }


def alpha_from_minimum_accuracy(minimum_accuracy: float) -> float:
    """Convert a target accuracy into the maximum allowed error risk."""
    if not 0.0 < minimum_accuracy < 1.0:
        raise ValueError(f"minimum_accuracy must be in (0, 1), got {minimum_accuracy}")
    return 1.0 - minimum_accuracy


def delta_from_confidence_level(confidence_level: float) -> float:
    """Convert a confidence level into the family-wise error level."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    return 1.0 - confidence_level


def errors_from_accuracy(*, accuracy: float, examples: int) -> int:
    """Recover the integer error count behind a recorded accuracy."""
    if examples <= 0:
        raise ValueError("examples must be positive")
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy must be in [0, 1], got {accuracy}")
    exact = (1.0 - accuracy) * examples
    errors = int(round(exact))
    if abs(exact - errors) > 1e-6:
        raise ValueError("accuracy does not correspond to an integer error count")
    return errors


def kl_divergence(a: float, b: float) -> float:
    """Return the binary KL divergence ``h1(a, b)`` with safe boundary handling."""
    if not 0.0 <= a <= 1.0 or not 0.0 < b < 1.0:
        raise ValueError("a must be in [0, 1] and b must be in (0, 1)")
    if a == 0.0:
        return -math.log1p(-b)
    if a == 1.0:
        return -math.log(b)
    return a * math.log(a / b) + (1.0 - a) * math.log((1.0 - a) / (1.0 - b))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """Return ``P(Bin(n, p) <= k)`` using a recurrence without large integers."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1)")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    term = (1.0 - p) ** n
    total = term
    odds = p / (1.0 - p)
    for index in range(1, min(k, n - 1) + 1):
        term *= (n - index + 1) / index * odds
        total += term
    return min(1.0, total)


def exact_binomial_p_value(*, errors: int, examples: int, alpha: float) -> float:
    """Return the exact one-sided binomial p-value for ``H: risk > alpha``."""
    if examples <= 0:
        raise ValueError("examples must be positive")
    if not 0 <= errors <= examples:
        raise ValueError("errors must be between 0 and examples")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return min(1.0, binomial_cdf(errors, examples, alpha))


def hb_p_value(*, errors: int, examples: int, alpha: float) -> float:
    """Return the Hoeffding-Bentkus p-value for ``H: risk > alpha``."""
    if examples <= 0:
        raise ValueError("examples must be positive")
    if not 0 <= errors <= examples:
        raise ValueError("errors must be between 0 and examples")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    empirical_risk = errors / examples
    clipped = min(empirical_risk, alpha)
    hoeffding = math.exp(-examples * kl_divergence(clipped, alpha))
    binomial = min(
        1.0, math.e * exact_binomial_p_value(errors=errors, examples=examples, alpha=alpha)
    )
    return min(hoeffding, binomial)


def required_examples_for_accuracy(
    *,
    target_accuracy: float,
    confidence_level: float,
    assumed_accuracy: float,
    max_examples: int = 100_000,
) -> int:
    """Find the smallest test size that can reject ``risk > 1 - target``.

    The assumed accuracy must exceed the target; its expected error count is
    rounded up so the calculation is conservative.
    """
    alpha = alpha_from_minimum_accuracy(target_accuracy)
    delta = delta_from_confidence_level(confidence_level)
    if not 0.0 < target_accuracy < assumed_accuracy < 1.0:
        raise ValueError("require 0 < target_accuracy < assumed_accuracy < 1")
    for examples in range(1, max_examples + 1):
        errors = math.ceil((1.0 - assumed_accuracy) * examples)
        if hb_p_value(errors=errors, examples=examples, alpha=alpha) <= delta:
            return examples
    raise ValueError(f"no sufficient sample size below {max_examples}")


def fixed_sequence_rejections(p_values: Sequence[float], delta: float) -> tuple[int, ...]:
    """Return indices rejected by a fixed sequence test stopped at the first failure."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    rejected: list[int] = []
    for index, p_value in enumerate(p_values):
        if not 0.0 <= p_value <= 1.0:
            raise ValueError(f"p_value must be in [0, 1], got {p_value}")
        if p_value > delta:
            break
        rejected.append(index)
    return tuple(rejected)


def certify_candidates(
    candidates: Sequence[Candidate],
    *,
    minimum_accuracy: float,
    delta: float,
    max_p95_latency_ms: float,
) -> CertificationOutcome:
    """Certify a pre-ordered candidate list using a single-use test split.

    The ordering must be learned from calibration data only. Among hypotheses
    rejected by the fixed sequence test, the cheapest candidate meeting the
    measured latency budget is selected.

    Args:
        candidates: Configurations in pre-registered order.
        minimum_accuracy: Required accuracy; risk ``alpha`` is its complement.
        delta: Already-corrected family-wise error level.
        max_p95_latency_ms: Maximum allowed measured p95 latency.

    Returns:
        A :class:`CertificationOutcome` with an explicit status.
    """
    alpha = alpha_from_minimum_accuracy(minimum_accuracy)
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if max_p95_latency_ms <= 0.0:
        raise ValueError("max_p95_latency_ms must be positive")
    if not candidates:
        return CertificationOutcome(
            status="NO_FEASIBLE_CONFIG",
            alpha=alpha,
            delta=delta,
            candidates=(),
            certified=(),
            chosen=None,
        )

    evaluations: list[CandidateEvaluation] = []
    for candidate in candidates:
        examples = candidate.certification.examples
        errors = errors_from_accuracy(
            accuracy=candidate.certification.accuracy,
            examples=examples,
        )
        if not 0 <= errors <= examples:
            raise ValueError("candidate accuracy implies an impossible error count")
        p_value = hb_p_value(errors=errors, examples=examples, alpha=alpha)
        exact = exact_binomial_p_value(errors=errors, examples=examples, alpha=alpha)
        evaluations.append(
            CandidateEvaluation(
                student=candidate.student,
                threshold=candidate.threshold,
                p_value=p_value,
                exact_binomial_p_value=exact,
                delta=delta,
                certified=False,
                latency_ok=candidate.certification.p95_latency_ms <= max_p95_latency_ms,
                examples=examples,
                errors=errors,
                measured_accuracy=candidate.certification.accuracy,
                calibration_accuracy=candidate.calibration.accuracy,
                fallback_rate=candidate.certification.fallback_rate,
                estimated_cost=candidate.certification.estimated_cost,
                p95_latency_ms=candidate.certification.p95_latency_ms,
            )
        )

    rejected = fixed_sequence_rejections([item.p_value for item in evaluations], delta)
    for index in rejected:
        evaluations[index] = replace(evaluations[index], certified=True)
    certified = tuple(evaluations[index] for index in rejected)
    feasible = [item for item in certified if item.latency_ok]
    chosen = (
        min(
            feasible,
            key=lambda item: (
                item.estimated_cost,
                item.fallback_rate,
                item.threshold,
                item.student,
            ),
        )
        if feasible
        else None
    )
    if chosen is not None:
        status: CertificationStatus = "CERTIFIED"
    elif not certified:
        status = "INSUFFICIENT_EVIDENCE"
    else:
        status = "NO_FEASIBLE_CONFIG"
    return CertificationOutcome(
        status=status,
        alpha=alpha,
        delta=delta,
        candidates=tuple(evaluations),
        certified=certified,
        chosen=chosen,
    )


def certify_candidates_with_recall(
    candidates: Sequence[Candidate],
    *,
    minimum_accuracy: float,
    minimum_recall: float,
    delta: float,
    max_p95_latency_ms: float,
    recall_counts: Sequence[tuple[int, int]],
) -> dict[str, object]:
    """Certify two co-primary endpoints with a Bonferroni-split fixed sequence.

    Each candidate must reject ``risk > 1 - minimum_accuracy`` for overall
    accuracy and ``risk > 1 - minimum_recall`` for positive-class recall. The
    family-wise error level ``delta`` is split evenly, so each endpoint is
    tested at ``delta / 2``. Candidates are walked in their pre-registered
    order and the sequence stops at the first candidate that fails either
    endpoint.

    Args:
        candidates: Configurations in pre-registered order.
        minimum_accuracy: Required overall accuracy.
        minimum_recall: Required positive-class recall.
        delta: Family-wise error level, already corrected for repeated looks.
        max_p95_latency_ms: Maximum allowed measured p95 latency.
        recall_counts: ``(errors, positives)`` for each candidate, in the same
            order as ``candidates``.

    Returns:
        A JSON-serializable outcome dictionary with an explicit status.
    """
    accuracy_alpha = alpha_from_minimum_accuracy(minimum_accuracy)
    recall_alpha = alpha_from_minimum_accuracy(minimum_recall)
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if max_p95_latency_ms <= 0.0:
        raise ValueError("max_p95_latency_ms must be positive")
    if len(recall_counts) != len(candidates):
        raise ValueError("recall_counts must have one entry per candidate")

    per_endpoint_delta = delta / 2.0
    evaluations: list[RecallCandidateEvaluation] = []
    for candidate, (recall_errors, positives) in zip(candidates, recall_counts, strict=True):
        examples = candidate.certification.examples
        accuracy_errors = errors_from_accuracy(
            accuracy=candidate.certification.accuracy,
            examples=examples,
        )
        if positives <= 0 or not 0 <= recall_errors <= positives:
            raise ValueError("recall_counts must have positive examples and valid errors")
        accuracy_p = hb_p_value(errors=accuracy_errors, examples=examples, alpha=accuracy_alpha)
        recall_p = hb_p_value(errors=recall_errors, examples=positives, alpha=recall_alpha)
        evaluations.append(
            RecallCandidateEvaluation(
                student=candidate.student,
                threshold=candidate.threshold,
                accuracy_errors=accuracy_errors,
                accuracy_examples=examples,
                accuracy_p_value=accuracy_p,
                accuracy_exact_p_value=exact_binomial_p_value(
                    errors=accuracy_errors, examples=examples, alpha=accuracy_alpha
                ),
                recall_errors=recall_errors,
                recall_examples=positives,
                recall_p_value=recall_p,
                recall_exact_p_value=exact_binomial_p_value(
                    errors=recall_errors, examples=positives, alpha=recall_alpha
                ),
                per_endpoint_delta=per_endpoint_delta,
                certified=False,
                latency_ok=candidate.certification.p95_latency_ms <= max_p95_latency_ms,
                measured_accuracy=candidate.certification.accuracy,
                measured_recall=1.0 - recall_errors / positives,
                calibration_accuracy=candidate.calibration.accuracy,
                fallback_rate=candidate.certification.fallback_rate,
                estimated_cost=candidate.certification.estimated_cost,
                p95_latency_ms=candidate.certification.p95_latency_ms,
            )
        )

    certified_list: list[RecallCandidateEvaluation] = []
    for index, item in enumerate(evaluations):
        if item.accuracy_p_value > per_endpoint_delta or item.recall_p_value > per_endpoint_delta:
            break
        updated = replace(item, certified=True)
        evaluations[index] = updated
        certified_list.append(updated)
    certified = tuple(certified_list)
    feasible = [item for item in certified if item.latency_ok]
    chosen = (
        min(
            feasible,
            key=lambda item: (
                item.estimated_cost,
                item.fallback_rate,
                item.threshold,
                item.student,
            ),
        )
        if feasible
        else None
    )
    if chosen is not None:
        status: CertificationStatus = "CERTIFIED"
    elif not certified:
        status = "INSUFFICIENT_EVIDENCE"
    else:
        status = "NO_FEASIBLE_CONFIG"
    return {
        "status": status,
        "minimum_accuracy": minimum_accuracy,
        "minimum_recall": minimum_recall,
        "alpha_accuracy": accuracy_alpha,
        "alpha_recall": recall_alpha,
        "family_wise_delta": delta,
        "per_endpoint_delta": per_endpoint_delta,
        "multiplicity": "Bonferroni split across accuracy and recall",
        "candidates": [asdict(item) for item in evaluations],
        "certified": [asdict(item) for item in certified],
        "chosen": None if chosen is None else asdict(chosen),
        "assumptions": [
            *CERTIFICATION_ASSUMPTIONS,
            "Accuracy and recall are co-primary endpoints sharing one family-wise error budget.",
        ],
    }
