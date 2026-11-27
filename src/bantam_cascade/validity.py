"""Monte Carlo validity experiment for the certification procedure."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from bantam_cascade.certification import alpha_from_minimum_accuracy, hb_p_value


@dataclass(frozen=True)
class ValidityScenario:
    """A finite population with known correctness probabilities.

    Student correctness is a linear function of confidence; teacher correctness
    is constant. Every threshold's true accuracy is therefore known before the
    test sample is drawn.
    """

    name: str
    student_intercept: float
    student_slope: float
    teacher_accuracy: float
    target_accuracy: float
    confidence_level: float
    examples: int
    calibration_examples: int
    trials: int
    grid_size: int
    population_size: int
    seed: int


@dataclass(frozen=True)
class ValidityTrial:
    """One repetition of valid and naive threshold selection."""

    chosen_threshold: float
    true_accuracy: float
    measured_accuracy: float
    errors: int
    p_value: float
    certified: bool
    false_certificate: bool
    naive_threshold: float
    naive_true_accuracy: float
    naive_measured_accuracy: float
    naive_p_value: float
    naive_certified: bool
    naive_false_certificate: bool


@dataclass(frozen=True)
class ValidityReport:
    """Aggregate validity rates for one scenario."""

    scenario: ValidityScenario
    trials: int
    false_certificate_rate: float
    naive_false_certificate_rate: float
    certification_rate: float
    naive_certification_rate: float
    max_false_certificate_rate_allowed: float
    mean_true_accuracy: float
    details: tuple[ValidityTrial, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report."""
        payload = asdict(self)
        payload["scenario"] = asdict(self.scenario)
        payload["details"] = [asdict(item) for item in self.details]
        return payload


def _threshold_accuracies(
    *,
    thresholds: np.ndarray,
    confidences: np.ndarray,
    student_ok: np.ndarray,
    teacher_ok: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return accuracy and fallback rate for each threshold on given indices."""
    accuracies = np.empty(len(thresholds), dtype=np.float64)
    fallbacks = np.empty(len(thresholds), dtype=np.float64)
    for index, threshold in enumerate(thresholds):
        use_student = confidences >= threshold
        outcome = np.where(use_student, student_ok, teacher_ok)
        accuracies[index] = float(np.mean(outcome))
        fallbacks[index] = 1.0 - float(np.mean(use_student))
    return accuracies, fallbacks


def run_validity_experiment(scenario: ValidityScenario) -> ValidityReport:
    """Repeat certification on resampled finite populations with known truth.

    The valid path selects a threshold on an independent calibration resample,
    then tests it once on a separate certification resample. The naive path
    selects the best threshold on the certification resample and tests that same
    resample, which is the overfitting behavior the ledger is designed to avoid.
    """
    if scenario.trials <= 0:
        raise ValueError("trials must be positive")
    if scenario.examples <= 0 or scenario.calibration_examples <= 0:
        raise ValueError("sample sizes must be positive")
    if scenario.grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    alpha = alpha_from_minimum_accuracy(scenario.target_accuracy)
    delta = 1.0 - scenario.confidence_level
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("target_accuracy and confidence_level must be in (0, 1)")

    rng = np.random.default_rng(scenario.seed)
    thresholds = np.linspace(0.0, 1.0, scenario.grid_size)
    details: list[ValidityTrial] = []
    false_certificates = 0
    naive_false_certificates = 0
    certified_count = 0
    naive_certified_count = 0
    true_accuracy_sum = 0.0

    for _ in range(scenario.trials):
        confidences = rng.random(scenario.population_size)
        student_probability = np.clip(
            scenario.student_intercept + scenario.student_slope * confidences, 0.0, 1.0
        )
        student_ok = rng.random(scenario.population_size) < student_probability
        teacher_ok = rng.random(scenario.population_size) < scenario.teacher_accuracy
        population_accuracy, _ = _threshold_accuracies(
            thresholds=thresholds,
            confidences=confidences,
            student_ok=student_ok,
            teacher_ok=teacher_ok,
        )

        calibration_indices = rng.choice(
            scenario.population_size, size=scenario.calibration_examples, replace=True
        )
        calibration_accuracy, calibration_fallback = _threshold_accuracies(
            thresholds=thresholds,
            confidences=confidences[calibration_indices],
            student_ok=student_ok[calibration_indices],
            teacher_ok=teacher_ok[calibration_indices],
        )
        chosen_index = int(np.lexsort((-calibration_fallback, -calibration_accuracy))[0])

        test_indices = rng.choice(scenario.population_size, size=scenario.examples, replace=True)
        keep = confidences[test_indices] >= thresholds[chosen_index]
        test_outcome = np.where(keep, student_ok[test_indices], teacher_ok[test_indices])
        errors = int(np.count_nonzero(~test_outcome))
        p_value = hb_p_value(errors=errors, examples=scenario.examples, alpha=alpha)
        certified = p_value <= delta
        true_accuracy = float(population_accuracy[chosen_index])
        false_certificate = certified and true_accuracy < scenario.target_accuracy
        false_certificates += int(false_certificate)
        certified_count += int(certified)
        true_accuracy_sum += true_accuracy

        naive_accuracy, naive_fallback = _threshold_accuracies(
            thresholds=thresholds,
            confidences=confidences[test_indices],
            student_ok=student_ok[test_indices],
            teacher_ok=teacher_ok[test_indices],
        )
        naive_index = int(np.lexsort((-naive_fallback, -naive_accuracy))[0])
        naive_keep = confidences[test_indices] >= thresholds[naive_index]
        naive_outcome = np.where(naive_keep, student_ok[test_indices], teacher_ok[test_indices])
        naive_errors = int(np.count_nonzero(~naive_outcome))
        naive_p_value = hb_p_value(errors=naive_errors, examples=scenario.examples, alpha=alpha)
        naive_certified = naive_p_value <= delta
        naive_true_accuracy = float(population_accuracy[naive_index])
        naive_false = naive_certified and naive_true_accuracy < scenario.target_accuracy
        naive_false_certificates += int(naive_false)
        naive_certified_count += int(naive_certified)

        details.append(
            ValidityTrial(
                chosen_threshold=float(thresholds[chosen_index]),
                true_accuracy=true_accuracy,
                measured_accuracy=float(test_outcome.mean()),
                errors=errors,
                p_value=p_value,
                certified=certified,
                false_certificate=false_certificate,
                naive_threshold=float(thresholds[naive_index]),
                naive_true_accuracy=naive_true_accuracy,
                naive_measured_accuracy=float(naive_outcome.mean()),
                naive_p_value=naive_p_value,
                naive_certified=naive_certified,
                naive_false_certificate=naive_false,
            )
        )

    trials = scenario.trials
    return ValidityReport(
        scenario=scenario,
        trials=trials,
        false_certificate_rate=false_certificates / trials,
        naive_false_certificate_rate=naive_false_certificates / trials,
        certification_rate=certified_count / trials,
        naive_certification_rate=naive_certified_count / trials,
        max_false_certificate_rate_allowed=delta,
        mean_true_accuracy=true_accuracy_sum / trials,
        details=tuple(details),
    )
