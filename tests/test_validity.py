import pytest

from bantam_cascade.validity import ValidityScenario, run_validity_experiment


def _scenario(*, seed: int) -> ValidityScenario:
    return ValidityScenario(
        name="test",
        student_intercept=0.9495,
        student_slope=0.0,
        teacher_accuracy=0.9495,
        target_accuracy=0.95,
        confidence_level=0.95,
        examples=300,
        calibration_examples=100,
        trials=20,
        grid_size=50,
        population_size=5_000,
        seed=seed,
    )


def test_validity_report_has_known_truth_and_naive_control() -> None:
    report = run_validity_experiment(_scenario(seed=7))
    assert report.trials == 20
    assert len(report.details) == 20
    assert report.max_false_certificate_rate_allowed == pytest.approx(0.05)
    assert 0.0 <= report.false_certificate_rate <= 1.0
    assert 0.0 <= report.naive_false_certificate_rate <= 1.0
    assert 0.0 <= report.certification_rate <= 1.0
    assert 0.0 <= report.naive_certification_rate <= 1.0
    for trial in report.details:
        assert 0.0 <= trial.true_accuracy <= 1.0
        assert 0.0 <= trial.measured_accuracy <= 1.0
        assert 0.0 <= trial.p_value <= 1.0
        assert trial.false_certificate == (trial.certified and trial.true_accuracy < 0.95)


def test_validity_rejects_invalid_configuration() -> None:
    scenario = _scenario(seed=7)
    with pytest.raises(ValueError, match="trials"):
        run_validity_experiment(ValidityScenario(**{**scenario.__dict__, "trials": 0}))
    with pytest.raises(ValueError, match="grid_size"):
        run_validity_experiment(ValidityScenario(**{**scenario.__dict__, "grid_size": 1}))
