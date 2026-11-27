import json
from pathlib import Path

import pytest

from bantam_cascade.gradio_app import read_thresholds, student_report
from bantam_cascade.students import TfidfBaselineStudent
from bantam_cascade.students.base import save_student


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "students").mkdir(parents=True)
    student = TfidfBaselineStudent()
    student.fit(
        [
            "billing invoice help",
            "billing charge dispute",
            "shipping parcel delayed",
            "shipping tracking",
        ],
        ["billing", "billing", "shipping", "shipping"],
    )
    save_student(student, run_dir / "students" / "tfidf_baseline.joblib")
    (run_dir / "cascade_results.jsonl").write_text(
        json.dumps({"student": "tfidf_baseline", "threshold": 0.5}) + "\n", encoding="utf-8"
    )
    return run_dir


def test_read_thresholds_and_student_report(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert read_thresholds(run_dir) == {"tfidf_baseline": 0.5}
    report = student_report(run_dir, "billing charge")
    assert "Offline student-only demo" in report
    assert "tfidf_baseline" in report
    assert "billing" in report


def test_read_thresholds_requires_results(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_thresholds(tmp_path)
