from pathlib import Path

import pytest

from bantam_cascade.teacher.cache import LabelCache
from bantam_cascade.teacher.client import LabelBatch, LabelResult, TeacherTotals
from bantam_cascade.teacher.select import (
    CandidateMetrics,
    evaluate_candidate,
    recommend,
    teacher_sanity_gate,
)


class _StubClient:
    """Duck-typed teacher client returning fixed labels in text order."""

    def __init__(self, predictions: list[str]) -> None:
        """Store predictions to return."""
        self.predictions = predictions

    def label_many(
        self,
        texts: list[str],
        labels: list[str],
        *,
        cache: LabelCache,
        model: str | None = None,
        prompt_version: str | None = None,
        limit: int | None = None,
    ) -> LabelBatch:
        """Return pre-baked results without network access."""
        results = tuple(
            LabelResult(
                text=text,
                label=self.predictions[index],
                input_tokens=10,
                output_tokens=2,
                latency_ms=5.0,
                cached=False,
            )
            for index, text in enumerate(texts)
        )
        total = len(results)
        return LabelBatch(
            results,
            TeacherTotals(total, 0, 10 * total, 2 * total, 5.0 * total, 0),
        )


def test_evaluate_candidate_and_recommend(tmp_path: Path) -> None:
    cache = LabelCache(tmp_path / "cache.sqlite3")
    texts = ["a", "b"]
    gold = ["billing", "shipping"]
    labels = ["billing", "shipping"]

    strong = evaluate_candidate(
        client=_StubClient(["billing", "shipping"]),
        cache=cache,
        texts=texts,
        gold_labels=gold,
        labels=labels,
        model="strong",
        price_per_million_input_tokens=0.0,
        price_per_million_output_tokens=0.0,
    )
    weak = evaluate_candidate(
        client=_StubClient(["billing", "billing"]),
        cache=cache,
        texts=texts,
        gold_labels=gold,
        labels=labels,
        model="weak",
        price_per_million_input_tokens=0.0,
        price_per_million_output_tokens=0.0,
    )
    assert strong.accuracy == 1.0
    assert weak.accuracy == 0.5
    assert strong.invalid_rate == 0.0
    assert strong.mean_latency_ms == 5.0
    assert recommend([weak, strong]) is strong
    assert recommend([]) is None


def test_evaluate_candidate_rejects_mismatched_lengths(tmp_path: Path) -> None:
    cache = LabelCache(tmp_path / "cache.sqlite3")
    with pytest.raises(ValueError, match="same length"):
        evaluate_candidate(
            client=_StubClient(["billing"]),
            cache=cache,
            texts=["a"],
            gold_labels=["billing", "shipping"],
            labels=["billing", "shipping"],
            model="m",
            price_per_million_input_tokens=0.0,
            price_per_million_output_tokens=0.0,
        )


def test_sanity_gate() -> None:
    assert teacher_sanity_gate(teacher_accuracy=0.9, best_student_accuracy=0.8)
    assert not teacher_sanity_gate(teacher_accuracy=0.7, best_student_accuracy=0.8)
    assert not teacher_sanity_gate(teacher_accuracy=0.8, best_student_accuracy=0.8, margin=0.01)
    with pytest.raises(ValueError, match="teacher_accuracy"):
        teacher_sanity_gate(teacher_accuracy=1.5, best_student_accuracy=0.5)


def test_metrics_reject_impossible_values() -> None:
    with pytest.raises(ValueError, match="examples must be positive"):
        CandidateMetrics("m", 0, 0, 0, 0, 0, 0.0, 0.0)
