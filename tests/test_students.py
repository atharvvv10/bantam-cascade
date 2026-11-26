import builtins
from pathlib import Path

import numpy as np
import pytest

from bantam_cascade.students import (
    MiniLMOnnxStudent,
    StaticEmbedStudent,
    WordTfidfStudent,
    build_student,
    restore_student,
    save_student,
)
from bantam_cascade.students.base import (
    StudentArtifact,
    StudentDependencyError,
    StudentReport,
    align_proba,
    confidence_scores,
    labels_from_proba,
    validate_training_input,
)


def test_static_embed_trains_and_predicts(tmp_path: Path) -> None:
    student = StaticEmbedStudent()
    texts = [
        "billing invoice issue",
        "billing charge dispute",
        "shipping delayed parcel",
        "shipping tracking number",
        "support reset password",
        "support account help",
    ]
    labels = ["billing", "billing", "shipping", "shipping", "support", "support"]
    student.fit(texts, labels)
    assert student.classes == ("billing", "shipping", "support")
    proba = student.predict_proba(["billing charge", "parcel delayed"])
    assert proba.shape == (2, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert labels_from_proba(proba, student.classes)[0] == "billing"

    path = tmp_path / "student.joblib"
    save_student(student, path)
    restored = restore_student(path)
    assert isinstance(restored, StaticEmbedStudent)
    assert np.allclose(restored.predict_proba(["billing charge"]), proba[:1])


def test_word_tfidf_trains_and_restores(tmp_path: Path) -> None:
    student = WordTfidfStudent()
    texts = [
        "free free prize prize now now",
        "urgent urgent win win cash cash",
        "free prize claim claim now",
        "urgent win cash offer offer",
        "meeting meeting at at ten ten",
        "lunch lunch tomorrow tomorrow noon noon",
        "meeting at ten today today",
        "lunch at noon today",
        "project notes attached attached",
        "please review review report report",
    ]
    labels = ["spam", "spam", "spam", "spam", "ham", "ham", "ham", "ham", "ham", "ham"]
    student.fit(texts, labels)
    assert student.classes == ("ham", "spam")
    proba = student.predict_proba(["claim your prize", "lunch at noon"])
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert labels_from_proba(proba, student.classes)[0] == "spam"

    path = tmp_path / "word_tfidf.joblib"
    save_student(student, path)
    restored = restore_student(path)
    assert isinstance(restored, WordTfidfStudent)
    assert np.allclose(restored.predict_proba(["claim your prize"]), proba[:1])

    with pytest.raises(ValueError, match="not a word_tfidf"):
        WordTfidfStudent.from_artifact(StudentArtifact("static_embed", ("ham", "spam"), {}))


def test_static_embed_rejects_bad_inputs() -> None:
    student = StaticEmbedStudent()
    with pytest.raises(RuntimeError, match="fit"):
        student.predict_proba(["x"])
    with pytest.raises(ValueError, match="same length"):
        student.fit(["a", "b"], ["billing"])
    with pytest.raises(ValueError, match="two distinct"):
        student.fit(["a", "b"], ["billing", "billing"])
    with pytest.raises(ValueError, match="missing vectorizer"):
        StaticEmbedStudent.from_artifact(StudentArtifact("static_embed", ("a", "b"), {}))


def test_helper_validation_and_alignment() -> None:
    assert validate_training_input(["a", "b"], ["z", "a"]) == ("a", "z")
    proba = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert np.allclose(confidence_scores(proba), [0.9, 0.8])
    assert labels_from_proba(proba, ("a", "z")) == ("z", "a")
    aligned = align_proba(proba, ("z", "a"), ("a", "z"))
    assert np.allclose(aligned, proba[:, ::-1])
    with pytest.raises(ValueError, match="non-empty 2-D"):
        confidence_scores(np.array([1.0]))


def test_student_report_validates() -> None:
    assert StudentReport("m", 2, 0.5, 0.7).accuracy == 0.5
    with pytest.raises(ValueError, match="accuracy"):
        StudentReport("m", 2, 1.5, 0.7)


def test_build_student_names() -> None:
    assert isinstance(build_student("static_embed"), StaticEmbedStudent)
    assert isinstance(build_student("word_tfidf"), WordTfidfStudent)
    int8 = build_student("minilm_onnx_int8")
    fp32 = build_student("minilm_onnx_fp32")
    assert isinstance(int8, MiniLMOnnxStudent) and int8.quantized
    assert isinstance(fp32, MiniLMOnnxStudent) and not fp32.quantized
    with pytest.raises(ValueError, match="Unsupported student"):
        build_student("nope")


def test_minilm_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def restricted_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"onnxruntime", "tokenizers"}:
            raise ImportError("simulated missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", restricted_import)
    student = MiniLMOnnxStudent()
    with pytest.raises(StudentDependencyError, match="optional dependencies"):
        student.embed(["hello"])
