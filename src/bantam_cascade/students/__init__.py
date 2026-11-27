"""CPU-only student classifiers and their persistence helpers."""

from __future__ import annotations

from pathlib import Path

from bantam_cascade.students.base import (
    StudentArtifact,
    StudentDependencyError,
    StudentModel,
    StudentReport,
    load_student,
    save_student,
)
from bantam_cascade.students.minilm_onnx import MiniLMOnnxStudent
from bantam_cascade.students.static_embed import StaticEmbedStudent, TfidfBaselineStudent
from bantam_cascade.students.word_tfidf import WordTfidfStudent

_TFIDF_BASELINE = TfidfBaselineStudent.canonical_name
_LEGACY_NAMES = {name: _TFIDF_BASELINE for name in TfidfBaselineStudent.legacy_names}
_STUDENT_TYPES: dict[
    str, type[TfidfBaselineStudent] | type[MiniLMOnnxStudent] | type[WordTfidfStudent]
] = {
    TfidfBaselineStudent.canonical_name: TfidfBaselineStudent,
    WordTfidfStudent.name: WordTfidfStudent,
    "minilm_onnx_int8": MiniLMOnnxStudent,
    "minilm_onnx_fp32": MiniLMOnnxStudent,
}


def _canonical_name(name: str) -> str:
    """Map historical student names to their current canonical identifiers."""
    return _LEGACY_NAMES.get(name, name)


def build_student(name: str) -> StudentModel:
    """Construct an untrained student by its configured name."""
    canonical = _canonical_name(name)
    if canonical not in _STUDENT_TYPES:
        raise ValueError(f"Unsupported student: {name!r}")
    if canonical == _TFIDF_BASELINE:
        return TfidfBaselineStudent()
    if canonical == WordTfidfStudent.name:
        return WordTfidfStudent()
    return MiniLMOnnxStudent(quantized=canonical == "minilm_onnx_int8")


def restore_student(path: Path) -> StudentModel:
    """Load a saved student artifact into the correct student class."""
    name, artifact = load_student(path)
    canonical = _canonical_name(name)
    if canonical not in _STUDENT_TYPES:
        raise ValueError(f"Unsupported saved student: {name!r}")
    if canonical == _TFIDF_BASELINE:
        return TfidfBaselineStudent.from_artifact(artifact)
    if canonical == WordTfidfStudent.name:
        return WordTfidfStudent.from_artifact(artifact)
    return MiniLMOnnxStudent.from_artifact(artifact)


__all__ = [
    "MiniLMOnnxStudent",
    "StaticEmbedStudent",
    "TfidfBaselineStudent",
    "WordTfidfStudent",
    "StudentArtifact",
    "StudentDependencyError",
    "StudentModel",
    "StudentReport",
    "build_student",
    "load_student",
    "restore_student",
    "save_student",
]
