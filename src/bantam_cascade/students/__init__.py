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
from bantam_cascade.students.static_embed import StaticEmbedStudent

_STUDENT_TYPES: dict[str, type[StaticEmbedStudent] | type[MiniLMOnnxStudent]] = {
    StaticEmbedStudent.name: StaticEmbedStudent,
    "minilm_onnx_int8": MiniLMOnnxStudent,
    "minilm_onnx_fp32": MiniLMOnnxStudent,
}


def build_student(name: str) -> StudentModel:
    """Construct an untrained student by its configured name."""
    if name not in _STUDENT_TYPES:
        raise ValueError(f"Unsupported student: {name!r}")
    if name == "static_embed":
        return StaticEmbedStudent()
    return MiniLMOnnxStudent(quantized=name == "minilm_onnx_int8")


def restore_student(path: Path) -> StudentModel:
    """Load a saved student artifact into the correct student class."""
    name, artifact = load_student(path)
    if name not in _STUDENT_TYPES:
        raise ValueError(f"Unsupported saved student: {name!r}")
    if name == "static_embed":
        return StaticEmbedStudent.from_artifact(artifact)
    return MiniLMOnnxStudent.from_artifact(artifact)


__all__ = [
    "MiniLMOnnxStudent",
    "StaticEmbedStudent",
    "StudentArtifact",
    "StudentDependencyError",
    "StudentModel",
    "StudentReport",
    "build_student",
    "load_student",
    "restore_student",
    "save_student",
]
