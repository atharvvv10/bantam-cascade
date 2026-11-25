"""Shared interfaces, validation, and persistence for CPU-only students."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np


class StudentDependencyError(RuntimeError):
    """Raised when an optional student dependency is not installed."""


class StudentModel(Protocol):
    """A CPU-only classifier that produces calibrated class probabilities."""

    name: str
    classes: tuple[str, ...]

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        """Train the student on gold-labeled texts."""
        ...

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n_texts, n_classes)`` probability array."""
        ...

    def artifact(self) -> StudentArtifact:
        """Return a picklable artifact for later inference."""
        ...


@dataclass(frozen=True)
class StudentArtifact:
    """Saved state of a trained student."""

    name: str
    classes: tuple[str, ...]
    payload: dict[str, object]


@dataclass(frozen=True)
class StudentReport:
    """Accuracy and confidence of one student on gold-labeled calibration data."""

    name: str
    examples: int
    accuracy: float
    mean_confidence: float

    def __post_init__(self) -> None:
        """Reject impossible report values."""
        if self.examples <= 0:
            raise ValueError("examples must be positive")
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError("accuracy must be in [0, 1]")
        if not 0.0 <= self.mean_confidence <= 1.0:
            raise ValueError("mean_confidence must be in [0, 1]")


def validate_training_input(texts: Sequence[str], labels: Sequence[str]) -> tuple[str, ...]:
    """Validate training texts and labels, returning sorted unique classes."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if not texts:
        raise ValueError("at least one training example is required")
    if any(not text.strip() for text in texts):
        raise ValueError("training texts must not be empty")
    classes = tuple(sorted(set(labels)))
    if len(classes) < 2:
        raise ValueError("at least two distinct labels are required")
    return classes


def confidence_scores(proba: np.ndarray) -> np.ndarray:
    """Return the maximum class probability for each row."""
    array = np.asarray(proba, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("proba must be a non-empty 2-D array")
    return np.asarray(array.max(axis=1), dtype=np.float64)


def labels_from_proba(proba: np.ndarray, classes: Sequence[str]) -> tuple[str, ...]:
    """Convert a probability array into predicted label names."""
    array = np.asarray(proba, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("proba must be a non-empty 2-D array")
    if array.shape[1] != len(classes):
        raise ValueError("proba columns must match the number of classes")
    return tuple(classes[int(index)] for index in array.argmax(axis=1))


def align_proba(
    proba: np.ndarray, source_classes: Sequence[str], target_classes: Sequence[str]
) -> np.ndarray:
    """Reorder probability columns to match ``target_classes``."""
    array = np.asarray(proba, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(source_classes):
        raise ValueError("proba columns must match source classes")
    mapping = {name: index for index, name in enumerate(source_classes)}
    missing = [name for name in target_classes if name not in mapping]
    if missing:
        raise ValueError(f"classes missing from probability output: {missing}")
    return np.asarray(array[:, [mapping[name] for name in target_classes]], dtype=np.float64)


def save_student(student: StudentModel, path: Path) -> None:
    """Persist a trained student's artifact to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(student.artifact(), path)


def load_student(path: Path) -> tuple[str, StudentArtifact]:
    """Load a saved artifact, verifying it has the expected shape."""
    if not path.is_file():
        raise FileNotFoundError(f"Saved student does not exist: {path}")
    loaded = joblib.load(path)
    if not isinstance(loaded, StudentArtifact):
        raise ValueError(f"Unexpected saved student format in {path}")
    artifact = loaded
    if not isinstance(artifact.payload, dict):
        raise ValueError(f"Unexpected saved student payload in {path}")
    return artifact.name, artifact
