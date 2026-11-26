"""Word n-gram TF-IDF plus logistic-regression student.

This is the strongest lightweight headline observed in calibration-only probes
on the SMS Spam benchmark. It uses only scikit-learn and runs on CPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from bantam_cascade.students.base import (
    StudentArtifact,
    align_proba,
    validate_training_input,
)


class WordTfidfStudent:
    """A word n-gram linear student with no model downloads."""

    name = "word_tfidf"

    def __init__(self) -> None:
        """Create an untrained student."""
        self.classes: tuple[str, ...] = ()
        self._vectorizer: TfidfVectorizer | None = None
        self._classifier: LogisticRegression | None = None

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        """Train the vectorizer and linear classifier on gold labels."""
        self.classes = validate_training_input(texts, labels)
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self._classifier = LogisticRegression(
            max_iter=3000,
            C=16.0,
            random_state=42,
        )
        features = self._vectorizer.fit_transform(list(texts))
        self._classifier.fit(features, list(labels))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return class probabilities for each text."""
        if self._vectorizer is None or self._classifier is None:
            raise RuntimeError("WordTfidfStudent.fit() must be called first")
        if not texts:
            raise ValueError("at least one text is required")
        features = self._vectorizer.transform(list(texts))
        proba = self._classifier.predict_proba(features)
        source = tuple(str(name) for name in self._classifier.classes_)
        return align_proba(proba, source, self.classes)

    def artifact(self) -> StudentArtifact:
        """Return a picklable artifact containing the fitted sklearn objects."""
        if self._vectorizer is None or self._classifier is None:
            raise RuntimeError("WordTfidfStudent.fit() must be called first")
        return StudentArtifact(
            name=self.name,
            classes=self.classes,
            payload={"vectorizer": self._vectorizer, "classifier": self._classifier},
        )

    @classmethod
    def from_artifact(cls, artifact: StudentArtifact) -> WordTfidfStudent:
        """Rebuild a trained student from a saved artifact."""
        if artifact.name != cls.name:
            raise ValueError(f"Artifact {artifact.name!r} is not a {cls.name} artifact")
        student = cls()
        student.classes = artifact.classes
        vectorizer = artifact.payload.get("vectorizer")
        classifier = artifact.payload.get("classifier")
        if vectorizer is None or classifier is None:
            raise ValueError("Saved student is missing vectorizer or classifier")
        student._vectorizer = vectorizer
        student._classifier = classifier
        return student
