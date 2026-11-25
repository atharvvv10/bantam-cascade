"""MiniLM ONNX student with int8 or fp32 CPU inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from bantam_cascade.students.base import (
    StudentArtifact,
    StudentDependencyError,
    align_proba,
    validate_training_input,
)

_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_FP32_FILE = "onnx/model.onnx"
_INT8_FILE = "onnx/model_quint8_avx2.onnx"
_TOKENIZER_FILE = "tokenizer.json"
_MAX_LENGTH = 256


class MiniLMOnnxStudent:
    """MiniLM ONNX embeddings plus a logistic-regression classification head."""

    def __init__(self, *, quantized: bool = True) -> None:
        """Create an untrained student; ONNX assets load lazily on first use."""
        self.name = "minilm_onnx_int8" if quantized else "minilm_onnx_fp32"
        self.quantized = quantized
        self.classes: tuple[str, ...] = ()
        self._classifier: LogisticRegression | None = None
        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: tuple[str, ...] = ()

    def _load_embedder(self) -> None:
        """Download (once) and initialize the ONNX Runtime embedder."""
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise StudentDependencyError(
                "MiniLM ONNX students need the optional dependencies. Install them with "
                '`pip install -e ".[students]"`.'
            ) from exc

        model_file = _INT8_FILE if self.quantized else _FP32_FILE
        try:
            model_path = hf_hub_download(_MODEL_REPO, model_file)
        except Exception as exc:  # pragma: no cover - network/backend dependent
            if self.quantized:
                raise StudentDependencyError(
                    f"Could not download the int8 ONNX model {model_file!r}; "
                    "the CPU may not support AVX2. Use minilm_onnx_fp32 instead."
                ) from exc
            raise StudentDependencyError(
                f"Could not download the ONNX model {model_file!r}."
            ) from exc
        tokenizer_path = hf_hub_download(_MODEL_REPO, _TOKENIZER_FILE)
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=_MAX_LENGTH)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_names = tuple(item.name for item in self._session.get_inputs())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized mean-pooled MiniLM embeddings."""
        if not texts:
            raise ValueError("at least one text is required")
        self._load_embedder()
        encodings = self._tokenizer.encode_batch(list(texts))
        input_ids = np.asarray([item.ids for item in encodings], dtype=np.int64)
        attention_mask = np.asarray([item.attention_mask for item in encodings], dtype=np.int64)
        token_type_ids = np.asarray([item.type_ids for item in encodings], dtype=np.int64)
        available = set(self._input_names)
        feeds: dict[str, np.ndarray] = {}
        if "input_ids" in available:
            feeds["input_ids"] = input_ids
        if "attention_mask" in available:
            feeds["attention_mask"] = attention_mask
        if "token_type_ids" in available:
            feeds["token_type_ids"] = token_type_ids
        outputs = self._session.run(None, feeds)
        hidden = np.asarray(outputs[0], dtype=np.float64)
        mask = attention_mask[:, :, np.newaxis]
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / counts
        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        return np.asarray(pooled / norms, dtype=np.float64)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        """Embed gold-labeled texts and train the classification head."""
        self.classes = validate_training_input(texts, labels)
        features = self.embed(texts)
        self._classifier = LogisticRegression(
            max_iter=1000,
            C=4.0,
            random_state=42,
        )
        self._classifier.fit(features, list(labels))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return class probabilities for each text."""
        if self._classifier is None:
            raise RuntimeError("MiniLMOnnxStudent.fit() must be called first")
        features = self.embed(texts)
        proba = self._classifier.predict_proba(features)
        source = tuple(str(name) for name in self._classifier.classes_)
        return align_proba(proba, source, self.classes)

    def artifact(self) -> StudentArtifact:
        """Return a picklable artifact; ONNX assets are reloaded on demand."""
        if self._classifier is None:
            raise RuntimeError("MiniLMOnnxStudent.fit() must be called first")
        return StudentArtifact(
            name=self.name,
            classes=self.classes,
            payload={"classifier": self._classifier, "quantized": self.quantized},
        )

    @classmethod
    def from_artifact(cls, artifact: StudentArtifact) -> MiniLMOnnxStudent:
        """Rebuild a trained student from a saved artifact."""
        if artifact.name not in {"minilm_onnx_int8", "minilm_onnx_fp32"}:
            raise ValueError(f"Artifact {artifact.name!r} is not a MiniLM artifact")
        student = cls(quantized=bool(artifact.payload.get("quantized", True)))
        student.name = artifact.name
        student.classes = artifact.classes
        classifier = artifact.payload.get("classifier")
        if classifier is None:
            raise ValueError("Saved student is missing classifier")
        student._classifier = classifier
        return student
