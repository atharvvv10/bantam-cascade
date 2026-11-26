"""Pydantic validation for Bantam Cascade YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_SUPPORTED_STUDENTS = frozenset(
    {"static_embed", "word_tfidf", "minilm_onnx_int8", "minilm_onnx_fp32"}
)
GEMINI_DECISIONS_MODEL = "gemini-decision-1"


class BantamConfigError(ValueError):
    """Raised when a configuration file is missing, malformed, or incomplete."""


class Constraints(BaseModel):
    """Accuracy, confidence, and latency requirements."""

    model_config = ConfigDict(extra="forbid")

    minimum_accuracy: float = Field(gt=0.0, lt=1.0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    max_p95_latency_ms: float = Field(gt=0.0)


class RateLimits(BaseModel):
    """Proactive teacher rate limits supplied by the user."""

    model_config = ConfigDict(extra="forbid")

    requests_per_minute: int = Field(gt=0)
    requests_per_day: int = Field(gt=0)
    tokens_per_minute: int = Field(gt=0)


class TeacherConfig(BaseModel):
    """Pinned teacher settings. Prices are always user-supplied."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["groq", "gemini"] = "groq"
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    price_per_million_input_tokens: float = Field(ge=0.0)
    price_per_million_output_tokens: float = Field(ge=0.0)
    rate_limits: RateLimits

    @field_validator("model", "prompt_version")
    @classmethod
    def _reject_placeholders(cls, value: str) -> str:
        if "<" in value or ">" in value or value.strip().upper() == "TBD":
            raise ValueError("must be a concrete pinned value, not a placeholder or TBD")
        return value

    @model_validator(mode="after")
    def _check_provider_model(self) -> TeacherConfig:
        if self.provider == "gemini" and self.model != GEMINI_DECISIONS_MODEL:
            raise ValueError(
                "provider 'gemini' uses the Decisions API and supports only "
                f"model '{GEMINI_DECISIONS_MODEL}'"
            )
        return self


class SplitsConfig(BaseModel):
    """Sizes and seed for the three disjoint gold/training splits."""

    model_config = ConfigDict(extra="forbid")

    student_train: int = Field(gt=0)
    calibration: int = Field(gt=0)
    certification: int = Field(gt=0)
    seed: int


class Config(BaseModel):
    """Top-level validated configuration."""

    model_config = ConfigDict(extra="forbid")

    task: Literal["intent_classification"] = "intent_classification"
    dataset: str = Field(min_length=1)
    dataset_config: str | None = None
    exclude_labels: list[str] = Field(default_factory=list)
    constraints: Constraints
    objective: Literal["minimize_cost"] = "minimize_cost"
    teacher: TeacherConfig
    students: list[str]
    splits: SplitsConfig
    threshold_grid_size: int = Field(gt=1)
    seed: int

    @field_validator("dataset")
    @classmethod
    def _reject_placeholder_dataset(cls, value: str) -> str:
        if "<" in value or ">" in value or value.strip().upper() == "TBD":
            raise ValueError("must be a concrete dataset id, not a placeholder or TBD")
        return value

    @model_validator(mode="after")
    def _check_candidates(self) -> Config:
        if not self.students:
            raise ValueError("students must contain at least one candidate")
        unknown = sorted(set(self.students) - _SUPPORTED_STUDENTS)
        if unknown:
            raise ValueError(f"unsupported student candidates: {', '.join(unknown)}")
        return self


def _placeholder_paths(value: object, path: str = "") -> list[str]:
    """Return config paths whose scalar values are unresolved placeholders."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            found.extend(_placeholder_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_placeholder_paths(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.upper() == "TBD" or "<" in stripped or ">" in stripped:
            found.append(path)
    return found


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config, failing fast with a clear error.

    Args:
        path: Path to the YAML file.

    Returns:
        A validated :class:`Config`.

    Raises:
        BantamConfigError: If the file is missing, malformed, or incomplete.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise BantamConfigError(f"Config file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BantamConfigError(f"Could not read config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BantamConfigError(f"Config {config_path} must contain a YAML mapping")

    placeholders = _placeholder_paths(raw)
    if placeholders:
        raise BantamConfigError(
            "Config has unresolved placeholders: "
            + ", ".join(placeholders)
            + ". Replace them with concrete pinned values (user-supplied prices and model id)."
        )
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise BantamConfigError(f"Invalid config {config_path}: {exc}") from exc
