"""Gold-labeled dataset loading for intent classification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from datasets import load_dataset

from bantam_cascade.config import Config


class DataError(ValueError):
    """Raised for unsupported datasets or unexpected dataset schemas."""


@dataclass(frozen=True)
class Example:
    """A single text with its gold human label."""

    text: str
    label: str


@dataclass(frozen=True)
class Dataset:
    """Gold-labeled examples split into a training pool and an official test set."""

    train: tuple[Example, ...]
    test: tuple[Example, ...]
    labels: tuple[str, ...]


def make_smoke_dataset(*, n_per_label: int = 40, seed: int = 42) -> Dataset:
    """Return a deterministic synthetic dataset for offline development.

    Args:
        n_per_label: Number of examples to generate for each synthetic label.
        seed: Value embedded in the generated texts for reproducibility.

    Returns:
        A :class:`Dataset` with three labels and an 80/20 train/test split.
    """
    if n_per_label < 1:
        raise DataError("n_per_label must be at least 1")
    labels = ("billing", "shipping", "support")
    examples = tuple(
        Example(f"{label} request number {index} (seed {seed})", label)
        for index in range(n_per_label)
        for label in labels
    )
    train = tuple(example for index, example in enumerate(examples) if index % 5 != 0)
    test = tuple(example for index, example in enumerate(examples) if index % 5 == 0)
    return Dataset(train=train, test=test, labels=labels)


def _rows_to_examples(
    rows: object,
    *,
    text_field: str,
    label_field: str,
    all_names: tuple[str, ...],
    excluded: frozenset[str],
) -> tuple[Example, ...]:
    """Convert Hugging Face rows into typed examples, dropping excluded labels."""
    result: list[Example] = []
    for row in cast(Iterable[Mapping[str, object]], rows):
        text_value = row[text_field]
        label_value = row[label_field]
        if not isinstance(text_value, str):
            raise DataError(f"Expected a string in field {text_field!r}")
        label_index = cast(int, label_value)
        if not isinstance(label_index, int) or not 0 <= label_index < len(all_names):
            raise DataError(f"Label index out of range for field {label_field!r}: {label_value!r}")
        label = all_names[label_index]
        if label in excluded:
            continue
        result.append(Example(text=text_value, label=label))
    return tuple(result)


def _load_hf_dataset(
    config: Config,
    *,
    path: str,
    name: str | None,
    text_field: str,
    label_field: str,
) -> Dataset:
    """Load a Hugging Face dataset and map ClassLabel ids to human-readable names."""
    raw = load_dataset(path, name) if name is not None else load_dataset(path)
    try:
        train_rows = raw["train"]
        test_rows = raw.get("test", [])
        names_value = train_rows.features[label_field].names
    except (KeyError, AttributeError, TypeError) as exc:
        raise DataError(f"Unexpected schema for dataset {path!r}") from exc

    all_names = tuple(str(item) for item in cast(Iterable[object], names_value))
    excluded = frozenset(config.exclude_labels)
    in_scope = tuple(label for label in all_names if label not in excluded)
    if not in_scope:
        raise DataError(f"No in-scope labels remain for dataset {path!r}")

    train = _rows_to_examples(
        train_rows,
        text_field=text_field,
        label_field=label_field,
        all_names=all_names,
        excluded=excluded,
    )
    test = _rows_to_examples(
        test_rows,
        text_field=text_field,
        label_field=label_field,
        all_names=all_names,
        excluded=excluded,
    )
    return Dataset(train=train, test=test, labels=in_scope)


def load_dataset_for_config(config: Config) -> Dataset:
    """Load the dataset named in ``config`` with its gold labels intact.

    Args:
        config: Validated configuration containing dataset id, config, and exclusions.

    Returns:
        A :class:`Dataset` with the official train and test splits.

    Raises:
        DataError: If the dataset id is unsupported or its schema is unexpected.
    """
    if config.dataset == "smoke-tiny":
        return make_smoke_dataset(seed=config.seed)
    if config.dataset == "PolyAI/banking77":
        return _load_hf_dataset(
            config,
            path="PolyAI/banking77",
            name=None,
            text_field="text",
            label_field="label",
        )
    if config.dataset == "clinc/clinc_oos":
        return _load_hf_dataset(
            config,
            path="clinc/clinc_oos",
            name=config.dataset_config or "imbalanced",
            text_field="text",
            label_field="intent",
        )
    if config.dataset == "ucirvine/sms_spam":
        return _load_hf_dataset(
            config,
            path="ucirvine/sms_spam",
            name=config.dataset_config or "plain_text",
            text_field="sms",
            label_field="label",
        )
    raise DataError(f"Unsupported dataset: {config.dataset!r}")
