from pathlib import Path
from types import SimpleNamespace

import pytest

import bantam_cascade.data as data
from bantam_cascade.config import load_config
from bantam_cascade.data import DataError, load_dataset_for_config, make_smoke_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_dataset_is_deterministic() -> None:
    first = make_smoke_dataset(n_per_label=5, seed=7)
    second = make_smoke_dataset(n_per_label=5, seed=7)
    assert first == second
    assert first.labels == ("billing", "shipping", "support")
    assert len(first.train) == 12
    assert len(first.test) == 3
    assert not set(first.train) & set(first.test)


def test_smoke_config_dataset_has_enough_training_texts() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    dataset = load_dataset_for_config(config)
    required = config.splits.student_train + config.splits.calibration + config.splits.certification
    assert len(dataset.train) >= required
    assert set(dataset.labels) == {"billing", "shipping", "support"}


def test_unsupported_dataset_raises() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml").model_copy(update={"dataset": "nope"})
    with pytest.raises(DataError, match="Unsupported dataset"):
        load_dataset_for_config(config)


class _FakeSplit:
    """Minimal stand-in for a Hugging Face dataset split."""

    def __init__(self, rows: list[dict[str, object]], names: list[str]) -> None:
        """Store rows and ClassLabel names."""
        self.rows = rows
        self.features = {"intent": SimpleNamespace(names=names)}

    def __iter__(self) -> object:
        """Iterate over fake rows."""
        return iter(self.rows)


def test_hf_label_mapping_and_exclusion(monkeypatch: pytest.MonkeyPatch) -> None:
    names = ["billing", "oos", "support", "shipping"]
    train = _FakeSplit(
        [
            {"text": "a", "intent": 0},
            {"text": "skip", "intent": 1},
            {"text": "c", "intent": 3},
        ],
        names,
    )
    test = _FakeSplit([{"text": "z", "intent": 2}], names)
    monkeypatch.setattr(data, "load_dataset", lambda *args: {"train": train, "test": test})

    config = load_config(ROOT / "configs" / "clinc150.yaml")
    dataset = load_dataset_for_config(config)

    assert dataset.labels == ("billing", "support", "shipping")
    assert [example.text for example in dataset.train] == ["a", "c"]
    assert [example.label for example in dataset.train] == ["billing", "shipping"]
    assert [example.label for example in dataset.test] == ["support"]
