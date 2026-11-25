from pathlib import Path

import pytest
import yaml

from bantam_cascade.config import BantamConfigError, SplitsConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def _smoke_raw() -> dict:
    return yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_smoke_config_loads() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    assert config.dataset == "smoke-tiny"
    assert config.teacher.model == "smoke-fake-model"
    assert config.splits == SplitsConfig(
        student_train=40, calibration=20, certification=30, seed=42
    )


def test_placeholder_config_rejected(tmp_path: Path) -> None:
    raw = _smoke_raw()
    raw["teacher"]["model"] = "<exact model id>"
    with pytest.raises(BantamConfigError, match="unresolved placeholders"):
        load_config(_write(tmp_path, raw))


def test_real_dataset_configs_load() -> None:
    banking = load_config(ROOT / "configs" / "banking77.yaml")
    clinc = load_config(ROOT / "configs" / "clinc150.yaml")
    assert banking.teacher.model == "openai/gpt-oss-20b"
    assert banking.teacher.price_per_million_input_tokens == 0.0
    assert clinc.dataset == "clinc/clinc_oos"
    assert clinc.exclude_labels == ["oos"]


def test_clinc_config_declares_dataset_options(tmp_path: Path) -> None:
    raw = _smoke_raw()
    raw["dataset"] = "clinc/clinc_oos"
    raw["dataset_config"] = "imbalanced"
    raw["exclude_labels"] = ["oos"]
    config = load_config(_write(tmp_path, raw))
    assert config.dataset_config == "imbalanced"
    assert config.exclude_labels == ["oos"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_accuracy", 1.0),
        ("confidence_level", 0.0),
        ("max_p95_latency_ms", -1),
    ],
)
def test_invalid_constraints_rejected(tmp_path: Path, field: str, value: object) -> None:
    raw = _smoke_raw()
    raw["constraints"][field] = value
    with pytest.raises(BantamConfigError):
        load_config(_write(tmp_path, raw))


def test_unknown_student_rejected(tmp_path: Path) -> None:
    raw = _smoke_raw()
    raw["students"] = ["not_a_student"]
    with pytest.raises(BantamConfigError, match="unsupported student"):
        load_config(_write(tmp_path, raw))


def test_extra_field_rejected(tmp_path: Path) -> None:
    raw = _smoke_raw()
    raw["unexpected"] = 1
    with pytest.raises(BantamConfigError, match="Invalid config"):
        load_config(_write(tmp_path, raw))


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(BantamConfigError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")
