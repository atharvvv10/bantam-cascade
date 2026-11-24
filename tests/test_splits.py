import pytest

from bantam_cascade.config import SplitsConfig
from bantam_cascade.splits import SplitError, split_by_hash, splits_fingerprint, text_hash


def _config() -> SplitsConfig:
    return SplitsConfig(student_train=3, calibration=2, certification=2, seed=42)


def _texts() -> tuple[str, ...]:
    return tuple(f"text number {i}" for i in range(10))


def test_split_is_deterministic_and_disjoint() -> None:
    texts = _texts()
    first = split_by_hash(texts, config=_config())
    second = split_by_hash(texts, config=_config())
    assert first == second
    groups = [first.student_train, first.calibration, first.certification, first.official_test]
    assert sum(len(group) for group in groups[:3]) == 7
    assert all(not set(a) & set(b) for i, a in enumerate(groups) for b in groups[i + 1 :])


def test_official_test_included_in_overlap_check() -> None:
    texts = _texts()
    result = split_by_hash(texts, config=_config(), official_test=("leftover",))
    assert result.official_test == ("leftover",)
    overlap = texts[0]
    with pytest.raises(SplitError, match="both training pool and official test"):
        split_by_hash(texts, config=_config(), official_test=(overlap,))


def test_duplicate_texts_are_an_error() -> None:
    with pytest.raises(SplitError, match="Duplicate text"):
        split_by_hash(["a", "a"] + [f"x{i}" for i in range(8)], config=_config())


def test_insufficient_data_is_an_error() -> None:
    with pytest.raises(SplitError, match="Need at least"):
        split_by_hash(
            ["a", "b"], config=SplitsConfig(student_train=2, calibration=2, certification=2, seed=1)
        )


def test_fingerprint_stable_and_seed_sensitive() -> None:
    texts = _texts()
    result = split_by_hash(texts, config=_config())
    assert splits_fingerprint(result, seed=42) == splits_fingerprint(result, seed=42)
    assert splits_fingerprint(result, seed=42) != splits_fingerprint(result, seed=43)
    assert text_hash("a", 1) != text_hash("a", 2)
