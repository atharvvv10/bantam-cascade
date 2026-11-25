from bantam_cascade.teacher.parse import INVALID_LABEL, parse_label

LABELS = ("billing", "shipping", "reverted_card_payment?")


def test_exact_and_wrapped_outputs() -> None:
    assert parse_label("billing", LABELS) == "billing"
    assert parse_label("  `shipping`  ", LABELS) == "shipping"
    assert parse_label("Label: billing", LABELS) == "billing"
    assert parse_label("```\nbilling\n```", LABELS) == "billing"


def test_case_insensitive_and_embedded_match() -> None:
    assert parse_label("BILLING", LABELS) == "billing"
    assert parse_label("The intent is shipping.", LABELS) == "shipping"
    assert parse_label("reverted_card_payment?", LABELS) == "reverted_card_payment?"


def test_ambiguous_and_out_of_list_outputs_are_invalid() -> None:
    assert parse_label("unknown", LABELS) == INVALID_LABEL
    assert parse_label("", LABELS) == INVALID_LABEL
    assert parse_label("billing shipping", LABELS) == INVALID_LABEL
