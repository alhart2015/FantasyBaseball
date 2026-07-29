"""Tests for player-name normalization."""

from fantasy_baseball.utils.name_utils import normalize_name, repair_double_encoded


def test_strips_accents_and_lowercases():
    assert normalize_name("Jose Ramirez") == "jose ramirez"
    assert normalize_name("Julio Rodriguez") == "julio rodriguez"


def test_collapses_surrounding_whitespace():
    assert normalize_name("  Aaron Judge  ") == "aaron judge"


def test_nan_name_returns_empty_string():
    """pandas yields float('nan') for a blank name cell. A bad/blank row in a
    projection CSV must not crash name normalization (regression: the ROS blend
    quality check called normalize_name on a NaN name and raised
    'normalize() argument 2 must be str, not float')."""
    assert normalize_name(float("nan")) == ""


def test_none_name_returns_empty_string():
    assert normalize_name(None) == ""


def test_repair_double_encoded_restores_utf8_bytes():
    """Baseball Reference spells the bytes out literally: "Acu\\xc3\\xb1a" is 17
    characters, not "Acuna" with a tilde."""
    assert repair_double_encoded("Luisangel Acu\\xc3\\xb1a") == "Luisangel Acu\u00f1a"
    assert repair_double_encoded("Jos\\xc3\\xa9 Alvarado") == "Jos\u00e9 Alvarado"


def test_repair_leaves_clean_names_untouched():
    assert repair_double_encoded("Andrew Abbott") == "Andrew Abbott"
    assert repair_double_encoded("Jos\u00e9 Alvarado") == "Jos\u00e9 Alvarado"


def test_repair_returns_input_when_it_does_not_round_trip():
    assert repair_double_encoded("Weird\\xff") == "Weird\\xff"


def test_repair_tolerates_a_non_string():
    assert repair_double_encoded(None) is None


def test_repair_then_normalize_matches_the_board():
    """The whole point: a repaired name normalizes to the same key as the clean
    one, so a name join stops silently dropping ~7% of players."""
    assert normalize_name(repair_double_encoded("Luisangel Acu\\xc3\\xb1a")) == "luisangel acuna"
