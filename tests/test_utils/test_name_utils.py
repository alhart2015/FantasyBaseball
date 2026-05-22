"""Tests for player-name normalization."""

from fantasy_baseball.utils.name_utils import normalize_name


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
