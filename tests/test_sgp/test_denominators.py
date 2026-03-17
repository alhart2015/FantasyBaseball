import pytest
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.constants import DEFAULT_SGP_DENOMINATORS, ALL_CATEGORIES


def test_returns_defaults_with_no_overrides():
    denoms = get_sgp_denominators()
    assert denoms == DEFAULT_SGP_DENOMINATORS


def test_overrides_specific_categories():
    overrides = {"HR": 10.0, "SV": 8.0}
    denoms = get_sgp_denominators(overrides)
    assert denoms["HR"] == 10.0
    assert denoms["SV"] == 8.0
    assert denoms["R"] == DEFAULT_SGP_DENOMINATORS["R"]


def test_all_categories_present():
    denoms = get_sgp_denominators()
    assert set(denoms.keys()) == set(ALL_CATEGORIES)


def test_all_denominators_positive():
    denoms = get_sgp_denominators()
    for cat, val in denoms.items():
        assert val > 0, f"SGP denominator for {cat} must be positive"
