from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.constants import ALL_CATEGORIES, DEFAULT_SGP_DENOMINATORS


def test_returns_defaults():
    denoms = get_sgp_denominators()
    assert denoms == DEFAULT_SGP_DENOMINATORS


def test_returns_a_fresh_copy():
    """Callers may mutate the result without corrupting the shared default."""
    get_sgp_denominators()["HR"] = 999
    assert get_sgp_denominators() == DEFAULT_SGP_DENOMINATORS


def test_all_categories_present():
    denoms = get_sgp_denominators()
    assert set(denoms.keys()) == set(ALL_CATEGORIES)


def test_all_denominators_positive():
    denoms = get_sgp_denominators()
    for cat, val in denoms.items():
        assert val > 0, f"SGP denominator for {cat} must be positive"
