from fantasy_baseball.utils.constants import DEFAULT_SGP_DENOMINATORS, Category


def get_sgp_denominators() -> dict[Category, float]:
    """Return a fresh copy of the league SGP denominators (the code defaults)."""
    return dict(DEFAULT_SGP_DENOMINATORS)
