from fantasy_baseball.utils.constants import DEFAULT_SGP_DENOMINATORS


def get_sgp_denominators(
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """Get SGP denominators, optionally overriding defaults."""
    denoms = dict(DEFAULT_SGP_DENOMINATORS)
    if overrides:
        denoms.update(overrides)
    return denoms
