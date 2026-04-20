from fantasy_baseball.utils.constants import DEFAULT_SGP_DENOMINATORS, Category


def get_sgp_denominators(
    overrides: dict[Category, float] | dict[str, float] | None = None,
) -> dict[Category, float]:
    """Get SGP denominators, optionally overriding defaults.

    Overrides may key on either :class:`Category` members or bare strings
    (``"R"``, ``"HR"``, …). String keys are normalized to enum members so
    the returned dict is uniformly typed.
    """
    denoms = dict(DEFAULT_SGP_DENOMINATORS)
    if overrides:
        for k, v in overrides.items():
            denoms[Category(k)] = v
    return denoms
