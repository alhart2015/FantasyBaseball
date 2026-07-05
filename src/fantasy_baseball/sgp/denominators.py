from fantasy_baseball.utils.constants import DEFAULT_SGP_DENOMINATORS, Category

# Override mapping accepted everywhere an ``sgp_overrides`` param exists:
# keys may be Category members or their string values; values are absolute
# stat-per-standings-place denominators.
SgpOverrides = dict[Category, float] | dict[str, float]


def get_sgp_denominators(
    overrides: SgpOverrides | None = None,
) -> dict[Category, float]:
    """Get SGP denominators, optionally overriding defaults.

    Overrides may key on either :class:`Category` members or bare strings
    (``"R"``, ``"HR"``, ...). String keys are normalized to enum members so
    the returned dict is uniformly typed.
    """
    denoms = dict(DEFAULT_SGP_DENOMINATORS)
    if overrides:
        for k, v in overrides.items():
            denoms[Category(k)] = v
    return denoms
