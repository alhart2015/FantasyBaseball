from collections.abc import Mapping

from fantasy_baseball.utils.constants import DEFAULT_SGP_DENOMINATORS, Category

# Override mapping accepted everywhere an ``sgp_overrides`` param exists:
# keys may be Category members or their string values (mixed is fine);
# values are absolute stat-per-standings-place denominators. Mapping (not
# dict) so a mixed-key dict typechecks and callees cannot mutate config.
SgpOverrides = Mapping[str, float] | Mapping[Category, float] | Mapping[Category | str, float]


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
