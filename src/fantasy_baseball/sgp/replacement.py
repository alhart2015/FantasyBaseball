import pandas as pd
from fantasy_baseball.utils.constants import STARTERS_PER_POSITION
from fantasy_baseball.utils.positions import is_pitcher


def calculate_replacement_levels(
    player_pool: pd.DataFrame,
    starters_per_position: dict[str, int] | None = None,
) -> dict[str, float]:
    """Calculate replacement-level SGP for each position."""
    if starters_per_position is None:
        starters_per_position = dict(STARTERS_PER_POSITION)

    replacement_levels: dict[str, float] = {}

    for position, num_starters in starters_per_position.items():
        if position in ("IF", "UTIL"):
            continue

        eligible = _get_eligible_players(player_pool, position)
        eligible = eligible.sort_values("total_sgp", ascending=False).reset_index(drop=True)

        if len(eligible) > num_starters:
            replacement_levels[position] = eligible.iloc[num_starters]["total_sgp"]
        elif len(eligible) > 0:
            replacement_levels[position] = eligible.iloc[-1]["total_sgp"]
        else:
            replacement_levels[position] = 0.0

    return replacement_levels


def _get_eligible_players(pool: pd.DataFrame, position: str) -> pd.DataFrame:
    if position == "P":
        return pool[pool["positions"].apply(lambda pos: any(p in ("P", "SP", "RP") for p in pos))]
    if position == "OF":
        return pool[pool["positions"].apply(lambda pos: "OF" in pos)]
    return pool[pool["positions"].apply(lambda pos: position in pos)]
