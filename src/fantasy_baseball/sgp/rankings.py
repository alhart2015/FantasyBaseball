"""Compute ordinal SGP rankings across the full player pool."""

import pandas as pd
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.name_utils import normalize_name


def compute_sgp_rankings(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
) -> dict[str, int]:
    """Rank all players by unweighted SGP within hitter/pitcher pools.

    Returns {normalized_name: rank} where rank is 1-based ordinal
    (1 = highest SGP in that pool).
    """
    rankings = {}

    for df in [hitters, pitchers]:
        if df.empty:
            continue

        sgp_list = []
        for _, row in df.iterrows():
            sgp = calculate_player_sgp(row)
            sgp_list.append((normalize_name(row["name"]), sgp))

        sgp_list.sort(key=lambda x: x[1], reverse=True)

        for rank, (norm_name, _sgp) in enumerate(sgp_list, start=1):
            rankings[norm_name] = rank

    return rankings
