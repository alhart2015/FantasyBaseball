"""Compute ordinal SGP rankings across the full player pool.

Rankings are keyed by ``name::player_type`` (e.g., ``"juan soto::hitter"``)
to disambiguate same-name players across hitter/pitcher pools.
Use ``rank_key()`` to build lookup keys.
"""

import pandas as pd
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.name_utils import normalize_name

PITCHER_POSITIONS = {"P", "SP", "RP"}


def rank_key(name: str, player_type: str) -> str:
    """Build a ranking lookup key from player name and type."""
    return f"{normalize_name(name)}::{player_type}"


def rank_key_from_positions(name: str, positions: list[str]) -> str:
    """Build a ranking lookup key, inferring player_type from positions."""
    ptype = "pitcher" if set(positions) & PITCHER_POSITIONS else "hitter"
    return f"{normalize_name(name)}::{ptype}"


def compute_sgp_rankings(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
) -> dict[str, int]:
    """Rank all players by unweighted SGP within hitter/pitcher pools.

    Returns {name::player_type: rank} where rank is 1-based ordinal
    (1 = highest SGP in that pool).
    """
    rankings = {}

    for df, ptype in [(hitters, "hitter"), (pitchers, "pitcher")]:
        if df.empty:
            continue

        sgp_list = []
        for _, row in df.iterrows():
            sgp = calculate_player_sgp(row)
            key = rank_key(row["name"], ptype)
            sgp_list.append((key, sgp))

        sgp_list.sort(key=lambda x: x[1], reverse=True)

        for rank_num, (key, _sgp) in enumerate(sgp_list, start=1):
            rankings[key] = rank_num

    return rankings


def compute_rankings_from_game_logs(
    hitter_logs: dict[str, dict],
    pitcher_logs: dict[str, dict],
) -> dict[str, int]:
    """Rank players by SGP of actual accumulated stats from game logs.

    Args:
        hitter_logs: {normalized_name: {pa, ab, h, r, hr, rbi, sb}}
        pitcher_logs: {normalized_name: {ip, k, w, sv, er, bb, h_allowed}}

    Returns {name::player_type: rank} where rank is 1-based ordinal.
    """
    rankings = {}

    for logs, player_type in [(hitter_logs, "hitter"), (pitcher_logs, "pitcher")]:
        if not logs:
            continue

        sgp_list = []
        for norm_name, stats in logs.items():
            player_dict = dict(stats)
            player_dict["player_type"] = player_type
            if player_type == "hitter":
                ab = player_dict.get("ab", 0) or 0
                h = player_dict.get("h", 0) or 0
                player_dict["avg"] = h / ab if ab > 0 else 0.0
            else:
                ip = player_dict.get("ip", 0) or 0
                if ip > 0:
                    er = player_dict.get("er", 0) or 0
                    bb = player_dict.get("bb", 0) or 0
                    ha = player_dict.get("h_allowed", 0) or 0
                    player_dict["era"] = er * 9.0 / ip
                    player_dict["whip"] = (bb + ha) / ip
                else:
                    player_dict["era"] = 0.0
                    player_dict["whip"] = 0.0

            sgp = calculate_player_sgp(pd.Series(player_dict))
            key = f"{norm_name}::{player_type}"
            sgp_list.append((key, sgp))

        sgp_list.sort(key=lambda x: x[1], reverse=True)

        for rank_num, (key, _sgp) in enumerate(sgp_list, start=1):
            rankings[key] = rank_num

    return rankings
