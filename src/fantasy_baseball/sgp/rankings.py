"""Compute ordinal SGP rankings across the full player pool.

Rankings are keyed by ``fg_id`` (primary, unique) with a secondary
``name::player_type`` index for lookups that lack fg_id.
Use ``rank_key()`` to build name-based lookup keys.
"""

from collections.abc import Sequence

import pandas as pd
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

from fantasy_baseball.utils.positions import PITCHER_POSITIONS


def rank_key(name: str, player_type: str) -> str:
    """Build a name-based ranking lookup key."""
    return f"{normalize_name(name)}::{player_type}"


def rank_key_from_positions(name: str, positions: Sequence[Position | str]) -> str:
    """Build a name-based ranking lookup key, inferring player_type from positions."""
    ptype = PlayerType.PITCHER if set(positions) & PITCHER_POSITIONS else PlayerType.HITTER
    return f"{normalize_name(name)}::{ptype}"


def lookup_rank(
    rankings: dict[str, int | dict],
    fg_id: str | None,
    name: str,
    player_type: str,
) -> dict:
    """Look up rank data, trying fg_id first then name::player_type fallback."""
    if fg_id:
        result = rankings.get(str(fg_id))
        if result is not None:
            return result if isinstance(result, dict) else {}
    fallback = rankings.get(rank_key(name, player_type), {})
    return fallback if isinstance(fallback, dict) else {}


def compute_sgp_rankings(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
) -> dict[str, int]:
    """Rank all players by unweighted SGP within hitter/pitcher pools.

    Returns dict with two types of keys pointing to the same ranks:
    - fg_id (e.g., "31757") — primary, unique per player
    - name::player_type (e.g., "mason miller::pitcher") — fallback

    When two players share a name and type (e.g., two Mason Miller pitchers),
    the fg_id keys are distinct but the name key gets the better rank.
    """
    rankings = {}

    for df, ptype in [(hitters, PlayerType.HITTER), (pitchers, PlayerType.PITCHER)]:
        if df.empty:
            continue

        sgp_list = []
        for _, row in df.iterrows():
            sgp = calculate_player_sgp(row)
            fg_id = str(row.get("fg_id", "")) if pd.notna(row.get("fg_id")) else None
            name_key = rank_key(row["name"], ptype)
            sgp_list.append((fg_id, name_key, sgp))

        sgp_list.sort(key=lambda x: x[2], reverse=True)

        for rank_num, (fg_id, name_key, _sgp) in enumerate(sgp_list, start=1):
            # fg_id key — always unique
            if fg_id:
                rankings[fg_id] = rank_num
            # name key — keep the better (lower) rank on collision
            if name_key not in rankings or rank_num < rankings[name_key]:
                rankings[name_key] = rank_num

    return rankings


def compute_combined_sgp_rankings(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
) -> dict[str, int]:
    """Rank all players in a single combined pool by unweighted SGP.

    Unlike ``compute_sgp_rankings`` which ranks hitters and pitchers
    separately, this produces a single ranking across both types.
    Useful for trade filtering where cross-type comparisons must be
    meaningful (hitter #5 and pitcher #5 may have very different SGP).
    """
    rankings = {}
    sgp_list = []

    for df, ptype in [(hitters, PlayerType.HITTER), (pitchers, PlayerType.PITCHER)]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            sgp = calculate_player_sgp(row)
            fg_id = str(row.get("fg_id", "")) if pd.notna(row.get("fg_id")) else None
            name_key = rank_key(row["name"], ptype)
            sgp_list.append((fg_id, name_key, sgp))

    sgp_list.sort(key=lambda x: x[2], reverse=True)

    for rank_num, (fg_id, name_key, _sgp) in enumerate(sgp_list, start=1):
        if fg_id:
            rankings[fg_id] = rank_num
        if name_key not in rankings or rank_num < rankings[name_key]:
            rankings[name_key] = rank_num

    return rankings


def compute_rankings_from_game_logs(
    hitter_logs: dict[str, dict],
    pitcher_logs: dict[str, dict],
) -> dict[str, int]:
    """Rank players by SGP of actual accumulated stats from game logs.

    Game logs are keyed by normalized name (no fg_id available), so these
    rankings use name::player_type keys only.

    Args:
        hitter_logs: {normalized_name: {pa, ab, h, r, hr, rbi, sb}}
        pitcher_logs: {normalized_name: {ip, k, w, sv, er, bb, h_allowed}}

    Returns {name::player_type: rank} where rank is 1-based ordinal.
    """
    rankings = {}

    for logs, player_type in [(hitter_logs, PlayerType.HITTER), (pitcher_logs, PlayerType.PITCHER)]:
        if not logs:
            continue

        sgp_list = []
        for norm_name, stats in logs.items():
            player_dict = dict(stats)
            player_dict["player_type"] = player_type
            if player_type == PlayerType.HITTER:
                ab = player_dict.get("ab", 0) or 0
                h = player_dict.get("h", 0) or 0
                player_dict["avg"] = calculate_avg(h, ab, default=0.0)
            else:
                ip = player_dict.get("ip", 0) or 0
                er = player_dict.get("er", 0) or 0
                bb = player_dict.get("bb", 0) or 0
                ha = player_dict.get("h_allowed", 0) or 0
                player_dict["era"] = calculate_era(er, ip, default=0.0)
                player_dict["whip"] = calculate_whip(bb, ha, ip, default=0.0)

            sgp = calculate_player_sgp(pd.Series(player_dict))
            key = f"{norm_name}::{player_type}"
            sgp_list.append((key, sgp))

        sgp_list.sort(key=lambda x: x[1], reverse=True)

        for rank_num, (key, _sgp) in enumerate(sgp_list, start=1):
            rankings[key] = rank_num

    return rankings
