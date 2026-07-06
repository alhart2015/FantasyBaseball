"""Compute ordinal SGP rankings across the full player pool.

Rankings are keyed by ``fg_id`` (primary, unique) with a secondary
``name::player_type`` index for lookups that lack fg_id.
Use ``rank_key()`` to build name-based lookup keys.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.constants import Category
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import PITCHER_POSITIONS
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip


def rank_key(name: str, player_type: str) -> str:
    """Build a name-based ranking lookup key."""
    return f"{normalize_name(name)}::{player_type}"


def fg_key(fg_id: str, player_type: str) -> str:
    """Build an fg_id-based ranking lookup key, namespaced by pool.

    Keeps the writer (:func:`compute_sgp_rankings`) and reader
    (:func:`lookup_rank`) on one format so they can't silently drift.
    """
    return f"{fg_id}::{player_type}"


def rank_key_from_positions(name: str, positions: Sequence[Position | str]) -> str:
    """Build a name-based ranking lookup key, inferring player_type from positions."""
    ptype = PlayerType.PITCHER if set(positions) & PITCHER_POSITIONS else PlayerType.HITTER
    return f"{normalize_name(name)}::{ptype}"


def lookup_rank(
    rankings: Mapping[str, int | dict[str, Any]],
    fg_id: str | None,
    name: str,
    player_type: str,
) -> dict[str, Any]:
    """Look up rank data, trying fg_id first then name::player_type fallback."""
    if fg_id:
        result = rankings.get(fg_key(fg_id, player_type))
        if result is not None:
            return result if isinstance(result, dict) else {}
    fallback = rankings.get(rank_key(name, player_type), {})
    return fallback if isinstance(fallback, dict) else {}


def compute_sgp_rankings(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
    denoms: dict[Category, float] | None = None,
) -> dict[str, int]:
    """Rank all players by unweighted SGP within hitter/pitcher pools.

    Returns dict with two types of keys pointing to the ranks:
    - fg_id::player_type (e.g., "31757::pitcher") — primary
    - name::player_type (e.g., "mason miller::pitcher") — fallback

    Both keys are namespaced by pool. A single fg_id can appear in BOTH
    pools — a two-way player, or a position player charged with mop-up
    innings — and each pool's rank is a pool-relative ordinal (not
    comparable across the differently-sized pools), so they are stored
    separately and ``lookup_rank`` selects by the caller's player_type.
    An un-namespaced fg_id key would let the pitcher pass overwrite a real
    hitter rank with a junk 1-IP line (a catcher's fg_id resolving to rank
    ~6000 instead of his real rank).

    When two players share a name and type (e.g., two Mason Miller
    pitchers), the fg_id keys are distinct but the name key gets the
    better rank.

    ``denoms``: league-specific SGP denominators (from
    ``get_sgp_denominators(config.sgp_overrides)``). ``None`` keeps the
    code defaults.
    """
    rankings: dict[str, int] = {}

    for df, ptype in [(hitters, PlayerType.HITTER), (pitchers, PlayerType.PITCHER)]:
        if df.empty:
            continue

        sgp_list = []
        for _, row in df.iterrows():
            sgp = calculate_player_sgp(row, denoms=denoms)
            fg_id = str(row.get("fg_id", "")) if pd.notna(row.get("fg_id")) else None
            name_key = rank_key(row["name"], ptype)
            sgp_list.append((fg_id, name_key, sgp))

        sgp_list.sort(key=lambda x: x[2], reverse=True)

        for rank_num, (fg_id, name_key, _sgp) in enumerate(sgp_list, start=1):
            # fg_id key namespaced by pool: a player in both pools keeps a
            # separate rank per pool (lookup_rank picks by player_type).
            if fg_id:
                rankings[fg_key(fg_id, ptype)] = rank_num
            # name key — keep the better (lower) rank on same-name collision
            if name_key not in rankings or rank_num < rankings[name_key]:
                rankings[name_key] = rank_num

    return rankings


def build_rankings_lookup(
    ros: dict[str, Any],
    preseason: dict[str, Any],
    current: dict[str, Any],
    total: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Four-way merge of player ranking dicts keyed by ``name::player_type``.

    The output maps each player key to a dict with four keys
    (``rest_of_season``, ``preseason``, ``current``, ``total``); missing
    entries are ``None``. ``total`` is optional so legacy 3-arg callers keep
    working (they get ``total=None`` for every player).
    """
    total = total if total is not None else {}
    all_keys = set(ros) | set(preseason) | set(current) | set(total)
    return {
        key: {
            "rest_of_season": ros.get(key),
            "preseason": preseason.get(key),
            "current": current.get(key),
            "total": total.get(key),
        }
        for key in all_keys
    }


def compute_rankings_from_game_logs(
    hitter_logs: dict[str, dict[str, Any]],
    pitcher_logs: dict[str, dict[str, Any]],
    denoms: dict[Category, float] | None = None,
) -> dict[str, int]:
    """Rank players by SGP of actual accumulated stats from game logs.

    Game logs are keyed by normalized name (no fg_id available), so these
    rankings use name::player_type keys only.

    Args:
        hitter_logs: {normalized_name: {pa, ab, h, r, hr, rbi, sb}}
        pitcher_logs: {normalized_name: {ip, k, w, sv, er, bb, h_allowed}}
        denoms: league-specific SGP denominators; None keeps defaults.

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

            sgp = calculate_player_sgp(pd.Series(player_dict), denoms=denoms)
            key = f"{norm_name}::{player_type}"
            sgp_list.append((key, sgp))

        sgp_list.sort(key=lambda x: x[1], reverse=True)

        for rank_num, (key, _sgp) in enumerate(sgp_list, start=1):
            rankings[key] = rank_num

    return rankings
