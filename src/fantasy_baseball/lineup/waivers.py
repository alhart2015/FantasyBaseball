from typing import Callable

import pandas as pd

from fantasy_baseball.lineup.yahoo_roster import fetch_free_agents
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.utils.name_utils import normalize_name


def fetch_and_match_free_agents(
    league,
    hitters_proj: pd.DataFrame,
    pitchers_proj: pd.DataFrame,
    fa_per_position: int = 100,
    on_position_loaded: Callable[[str, int], None] | None = None,
) -> tuple[list[Player], int]:
    """Fetch available players from Yahoo, match to projections.

    Fetches FA + waiver players across 8 positions, deduplicates by
    normalized name, and matches each to projections using position-aware
    search order (pitcher positions check pitchers_proj first).

    Projection DataFrames must have a ``_name_norm`` column precomputed
    via ``df["_name_norm"] = df["name"].apply(normalize_name)``.

    Args:
        league: Yahoo league object.
        hitters_proj: Blended hitter projections with _name_norm column.
        pitchers_proj: Blended pitcher projections with _name_norm column.
        fa_per_position: Number of players to fetch per position.
        on_position_loaded: Optional callback(position, count) for progress.

    Returns:
        Tuple of (matched_fa_players as list[Player], total_fetched_count).
    """
    fa_players: list[Player] = []
    fa_fetched = 0
    seen_names: set[str] = set()

    for pos in ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]:
        fas = fetch_free_agents(league, pos, count=fa_per_position)
        fa_fetched += len(fas)
        if on_position_loaded:
            on_position_loaded(pos, len(fas))

        if pos in ("SP", "RP"):
            search_order = [pitchers_proj, hitters_proj]
        else:
            search_order = [hitters_proj, pitchers_proj]

        for fa in fas:
            fa_name_norm = normalize_name(fa["name"])
            if fa_name_norm in seen_names:
                continue
            seen_names.add(fa_name_norm)

            proj_row = None
            ptype = None
            for df in search_order:
                if df.empty:
                    continue
                matches = df[df["_name_norm"] == fa_name_norm]
                if not matches.empty:
                    proj_row = matches.iloc[0]
                    ptype = PlayerType.PITCHER if df is pitchers_proj else PlayerType.HITTER
                    break

            if proj_row is not None:
                if ptype == PlayerType.HITTER:
                    ros = HitterStats.from_dict(proj_row.to_dict())
                else:
                    ros = PitcherStats.from_dict(proj_row.to_dict())
                p = Player(
                    name=fa["name"],
                    player_type=ptype,
                    positions=fa["positions"],
                    rest_of_season=ros,
                    status=fa.get("status", ""),
                )
                fa_players.append(p)

    return fa_players, fa_fetched
