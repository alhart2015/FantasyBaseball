from collections.abc import Callable
from typing import Any

import pandas as pd

from fantasy_baseball.lineup.yahoo_roster import fetch_free_agents
from fantasy_baseball.models.player import Player
from fantasy_baseball.utils.name_utils import normalize_name


def fetch_and_match_free_agents(
    league,
    hitters_proj: pd.DataFrame,
    pitchers_proj: pd.DataFrame,
    fa_per_position: int = 100,
    on_position_loaded: Callable[[str, int], None] | None = None,
    *,
    preseason_hitters_proj: pd.DataFrame | None = None,
    preseason_pitchers_proj: pd.DataFrame | None = None,
) -> tuple[list[Player], int]:
    """Fetch available players from Yahoo, match to projections.

    Fetches FA + waiver players across 8 positions, deduplicates by normalized
    name, and runs them through :func:`data.projections.match_roster_to_projections`
    -- the same matcher the roster path uses -- so free agents get identical
    name-normalization, suffix stripping, same-name-collision tie-break
    (``_pick_best_match``), and identity-keyed full-season / preseason attach.
    Attaching ``.preseason`` lets the stash board score FA candidates with the
    same remaining-season slot-share displacement model as owned IL arms.

    ``warn_unmatched=False``: most fetched FAs are unprojectable waiver fodder,
    so per-player no-match warnings would flood the refresh log (the roster path
    keeps them on, since a rostered player missing a projection is notable).

    Projection DataFrames must have a ``_name_norm`` column precomputed
    via ``df["_name_norm"] = df["name"].apply(normalize_name)``.

    Args:
        league: Yahoo league object.
        hitters_proj: Blended hitter projections with _name_norm column.
        pitchers_proj: Blended pitcher projections with _name_norm column.
        fa_per_position: Number of players to fetch per position.
        on_position_loaded: Optional callback(position, count) for progress.
        preseason_hitters_proj: Optional blended preseason hitter frame.
        preseason_pitchers_proj: Optional blended preseason pitcher frame.

    Returns:
        Tuple of (matched_fa_players as list[Player], total_fetched_count).
    """
    from fantasy_baseball.data.projections import match_roster_to_projections

    fa_dicts: list[dict[str, Any]] = []
    fa_fetched = 0
    seen_names: set[str] = set()

    for pos in ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]:
        fas = fetch_free_agents(league, pos, count=fa_per_position)
        fa_fetched += len(fas)
        if on_position_loaded:
            on_position_loaded(pos, len(fas))

        for fa in fas:
            name_norm = normalize_name(fa["name"])
            if name_norm in seen_names:
                continue
            seen_names.add(name_norm)
            fa_dicts.append(
                {
                    "name": fa["name"],
                    "positions": fa["positions"],
                    "status": fa.get("status", ""),
                }
            )

    fa_players = match_roster_to_projections(
        fa_dicts,
        hitters_proj,
        pitchers_proj,
        preseason_hitters_proj=preseason_hitters_proj,
        preseason_pitchers_proj=preseason_pitchers_proj,
        warn_unmatched=False,
        context="fa",
    )
    return fa_players, fa_fetched
