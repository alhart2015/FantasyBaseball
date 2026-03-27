"""Buy-low candidate detection — players underperforming projections."""

import pandas as pd

from fantasy_baseball.analysis.pace import compute_player_pace
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.utils.constants import (
    HITTING_CATEGORIES, HITTER_PROJ_KEYS,
    PITCHING_CATEGORIES, PITCHER_PROJ_KEYS,
)
from fantasy_baseball.utils.name_utils import normalize_name


def find_buy_low_candidates(
    players: list[dict],
    game_log_lookup: dict,
    leverage: dict,
    owner: str = "Free Agent",
) -> list[dict]:
    """Find players underperforming projections by > 1 SD.

    Args:
        players: Roster entries with projection stats (dict with lowercase stat keys).
        game_log_lookup: {normalized_name: {stat: value}} from bulk game log query.
        leverage: Per-category leverage weights for wSGP computation.
        owner: Team name or "Free Agent" for display.

    Returns:
        List of candidate dicts sorted by avg_z ascending (most negative first).
    """
    candidates = []

    for player in players:
        name = player.get("name", "")
        ptype = player.get("player_type", "")
        if ptype not in ("hitter", "pitcher"):
            continue

        norm = normalize_name(name)
        actuals = game_log_lookup.get(norm, {})

        # Build projection dict from player entry
        if ptype == "hitter":
            proj_keys = HITTER_PROJ_KEYS
            categories = HITTING_CATEGORIES
        else:
            proj_keys = PITCHER_PROJ_KEYS
            categories = PITCHING_CATEGORIES

        projected = {k: player.get(k, 0) or 0 for k in proj_keys}
        pace = compute_player_pace(actuals, projected, ptype)

        # Average z-scores, excluding stats where z=0 and color=neutral
        # (below sample threshold or no projection — not informative)
        z_scores = []
        for cat in categories:
            st = pace.get(cat, {})
            z = st.get("z_score", 0.0)
            color = st.get("color_class", "stat-neutral")
            if z == 0.0 and color == "stat-neutral":
                continue  # skip non-informative stats
            z_scores.append(z)

        if not z_scores:
            continue  # no stats with enough sample

        avg_z = round(sum(z_scores) / len(z_scores), 2)

        if avg_z >= -1.0:
            continue  # not underperforming enough

        # Compute wSGP using projection stats and user's leverage
        try:
            wsgp = round(calculate_weighted_sgp(pd.Series(player), leverage), 2)
        except (KeyError, ZeroDivisionError, ValueError):
            wsgp = 0.0

        candidates.append({
            "name": name,
            "positions": player.get("positions", []),
            "owner": owner,
            "player_type": ptype,
            "avg_z": avg_z,
            "stats": pace,
            "wsgp": wsgp,
        })

    candidates.sort(key=lambda c: c["avg_z"])
    return candidates
