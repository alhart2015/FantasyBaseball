"""Player classification — categorize roster players by league-wide value vs team fit."""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball.models.player import Player

from fantasy_baseball.utils.constants import IL_STATUSES
from fantasy_baseball.utils.name_utils import normalize_name


def classify_roster(
    roster: list[Player],
    rankings: dict[str, int],
    rosterable_threshold: int = 50,
) -> dict[str, str]:
    """Classify each roster player into one of four categories.

    Uses two axes:
    - **SGP rank** (league-wide quality): ROS rank <= rosterable_threshold = high SGP
    - **wSGP** (team fit): above roster median wSGP = high wSGP

    Returns {player_name: classification} where classification is one of:
    "core", "trade_candidate", "role_player", "droppable".
    """
    if not roster:
        return {}

    # Compute median wSGP from active (non-IL) players only
    active_wsgps = [p.wsgp for p in roster if p.status not in IL_STATUSES]
    median_wsgp = statistics.median(active_wsgps) if active_wsgps else 0.0

    result: dict[str, str] = {}
    for player in roster:
        # Look up ROS rank using normalized name::player_type key
        rank_key = f"{normalize_name(player.name)}::{player.player_type.value}"
        rest_of_season_rank = rankings.get(rank_key)
        high_sgp = rest_of_season_rank is not None and rest_of_season_rank <= rosterable_threshold

        high_wsgp = player.wsgp > median_wsgp

        if high_sgp and high_wsgp:
            result[player.name] = "core"
        elif high_sgp and not high_wsgp:
            result[player.name] = "trade_candidate"
        elif not high_sgp and high_wsgp:
            result[player.name] = "role_player"
        else:
            result[player.name] = "droppable"

    return result
