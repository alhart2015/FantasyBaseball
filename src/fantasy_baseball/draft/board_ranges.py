"""Each board player's big-year line (#505): his P90 season, from
``scripts/build_board_ranges.py``'s ranges around the board's own lines.

"Big year" is the good tail of every stat: P90 of R, HR, RBI, SB, AVG, W, K and SV,
and P10 of ERA and WHIP (lower is better). Each stat's tail is its own: no single
season has all of them at once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fantasy_baseball.draft import our_players
from fantasy_baseball.sgp.player_value import (
    calculate_counting_sgp,
    calculate_hitting_rate_sgp,
    calculate_pitching_rate_sgp,
)
from fantasy_baseball.utils.constants import Category

if TYPE_CHECKING:
    from fantasy_baseball.draft.board import ScaleInputs

logger = logging.getLogger(__name__)

HITTING = ("r", "hr", "rbi", "sb", "avg")
PITCHING = ("w", "k", "sv", "era", "whip")
LOWER_IS_BETTER = ("era", "whip")
COUNTING_HIT = (("r", Category.R), ("hr", Category.HR), ("rbi", Category.RBI), ("sb", Category.SB))
COUNTING_PIT = (("w", Category.W), ("k", Category.K), ("sv", Category.SV))


def ranges_path(season: int) -> Path:
    """Where ``scripts/build_board_ranges.py`` writes ``season``'s ranges."""
    return our_players.DATA / "board_ranges" / f"{season}.parquet"


def big_year_column(stat: str) -> str:
    """The ranges column a stat's big year is: P10 for ERA/WHIP, else P90."""
    return f"{stat}_p10" if stat in LOWER_IS_BETTER else f"{stat}_p90"


def attach(
    conn, season: int, hitters: pd.DataFrame, pitchers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The board's hitters and pitchers with ``p90_<stat>`` columns (each stat's big
    year), matched by MLBAM id; NaN for a player without a range. Unchanged, with a
    warning, when ``season`` has no ranges built."""
    path = ranges_path(season)
    if not path.exists():
        logger.warning(
            "no %s: the board has no big-year lines. Build them with "
            "python scripts/build_board_ranges.py --season %d",
            path,
            season,
        )
        return hitters, pitchers
    for player_type in ("hitter", "pitcher"):
        ours = our_players.ours_path(player_type, season)
        if ours.exists() and ours.stat().st_mtime > path.stat().st_mtime:
            logger.warning(
                "%s is newer than %s: the big-year lines are around an older board. "
                "Rebuild them with python scripts/build_board_ranges.py --season %d",
                ours,
                path,
                season,
            )
    ranges = pd.read_parquet(path)
    out = []
    for frame, player_type, stats in (
        (hitters, "hitter", HITTING),
        (pitchers, "pitcher", PITCHING),
    ):
        mine = ranges[ranges["player_type"] == player_type].set_index("player_id")
        if mine.index.duplicated().any():
            raise ValueError(f"{path}: a {player_type} has two ranges")
        ids = frame["fg_id"].astype(str).map(our_players.season_mlbam(conn, season, player_type))
        frame = frame.copy()
        for stat in stats:
            frame[f"p90_{stat}"] = ids.map(mine[big_year_column(stat)]).to_numpy()
        # His P90 season's SGP (whole seasons: build_board_ranges.py); the board turns it
        # into P90 VAR with his own replacement level (board.with_p90_var).
        if "sgp_p90" in mine.columns:
            frame["p90_sgp"] = ids.map(mine["sgp_p90"]).to_numpy()
        else:
            logger.warning(
                "%s has no sgp_p90 (built before P90 VAR): rebuild it with "
                "python scripts/build_board_ranges.py --season %d",
                path,
                season,
            )
        out.append(frame)
    return out[0], out[1]


def season_sgp(draws: dict[str, np.ndarray], player_type: str, scale: ScaleInputs) -> np.ndarray:
    """Each simulated whole season's SGP (rows x draws, ``pt_model.ranges.draw``), scored
    exactly as the board scores a line (``sgp.player_value``, the board's ``scale``):
    counting stats over their denominators, AVG / ERA / WHIP as marginal value over
    the pool's replacement rates on the season's own AB / IP."""
    d = scale.denoms
    rep = scale.repl_rates
    if player_type == "hitter":
        sgp = sum(calculate_counting_sgp(draws[s], d[c]) for s, c in COUNTING_HIT)
        return np.asarray(
            sgp
            + calculate_hitting_rate_sgp(
                draws["avg"], draws["avg_trials"], rep["avg"], d[Category.AVG], scale.team_ab
            )
        )
    sgp = sum(calculate_counting_sgp(draws[s], d[c]) for s, c in COUNTING_PIT)
    for stat, cat, divisor in (("era", Category.ERA, 9), ("whip", Category.WHIP, 1)):
        sgp = sgp + calculate_pitching_rate_sgp(
            draws[stat], draws[f"{stat}_trials"], rep[stat], d[cat], scale.team_ip, divisor
        )
    return np.asarray(sgp)
