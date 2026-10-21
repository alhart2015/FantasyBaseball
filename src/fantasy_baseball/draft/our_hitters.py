"""Our hitters on the draft board: the board's FanGraphs blend mixed with ours (#459).

As for pitchers (``our_pitchers``): the board's hitters start as its FanGraphs blend
(``blended_projections``), which is never edited. This takes their rates (R, HR, RBI,
SB per PA; AVG), blends them with ours (``scripts/build_our_hitters.py``: the hitter
net's preseason rates, seeds averaged) by ``hitter_ros.final.board_rates``, and
rebuilds the counting stats on the FanGraphs blend's own playing time (its PA and AB).
Storage and matching are shared: ``our_players``. A hitter ours doesn't project keeps
the FanGraphs blend's line as is.
"""

from __future__ import annotations

import logging

import pandas as pd

from fantasy_baseball.draft.our_players import match
from fantasy_baseball.hitter_ros import final
from fantasy_baseball.hitter_ros.features import TARGETS, rates_from_counts

logger = logging.getLogger(__name__)

COUNTS = ("pa", "ab", "h", "r", "hr", "rbi", "sb")


def fg_rates(fg: pd.DataFrame) -> pd.DataFrame:
    """A FanGraphs blend's hitter lines as rates: R, HR, RBI, SB per PA and AVG = H / AB
    (``features.rates_from_counts``, the definition ours is trained and scored on). The
    fit (``scripts/fit_board_hitter_blend.py``) and the board share it."""
    return rates_from_counts(fg[list(COUNTS)].astype(float))


def blend_board_hitters(
    hitters: pd.DataFrame, ours: pd.DataFrame, mlbam: pd.Series
) -> pd.DataFrame:
    """``hitters`` (the board's FanGraphs blend, as ``get_blended_projections``) with
    each hitter ours projects re-lined at ``final.board_rates``: R, HR, RBI, SB = rate x
    FanGraphs PA; AVG blended; H = AVG x FanGraphs AB. PA, AB, names and the rest are
    FanGraphs'. Returns a new frame."""
    out = hitters.copy()
    lines = [*COUNTS, "avg"]
    out[lines] = out[lines].astype(float)
    rows, key = match(out, ours.index, mlbam, (out["pa"] > 0) & (out["ab"] > 0))
    if rows.empty:
        logger.warning("no board hitter matched ours: the board's hitters are FanGraphs only")
        return out
    fg = fg_rates(out.loc[rows]).set_axis(key)
    mixed = final.board_rates(ours.loc[key, list(TARGETS)], fg).loc[key].set_axis(rows)
    for stat in ("r", "hr", "rbi", "sb"):
        out.loc[rows, stat] = mixed[stat] * out.loc[rows, "pa"]
    out.loc[rows, "avg"] = mixed["avg"]
    out.loc[rows, "h"] = mixed["avg"] * out.loc[rows, "ab"]
    logger.info("board: %d of %d hitters blended with ours", len(rows), len(out))
    return out
