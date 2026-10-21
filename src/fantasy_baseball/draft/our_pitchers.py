"""Our pitchers on the draft board: the board's FanGraphs blend mixed with ours.

The board's pitchers start as its FanGraphs blend (``blended_projections``), which is
never edited. This takes their rates (K, W, SV per batter faced; ERA, WHIP), blends
them with ours (``scripts/build_our_pitchers.py``: the pitcher net, starters
age-corrected) by ``pitcher_ros.final.board_rates``, and rebuilds the counting stats
on the FanGraphs blend's own playing time (its IP and batters faced). Storage and
matching are shared with hitters: ``our_players``.

Batters faced is the FanGraphs blend's own (its systems' TBF, blended), what ours'
rates are per: :func:`batters_faced`, shared with the fit (``scripts/fit_board_blend.py``).
A pitcher ours doesn't project keeps the FanGraphs blend's line as is.
"""

from __future__ import annotations

import logging

import pandas as pd

from fantasy_baseball.draft.our_players import match
from fantasy_baseball.pitcher_ros import final

logger = logging.getLogger(__name__)

RATES = ("k", "w", "sv", "era", "whip")


def batters_faced(fg: pd.DataFrame) -> pd.Series:
    """A FanGraphs blend line's batters faced: its ``bf`` (the systems' TBF, blended).
    Ours' K, W and SV rates are per true batter faced, so this is what they multiply.
    The board blend's weights were fitted on it (``scripts/fit_board_blend.py``), so the
    fit and the board share it. A line without one (a blend built before ``bf`` was
    stored: rebuild with python scripts/build_db.py) falls back to ``3 x IP + H + BB``,
    which runs about 1% high, with a warning."""
    pseudo = 3 * fg["ip"].astype(float) + fg["h_allowed"].astype(float) + fg["bb"].astype(float)
    if "bf" not in fg.columns:
        logger.warning(
            "FanGraphs blend has no bf (TBF) column: batters faced = 3 x IP + H + BB for "
            "every pitcher. Rebuild it: python scripts/build_db.py"
        )
        return pseudo
    bf = pd.to_numeric(fg["bf"], errors="coerce")
    lacking = ~(bf > 0) & (pseudo > 0)
    if lacking.any():
        logger.warning(
            "%d FanGraphs blend lines have no bf (TBF): 3 x IP + H + BB for those",
            int(lacking.sum()),
        )
    return bf.where(bf > 0, pseudo)


def fg_rates(fg: pd.DataFrame) -> pd.DataFrame:
    """A FanGraphs blend's pitcher lines as rates: K, W, SV per :func:`batters_faced`;
    ERA, WHIP as published. The fit and the board share it."""
    bf = batters_faced(fg)
    return pd.DataFrame(
        {
            "k": fg["k"].astype(float) / bf,
            "w": fg["w"].astype(float) / bf,
            "sv": fg["sv"].astype(float) / bf,
            "era": fg["era"].astype(float),
            "whip": fg["whip"].astype(float),
        }
    )


def blend_board_pitchers(
    pitchers: pd.DataFrame, ours: pd.DataFrame, mlbam: pd.Series
) -> pd.DataFrame:
    """``pitchers`` (the board's FanGraphs blend, as ``get_blended_projections``) with
    each pitcher ours projects re-lined at ``final.board_rates``: K, W, SV = rate x
    FanGraphs BF; ER = ERA x IP / 9; hits and walks = WHIP x IP, split as FanGraphs
    splits them. IP, names and the rest are FanGraphs'. Returns a new frame."""
    out = pitchers.copy()
    lines = ["ip", "k", "w", "sv", "era", "whip", "er", "h_allowed", "bb"]
    out[lines] = out[lines].astype(float)
    bf = batters_faced(out)
    rows, key = match(out, ours.index, mlbam, bf > 0)
    if rows.empty:
        logger.warning("no board pitcher matched ours: the board's pitchers are FanGraphs only")
        return out
    fg = fg_rates(out.loc[rows]).set_axis(key)
    mixed = final.board_rates(ours.loc[key, list(RATES)], fg).loc[key].set_axis(rows)
    ip = out.loc[rows, "ip"]
    for stat in ("k", "w", "sv"):
        out.loc[rows, stat] = mixed[stat] * bf[rows]
    out.loc[rows, "era"] = mixed["era"]
    out.loc[rows, "er"] = mixed["era"] * ip / 9
    out.loc[rows, "whip"] = mixed["whip"]
    on_base = out.loc[rows, "h_allowed"] + out.loc[rows, "bb"]
    h_share = (out.loc[rows, "h_allowed"] / on_base).where(on_base > 0, 0.75)
    out.loc[rows, "h_allowed"] = mixed["whip"] * ip * h_share
    out.loc[rows, "bb"] = mixed["whip"] * ip * (1 - h_share)
    logger.info("board: %d of %d pitchers blended with ours", len(rows), len(out))
    return out
