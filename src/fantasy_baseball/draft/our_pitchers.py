"""Our pitchers on the draft board: the board's FanGraphs blend mixed with ours.

The board's pitchers start as its FanGraphs blend (``blended_projections``), which is
never edited. This takes their rates (K, W, SV per batter faced; ERA, WHIP), blends
them with ours (``scripts/build_our_pitchers.py``: the pitcher net, starters
age-corrected) by ``pitcher_ros.final.board_rates``, and rebuilds the counting stats
on the FanGraphs blend's own playing time (its IP and batters faced): playing time
stays FanGraphs Depth-Charts-style, the rates are the blend.

Batters faced is ``3 x IP + H + BB`` (the FanGraphs blend has no BF), the same
definition the board blend's weights were fitted on (``scripts/fit_board_blend.py``).
A pitcher ours doesn't project keeps the FanGraphs blend's line as is.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fantasy_baseball.pitcher_ros import final

logger = logging.getLogger(__name__)

OURS_DIR = Path(__file__).resolve().parents[3] / "data" / "pitcher_ros"
RATES = ("k", "w", "sv", "era", "whip")


def ours_path(season: int) -> Path:
    """Where ``scripts/build_our_pitchers.py`` writes our pitchers for ``season``."""
    return OURS_DIR / f"ours-{season}.parquet"


def load_ours(season: int) -> pd.DataFrame | None:
    """Our pitcher rates for ``season`` indexed by MLBAM id, or None (with a warning)
    when they haven't been built."""
    path = ours_path(season)
    if not path.exists():
        logger.warning(
            "no %s: the board's pitchers are FanGraphs only. Build ours with "
            "python scripts/build_our_pitchers.py --season %d --run <run>",
            path,
            season,
        )
        return None
    return pd.read_parquet(path).set_index("player_id")


def pseudo_bf(fg: pd.DataFrame) -> pd.Series:
    """Batters faced off a FanGraphs blend line (it has none): ``3 x IP + H + BB``. The
    board blend's weights were fitted on this definition (``scripts/fit_board_blend.py``),
    so the fit and the board must share it."""
    return 3 * fg["ip"] + fg["h_allowed"] + fg["bb"]


def fg_to_mlbam(raw: pd.DataFrame) -> pd.Series:
    """FanGraphs id -> MLBAM id from raw projection rows (``fg_id``, ``mlbam_id``).
    An id that maps to more than one of the other is dropped, never guessed."""
    pairs = raw[["fg_id", "mlbam_id"]].dropna().drop_duplicates()
    pairs = pairs.assign(fg_id=pairs["fg_id"].astype(str), mlbam_id=pairs["mlbam_id"].astype(int))
    pairs = pairs[
        ~pairs["fg_id"].duplicated(keep=False) & ~pairs["mlbam_id"].duplicated(keep=False)
    ]
    return pairs.set_index("fg_id")["mlbam_id"]


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
    ids = out["fg_id"].astype(str).map(mlbam)
    bf = pseudo_bf(out)
    keep = ids.notna() & ids.isin(ours.index) & (bf > 0)
    if not keep.any():
        logger.warning("no board pitcher matched ours: the board's pitchers are FanGraphs only")
        return out
    rows = out.index[keep]
    key = ids[keep].astype(int)
    if key.duplicated().any():
        raise ValueError(f"board pitchers share an MLBAM id: {sorted(key[key.duplicated()])}")
    fg_rates = pd.DataFrame(
        {
            "k": out.loc[rows, "k"] / bf[rows],
            "w": out.loc[rows, "w"] / bf[rows],
            "sv": out.loc[rows, "sv"] / bf[rows],
            "era": out.loc[rows, "era"],
            "whip": out.loc[rows, "whip"],
        }
    ).set_index(pd.Index(key.to_numpy(), name="player_id"))
    mixed = final.board_rates(ours.loc[key.to_numpy(), list(RATES)], fg_rates)
    mixed = mixed.loc[key.to_numpy()].set_axis(rows)
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
