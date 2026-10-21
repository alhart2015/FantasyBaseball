"""Ours on the draft board, shared by hitters (``our_hitters``) and pitchers
(``our_pitchers``): where ours is stored, how a board row finds its ours row, the
FanGraphs blend the board ranks, and the step that mixes the two into the board.

The board's FanGraphs blend is never edited: each player type's module takes its
rates, blends them with ours at the weights fitted held out
(``scripts/fit_board_hitter_blend.py``, ``scripts/fit_board_blend.py``), and rebuilds
its counts on the FanGraphs blend's own playing time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fantasy_baseball.data.db import get_id_pairs
from fantasy_baseball.data.projections import blend_projections

logger = logging.getLogger(__name__)

# The repo's data folder (tests point it at a temp folder: tests/conftest.py).
DATA = Path(__file__).resolve().parents[3] / "data"
_FOLDER = {"hitter": "hitter_ros", "pitcher": "pitcher_ros"}
_BUILD = {"hitter": "build_our_hitters.py", "pitcher": "build_our_pitchers.py"}


def ours_path(player_type: str, season: int) -> Path:
    """Where ``scripts/build_our_<type>s.py`` writes ours for ``season``."""
    return DATA / _FOLDER[player_type] / f"ours-{season}.parquet"


def load_ours(player_type: str, season: int) -> pd.DataFrame | None:
    """Our rates for ``season`` indexed by MLBAM id, or None (with a warning) when they
    haven't been built."""
    path = ours_path(player_type, season)
    if not path.exists():
        logger.warning(
            "no %s: the board's %ss are FanGraphs only. Build ours with "
            "python scripts/%s --season %d --run <run>",
            path,
            player_type,
            _BUILD[player_type],
            season,
        )
        return None
    return pd.read_parquet(path).set_index("player_id")


def fg_to_mlbam(raw: pd.DataFrame) -> pd.Series:
    """FanGraphs id -> MLBAM id from raw projection rows (``fg_id``, ``mlbam_id``).
    An id that maps to more than one of the other is dropped, never guessed."""
    pairs = raw[["fg_id", "mlbam_id"]].dropna().drop_duplicates()
    pairs = pairs.assign(fg_id=pairs["fg_id"].astype(str), mlbam_id=pairs["mlbam_id"].astype(int))
    pairs = pairs[
        ~pairs["fg_id"].duplicated(keep=False) & ~pairs["mlbam_id"].duplicated(keep=False)
    ]
    return pairs.set_index("fg_id")["mlbam_id"]


def season_mlbam(conn, season: int, player_type: str) -> pd.Series:
    """``fg_to_mlbam`` of ``season``'s raw projections of ``player_type``: how the board
    (and the fits, on past seasons) find a FanGraphs blend row's MLBAM id."""
    return fg_to_mlbam(get_id_pairs(conn, season, player_type))


def match(
    board: pd.DataFrame, ours_index: pd.Index, mlbam: pd.Series, usable: pd.Series
) -> tuple[pd.Index, pd.Index]:
    """The ``board`` rows ours can re-line -- FanGraphs id mapped to an MLBAM id ours
    projects, and ``usable`` (playing time to rebuild on) -- and their MLBAM ids,
    aligned. Two board rows on one MLBAM id are refused, never guessed between."""
    ids = board["fg_id"].astype(str).map(mlbam)
    keep = ids.notna() & ids.isin(ours_index) & usable
    rows = board.index[keep]
    key = pd.Index(ids[keep].astype(int).to_numpy(), name="player_id")
    if key.duplicated().any():
        raise ValueError(f"board rows share an MLBAM id: {sorted(key[key.duplicated()])}")
    return rows, key


def board_blend(
    folder: Path, systems: list[str], weights: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The FanGraphs blend the board ranks, for the projections in ``folder``: the
    league's ``systems`` that have files there, at the league's ``weights`` as given (a
    system with no weight gets none: ``blend_projections``), exactly as
    ``data.db.load_blended_projections`` builds the board's. Returns (hitters,
    pitchers, the systems used)."""
    on_file = [s for s in systems if any(folder.glob(f"{s}-*.csv"))]
    if not on_file:
        raise FileNotFoundError(f"no league projection system on file in {folder}")
    hitters, pitchers, _ = blend_projections(folder, on_file, weights)
    return hitters, pitchers, on_file
