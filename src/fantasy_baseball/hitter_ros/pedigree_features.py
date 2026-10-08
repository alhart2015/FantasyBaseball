"""Prospect-pedigree inputs for every hitter-table row (#433).

Our biggest AVG misses against FanGraphs are hitters with little MLB history. Their
minor-league lines are inputs already (``milb_features``); this adds what scouts
thought of them, from ``pitch_data.pedigree``: MLB Pipeline rankings and the June draft.

For a row in season S, only lists from seasons before S count, the same for every
season: the current season's lists are edited as players graduate, so for 2026 only
the 2025 and earlier lists are a fair input, and using last season's everywhere keeps
training and testing alike. A list's ranks become values from 1 (No. 1) down, 0 when
not on it: ``1 - (rank - 1) / 100`` on the Top 100 (a Top 50 in 2011), ``1 - (rank -
1) / 30`` on a club's Top 30 (Top 10 / 20 in the first years).

* ``ped_t100_p1`` / ``ped_org_p1``: value on last season's Top 100 / club list.
* ``ped_t100_best`` / ``ped_org_best``: best value on any earlier list.
* ``ped_seasons_listed``: seasons on any earlier list.
  All five are NaN when last season had no lists (before 2012).
* ``ped_draft_log_pick``: log of the overall pick in his last June draft before S;
  ``ped_draft_age``: his age on July 1 of that draft; ``ped_draft_years``: S minus its
  year. NaN when he wasn't drafted (an international signing, mostly).

Build option ``vets_blank_from`` (``PEDIGREE_PRESETS``): rows of players with at least
that many MLB PA when projected, or whose career count can't be trusted (as in
``milb_features``), get every input blank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.backtest import MIN_HISTORY_SEASONS
from fantasy_baseball.hitter_ros.features import ROW_KEYS, load_feature_file
from fantasy_baseball.pitch_data.pedigree import FIRST_RANKINGS_SEASON, TOP100

PEDIGREE_FEATURES = [
    "ped_t100_p1",
    "ped_t100_best",
    "ped_org_p1",
    "ped_org_best",
    "ped_seasons_listed",
    "ped_draft_log_pick",
    "ped_draft_age",
    "ped_draft_years",
]
_LIST_FEATURES = PEDIGREE_FEATURES[:5]
# Named build options (scripts/build_hitter_ros_pedigree.py --name), checked on load.
PEDIGREE_PRESETS: dict[str, dict[str, float | None]] = {
    "rookies": {"vets_blank_from": 300.0},
    "all": {"vets_blank_from": None},
}


def pedigree_path(root: Path, name: str) -> Path:
    """Where ``scripts/build_hitter_ros_pedigree.py --name <name>`` writes the features."""
    return root / f"pedigree_{name}.parquet"


def load_pedigree_inputs(
    table: pd.DataFrame, root: Path, name: str
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """``PEDIGREE_FEATURES`` from ``pedigree_<name>.parquet`` under ``root``, aligned to
    ``table``'s rows, and its build options; ValueError saying how to fix a bad file
    (see ``features.load_feature_file``)."""
    return load_feature_file(
        table,
        pedigree_path(root, name),
        PEDIGREE_FEATURES,
        f"run scripts/build_hitter_ros_pedigree.py --name {name}",
        expected_options=PEDIGREE_PRESETS.get(name),
        expected_label=f"the {name} preset",
    )


def _list_values(rankings: pd.DataFrame) -> pd.DataFrame:
    """Per (player, season): Top 100 value and best club-list value that season."""
    top = rankings["list"] == TOP100
    value = np.where(top, 1 - (rankings["rank"] - 1) / 100, 1 - (rankings["rank"] - 1) / 30)
    df = rankings.assign(value=value, kind=np.where(top, "t100", "org"))
    wide = df.pivot_table(
        index=["player_id", "season"], columns="kind", values="value", aggfunc="max"
    )
    return wide.reindex(columns=["t100", "org"]).fillna(0.0).reset_index()


def _list_inputs(seasons: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """The list inputs per (player_id, season) in ``seasons``."""
    vals = _list_values(rankings)
    # Every earlier list season per row, then summarize.
    m = seasons.merge(vals, on="player_id", how="left", suffixes=("", "_list"))
    m = m[m["season_list"] < m["season"]]
    last = m[m["season_list"] == m["season"] - 1]
    g = m.groupby(["player_id", "season"])
    summary = pd.DataFrame(
        {
            "ped_t100_best": g["t100"].max(),
            "ped_org_best": g["org"].max(),
            "ped_seasons_listed": g["season_list"].nunique().astype(float),
        }
    ).join(
        last.set_index(["player_id", "season"])[["t100", "org"]].rename(
            columns={"t100": "ped_t100_p1", "org": "ped_org_p1"}
        )
    )
    out = seasons.merge(summary.reset_index(), on=["player_id", "season"], how="left")
    out[_LIST_FEATURES] = out[_LIST_FEATURES].fillna(0.0)
    out.loc[out["season"] - 1 < FIRST_RANKINGS_SEASON, _LIST_FEATURES] = np.nan
    return out


def _draft_inputs(seasons: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    """The draft inputs per (player_id, season): his last June draft before the season."""
    d = draft[["player_id", "year", "pick_number", "birth_date"]].dropna(subset=["pick_number"])
    m = seasons.merge(d, on="player_id", how="inner")
    m = (
        m[m["year"] < m["season"]]
        .sort_values("year")
        .drop_duplicates(["player_id", "season"], keep="last")
    )
    born = pd.to_datetime(m["birth_date"], errors="coerce")
    july1 = pd.to_datetime(m["year"].astype(str) + "-07-01")
    m = m.assign(
        ped_draft_log_pick=np.log(m["pick_number"].astype(float)),
        ped_draft_age=(july1 - born).dt.days / 365.25,
        ped_draft_years=(m["season"] - m["year"]).astype(float),
    )
    cols = ["ped_draft_log_pick", "ped_draft_age", "ped_draft_years"]
    return seasons.merge(m[["player_id", "season", *cols]], on=["player_id", "season"], how="left")


def build_pedigree_features(
    table: pd.DataFrame,
    rankings: pd.DataFrame,
    draft: pd.DataFrame,
    vet_min_pa: float | None = None,
) -> pd.DataFrame:
    """``PEDIGREE_FEATURES`` for every ``table`` row, keyed by ``ROW_KEYS`` plus
    ``as_of``. ``rankings`` / ``draft``: the store's ``prospect_rankings`` / ``draft``
    tables. ``vet_min_pa`` needs the table's ``car_pa``, ``std_pa`` and
    ``car_seasons_in_store``."""
    rows = table[[*ROW_KEYS, "as_of"]].copy()
    seasons = rows[["player_id", "season"]].drop_duplicates()
    per_season = _list_inputs(seasons, rankings).merge(
        _draft_inputs(seasons, draft), on=["player_id", "season"], how="left"
    )
    out = rows.merge(per_season, on=["player_id", "season"], how="left", validate="many_to_one")
    if vet_min_pa is not None:
        mlb_pa = table["car_pa"].astype(float) + table["std_pa"].astype(float)
        short_history = table["car_seasons_in_store"] < MIN_HISTORY_SEASONS
        blank = ((mlb_pa >= vet_min_pa) | short_history).to_numpy()
        out.loc[blank, PEDIGREE_FEATURES] = np.nan
    return out[[*ROW_KEYS, "as_of", *PEDIGREE_FEATURES]]
