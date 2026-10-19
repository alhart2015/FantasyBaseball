"""Rows for ``ranges`` from the files (#505): projected playing time, our projected
rates, each hitter's history and, for past seasons, what he did.

* Season lines (games and PA per scheduled game) come from ``data/stats/hitters-<y>.csv``
  when it exists, else from the playing-time panel (``scripts/build_pt_panel.py``),
  whose 2020 rows are per 60 games. The stats files win where both exist: the panel's
  last season can be a partial one (it was built mid-2026).
* Ages are on the June 30 convention (as the panel's), from the people file's birth
  dates; debut year and primary position from the panel.
* Our rates: a hitter run's preseason (week 0) rows, averaged over its seeds.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.evaluate import load_fangraphs_hitters
from fantasy_baseball.pt_model.ranges import HISTORY_SEASONS, STATS, history, top_ranked

# Columns read from a season stats file (data/stats/hitters-<y>.csv).
STAT_COLUMNS = {
    "G": "g",
    "PA": "pa",
    "AB": "ab",
    "H": "h",
    "R": "r",
    "HR": "hr",
    "RBI": "rbi",
    "SB": "sb",
}
SCHEDULED_GAMES = 162


def load_season_stats(stats_dir: Path, season: int) -> pd.DataFrame:
    """A season's hitting lines by MLBAM id, lower-case columns (``STAT_COLUMNS``)."""
    raw = pd.read_csv(stats_dir / f"hitters-{season}.csv")
    if not raw["MLBAMID"].is_unique:
        raise ValueError(f"hitters-{season}.csv has repeated MLBAM ids")
    return raw.set_index("MLBAMID")[list(STAT_COLUMNS)].rename(columns=STAT_COLUMNS).astype(float)


def season_lines(
    panel: pd.DataFrame, stats_dir: Path, seasons: Iterable[int]
) -> dict[int, pd.DataFrame]:
    """``g`` and ``pa`` per scheduled game, by MLBAM id, for each season: the stats file
    if there is one, else the panel's observed non-pitcher rows."""
    out = {}
    for season in seasons:
        if (stats_dir / f"hitters-{season}.csv").exists():
            s = load_season_stats(stats_dir, season)
            out[season] = pd.DataFrame({"g": s["g"], "pa": s["pa"]}) / SCHEDULED_GAMES
            continue
        p = panel[(panel["season"] == season) & panel["observed"] & ~panel["is_pitcher"]]
        if p.empty:
            raise ValueError(f"no stats file and no panel rows for {season}")
        p = p.set_index("mlbam_id")
        out[season] = pd.DataFrame(
            {"g": p["games"] / p["scheduled_games"], "pa": p["pa"] / p["scheduled_games"]}
        )
    return out


def player_info(panel: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """Per MLBAM id: ``birth`` (people file's ``birthDate``), ``debut_year`` and ``pos``
    (the panel's latest primary position)."""
    hitters = panel[~panel["is_pitcher"]].sort_values("season").groupby("mlbam_id")
    info = pd.DataFrame(
        {
            "debut_year": pd.to_datetime(hitters["debut_date"].last()).dt.year,
            "pos": hitters["primary_position"].last(),
        }
    )
    births = pd.to_datetime(people.set_index("id")["birthDate"])
    ids = info.index.union(births.index)
    return info.reindex(ids).assign(birth=births.reindex(ids))


def load_rates(run_dirs: Iterable[Path]) -> pd.DataFrame:
    """Our preseason rates by (player_id, season): the runs' week-0 predictions,
    averaged over the runs (one per seed). Columns ``p_<stat>``."""
    frames = []
    for run in run_dirs:
        preds = pd.read_parquet(run / "predictions.parquet")
        week0 = preds[preds["week"] == 0]
        if week0.duplicated(["player_id", "season"]).any():
            raise ValueError(f"{run}: more than one preseason row per player-season")
        frames.append(week0[["player_id", "season", *STATS]])
    if not frames:
        raise ValueError("no runs to read rates from")
    rates = pd.concat(frames).groupby(["player_id", "season"])[list(STATS)].mean()
    return rates.rename(columns={c: f"p_{c}" for c in STATS})


def load_pt(path: Path) -> pd.DataFrame:
    """A FanGraphs hitter projection's ``pa`` and ``ab`` by MLBAM id."""
    return load_fangraphs_hitters(path)[["pa", "ab"]]


def build_rows(
    season: int,
    pt: pd.DataFrame,
    rates: pd.DataFrame,
    lines: Mapping[int, pd.DataFrame],
    info: pd.DataFrame,
    actual: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, int]:
    """One season's rows for ``ranges``, indexed by MLBAM id, and how many ranked
    hitters were dropped for having no projected rates.

    ``pt``: the PT source's ``pa``/``ab``; its top ``ranges.TOP_N`` by PA are ranked
    (before anyone is dropped, so a rank means the same as in training). ``rates``:
    :func:`load_rates`. ``actual``: :func:`load_season_stats` for a past season (a hitter
    missing from it batted 0 times), None for one not played yet."""
    rank = top_ranked(pt["pa"])
    ids = rank.index
    ab_pa = (pt["ab"] / pt["pa"]).reindex(ids)
    i = info.reindex(ids)
    rows = pd.DataFrame(
        {
            "season": season,
            "proj_rank": rank,
            "ab_pa": ab_pa,
            "age": (pd.Timestamp(f"{season}-06-30") - i["birth"]).dt.days / 365.25,
            "catcher": (i["pos"] == "C").astype(float),
            "dh": (i["pos"] == "DH").astype(float),
        },
        index=ids,
    )
    rows = rows.join(history(ids, season, lines, i["debut_year"]))
    if season not in rates.index.get_level_values("season"):
        raise ValueError(f"the rates have no {season} rows")
    season_rates = rates.xs(season, level="season")
    rows = rows.join(season_rates)
    if actual is not None:
        a = actual.reindex(ids).fillna(0.0)
        for c in ("pa", "ab", "h", "r", "hr", "rbi", "sb"):
            rows[f"a_{c}"] = a[c]
    no_rates = rows[[f"p_{c}" for c in STATS]].isna().any(axis=1)
    rows = rows[~no_rates]
    rows.index.name = "player_id"
    return rows, int(no_rates.sum())


def history_seasons(seasons: Iterable[int]) -> list[int]:
    """Every season whose lines the rows for ``seasons`` read (the ``HISTORY_SEASONS``
    before each)."""
    return sorted({s - k for s in seasons for k in range(1, HISTORY_SEASONS + 1)})


def actual_pa(stats_dir: Path, seasons: Iterable[int]) -> dict[int, np.ndarray]:
    """Every hitter's PA by season, for ``ranges.pa_by_rank``."""
    return {s: load_season_stats(stats_dir, s)["pa"].to_numpy() for s in seasons}
