"""Rows for ``ranges`` from the files (#505): projected playing time, our projected
rates, each hitter's history and, for past seasons, what he did.

* Seasons (games, PA, AB, H, R, HR, RBI, SB and the season's position) come from the
  MLB Stats API season leaderboards (``keepers.mlb_stats``, cached as
  ``data/cache/keeper_skills/mlb_hitting_<y>.csv``): every hitter who batted, not just
  those with 50+ PA as in ``data/stats`` (so a 30-PA season reads as 30, not 0), the
  same source for every season, and finished seasons only once refreshed after them.
  A ranked hitter missing from a played season's leaderboard batted 0 times.
* Position is the latest one before the row's season (none later: a catcher who moved
  to DH afterwards is still a catcher on his earlier rows). The leaderboard's
  per-season position also reads a two-way player by his hitting job (Ohtani: DH).
* Ages are on the June 30 convention, from the people file's birth dates; debut year
  from its ``mlbDebutDate``.
* Our rates: a hitter run's preseason (week 0) rows, averaged over its seeds.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.evaluate import load_fangraphs_hitters
from fantasy_baseball.keepers.actuals import coerce_numeric, index_by_mlbam
from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season
from fantasy_baseball.pt_model.panel import FULL_SEASON_GAMES, SHORTENED_SEASONS
from fantasy_baseball.pt_model.ranges import HISTORY_SEASONS, STATS, history, top_ranked

# Leaderboard stat columns (``stat.<name>``) read, and what they're called here.
SEASON_COLUMNS = {
    "gamesPlayed": "g",
    "plateAppearances": "pa",
    "atBats": "ab",
    "hits": "h",
    "runs": "r",
    "homeRuns": "hr",
    "rbi": "rbi",
    "stolenBases": "sb",
}
COUNTS = tuple(SEASON_COLUMNS.values())


def season_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """A raw hitting leaderboard (``fetch_mlb_season(..., "hitting")``) by MLBAM id: the
    ``COUNTS`` as floats and ``pos``, the season's position abbreviation."""
    frame = index_by_mlbam(raw, "player.id")
    if not frame.index.is_unique:
        raise ValueError("the hitting leaderboard repeats a player")
    out = pd.DataFrame(
        {name: frame[f"stat.{col}"].map(coerce_numeric) for col, name in SEASON_COLUMNS.items()},
        index=frame.index,
    )
    out["pos"] = frame["position.abbreviation"]
    return out


def load_seasons(cache_dir: Path, seasons: Iterable[int]) -> dict[int, pd.DataFrame]:
    """:func:`season_frame` per season, from the leaderboard cache (fetched on a miss)."""
    out = {}
    for season in sorted(set(seasons)):
        frame = season_frame(fetch_mlb_season(cache_dir, season, "hitting"))
        if frame.empty:
            raise ValueError(f"no {season} hitting leaderboard")
        out[season] = frame
    return out


def season_lines(
    seasons: Mapping[int, pd.DataFrame], years: Iterable[int]
) -> dict[int, pd.DataFrame]:
    """``g`` and ``pa`` per scheduled game, by MLBAM id, for each of ``years``."""
    out = {}
    for year in years:
        if year not in seasons:
            raise ValueError(f"no {year} season loaded")
        games = SHORTENED_SEASONS.get(year, FULL_SEASON_GAMES)
        out[year] = seasons[year][["g", "pa"]] / games
    return out


def position_before(seasons: Mapping[int, pd.DataFrame], season: int, ids: pd.Index) -> pd.Series:
    """Each id's position in the latest of the ``HISTORY_SEASONS`` before ``season``
    he batted in (NaN if none): never one from ``season`` or later."""
    pos = pd.Series(np.nan, index=ids, dtype=object)
    for k in range(HISTORY_SEASONS, 0, -1):  # oldest first, so a later season wins
        year = season - k
        if year in seasons:
            known = seasons[year]["pos"].reindex(ids)
            pos = known.where(known.notna(), pos)
    return pos


def player_info(people: pd.DataFrame) -> pd.DataFrame:
    """Per MLBAM id (the people file's ``id``): ``birth`` and ``debut_year``."""
    p = people.set_index("id")
    if not p.index.is_unique:
        raise ValueError("the people file repeats an id")
    return pd.DataFrame(
        {
            "birth": pd.to_datetime(p["birthDate"]),
            "debut_year": pd.to_datetime(p["mlbDebutDate"]).dt.year,
        },
        index=p.index,
    )


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
    seasons: Mapping[int, pd.DataFrame],
    info: pd.DataFrame,
    *,
    with_actuals: bool,
) -> tuple[pd.DataFrame, int]:
    """One season's rows for ``ranges``, indexed by MLBAM id, and how many ranked
    hitters were dropped for having no projected rates.

    ``pt``: the PT source's ``pa``/``ab``; its top ``ranges.TOP_N`` by PA are ranked
    (before anyone is dropped, so a rank means the same as in training). ``rates``:
    :func:`load_rates`. ``seasons``: :func:`load_seasons` with the ``HISTORY_SEASONS``
    before ``season`` and, ``with_actuals``, ``season`` itself (a ranked hitter missing
    from it batted 0 times). ``info``: :func:`player_info`."""
    rank = top_ranked(pt["pa"])
    ids = rank.index
    i = info.reindex(ids)
    pos = position_before(seasons, season, ids)
    rows = pd.DataFrame(
        {
            "season": season,
            "proj_rank": rank,
            "ab_pa": (pt["ab"] / pt["pa"]).reindex(ids),
            "age": (pd.Timestamp(f"{season}-06-30") - i["birth"]).dt.days / 365.25,
            "catcher": (pos == "C").astype(float),
            "dh": (pos == "DH").astype(float),
        },
        index=ids,
    )
    lines = season_lines(seasons, history_seasons([season]))
    rows = rows.join(history(ids, season, lines, i["debut_year"]))
    if season not in rates.index.get_level_values("season"):
        raise ValueError(f"the rates have no {season} rows")
    rows = rows.join(rates.xs(season, level="season"))
    if with_actuals:
        if season not in seasons:
            raise ValueError(f"no {season} season loaded for the actuals")
        actual = seasons[season][list(COUNTS)].reindex(ids).fillna(0.0)
        for c in ("pa", "ab", "h", "r", "hr", "rbi", "sb"):
            rows[f"a_{c}"] = actual[c]
    no_rates = rows[[f"p_{c}" for c in STATS]].isna().any(axis=1)
    rows = rows[~no_rates]
    rows.index.name = "player_id"
    return rows, int(no_rates.sum())


def history_seasons(seasons: Iterable[int]) -> list[int]:
    """Every season whose lines the rows for ``seasons`` read (the ``HISTORY_SEASONS``
    before each)."""
    return sorted({s - k for s in seasons for k in range(1, HISTORY_SEASONS + 1)})


def actual_pa(seasons: Mapping[int, pd.DataFrame], years: Iterable[int]) -> dict[int, np.ndarray]:
    """Every hitter's PA by season, for ``ranges.pa_by_rank``."""
    return {y: seasons[y]["pa"].to_numpy() for y in years}
