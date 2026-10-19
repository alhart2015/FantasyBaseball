"""Rows for ``ranges`` from the files (#505): projected playing time, our projected
rates, each player's history and, for past seasons, what he did.

* Seasons come from the MLB Stats API season leaderboards (``keepers.mlb_stats``,
  cached as ``data/cache/keeper_skills/mlb_<hitting|pitching>_<y>.csv``): everyone who
  played, not just the 50+ PA hitters of ``data/stats`` (so a 30-PA season reads as
  30, not 0), the same source for every season, and finished seasons only once
  refreshed after them. A ranked player missing from a played season's leaderboard
  didn't play.
* A hitter's position is the latest one before the row's season (none later: a catcher
  who moved to DH afterwards is still a catcher on his earlier rows). The leaderboard's
  per-season position also reads a two-way player by his hitting job (Ohtani: DH).
* A pitcher's role (starter or reliever, ``range_specs.STARTER_SHARE``) is the PT
  source's: its projected GS / G, or for last season's line, its GS / G.
* Ages are on the June 30 convention, from the people file's birth dates; debut year
  from its ``mlbDebutDate``.
* Our rates: a run's preseason (week 0) rows, averaged over its seeds (hitters: a run
  per seed; pitchers: the run's own seed-averaged predictions).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.evaluate import load_fangraphs_hitters
from fantasy_baseball.keepers.actuals import coerce_numeric, index_by_mlbam, innings_to_float
from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season
from fantasy_baseball.pitcher_ros.evaluate import load_fangraphs_pitchers, read_fangraphs_pitchers
from fantasy_baseball.pt_model.panel import FULL_SEASON_GAMES, SHORTENED_SEASONS
from fantasy_baseball.pt_model.range_specs import HITTER, STARTER, STARTER_SHARE
from fantasy_baseball.pt_model.ranges import HISTORY_SEASONS, Spec, history, top_ranked

STATS = HITTER.stat_names

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
PITCHING_COLUMNS = {
    "gamesPitched": "g",
    "gamesStarted": "gs",
    "battersFaced": "bf",
    "strikeOuts": "k",
    "wins": "w",
    "saves": "sv",
    "earnedRuns": "er",
    "hits": "h",
    "baseOnBalls": "bb",
}


def season_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """A raw hitting leaderboard (``fetch_mlb_season(..., "hitting")``) by MLBAM id: the
    ``SEASON_COLUMNS`` as floats and ``pos``, the season's position abbreviation."""
    frame = index_by_mlbam(raw, "player.id")
    if not frame.index.is_unique:
        raise ValueError("the hitting leaderboard repeats a player")
    out = pd.DataFrame(
        {name: frame[f"stat.{col}"].map(coerce_numeric) for col, name in SEASON_COLUMNS.items()},
        index=frame.index,
    )
    out["pos"] = frame["position.abbreviation"]
    return out


def pitching_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """A raw pitching leaderboard by MLBAM id: ``ip`` (from baseball-notation innings)
    and ``PITCHING_COLUMNS`` as floats."""
    frame = index_by_mlbam(raw, "player.id")
    if not frame.index.is_unique:
        raise ValueError("the pitching leaderboard repeats a player")
    out = pd.DataFrame(
        {name: frame[f"stat.{col}"].map(coerce_numeric) for col, name in PITCHING_COLUMNS.items()},
        index=frame.index,
    )
    out.insert(0, "ip", frame["stat.inningsPitched"].map(innings_to_float))
    return out


def _load(cache_dir: Path, seasons: Iterable[int], group: str) -> dict[int, pd.DataFrame]:
    parse = season_frame if group == "hitting" else pitching_frame
    out = {}
    for season in sorted(set(seasons)):
        frame = parse(fetch_mlb_season(cache_dir, season, group))
        if frame.empty:
            raise ValueError(f"no {season} {group} leaderboard")
        out[season] = frame
    return out


def load_seasons(cache_dir: Path, seasons: Iterable[int]) -> dict[int, pd.DataFrame]:
    """:func:`season_frame` per season, from the leaderboard cache (fetched on a miss)."""
    return _load(cache_dir, seasons, "hitting")


def load_pitching_seasons(cache_dir: Path, seasons: Iterable[int]) -> dict[int, pd.DataFrame]:
    """:func:`pitching_frame` per season, from the leaderboard cache (fetched on a miss)."""
    return _load(cache_dir, seasons, "pitching")


def season_lines(
    seasons: Mapping[int, pd.DataFrame],
    years: Iterable[int],
    columns: Iterable[str] = ("g", "pa"),
) -> dict[int, pd.DataFrame]:
    """``columns`` per scheduled game, by MLBAM id, for each of ``years``."""
    cols = list(columns)
    out = {}
    for year in years:
        if year not in seasons:
            raise ValueError(f"no {year} season loaded")
        games = SHORTENED_SEASONS.get(year, FULL_SEASON_GAMES)
        out[year] = seasons[year][cols] / games
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


def _week0(preds: pd.DataFrame, stats: Iterable[str], where: object) -> pd.DataFrame:
    week0 = preds[preds["week"] == 0]
    if week0.duplicated(["player_id", "season"]).any():
        raise ValueError(f"{where}: more than one preseason row per player-season")
    return week0[["player_id", "season", *stats]]


def load_rates(run_dirs: Iterable[Path]) -> pd.DataFrame:
    """Our preseason rates by (player_id, season): the runs' week-0 predictions,
    averaged over the runs (one per seed). Columns ``p_<stat>``."""
    frames = [_week0(pd.read_parquet(run / "predictions.parquet"), STATS, run) for run in run_dirs]
    if not frames:
        raise ValueError("no runs to read rates from")
    rates = pd.concat(frames).groupby(["player_id", "season"])[list(STATS)].mean()
    return rates.rename(columns={c: f"p_{c}" for c in STATS})


def load_pitcher_rates(run_dir: Path) -> pd.DataFrame:
    """Our preseason pitcher rates by (player_id, season): a pitcher run's week-0
    predictions (``predictions.parquet``, its seeds already averaged). Columns
    ``p_<stat>``."""
    stats = STARTER.stat_names
    week0 = _week0(pd.read_parquet(run_dir / "predictions.parquet"), stats, run_dir)
    rates = week0.set_index(["player_id", "season"])[list(stats)]
    return rates.rename(columns={c: f"p_{c}" for c in stats})


def load_pt(path: Path) -> pd.DataFrame:
    """A FanGraphs hitter projection's ``pa`` and ``ab`` by MLBAM id."""
    return load_fangraphs_hitters(path)[["pa", "ab"]]


def load_pitcher_pt(path: Path) -> pd.DataFrame:
    """A FanGraphs pitcher projection's ``ip``, ``bf`` and ``gs_share`` (GS / G) by
    MLBAM id."""
    raw = read_fangraphs_pitchers(path)
    pt = load_fangraphs_pitchers(path, raw)[["ip", "bf"]]
    games = raw["G"].astype(float)
    share = raw["GS"].astype(float) / games.where(games > 0)
    return pt.assign(gs_share=share.to_numpy())


def last_season_pitcher_pt(seasons: Mapping[int, pd.DataFrame], season: int) -> pd.DataFrame:
    """Last season's line as a pitcher PT source: ``ip``, ``bf``, ``gs_share``."""
    last = seasons[season - 1]
    return pd.DataFrame(
        {
            "ip": last["ip"],
            "bf": last["bf"],
            "gs_share": last["gs"] / last["g"].where(last["g"] > 0),
        }
    )


def _rows(
    season: int,
    spec: Spec,
    proj: pd.Series,
    per_volume: Mapping[str, pd.Series],
    extra: Mapping[str, pd.Series],
    rates: pd.DataFrame,
    seasons: Mapping[int, pd.DataFrame],
    info: pd.DataFrame,
    with_actuals: bool,
) -> tuple[pd.DataFrame, int]:
    """The rows for one season and spec: rank on ``proj`` (before anyone is dropped, so
    a rank means the same as in training), age, ``extra`` inputs, history, rates and,
    ``with_actuals``, what he did (0 if missing from the season)."""
    rank = top_ranked(proj, spec.top_n)
    ids = rank.index
    i = info.reindex(ids)
    rows = pd.DataFrame(
        {
            "season": season,
            "proj_rank": rank,
            **{col: v.reindex(ids) for col, v in per_volume.items()},
            "age": (pd.Timestamp(f"{season}-06-30") - i["birth"]).dt.days / 365.25,
            **{col: v.reindex(ids) for col, v in extra.items()},
        },
        index=ids,
    )
    lines = season_lines(seasons, history_seasons([season]), spec.history_columns)
    rows = rows.join(history(ids, season, lines, i["debut_year"], spec.history_columns))
    if season not in rates.index.get_level_values("season"):
        raise ValueError(f"the rates have no {season} rows")
    rows = rows.join(rates.xs(season, level="season"))
    if with_actuals:
        if season not in seasons:
            raise ValueError(f"no {season} season loaded for the actuals")
        cols = [c.removeprefix("a_") for c in spec.actuals]
        actual = seasons[season][cols].reindex(ids).fillna(0.0)
        for c in cols:
            rows[f"a_{c}"] = actual[c]
    no_rates = rows[[f"p_{c}" for c in spec.stat_names]].isna().any(axis=1)
    rows = rows[~no_rates]
    rows.index.name = "player_id"
    return rows, int(no_rates.sum())


def build_rows(
    season: int,
    pt: pd.DataFrame,
    rates: pd.DataFrame,
    seasons: Mapping[int, pd.DataFrame],
    info: pd.DataFrame,
    *,
    with_actuals: bool,
) -> tuple[pd.DataFrame, int]:
    """One season's hitter rows for ``ranges``, indexed by MLBAM id, and how many ranked
    hitters were dropped for having no projected rates.

    ``pt``: the PT source's ``pa``/``ab``; its top ``HITTER.top_n`` by PA are ranked.
    ``rates``: :func:`load_rates`. ``seasons``: :func:`load_seasons` with the
    ``HISTORY_SEASONS`` before ``season`` and, ``with_actuals``, ``season`` itself.
    ``info``: :func:`player_info`."""
    pos = position_before(seasons, season, pt.index)  # _rows keeps the ranked ones
    return _rows(
        season,
        HITTER,
        pt["pa"],
        {"ab_pa": pt["ab"] / pt["pa"]},
        {"catcher": (pos == "C").astype(float), "dh": (pos == "DH").astype(float)},
        rates,
        seasons,
        info,
        with_actuals,
    )


def _starter(spec: Spec) -> bool:
    """Whether a pitcher spec is the starter role (else the reliever)."""
    if spec.role not in ("starter", "reliever"):
        raise ValueError(f"{spec.name}: not a pitcher role ({spec.role!r})")
    return spec.role == "starter"


def build_pitcher_rows(
    season: int,
    spec: Spec,
    pt: pd.DataFrame,
    rates: pd.DataFrame,
    seasons: Mapping[int, pd.DataFrame],
    info: pd.DataFrame,
    *,
    with_actuals: bool,
) -> tuple[pd.DataFrame, int]:
    """One season's rows for a pitcher role (``range_specs.STARTER`` or ``RELIEVER``),
    as :func:`build_rows`. ``pt``: :func:`load_pitcher_pt` (or
    :func:`last_season_pitcher_pt`); the role is its ``gs_share``, and its top
    ``spec.top_n`` in the role by IP are ranked. ``seasons``:
    :func:`load_pitching_seasons`; ``rates``: :func:`load_pitcher_rates`."""
    starts = pt["gs_share"] >= STARTER_SHARE
    in_role = pt[starts] if _starter(spec) else pt[~starts & pt["gs_share"].notna()]
    return _rows(
        season,
        spec,
        in_role["ip"],
        {"bf_ip": in_role["bf"] / in_role["ip"]},
        {},
        rates,
        seasons,
        info,
        with_actuals,
    )


def history_seasons(seasons: Iterable[int]) -> list[int]:
    """Every season whose lines the rows for ``seasons`` read (the ``HISTORY_SEASONS``
    before each)."""
    return sorted({s - k for s in seasons for k in range(1, HISTORY_SEASONS + 1)})


def actual_pa(seasons: Mapping[int, pd.DataFrame], years: Iterable[int]) -> dict[int, np.ndarray]:
    """Every hitter's PA by season, for ``ranges.volume_by_rank``."""
    return {y: seasons[y]["pa"].to_numpy() for y in years}


def actual_ip(
    seasons: Mapping[int, pd.DataFrame], years: Iterable[int], spec: Spec
) -> dict[int, np.ndarray]:
    """IP by season of every pitcher who pitched in ``spec``'s role that season (his
    actual GS / G), for ``ranges.volume_by_rank``."""
    out = {}
    for y in years:
        s = seasons[y]
        starts = s["gs"] / s["g"].where(s["g"] > 0) >= STARTER_SHARE
        out[y] = s.loc[starts if _starter(spec) else ~starts, "ip"].to_numpy()
    return out
