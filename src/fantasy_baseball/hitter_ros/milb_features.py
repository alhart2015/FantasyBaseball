"""Minor-league inputs for every hitter-table row (#435 step 3).

For each row (a player on an as-of date in season S) and each window -- ``std`` (this
season before the date), ``p1`` (last season), ``p3`` (the three seasons before), ``car``
(every earlier season in the store, 2008 on) -- one graded minor-league line:

* ``milb_<w>_<stat>`` for each ``milb_grade.STATS``: the line's MLB-league-relative rate
  (league-relative, times its level's factor; ``milb_grade``), PA-weighted over the
  levels and leagues he played in. NaN with no graded PA.
* ``milb_<w>_log_pa``: log(1 + minor-league PA), 0 with none -- "no minor-league PA" is
  known, not missing.
* ``milb_<w>_level``: PA-weighted level, 1 = AAA .. 6 = Rookie (``LEVEL_RANK``), and
  ``milb_<w>_top_level``, the highest level reached. NaN with no PA.
* ``milb_<w>_age_vs_level``: PA-weighted age minus the level's mean age.

No hindsight:

* Factors are walk-forward: season S's rows are graded with factors measured on the
  ``FACTOR_SEASONS`` seasons before S only (``milb_grade.level_factors``).
* Past seasons are league-relative to their full-season league rates (known by S).
  ``std`` uses only the weekly windows that end before the as-of date, league-relative
  to the league's rates over those same windows.
* A level's mean age comes from the season the line is from; for ``std``, from the
  season before (this season's level ages are only known at its end).

Two build options (``build_milb_features``), for the vets these inputs can add noise to
(``MILB_PRESETS`` names the combinations in use; the default model's is ``rookies-s100``):

* ``shrink_pa``: each window's graded rates are pulled toward the average line at its
  levels (the factor itself: 1x the league, graded) by that many PA, so an 18-PA
  rehab stint can't read as an 11x home-run rate.
* ``vet_min_pa``: rows of players with at least that many MLB PA when projected (career
  before the season plus this season so far) get no minor-league inputs -- log PA 0 and
  everything else blank, as for a player who never played in the minors. So do rows
  whose career count can't be trusted (fewer than ``backtest.MIN_HISTORY_SEASONS``
  earlier seasons in the box-score store, i.e. 2008-2011): a veteran there reads as a
  rookie, and only known rookies get the inputs.

There was no minor-league season in 2020, so a 2021 row's ``p1`` window is unknown --
every ``milb_p1_*`` blank, log PA included -- not "no minor-league PA".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.backtest import MIN_HISTORY_SEASONS
from fantasy_baseball.hitter_ros.features import ROW_KEYS, load_feature_file
from fantasy_baseball.hitter_ros.milb_grade import (
    COUNTS,
    MLB,
    STATS,
    league_rates,
    level_factors,
    level_mean_ages,
    player_levels,
    rates,
    relative_lines,
)
from fantasy_baseball.pitch_data.milb import NO_MILB_SEASONS

MILB_WINDOWS = ("std", "p1", "p3", "car")
LEVEL_RANK = {11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6}
FACTOR_SEASONS = 7
_PER_WINDOW = (*STATS, "log_pa", "level", "top_level", "age_vs_level")
MILB_FEATURES = [f"milb_{w}_{name}" for w in MILB_WINDOWS for name in _PER_WINDOW]
_LEAGUE = ["season", "sport_id", "league_id"]
# Named build options (scripts/build_hitter_ros_milb.py --name), checked on load.
MILB_PRESETS: dict[str, dict[str, float | None]] = {
    "rookies-s100": {"shrink_pa": 100.0, "vets_blank_from": 300.0},
}


def milb_path(root: Path, name: str) -> Path:
    """Where ``scripts/build_hitter_ros_milb.py --name <name>`` writes the features."""
    return root / f"milb_{name}.parquet"


def load_milb_inputs(
    table: pd.DataFrame, root: Path, name: str
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """``MILB_FEATURES`` from ``milb_<name>.parquet`` under ``root``, aligned to
    ``table``'s rows, and the build options saved next to it (``milb_<name>.json``).
    ValueError, saying how to fix it, when the file is missing, lacks a feature, was
    built for another table, or -- for a ``MILB_PRESETS`` name -- with other options."""
    path = milb_path(root, name)
    rebuild = f"run scripts/build_hitter_ros_milb.py --name {name}"
    if not path.exists():
        raise ValueError(f"{path} is missing; {rebuild}")
    options_path = path.with_suffix(".json")
    options = json.loads(options_path.read_text()) if options_path.exists() else {}
    if name in MILB_PRESETS and options != MILB_PRESETS[name]:
        raise ValueError(
            f"{path} was built with {options or 'unknown options'}, not the {name} preset "
            f"{MILB_PRESETS[name]}; {rebuild}"
        )
    return load_feature_file(table, path, MILB_FEATURES, rebuild), options


def _grade(rel: pd.DataFrame, factors: pd.DataFrame, level_age: pd.Series) -> pd.DataFrame:
    """Per line: graded rates ``g_<stat>``, level rank and age vs. level. ``level_age``
    is indexed by (season, sport_id) and matched on the line's ``age_season``."""
    f = factors.reindex(rel["sport_id"]).set_axis(rel.index)
    mean_age = level_age.reindex(pd.MultiIndex.from_frame(rel[["age_season", "sport_id"]]))
    return rel.assign(
        **{f"g_{s}": rel[f"rel_{s}"] * f[s] for s in STATS},
        **{f"f_{s}": f[s] for s in STATS},
        rank=rel["sport_id"].map(LEVEL_RANK),
        age_vs_level=rel["age"].to_numpy() - mean_age.to_numpy(),
    )


def _summarize(
    graded: pd.DataFrame, keys: list[str], window: str, shrink_pa: float = 0.0
) -> pd.DataFrame:
    """One graded line per ``keys`` group (see the module doc for each column)."""
    pa = graded["pa"].astype(float)
    group = [graded[k] for k in keys]

    def weighted(col: pd.Series) -> pd.Series:
        ok = col.notna()
        num = (col.where(ok, 0.0) * pa).groupby(group).sum()
        den = pa.where(ok, 0.0).groupby(group).sum()
        return num / den.where(den > 0)

    def graded_rate(s: str) -> pd.Series:
        """PA-weighted graded rate, shrunk ``shrink_pa`` PA toward the levels' factor."""
        g, f = graded[f"g_{s}"], graded[f"f_{s}"]
        ok = g.notna() & f.notna()
        pa_ok = pa.where(ok, 0.0).groupby(group).sum()
        sum_g = (g.where(ok, 0.0) * pa).groupby(group).sum()
        prior = (f.where(ok, 0.0) * pa).groupby(group).sum() / pa_ok.where(pa_ok > 0)
        return (sum_g + shrink_pa * prior) / (pa_ok + shrink_pa).where(pa_ok > 0)

    out = pd.DataFrame(
        {
            **{s: graded_rate(s) for s in STATS},
            "log_pa": np.log1p(pa.groupby(group).sum()),
            "level": weighted(graded["rank"].astype(float)),
            "top_level": graded["rank"].groupby(group).min().astype(float),
            "age_vs_level": weighted(graded["age_vs_level"]),
        }
    )
    return out.add_prefix(f"milb_{window}_").reset_index()


def _past_windows(
    rows: pd.DataFrame,
    season_rel: pd.DataFrame,
    factors: dict[int, pd.DataFrame],
    level_age: pd.Series,
    shrink_pa: float,
) -> pd.DataFrame:
    """p1 / p3 / car features per (player, season) of ``rows``."""
    parts = []
    for season in sorted(rows["season"].unique()):
        players = rows.loc[rows["season"] == season, "player_id"].unique()
        before = season_rel[(season_rel["season"] < season) & season_rel["player_id"].isin(players)]
        graded = _grade(before.assign(age_season=before["season"]), factors[season], level_age)
        merged = pd.DataFrame({"player_id": players, "season": season})
        for window, first in (("p1", season - 1), ("p3", season - 3), ("car", None)):
            sub = graded if first is None else graded[graded["season"] >= first]
            summary = _summarize(sub, ["player_id"], window, shrink_pa).assign(season=season)
            merged = merged.merge(summary, on=["player_id", "season"], how="left")
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def _cumulative_league(window_lines: pd.DataFrame) -> pd.DataFrame:
    """Each league's counts over every window up to and including each ``window_end``
    of its season, on a dense grid (a league with no games in a window still gets a
    row), so a level's total at a date sums all its leagues."""
    counts = list(COUNTS)
    per = window_lines.groupby([*_LEAGUE, "window_end"])[counts].sum().reset_index()
    parts = []
    for season, g in per.groupby("season"):
        ends = np.sort(window_lines.loc[window_lines["season"] == season, "window_end"].unique())
        leagues = g[["sport_id", "league_id"]].drop_duplicates()
        grid = leagues.merge(pd.DataFrame({"window_end": ends}), how="cross").assign(season=season)
        dense = grid.merge(g, on=[*_LEAGUE, "window_end"], how="left").fillna(
            dict.fromkeys(counts, 0)
        )
        dense = dense.sort_values("window_end")
        dense[counts] = dense.groupby(["sport_id", "league_id"])[counts].cumsum()
        parts.append(dense)
    return pd.concat(parts, ignore_index=True)


def _no_std() -> pd.DataFrame:
    """The std features with no rows (float columns, so a merge keeps them numeric)."""
    return pd.DataFrame(
        {
            **{k: pd.Series(dtype="int64") for k in ROW_KEYS},
            **{f"milb_std_{n}": pd.Series(dtype="float64") for n in _PER_WINDOW},
        }
    )


def _std_window(
    rows: pd.DataFrame,
    window_lines: pd.DataFrame,
    factors: dict[int, pd.DataFrame],
    level_age: pd.Series,
    shrink_pa: float,
) -> pd.DataFrame:
    """std features per table row: minor-league windows ending before the as-of date."""
    counts = list(COUNTS)
    lines = window_lines[window_lines["player_id"].isin(rows["player_id"].unique())]
    lines = lines.sort_values("window_end")
    keys = ["season", "player_id", "sport_id", "league_id"]
    cum = lines.copy()
    cum[counts] = lines.groupby(keys)[counts].cumsum()
    # Each table row x each (level, league) the player had minor-league lines at.
    combos = lines[[*keys, "age"]].drop_duplicates(keys)
    left = rows[[*ROW_KEYS, "as_of"]].merge(combos, on=["season", "player_id"])
    if left.empty:
        return _no_std()
    left = left.assign(as_of=pd.to_datetime(left["as_of"]).astype("datetime64[ns]"))
    cum = cum.assign(
        end=pd.to_datetime(cum["window_end"]).astype("datetime64[ns]"),
        **{k: cum[k].astype("int64") for k in keys},
    ).drop(columns=["window_end", "age"])
    left = left.astype({k: "int64" for k in keys})
    player = pd.merge_asof(
        left.sort_values("as_of"),
        cum.sort_values("end"),
        left_on="as_of",
        right_on="end",
        by=keys,
        allow_exact_matches=False,  # a window ending on the as-of date isn't over yet
    ).dropna(subset=["pa"])
    player = player[player["pa"] > 0]
    if player.empty:  # lines this season, but none in a window over before any date
        return _no_std()
    # The league's rates over the same windows; a small league uses its level's.
    league_cum = _cumulative_league(window_lines)
    league_cum = league_cum.assign(
        end=pd.to_datetime(league_cum["window_end"]).astype("datetime64[ns]")
    )
    league_cum = league_cum.astype({k: "int64" for k in _LEAGUE})
    pooled_keys = ["season", "end", "sport_id", "league_id"]
    league_ref = league_rates(league_cum, pooled_keys)
    at = pd.merge_asof(
        player.sort_values("as_of")[[*keys, "as_of", "end"]].reset_index(),
        league_cum[["season", "sport_id", "league_id", "end"]]
        .drop_duplicates()
        .rename(columns={"end": "league_end"})
        .sort_values("league_end"),
        left_on="as_of",
        right_on="league_end",
        by=_LEAGUE,
        allow_exact_matches=False,
    ).set_index("index")
    ref = league_ref.reindex(
        pd.MultiIndex.from_arrays(
            [at["season"], at["league_end"], at["sport_id"], at["league_id"]],
            names=pooled_keys,
        )
    ).set_axis(at.index)
    own = rates(player)
    rel = player.assign(**{f"rel_{s}": own[s] / ref.loc[player.index, s] for s in STATS})
    graded = []
    for season, g in rel.groupby("season"):
        graded.append(_grade(g.assign(age_season=season - 1), factors[season], level_age))
    return _summarize(pd.concat(graded), ROW_KEYS, "std", shrink_pa)


def build_milb_features(
    table: pd.DataFrame,
    season_lines: pd.DataFrame,
    window_lines: pd.DataFrame,
    mlb_lines: pd.DataFrame,
    shrink_pa: float = 0.0,
    vet_min_pa: float | None = None,
) -> pd.DataFrame:
    """``MILB_FEATURES`` for every ``table`` row, keyed by ``ROW_KEYS`` plus ``as_of``.

    ``season_lines`` / ``window_lines``: ``milb_grade.load_milb_lines`` without and with
    ``by_window``; ``mlb_lines``: ``milb_grade.load_mlb_lines`` (for the factors).
    ``shrink_pa`` / ``vet_min_pa``: see the module doc; ``vet_min_pa`` needs the table's
    ``car_pa``, ``std_pa`` and ``car_seasons_in_store``."""
    rows = table[[*ROW_KEYS, "as_of"]].copy()
    rel = relative_lines(pd.concat([season_lines, mlb_lines], ignore_index=True))
    levels = player_levels(rel)
    level_age = level_mean_ages(levels)
    factors = {
        int(s): level_factors(levels, (int(s) - FACTOR_SEASONS, int(s) - 1))
        for s in rows["season"].unique()
    }
    season_rel = rel[rel["sport_id"] != MLB]  # leagues never mix levels: MLB can't move it
    past = _past_windows(rows, season_rel, factors, level_age, shrink_pa)
    std = _std_window(rows, window_lines, factors, level_age, shrink_pa)
    out = rows.merge(past, on=["player_id", "season"], how="left").merge(
        std, on=ROW_KEYS, how="left"
    )
    blank = np.zeros(len(out), dtype=bool)
    if vet_min_pa is not None:
        mlb_pa = table["car_pa"].astype(float) + table["std_pa"].astype(float)
        short_history = table["car_seasons_in_store"] < MIN_HISTORY_SEASONS
        blank = ((mlb_pa >= vet_min_pa) | short_history).to_numpy()
        out.loc[blank, MILB_FEATURES] = np.nan
    for w in MILB_WINDOWS:
        out[f"milb_{w}_log_pa"] = out[f"milb_{w}_log_pa"].fillna(0.0)
    # Last season had no minor leagues: unknown, not zero (blanked rows stay "none").
    no_p1 = out["season"].sub(1).isin(NO_MILB_SEASONS).to_numpy() & ~blank
    out.loc[no_p1, [f"milb_p1_{n}" for n in _PER_WINDOW]] = np.nan
    return out[[*ROW_KEYS, "as_of", *MILB_FEATURES]]
