"""Saves as a reliever's share of his team's saves x the team's saves (#474).

Saves come from role: who closes, and how many save chances the team gets. So a
pitcher's projected saves over a span = his projected **share** of his team's saves x
the team's projected **total**.

* Team total (:func:`team_weeks`, :func:`fit_team_saves`): our own model, saves per
  remaining game from the team's saves and wins so far and last season, fit on every
  complete team-season in the store (2008 on) before the projected one, times the
  games the team has left on the schedule. The team numbers are the training table's
  (``pitcher_ros/table.py``: ``std_team_*``, ``p1_team_*``, ``team_games_left``,
  ``ros_team_*``), so there is one definition of a team's saves and games.
* Share, two ways, to compare:

  * ``fg``: FanGraphs' (:func:`fangraphs_shares`). Their projected saves already carry
    their closer calls (news, depth charts); his share is his projected saves over his
    FanGraphs team's. Only where FanGraphs files exist (preseason 2022 on, 2026's
    rest-of-season snapshots).
  * ``usage``: ours, with no FanGraphs (:func:`fit_usage_shares`): gradient-boosted
    trees on his recent and past usage (saves, save chances, games finished, holds,
    his share of his team's saves), fit on every complete season before the projected
    one; the answer is his saves for his current team over its rest-of-season saves.

Everything is dated like the training table: inputs only from games before the date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from fantasy_baseball.pitcher_ros.features import div, input_frame
from fantasy_baseball.ros_training import holdout_players

# Games of last season's rate added to this season's team rates, so a team 10 games in
# isn't read off 10 games.
TEAM_PRIOR_GAMES = 20
TEAM_FEATURES = ("p1_sv_per_g", "p1_wpct", "std_sv_per_g", "std_wpct", "lg_p1_sv_per_g")
TEAM_COLUMNS = (
    "std_team_sv",
    "std_team_games",
    "std_team_w",
    "p1_team_sv",
    "p1_team_games",
    "p1_team_w",
    "team_games_left",
    "ros_team_sv",
    "ros_team_games",
)
TEAM_KEY = ["team_id", "season", "week"]


def team_weeks(table: pd.DataFrame) -> pd.DataFrame:
    """One row per team and as-of date, read off the training table: every pitcher row
    of a team on a date carries the same team columns (``TEAM_COLUMNS``)."""
    rows = table.dropna(subset=["team_id"]).drop_duplicates(TEAM_KEY)
    cols = [*TEAM_KEY, "as_of", "season_complete", *TEAM_COLUMNS]
    return rows[cols].astype({"team_id": "int64"}).reset_index(drop=True)


def _team_features(tw: pd.DataFrame) -> pd.DataFrame:
    week0 = tw[tw["week"] == 0]
    sums = week0.groupby("season")[["p1_team_sv", "p1_team_games"]].sum()
    lg = div(sums["p1_team_sv"], sums["p1_team_games"])
    lg_p1 = tw["season"].map(lg)
    p1_sv = div(tw["p1_team_sv"], tw["p1_team_games"]).fillna(lg_p1)
    p1_w = div(tw["p1_team_w"], tw["p1_team_games"]).fillna(0.5)
    k = TEAM_PRIOR_GAMES
    return pd.DataFrame(
        {
            "p1_sv_per_g": p1_sv,
            "p1_wpct": p1_w,
            "std_sv_per_g": (tw["std_team_sv"] + k * p1_sv) / (tw["std_team_games"] + k),
            "std_wpct": (tw["std_team_w"] + k * p1_w) / (tw["std_team_games"] + k),
            "lg_p1_sv_per_g": lg_p1,
        },
        index=tw.index,
    )[list(TEAM_FEATURES)]


def fit_team_saves(tw: pd.DataFrame, test_season: int) -> pd.Series:
    """Projected saves per remaining game for every team-week of ``test_season``: a
    linear model fit (weighted by remaining games) on the team-weeks of the complete
    seasons before it. Times the team's games left, its projected saves."""
    x = _team_features(tw)
    train = tw["season_complete"] & (tw["season"] < test_season) & (tw["ros_team_games"] > 0)
    train &= x.notna().all(axis=1)
    if not train.any():
        raise ValueError(
            f"no team-week before {test_season} to fit on (a complete season with the "
            "season before it in the store)"
        )
    games = tw.loc[train, "ros_team_games"]
    model = LinearRegression().fit(
        x[train], tw.loc[train, "ros_team_sv"] / games, sample_weight=games
    )
    test = tw["season"] == test_season
    rate = np.clip(model.predict(x[test].fillna(x[train].mean())), 0, None)
    return pd.Series(rate, index=tw.index[test], name="team_sv_per_g")


def scheduled_games_left(schedule: pd.DataFrame, season: int, when: pd.Timestamp) -> pd.Series:
    """Games each team has left on ``season``'s stored schedule from ``when`` on (every
    game listed, played or not), by ``team_id``. ``schedule``: the store's ``games``
    rows (``season``, ``game_date``, ``home_team_id``, ``away_team_id``)."""
    g = schedule[(schedule["season"] == season) & (pd.to_datetime(schedule["game_date"]) >= when)]
    teams = pd.concat([g["home_team_id"], g["away_team_id"]])
    out: pd.Series = teams.value_counts().rename("games_left")
    out.index.name = "team_id"
    return out


def score_methods(
    units: pd.DataFrame,
    methods: list[str],
    key: list[str],
    n_boot: int = 2000,
    pairs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Saves projections scored on actual saves: per method, MSE (squared miss in saves,
    pooled over ``units``) and its square root; and, with ``pairs``, each pair of
    methods' MSE gap (``a`` minus ``b``; ``mse_a``, ``mse_b``) with a 95% interval and
    how sure ``a`` is better (``projection_eval.mse_bootstrap``, redrawing pitchers).
    ``units``: one row per pitcher-unit, with ``key`` columns, ``actual`` and a column
    per method."""
    from fantasy_baseball.projection_eval import mse_bootstrap

    gaps = {c: int(n) for c, n in units[["actual", *methods]].isna().sum().items() if n}
    if gaps:
        raise ValueError(f"missing values to score (rows per column): {gaps}")
    table = pd.DataFrame(
        {m: {"mse": float(((units[m] - units["actual"]) ** 2).mean())} for m in methods}
    ).T
    table["typical_miss"] = table["mse"] ** 0.5
    if not pairs:
        return table, None
    parts = pd.concat(
        [
            units[key].assign(system=m, stat="sv", sq=(units[m] - units["actual"]) ** 2, w=1.0)
            for m in methods
        ],
        ignore_index=True,
    )
    rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            r = mse_bootstrap(parts, key, a, b, ("sv",), n_boot=n_boot).iloc[0]
            rows.append({**r.to_dict(), "a": a, "b": b, "mse_a": r["a"], "mse_b": r["b"]})
    return table, pd.DataFrame(rows)


def fangraphs_shares(
    directory: Path, systems: tuple[str, ...] | None = None
) -> pd.DataFrame | None:
    """FanGraphs' projected saves and share of their team's, per pitcher (``player_id``
    index), each averaged over the same systems -- the ones that project him:
    ``sv_fg``; ``share_fg``, his saves over his FanGraphs team's total in each (0 in a
    system that has him on no team: a free agent, a minor leaguer); and ``team_fg``,
    his FanGraphs team (the one most of those systems list; NaN with none).
    ``systems``: just these (None: every ``*-pitchers*.csv`` with saves). None when no
    file has saves."""
    parts = []
    for path in sorted(directory.glob("*-pitchers*.csv")):
        name = path.name.split("-pitchers")[0]
        if systems is not None and name not in systems:
            continue
        raw = pd.read_csv(path, encoding="utf-8-sig", usecols=["Team", "SV", "MLBAMID"])
        raw = raw.assign(
            MLBAMID=pd.to_numeric(raw["MLBAMID"], errors="coerce"),
            SV=pd.to_numeric(raw["SV"], errors="coerce"),
        ).dropna(subset=["MLBAMID"])
        if raw["SV"].isna().all():
            continue
        raw = raw.drop_duplicates("MLBAMID").assign(MLBAMID=lambda d: d["MLBAMID"].astype(int))
        team = raw["Team"].astype("string").str.strip().replace("", pd.NA)
        team_sv = raw["SV"].groupby(team).transform("sum")
        share = div(raw["SV"], team_sv).where(team.notna(), 0.0).fillna(0.0)
        parts.append(
            pd.DataFrame(
                {"sv_fg": raw["SV"].to_numpy(), "share_fg": share.to_numpy(),
                 "team_fg": team.to_numpy(), "system": name},
                index=pd.Index(raw["MLBAMID"].to_numpy(), name="player_id"),
            )
        )  # fmt: skip
    if not parts:
        return None
    stacked = pd.concat(parts)
    out = stacked.groupby(level="player_id")[["sv_fg", "share_fg"]].mean()
    out["team_fg"] = stacked.groupby(level="player_id")["team_fg"].agg(
        lambda s: s.mode().iloc[0] if s.notna().any() else pd.NA
    )
    return out


def fangraphs_team_ids(fg: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """FanGraphs team abbreviation -> our ``team_id``: for each abbreviation, the team
    most of its pitchers in ``rows`` (table rows: ``player_id``, ``team_id``) are on."""
    pairs = rows[["player_id", "team_id"]].dropna().join(fg["team_fg"], on="player_id")
    pairs = pairs.dropna(subset=["team_fg"])
    out: pd.Series = pairs.groupby("team_fg")["team_id"].agg(lambda s: s.mode().iloc[0])
    return out.astype("int64")


USAGE_WINDOWS = ("std", "l30", "p1", "p3")
# From features.input_frame, per window.
_USAGE_RATES = (
    "log_g",
    "gs_share",
    "sv_bf",
    "svo_per_g",
    "gf_per_g",
    "hld_per_g",
    "bs_per_g",
    "sv_per_svo",
    "k_bf",
    "bb_bf",
    "era",
)


def usage_inputs(table: pd.DataFrame, inputs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Role inputs per pitcher-table row: per window, the pitcher net's usage and
    quality rates (``features.input_frame``: saves, save chances, games finished,
    holds, blown saves, starts, K and walk rates, ERA), his team's save chances and
    saves per game this season and last, and his share of the team's saves and save
    chances this season and last. ``inputs``: ``input_frame(table)`` if already built.
    NaN = unknown."""
    x = input_frame(table) if inputs is None else inputs
    cols = [f"{w}_{r}" for w in USAGE_WINDOWS for r in _USAGE_RATES]
    cols += [f"{w}_team_{r}" for w in ("std", "p1") for r in ("svo_per_g", "sv_per_g")]
    out = x[[*cols, "frac_season_left", "age"]].copy()
    for w in ("std", "p1"):
        out[f"{w}_share_team_sv"] = div(table[f"{w}_sv"], table[f"{w}_team_sv"])
        out[f"{w}_share_team_svo"] = div(table[f"{w}_svo"], table[f"{w}_team_svo"])
    return out.replace([np.inf, -np.inf], np.nan)


def usage_answer(table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """His share of his current team's rest-of-season saves -- his saves **for that
    team** (``ros_sv_for_team``; a pitcher traded later saves for another) over the
    team's (``ros_team_sv``) -- and its weight, the team's saves (0 where it has none)."""
    team = table["ros_team_sv"].astype(float)
    share = div(table["ros_sv_for_team"], team).clip(upper=1.0)
    return share, team.where(share.notna(), 0.0)


# Rounds of trees tried at most, and the share of training pitchers held out to pick
# how many to keep.
USAGE_MAX_ITER = 400
USAGE_VAL_FRAC = 0.15


def _usage_model(max_iter: int, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=seed,
    )  # fmt: skip


def fit_usage_shares(
    table: pd.DataFrame,
    x: pd.DataFrame,
    share: pd.Series,
    weight: pd.Series,
    test_season: int,
    seed: int = 0,
) -> pd.Series:
    """Projected share of his team's rest-of-season saves for every ``test_season``
    row, from trees fit on the complete seasons before it, weighted by the team's
    saves (a team's closer race counts by how many saves are in it). How many rounds of
    trees to keep is picked on held-out **pitchers** (``ros_training.holdout_players``):
    a random split of rows would put a pitcher's neighboring weeks on both sides."""
    train = table["season_complete"] & (table["season"] < test_season) & (weight > 0)
    held = table["player_id"].isin(holdout_players(table, train, USAGE_VAL_FRAC, seed))
    fit, val = train & ~held, train & held
    probe = _usage_model(USAGE_MAX_ITER, seed).fit(x[fit], share[fit], sample_weight=weight[fit])
    w_val = weight[val].to_numpy()
    losses = [
        np.average((p - share[val].to_numpy()) ** 2, weights=w_val)
        for p in probe.staged_predict(x[val])
    ]
    model = _usage_model(int(np.argmin(losses)) + 1, seed).fit(
        x[train], share[train], sample_weight=weight[train]
    )
    test = table["season"] == test_season
    return pd.Series(
        np.clip(model.predict(x[test]), 0.0, 1.0), index=table.index[test], name="share_usage"
    )
