"""Saves as a reliever's share of his team's saves x the team's saves (#474).

Saves come from role: who closes, and how many save chances the team gets. So a
pitcher's projected saves over a span = his projected **share** of his team's saves x
the team's projected **total**.

* Team total (:func:`team_weeks`, :func:`fit_team_saves`): our own model, saves per
  remaining game from the team's saves and wins so far and last season, fit on every
  complete team-season in the store (2008 on) before the projected one, times the
  team's remaining games.
* Share, two ways, to compare:

  * ``fg``: FanGraphs' (:func:`fangraphs_shares`). Their projected saves already carry
    their closer calls (news, depth charts); his share is his projected saves over his
    FanGraphs team's. Only where FanGraphs files exist (preseason 2022 on, 2026's
    rest-of-season snapshots).
  * ``usage``: ours, with no FanGraphs (:func:`fit_usage_shares`): gradient-boosted
    trees on his recent and past usage (saves, save chances, games finished, holds,
    his share of his team's saves), fit on every complete season before the projected
    one, answer = his share of his team's rest-of-season saves.

Everything is dated like the training table: inputs only from games before the date.
The team's remaining games are its scheduled games, read off the games it played.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Games of last season's rate added to this season's team rates, so a team 10 games in
# isn't read off 10 games.
TEAM_PRIOR_GAMES = 20
TEAM_FEATURES = ("p1_sv_per_g", "p1_wpct", "std_sv_per_g", "std_wpct", "lg_p1_sv_per_g")


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team and game: ``team_id``, ``game_pk``, ``game_date``, ``season``,
    and the team's ``sv``, ``svo`` and ``w`` (1 for a win) in it. ``games``: pitcher-game
    lines as from ``stats.load_games``."""
    out: pd.DataFrame = (
        games.groupby(["team_id", "game_pk"], as_index=False)
        .agg(game_date=("game_date", "max"), season=("season", "max"), sv=("sv", "sum"),
             svo=("svo", "sum"), w=("w", "sum"))
    )  # fmt: skip
    return out


def team_weeks(tg: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """One row per team and grid date (``grid``: ``season``, ``week``, ``as_of``): the
    team's saves, save chances, wins and games before the date (``std_``) and on or
    after it (``ros_``), and last season's (``p1_``). ``tg``: :func:`team_games`."""
    cols = ("sv", "svo", "w")
    rows = []
    season_totals = tg.groupby(["team_id", "season"])[list(cols)].sum()
    season_games = tg.groupby(["team_id", "season"]).size()
    for (season, week, as_of), _ in grid.groupby(["season", "week", "as_of"]):
        g = tg[tg["season"] == season]
        before = g[g["game_date"] < as_of].groupby("team_id")
        after = g[g["game_date"] >= as_of].groupby("team_id")
        teams = pd.Index(g["team_id"].unique(), name="team_id")
        frame = pd.DataFrame(index=teams)
        for prefix, grp in (("std", before), ("ros", after)):
            sums = grp[list(cols)].sum().reindex(teams, fill_value=0)
            for c in cols:
                frame[f"{prefix}_{c}"] = sums[c]
            frame[f"{prefix}_g"] = grp.size().reindex(teams, fill_value=0)
        for c in cols:
            frame[f"p1_{c}"] = [season_totals[c].get((t, season - 1), np.nan) for t in teams]
        frame["p1_g"] = [season_games.get((t, season - 1), np.nan) for t in teams]
        rows.append(frame.reset_index().assign(season=season, week=week, as_of=as_of))
    return pd.concat(rows, ignore_index=True)


def _team_features(tw: pd.DataFrame) -> pd.DataFrame:
    sums = tw.groupby("season")[["p1_sv", "p1_g"]].sum()
    lg = sums["p1_sv"] / sums["p1_g"].where(sums["p1_g"] > 0)
    lg_p1 = tw["season"].map(lg)
    p1_sv = (tw["p1_sv"] / tw["p1_g"]).fillna(lg_p1)
    p1_w = (tw["p1_w"] / tw["p1_g"]).fillna(0.5)
    k = TEAM_PRIOR_GAMES
    return pd.DataFrame(
        {
            "p1_sv_per_g": p1_sv,
            "p1_wpct": p1_w,
            "std_sv_per_g": (tw["std_sv"] + k * p1_sv) / (tw["std_g"] + k),
            "std_wpct": (tw["std_w"] + k * p1_w) / (tw["std_g"] + k),
            "lg_p1_sv_per_g": lg_p1,
        },
        index=tw.index,
    )[list(TEAM_FEATURES)]


def fit_team_saves(tw: pd.DataFrame, test_season: int, complete: set[int]) -> pd.Series:
    """Projected saves per remaining game for every team-week of ``test_season``: a
    linear model fit (weighted by remaining games) on the team-weeks of ``complete``
    seasons before it. Times the team's remaining games, its projected saves."""
    x = _team_features(tw)
    train = tw["season"].isin([s for s in complete if s < test_season]) & (tw["ros_g"] > 0)
    train &= x.notna().all(axis=1)
    if not train.any():
        raise ValueError(
            f"no team-week before {test_season} to fit on (a complete season with the "
            "season before it in the store)"
        )
    model = LinearRegression().fit(
        x[train],
        tw.loc[train, "ros_sv"] / tw.loc[train, "ros_g"],
        sample_weight=tw.loc[train, "ros_g"],
    )
    test = tw["season"] == test_season
    rate = np.clip(model.predict(x[test].fillna(x[train].mean())), 0, None)
    return pd.Series(rate, index=tw.index[test], name="team_sv_per_g")


def score_methods(
    units: pd.DataFrame, methods: list[str], key: list[str], n_boot: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Saves projections scored on actual saves: per method, MSE (squared miss in saves,
    pooled over ``units``) and its square root; and each pair of methods' MSE gap
    with a 95% interval and how sure (``projection_eval.mse_bootstrap``, redrawing
    pitchers). ``units``: one row per pitcher-unit, with ``key`` columns, ``actual``
    and a column per method."""
    from fantasy_baseball.projection_eval import mse_bootstrap

    gaps = {c: int(n) for c, n in units[["actual", *methods]].isna().sum().items() if n}
    if gaps:
        raise ValueError(f"missing values to score (rows per column): {gaps}")
    table = pd.DataFrame(
        {m: {"mse": float(((units[m] - units["actual"]) ** 2).mean())} for m in methods}
    ).T
    table["typical_miss"] = table["mse"] ** 0.5
    table["n"] = len(units)
    parts = pd.concat(
        [
            units[key].assign(system=m, stat="sv", sq=(units[m] - units["actual"]) ** 2, w=1.0)
            for m in methods
        ],
        ignore_index=True,
    )
    pairs = []
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            r = mse_bootstrap(parts, key, a, b, ("sv",), n_boot=n_boot).iloc[0]
            pairs.append({**r.to_dict(), "a": a, "b": b, "mse_a": r["a"], "mse_b": r["b"]})
    return table, pd.DataFrame(pairs)


def fangraphs_shares(
    directory: Path, systems: tuple[str, ...] | None = None
) -> pd.DataFrame | None:
    """FanGraphs' projected saves and share of their team's, per pitcher (``player_id``
    index): ``sv_fg`` (the average over the systems that project him) and
    ``share_fg`` (each system's saves over his FanGraphs team's total, averaged the
    same way). ``systems``: just these (None: every ``*-pitchers*.csv`` with saves).
    None when no file has saves."""
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
        on_team = raw[raw["Team"].notna() & (raw["Team"].astype(str).str.strip() != "")]
        team_sv = on_team.groupby("Team")["SV"].transform("sum")
        share = (on_team["SV"] / team_sv.where(team_sv > 0)).fillna(0.0)
        parts.append(
            pd.DataFrame(
                {"sv_fg": raw.set_index("MLBAMID")["SV"],
                 "share_fg": pd.Series(share.to_numpy(), index=on_team["MLBAMID"])}
            ).assign(system=name)
        )  # fmt: skip
    if not parts:
        return None
    stacked = pd.concat(parts)
    stacked.index.name = "player_id"
    out = stacked.groupby(level="player_id")[["sv_fg", "share_fg"]].mean()
    # On no FanGraphs team (a free agent, a minor leaguer) in any system: no share of
    # any team's saves.
    out["share_fg"] = out["share_fg"].fillna(0.0)
    return out


USAGE_WINDOWS = ("std", "l14", "l30", "p1", "p3")


def usage_inputs(table: pd.DataFrame) -> pd.DataFrame:
    """Role inputs per pitcher-table row: per window, saves, save chances, games
    finished, holds, blown saves and starts per game, and games; his share of his
    current team's saves and save chances this season and last; how good he has been
    (K and walk rates, ERA); and when in the season it is. NaN = unknown."""

    def div(a: pd.Series, b: pd.Series) -> pd.Series:
        return a.astype(float) / b.astype(float).where(b != 0)

    cols: dict[str, pd.Series] = {}
    for w in USAGE_WINDOWS:
        g = table[f"{w}_g"]
        for c in ("sv", "svo", "gf", "hld", "bs", "gs"):
            cols[f"{w}_{c}_per_g"] = div(table[f"{w}_{c}"], g)
        cols[f"{w}_log_g"] = np.log1p(g.astype(float))
        cols[f"{w}_sv_per_svo"] = div(table[f"{w}_sv"], table[f"{w}_svo"])
    for w in ("std", "p1"):
        cols[f"{w}_share_team_sv"] = div(table[f"{w}_sv"], table[f"{w}_team_sv"])
        cols[f"{w}_share_team_svo"] = div(table[f"{w}_svo"], table[f"{w}_team_svo"])
        cols[f"{w}_team_svo_per_g"] = div(table[f"{w}_team_svo"], table[f"{w}_team_games"])
    for w in ("std", "p1", "p3"):
        cols[f"{w}_k_bf"] = div(table[f"{w}_k"], table[f"{w}_bf"])
        cols[f"{w}_bb_bf"] = div(table[f"{w}_bb"], table[f"{w}_bf"])
        cols[f"{w}_era"] = 27 * div(table[f"{w}_er"], table[f"{w}_outs"])
    cols["frac_season_left"] = table["frac_season_left"].astype(float)
    cols["age"] = table["age"].astype(float)
    return pd.DataFrame(cols, index=table.index).replace([np.inf, -np.inf], np.nan)


def usage_answer(table: pd.DataFrame, team_ros_sv: pd.Series) -> tuple[pd.Series, pd.Series]:
    """His share of his current team's rest-of-season saves (capped at 1: a pitcher
    traded mid-span can out-save his old team) and its weight, the team's saves (0
    where the team has none). ``team_ros_sv``: per table row, his team's actual
    rest-of-season saves."""
    team = team_ros_sv.astype(float)
    share = (table["ros_sv"].astype(float) / team.where(team > 0)).clip(upper=1.0)
    return share, team.where(share.notna(), 0.0).fillna(0.0)


def fit_usage_shares(
    table: pd.DataFrame,
    x: pd.DataFrame,
    share: pd.Series,
    weight: pd.Series,
    test_season: int,
    seed: int = 0,
) -> pd.Series:
    """Projected share of his team's rest-of-season saves for every ``test_season``
    row, from trees fit on the complete seasons before it (weighted by the team's
    saves, so a team's closer race counts by how many saves are in it)."""
    train = table["season_complete"] & (table["season"] < test_season) & (weight > 0)
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=seed,
    )  # fmt: skip
    model.fit(x[train], share[train], sample_weight=weight[train])
    test = table["season"] == test_season
    return pd.Series(
        np.clip(model.predict(x[test]), 0.0, 1.0), index=table.index[test], name="share_usage"
    )
