"""Compare two ways of projecting a reliever's saves (#474): FanGraphs' closer calls vs
our own usage-based role model, each as share x our team-saves projection.

Per pitcher and span (preseason 2022-2026; every 2026 rest-of-season snapshot), on the
pitchers who pitched in the span and that FanGraphs projects:

* ``fg``: FanGraphs' own projected saves (preseason: Steamer + ZiPS, every year;
  snapshots: every system with saves).
* ``fg_share``: FanGraphs' share of his team's projected saves x our team projection.
* ``usage``: our usage model's share x our team projection (no FanGraphs).
* ``mix``: the two shares averaged, x our team projection.

and two checks that split the error: each share x the team's **actual** saves (no
team error), and the actual share x our team projection (no share error).
Writes data/pitcher_ros/saves/compare-s<seed>.md and prints it.

Usage:
    python scripts/compare_pitcher_saves.py
    python scripts/compare_pitcher_saves.py --seed 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.backtest import sure_text, to_markdown
from fantasy_baseball.pitcher_ros.saves import (
    fangraphs_shares,
    fit_team_saves,
    fit_usage_shares,
    score_methods,
    team_games,
    team_weeks,
    usage_answer,
    usage_inputs,
)
from fantasy_baseball.pitcher_ros.stats import load_games

TABLE = PROJECT_ROOT / "data" / "pitcher_ros" / "table.parquet"
STORE = PROJECT_ROOT / "data" / "pitch_data"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
OUT = PROJECT_ROOT / "data" / "pitcher_ros" / "saves"
SEASONS = (2022, 2023, 2024, 2025, 2026)
PRESEASON_SYSTEMS = ("steamer", "zips")  # on file every season
METHODS = ["fg", "fg_share", "usage", "mix"]
CHECKS = ["fg_share_actual_team", "usage_actual_team", "actual_share_our_team"]
# "Closer candidates": a unit where the actual saves or any method's projection is at
# least this many.
CANDIDATE_SV = 5


def _units(
    rows: pd.DataFrame,
    actual_sv: pd.Series,
    team_actual: pd.Series,
    team_proj: pd.Series,
    share_usage: pd.Series,
    fg: pd.DataFrame,
) -> pd.DataFrame:
    """One row per pitcher in ``rows`` (table rows, one per pitcher) that FanGraphs
    projects: every method's projected saves and the actual saves."""
    u = rows[["player_id", "season"]].copy()
    u["actual"] = actual_sv.reindex(u["player_id"]).fillna(0).to_numpy()
    u["team_actual"] = team_actual.to_numpy()
    u["team_proj"] = team_proj.to_numpy()
    u["share_usage"] = share_usage.to_numpy()
    u = u.join(fg, on="player_id", how="inner")
    u["fg"] = u["sv_fg"]
    u["fg_share"] = u["share_fg"] * u["team_proj"]
    u["usage"] = u["share_usage"] * u["team_proj"]
    u["mix"] = (u["share_fg"] + u["share_usage"]) / 2 * u["team_proj"]
    u["fg_share_actual_team"] = u["share_fg"] * u["team_actual"]
    u["usage_actual_team"] = u["share_usage"] * u["team_actual"]
    actual_share = (u["actual"] / u["team_actual"].where(u["team_actual"] > 0)).fillna(0)
    u["actual_share_our_team"] = actual_share.clip(upper=1) * u["team_proj"]
    return u


def _block(units: pd.DataFrame, key: list[str], title: str) -> list[str]:
    md = []
    for label, sub in (
        ("all pitchers", units),
        (
            f"closer candidates (actual or any projection >= {CANDIDATE_SV} SV)",
            units[units[["actual", *METHODS]].max(axis=1) >= CANDIDATE_SV],
        ),
    ):
        table, pairs = score_methods(sub, METHODS, key)
        checks, _ = score_methods(sub, CHECKS, key, n_boot=1)
        n_per = sub.groupby([c for c in key if c != "player_id"]).size().mean()
        md += [
            "",
            f"**{title}, {label}** ({n_per:.0f} pitchers per unit, {len(sub)} in all)",
            "",
            to_markdown(pd.concat([table, checks])[["mse", "typical_miss"]], digits=2),
            "",
            "MSE gaps (a minus b; negative = a better), 95% interval, how sure a is better:",
            "",
        ]
        md += [
            f"* {r.a} - {r.b}: {r['diff']:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] {sure_text(r.sure)}"
            for _, r in pairs.iterrows()
        ]
    return md


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    table = pd.read_parquet(TABLE)
    games = load_games(STORE)
    tg = team_games(games)
    grid = table[["season", "week", "as_of"]].drop_duplicates()
    tw = team_weeks(tg, grid)
    complete = {int(s) for s, ok in table.groupby("season")["season_complete"].all().items() if ok}
    tw_key = tw.set_index(["team_id", "season", "week"])
    row_key = pd.MultiIndex.from_frame(table[["team_id", "season", "week"]])
    team_ros_actual = pd.Series(
        tw_key["ros_sv"].reindex(row_key).to_numpy(), index=table.index, dtype=float
    )
    x = usage_inputs(table)
    share, weight = usage_answer(table, team_ros_actual)

    pre, mid = [], []
    for season in SEASONS:
        logging.info("season %s", season)
        rate = fit_team_saves(tw, season, complete)
        rate_key = tw.loc[rate.index].set_index(["team_id", "season", "week"])
        rate_key = rate_key.assign(rate=rate.to_numpy())
        rows_t = table["season"] == season
        row_rate = pd.Series(
            rate_key["rate"].reindex(row_key[rows_t.to_numpy()]).to_numpy(),
            index=table.index[rows_t],
        )
        # No team yet (a pitcher before his first MLB game): the league's average team.
        week_mean = rate_key.groupby("week")["rate"].mean()
        row_rate = row_rate.fillna(table.loc[rows_t, "week"].map(week_mean))
        share_usage = fit_usage_shares(table, x, share, weight, season, args.seed)

        # Preseason: week-0 rows; actuals and the team's games are the whole season.
        w0 = table.index[rows_t & (table["week"] == 0) & (table["ros_bf"] > 0)]
        fg = fangraphs_shares(PROJECTIONS / str(season), PRESEASON_SYSTEMS)
        if fg is not None:
            team_g = pd.Series(
                tw_key["ros_g"].reindex(row_key[table.index.get_indexer(w0)]).to_numpy(), index=w0
            )
            pre.append(
                _units(
                    table.loc[w0],
                    table.loc[w0].set_index("player_id")["ros_sv"],
                    # No team yet: the average team, as for the projection.
                    team_ros_actual[w0].fillna(team_ros_actual[w0].mean()),
                    row_rate[w0] * team_g.fillna(team_g.mean()),
                    share_usage[w0],
                    fg,
                )
            )
        # Snapshots: each pitcher's latest row on or before the date; actuals from it.
        root = PROJECTIONS / str(season) / "rest_of_season"
        for snap_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
            snap = pd.Timestamp(date.fromisoformat(snap_dir.name))
            fg = fangraphs_shares(snap_dir)
            if fg is None:
                continue
            after = games[(games["season"] == season) & (games["game_date"] >= snap)]
            played = after.groupby("player_id")["bf"].sum()
            known = table.loc[rows_t & (table["as_of"] <= snap)]
            latest = known.loc[known.groupby("player_id")["as_of"].idxmax()]
            latest = latest[latest["player_id"].isin(played.index[played > 0])]
            if latest.empty:
                continue
            tga = tg[(tg["season"] == season) & (tg["game_date"] >= snap)]
            team_left = tga.groupby("team_id").size()
            team_sv_left = tga.groupby("team_id")["sv"].sum()
            g_left = latest["team_id"].map(team_left)
            proj = row_rate[latest.index] * g_left.fillna(team_left.mean())
            u = _units(
                latest,
                after.groupby("player_id")["sv"].sum(),
                latest["team_id"].map(team_sv_left).fillna(0),
                proj,
                share_usage[latest.index],
                fg,
            )
            mid.append(u.assign(snapshot=snap.date().isoformat()))

    md = ["### Saves: FanGraphs' closer calls vs our usage model (#474)", ""]
    md.append(
        "Squared miss in saves over the span (preseason: the season; snapshot: the rest "
        "of it), per pitcher who pitched in it and that FanGraphs projects. `fg` = "
        "FanGraphs' saves; `fg_share` / `usage` / `mix` = that share x our team-saves "
        "projection. Checks: `*_actual_team` uses the team's actual saves (no team "
        "error); `actual_share_our_team` the actual share (no share error)."
    )
    if pre:
        md += _block(
            pd.concat(pre, ignore_index=True), ["player_id", "season"], "Preseason 2022-2026"
        )
    if mid:
        md += _block(
            pd.concat(mid, ignore_index=True),
            ["player_id", "season", "snapshot"],
            "2026 rest-of-season snapshots",
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"compare-s{args.seed}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
