"""Compare two ways of projecting a reliever's saves (#474): FanGraphs' closer calls vs
our own usage-based role model, each as share x our team-saves projection.

Per pitcher and span (preseason 2022-2026; every 2026 rest-of-season snapshot), on the
pitchers who pitched in the span and that FanGraphs projects:

* ``fg``: FanGraphs' own projected saves (preseason: Steamer + ZiPS, every year;
  snapshots: every system with saves).
* ``fg_share``: FanGraphs' share of their team's projected saves x our projection for
  that team (their team, matched to ours by ``saves.fangraphs_team_ids``).
* ``usage``: our usage model's share x our projection for his team (no FanGraphs).
* ``mix``: the average of ``fg_share`` and ``usage``.

Our team projection is saves per remaining game x the games left on the stored
schedule. Two checks split the error: each share x the team's **actual** saves (no
team error), and his actual share x our team projection (no share error). A pitcher
with no team yet (before his first MLB game) gets the average team's numbers.
Writes data/pitcher_ros/saves/compare-s<seed>.md and prints it, and the units (one row
per pitcher and span, every method) to units-pre-s<seed>.parquet / units-mid-s<seed>.parquet
there for scripts/score_pitcher_net_saves.py.

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

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.backtest import sure_text, to_markdown
from fantasy_baseball.pitch_data.store import connect
from fantasy_baseball.pitcher_ros.features import input_frame
from fantasy_baseball.pitcher_ros.saves import (
    TEAM_KEY,
    fangraphs_shares,
    fangraphs_team_ids,
    fit_team_saves,
    fit_usage_shares,
    scheduled_games_left,
    score_methods,
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
    sv_for_team: pd.Series,
    team_proj: pd.Series,
    team_actual: pd.Series,
    share_usage: pd.Series,
    fg: pd.DataFrame,
    fg_ids: pd.Series,
) -> pd.DataFrame:
    """One row per pitcher in ``rows`` (table rows, one per pitcher) that FanGraphs
    projects: every method's projected saves, the checks and the actual saves.
    ``actual_sv`` / ``sv_for_team``: his saves over the span (all, and for his team),
    by ``player_id``. ``team_proj`` / ``team_actual``: our projection and the actual
    team saves over the span, by ``team_id``."""
    proj_avg, actual_avg = team_proj.mean(), team_actual.mean()
    u = rows[["player_id", "season", "team_id"]].join(fg, on="player_id", how="inner")
    u["share_usage"] = share_usage.reindex(u.index)
    u["actual"] = u["player_id"].map(actual_sv).fillna(0.0)
    fg_team = u["team_fg"].map(fg_ids).fillna(u["team_id"])
    u["proj_his"] = u["team_id"].map(team_proj).fillna(proj_avg)
    u["actual_his"] = u["team_id"].map(team_actual).fillna(actual_avg)
    proj_fg = fg_team.map(team_proj).fillna(proj_avg)
    actual_fg = fg_team.map(team_actual).fillna(actual_avg)
    u["fg"] = u["sv_fg"]
    u["fg_share"] = u["share_fg"] * proj_fg
    u["usage"] = u["share_usage"] * u["proj_his"]
    u["mix"] = (u["fg_share"] + u["usage"]) / 2
    u["fg_share_actual_team"] = u["share_fg"] * actual_fg
    u["usage_actual_team"] = u["share_usage"] * u["actual_his"]
    for_team = u["player_id"].map(sv_for_team).fillna(0.0)
    actual_share = (for_team / u["actual_his"].where(u["actual_his"] > 0)).fillna(0.0)
    u["actual_share_our_team"] = actual_share.clip(upper=1.0) * u["proj_his"]
    return u.drop(columns=["team_id", "team_fg"])


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
        checks, _ = score_methods(sub, CHECKS, key, pairs=False)
        assert pairs is not None
        n_per = sub.groupby([c for c in key if c != "player_id"]).size().mean()
        md += [
            "",
            f"**{title}, {label}** ({n_per:.0f} pitchers per unit, {len(sub)} in all)",
            "",
            to_markdown(pd.concat([table, checks]), digits=2),
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
    missing = {"team_games_left", "ros_team_sv", "ros_sv_for_team"} - set(table.columns)
    if missing:
        parser.error(f"{TABLE} predates {sorted(missing)}; run scripts/build_pitcher_ros_table.py")
    if table["team_games_left"].isna().all():
        parser.error("no schedule in the store: fill it with scripts/fetch_pitch_data.py")
    games = load_games(STORE)
    conn = connect(STORE)
    try:
        schedule = conn.execute("SELECT * FROM games").df()
    finally:
        conn.close()
    x = usage_inputs(table, input_frame(table))
    share, weight = usage_answer(table)
    tw = team_weeks(table)

    pre, mid = [], []
    for season in SEASONS:
        logging.info("season %s", season)
        rates = tw.loc[:, [*TEAM_KEY, "as_of", "team_games_left", "ros_team_sv"]]
        rates = rates[rates["season"] == season].assign(rate=fit_team_saves(tw, season))
        rows_t = table["season"] == season
        share_usage = fit_usage_shares(table, x, share, weight, season, args.seed)

        # Preseason: week-0 rows; the whole season's saves and the team's games left.
        w0 = table[rows_t & (table["week"] == 0) & (table["ros_bf"] > 0)]
        fg = fangraphs_shares(PROJECTIONS / str(season), PRESEASON_SYSTEMS)
        if fg is not None:
            r0 = rates[rates["week"] == 0].set_index("team_id")
            pre.append(
                _units(
                    w0,
                    w0.set_index("player_id")["ros_sv"],
                    w0.set_index("player_id")["ros_sv_for_team"],
                    r0["rate"] * r0["team_games_left"],
                    r0["ros_team_sv"],
                    share_usage,
                    fg,
                    fangraphs_team_ids(fg, w0),
                )
            )
        # Snapshots: each pitcher's latest row on or before the date; spans from it.
        root = PROJECTIONS / str(season) / "rest_of_season"
        for snap_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
            snap = pd.Timestamp(date.fromisoformat(snap_dir.name))
            fg = fangraphs_shares(snap_dir)
            if fg is None:
                continue
            after = games[(games["season"] == season) & (games["game_date"] >= snap)]
            pitched = after.groupby("player_id")["bf"].sum()
            known = table.loc[rows_t & (table["as_of"] <= snap)]
            latest = known.loc[known.groupby("player_id")["as_of"].idxmax()]
            latest = latest[latest["player_id"].isin(pitched.index[pitched > 0])]
            if latest.empty:
                continue
            # Each team's rate from its latest week on or before the date.
            r = rates[rates["as_of"] <= snap]
            r = r.loc[r.groupby("team_id")["as_of"].idxmax()].set_index("team_id")["rate"]
            left = scheduled_games_left(schedule, season, snap)
            his_team = latest.set_index("player_id")["team_id"]
            for_team = after[after["team_id"] == after["player_id"].map(his_team)]
            u = _units(
                latest,
                after.groupby("player_id")["sv"].sum(),
                for_team.groupby("player_id")["sv"].sum(),
                (r * left.reindex(r.index)).dropna(),
                after.groupby("team_id")["sv"].sum(),
                share_usage,
                fg,
                fangraphs_team_ids(fg, latest),
            )
            mid.append(u.assign(snapshot=snap.date().isoformat()))

    OUT.mkdir(parents=True, exist_ok=True)
    # A span with no units removes the file, so an old run's units can't be read as these.
    for name, parts in (("pre", pre), ("mid", mid)):
        path = OUT / f"units-{name}-s{args.seed}.parquet"
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(path)
        else:
            path.unlink(missing_ok=True)
    md = ["### Saves: FanGraphs' closer calls vs our usage model (#474)", ""]
    md.append(
        "Squared miss in saves over the span (preseason: the season; snapshot: the rest "
        "of it), per pitcher who pitched in it and that FanGraphs projects. `fg` = "
        "FanGraphs' saves; `fg_share` / `usage` = that share x our projection of the "
        "team's saves (games left on the schedule x projected saves per game); `mix` = "
        "their average. Checks: `*_actual_team` uses the team's actual saves (no team "
        "error); `actual_share_our_team` his actual share (no share error)."
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
    sys.exit(main())
