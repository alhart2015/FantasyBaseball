"""Keeper report: every rostered player's projected next-season SGP and VAR
(analysis.keeper_board) -- my whole roster, and the top N on every other team.

Rates: our nets' preseason predictions for --season (hitters: --hitter-run's seeds
averaged; pitchers: --pitcher-run's seed-averaged predictions). Playing time:
--hitter-pt / --pitcher-pt, FanGraphs projection CSVs (Depth Charts or Steamer for the
season: default data/projections/<season>/steamer-*.csv). ZiPS works but is a weak PT
source: it gives minor leaguers full playing time, and its 2027 file is a March 2026
snapshot. VAR is over the measured waiver lines every model uses (#521). Our old starters' rates are corrected for
age (pitcher_ros.aging, #503; fit on --aging-seasons -- default every season from 2022
before --season -- with --aging-run's held-out projections, role from the --pitcher-pt
system's past files, or Steamer's when that system has none; --no-aging to skip). The
playing time is the PT file's as published -- its old pitchers' innings aren't cut --
except that a reliever's innings are capped (keeper_board.RP_IP_CAP).
Rosters: the live store on Upstash
(cache:roster, cache:opp_rosters, plus anyone else on the latest weekly_rosters_history
snapshot they were built from; creds from .env).

Writes data/keeper_reports/<season>-<name>.md and prints it.

    python scripts/keeper_report.py --season 2027 --name steamer
    python scripts/keeper_report.py --season 2027 --name zips \\
        --hitter-pt data/projections/2027/zips-hitters.csv \\
        --pitcher-pt data/projections/2027/zips-pitchers.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
# Read the live store (Upstash), not the local mirror; set before any KV read.
os.environ["RENDER"] = "true"

import pandas as pd
from backtest_pitcher_aging import aging_rows

from fantasy_baseball.analysis import keeper_board as kb
from fantasy_baseball.config import load_config
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.data.kv_store import get_kv
from fantasy_baseball.data.redis_store import _PLAYER_SUFFIX_RE, get_latest_weekly_rosters
from fantasy_baseball.hitter_ros.backtest import league_denominators
from fantasy_baseball.hitter_ros.evaluate import load_fangraphs_hitters
from fantasy_baseball.keepers.positions import load_positions
from fantasy_baseball.models.positions import Position, UnknownPositions, player_type_for_positions
from fantasy_baseball.pitcher_ros import aging
from fantasy_baseball.pitcher_ros.evaluate import read_fangraphs_pitchers
from fantasy_baseball.pt_model import range_inputs as ri
from fantasy_baseball.sgp.replacement import league_replacement_levels
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.web.season_data import read_cache_with_meta

CONFIG = PROJECT_ROOT / "config" / "league.yaml"
PEOPLE = PROJECT_ROOT / "data" / "cache" / "keeper_skills" / "mlb_people_all_2000_2026.csv"
HITTER_RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"
PITCHER_RUNS = PROJECT_ROOT / "data" / "pitcher_ros" / "runs"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
OUT = PROJECT_ROOT / "data" / "keeper_reports"
# The first season with held-out pitcher projections to fit the age corrections on.
FIRST_AGING_SEASON = 2022


def fangraphs_teams(path: Path) -> pd.Series:
    """A FanGraphs CSV's MLB team code by MLBAM id (first of a repeated id).
    ``read_fangraphs_pitchers`` reads any FanGraphs CSV that way, hitters' too."""
    raw = read_fangraphs_pitchers(path)
    return raw.set_index("MLBAMID")["Team"]


def _roster_key(name: str, player_type: str) -> tuple[str, str]:
    return normalize_name(_PLAYER_SUFFIX_RE.sub("", name)), str(player_type)


def rosters(team: str) -> tuple[list[kb.Rostered], str, list[str]]:
    """Every rostered player, when the rosters were written, and rows that couldn't
    be read. cache:opp_rosters keeps only the players the last refresh could project
    (prospects and the season-long injured fall out), so anyone else on the latest
    roster snapshot -- the one those blobs were built from -- is added from it, with
    no MLB team."""
    mine, mine_meta = read_cache_with_meta(CacheKey.ROSTER)
    opp, opp_meta = read_cache_with_meta(CacheKey.OPP_ROSTERS)
    if not isinstance(mine, list) or not isinstance(opp, dict):
        raise SystemExit("no rosters in the live store (cache:roster, cache:opp_rosters)")
    entries = [(team, p) for p in mine] + [(t, p) for t, ps in opp.items() for p in ps]
    out = [
        kb.Rostered(
            team=t,
            name=p["name"],
            player_type=p["player_type"],
            positions=tuple(p.get("positions") or ()),
            mlb_team=p.get("team") or "",
        )
        for t, p in entries
    ]
    seen = {_roster_key(r.name, r.player_type) for r in out}
    unread = []
    for row in get_latest_weekly_rosters(get_kv()):
        name, owner = row.get("player_name"), row.get("team")
        if not name or not owner:
            continue
        try:
            player_type = player_type_for_positions(str(row.get("positions", "")))
        except UnknownPositions:
            unread.append(f"{name} ({owner}): positions {row.get('positions')!r} unreadable")
            continue
        key = _roster_key(str(name), player_type)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            kb.Rostered(
                team=str(owner),
                name=_PLAYER_SUFFIX_RE.sub("", str(name)),
                player_type=str(player_type),
                positions=tuple(p.value for p in Position.parse_list(str(row["positions"]))),
                mlb_team="",
            )
        )

    def stamp(meta: dict) -> str:
        return str(meta.get("_written_at") or "?")[:10]

    return out, f"{stamp(mine_meta)} / {stamp(opp_meta)}", unread


def line(r: pd.Series) -> str:
    if r["kind"] == kb.HITTER:
        return (
            f"{r['pa']:.0f} PA: {r['r']:.0f} R, {r['hr']:.0f} HR, {r['rbi']:.0f} RBI, "
            f"{r['sb']:.0f} SB, {r['avg']:.3f}"
        )
    return (
        f"{r['ip']:.0f} IP: {r['k']:.0f} K, {r['w']:.0f} W, {r['sv']:.0f} SV, "
        f"{r['era']:.2f} ERA, {r['whip']:.2f} WHIP"
    )


def table(df: pd.DataFrame) -> list[str]:
    md = ["| Player | Pos | VAR | SGP | Line |", "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        pos = r["role"] if r["kind"] == kb.PITCHER else "/".join(r["positions"])
        md.append(f"| {r['name']} | {pos} | {r['var']:.1f} | {r['sgp']:.1f} | {line(r)} |")
    return md


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hitter-pt", type=Path)
    parser.add_argument("--pitcher-pt", type=Path)
    parser.add_argument("--hitter-run", default="proj2027seq")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--pitcher-run", default="proj-2027-sv")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--no-aging", action="store_true", help="skip the pitcher age corrections")
    parser.add_argument("--aging-run", default="035-saves-net")
    parser.add_argument(
        "--aging-seasons", type=int, nargs="+", help="default: 2022 through --season - 1"
    )
    args = parser.parse_args()
    hit_pt = args.hitter_pt or PROJECTIONS / str(args.season) / "steamer-hitters.csv"
    pit_pt = args.pitcher_pt or PROJECTIONS / str(args.season) / "steamer-pitchers.csv"
    for path in (hit_pt, pit_pt):
        if not path.exists():
            parser.error(
                f"{path} not found: download it from FanGraphs, or pass --hitter-pt / --pitcher-pt"
            )

    config = load_config(CONFIG)
    denoms = league_denominators(CONFIG)
    people = pd.read_csv(PEOPLE, usecols=["id", "fullName", "birthDate", "mlbDebutDate"])
    names = people.set_index("id")["fullName"]
    slots = load_positions()
    hit_rates = ri.load_rates(HITTER_RUNS / f"{args.hitter_run}-s{s}" for s in args.seeds)
    pit_rates = ri.load_pitcher_rates(PITCHER_RUNS / args.pitcher_run)
    for label, rates in (("hitter", hit_rates), ("pitcher", pit_rates)):
        if args.season not in rates.index.get_level_values("season"):
            raise SystemExit(f"the {label} rates have no {args.season} rows")

    def season_rates(rates: pd.DataFrame) -> pd.DataFrame:
        r = rates.xs(args.season, level="season")
        return r.rename(columns={c: c.removeprefix("p_") for c in r.columns})

    pit_season = season_rates(pit_rates)
    pit_pt_frame = ri.load_pitcher_pt(pit_pt)
    aging_note = "no age corrections"
    if not args.no_aging:
        given = args.aging_seasons
        seasons = sorted(given if given is not None else range(FIRST_AGING_SEASON, args.season))
        # Fitting on --season or later would read the actuals being projected.
        if len(seasons) < 2 or seasons[-1] >= args.season:
            parser.error(f"--aging-seasons: want 2+ seasons, all before --season {args.season}")
        # Role is read the way it is applied: from the PT system's GS / G -- or, when
        # that system has no past files (Depth Charts), from Steamer's, the role the
        # backtest scored; role is all the fit reads from them.
        system = pit_pt.name.split("-pitchers")[0]
        if not all(any((PROJECTIONS / str(s)).glob(f"{system}-pitchers*.csv")) for s in seasons):
            system = "steamer"
        birth = ri.player_info(people)["birth"]
        rows = aging_rows(seasons, args.aging_run, system, birth)
        rate_fit = aging.fit_rates(rows)
        age = ri.season_age(birth, args.season)
        roles = aging.role(pit_pt_frame["gs_share"])
        pit_season = aging.adjust_rates(pit_season, age, roles, rate_fit)
        aging_note = (
            f"our starters' rates age-corrected (fit on {seasons[0]}-{seasons[-1]}, role from "
            f"{system}): " + "; ".join(aging.describe(rate_fit, sorted(aging.CORRECTED)))
        )
    hit_ids = hit_rates.xs(args.season, level="season").index
    positions = {int(i): slots.get(normalize_name(str(names.get(i, ""))), []) for i in hit_ids}
    board = pd.concat(
        [
            kb.hitter_board(
                season_rates(hit_rates), load_fangraphs_hitters(hit_pt), positions, denoms
            ),
            kb.pitcher_board(pit_season, pit_pt_frame, denoms),
        ]
    )
    board["name"] = names.reindex(board.index).fillna("?").to_numpy()
    levels = league_replacement_levels(config.sgp_overrides)
    teams = pd.concat([fangraphs_teams(hit_pt), fangraphs_teams(pit_pt)])
    teams = teams[~teams.index.duplicated()]
    rostered, written, unread = rosters(config.team_name)
    matched, problems, team_checks = kb.match_rostered(
        rostered, board, names.to_dict(), teams.to_dict()
    )
    problems = unread + problems
    if matched.empty:
        raise SystemExit("no rostered player matched the board:\n" + "\n".join(problems))
    matched["var"] = kb.var(matched["sgp"], matched["positions"], levels, matched["role"])

    md = [
        f"## {args.season} keeper report ({args.name})",
        "",
        f"Rates: hitters {args.hitter_run} (seeds {args.seeds}), pitchers {args.pitcher_run}. "
        f"Playing time: {hit_pt.name}, {pit_pt.name} (relievers capped at {kb.RP_IP_CAP:g} IP). "
        f"Rosters written {written}.",
        "",
        f"Pitchers: {aging_note}.",
        "",
        "Replacement (the measured waiver line at each position):",
        "",
        *[f"- {pos}: {sgp:.1f} SGP" for pos, sgp in levels.items()],
    ]
    if problems:
        md += ["", "Left out:", "", *[f"- {p}" for p in problems]]
    if team_checks:
        md += [
            "",
            f"Matched by name, MLB team differs from {hit_pt.name} / {pit_pt.name} (usually a "
            "trade since the file was made; check any you don't recognize):",
            "",
            "- " + "; ".join(team_checks),
        ]
    me = config.team_name
    md += ["", f"### {me}: everyone, by VAR", ""]
    md += table(matched[matched["team"] == me].sort_values("var", ascending=False))
    for team in sorted(t for t in matched["team"].unique() if t != me):
        md += ["", f"### {team}: top {args.top} by VAR", ""]
        md += table(matched[matched["team"] == team].nlargest(args.top, "var"))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.season}-{args.name}.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
