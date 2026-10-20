"""Keeper report: every rostered player's projected next-season SGP and VAR
(analysis.keeper_board) -- my whole roster, and the top N on every other team.

Rates: our nets' preseason predictions for --season (hitters: --hitter-run's seeds
averaged; pitchers: --pitcher-run's seed-averaged predictions). Playing time:
--hitter-pt / --pitcher-pt, FanGraphs projection CSVs (Depth Charts or Steamer for the
season: default data/projections/<season>/steamer-*.csv). ZiPS works but is a weak PT
source: it gives minor leaguers full playing time, which lifts replacement level, and
its 2027 file is a March 2026 snapshot. Rosters: the live store on Upstash
(cache:roster, cache:opp_rosters; creds from .env).

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
# Read the live store (Upstash), not the local mirror; set before any KV read.
os.environ["RENDER"] = "true"

import pandas as pd

from fantasy_baseball.analysis import keeper_board as kb
from fantasy_baseball.config import load_config
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.hitter_ros.backtest import league_denominators
from fantasy_baseball.hitter_ros.evaluate import load_fangraphs_hitters
from fantasy_baseball.keepers.positions import load_positions
from fantasy_baseball.pitcher_ros.evaluate import read_fangraphs_pitchers
from fantasy_baseball.pt_model import range_inputs as ri
from fantasy_baseball.utils.constants import compute_starters_per_position
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.web.season_data import read_cache_with_meta

CONFIG = PROJECT_ROOT / "config" / "league.yaml"
PEOPLE = PROJECT_ROOT / "data" / "cache" / "keeper_skills" / "mlb_people_all_2000_2026.csv"
HITTER_RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"
PITCHER_RUNS = PROJECT_ROOT / "data" / "pitcher_ros" / "runs"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
OUT = PROJECT_ROOT / "data" / "keeper_reports"


def fangraphs_teams(path: Path) -> pd.Series:
    """A FanGraphs CSV's MLB team code by MLBAM id (first of a repeated id)."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw = raw.assign(MLBAMID=pd.to_numeric(raw["MLBAMID"], errors="coerce"))
    raw = raw[raw["MLBAMID"].notna()].drop_duplicates("MLBAMID", keep="first")
    return raw.set_index(raw["MLBAMID"].astype(int))["Team"]


def rosters() -> tuple[list[kb.Rostered], str]:
    """Every rostered player, and when the rosters were written."""
    mine, mine_meta = read_cache_with_meta(CacheKey.ROSTER)
    opp, opp_meta = read_cache_with_meta(CacheKey.OPP_ROSTERS)
    if not isinstance(mine, list) or not isinstance(opp, dict):
        raise SystemExit("no rosters in the live store (cache:roster, cache:opp_rosters)")
    team = load_config(CONFIG).team_name
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
    written = f"{mine_meta.get('_written_at', '?')[:10]} / {opp_meta.get('_written_at', '?')[:10]}"
    return out, written


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
    starters = compute_starters_per_position(config.roster_slots, config.num_teams)
    names = pd.read_csv(PEOPLE, usecols=["id", "fullName"]).set_index("id")["fullName"]
    slots = load_positions()
    hit_rates = ri.load_rates(HITTER_RUNS / f"{args.hitter_run}-s{s}" for s in args.seeds)
    pit_rates = ri.load_pitcher_rates(PITCHER_RUNS / args.pitcher_run)
    for label, rates in (("hitter", hit_rates), ("pitcher", pit_rates)):
        if args.season not in rates.index.get_level_values("season"):
            raise SystemExit(f"the {label} rates have no {args.season} rows")

    def season_rates(rates: pd.DataFrame) -> pd.DataFrame:
        r = rates.xs(args.season, level="season")
        return r.rename(columns={c: c.removeprefix("p_") for c in r.columns})

    hit_ids = hit_rates.xs(args.season, level="season").index
    positions = {int(i): slots.get(normalize_name(str(names.get(i, ""))), []) for i in hit_ids}
    board = pd.concat(
        [
            kb.hitter_board(
                season_rates(hit_rates), load_fangraphs_hitters(hit_pt), positions, denoms
            ),
            kb.pitcher_board(season_rates(pit_rates), ri.load_pitcher_pt(pit_pt), denoms),
        ]
    )
    board["name"] = names.reindex(board.index).fillna("?").to_numpy()
    levels, repl = kb.replacement(board, starters)
    teams = pd.concat([fangraphs_teams(hit_pt), fangraphs_teams(pit_pt)])
    teams = teams[~teams.index.duplicated()]
    rostered, written = rosters()
    matched, problems = kb.match_rostered(rostered, board, names.to_dict(), teams.to_dict())
    matched["var"] = kb.var(matched["sgp"], matched["positions"], levels)

    md = [
        f"## {args.season} keeper report ({args.name})",
        "",
        f"Rates: hitters {args.hitter_run} (seeds {args.seeds}), pitchers {args.pitcher_run}. "
        f"Playing time: {hit_pt.name}, {pit_pt.name} (relievers capped at {kb.RP_IP_CAP:g} IP). "
        f"Rosters written {written}.",
        "",
        "Replacement (first player past the starting slots):",
        "",
        *[
            f"- {pos} #{starters[pos] + 1}: {row['name']}, {row['total_sgp']:.1f} SGP"
            for pos, row in repl.items()
        ],
    ]
    if problems:
        md += ["", "Not matched:", "", *[f"- {p}" for p in problems]]
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
