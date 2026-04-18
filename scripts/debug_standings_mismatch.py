"""Debug why local standings page points don't match Yahoo's standings.

Pulls raw standings from Yahoo, extracts:
  - per-category stats (what we already cache and use for score_roto)
  - Yahoo's own points_for / outcome_totals (what Yahoo displays)

Then runs our score_roto over the same stats and diffs the totals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
from fantasy_baseball.lineup.yahoo_roster import YAHOO_STAT_ID_MAP, parse_standings_raw
from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import ALL_CATEGORIES


def dump_yahoo_standings():
    session = get_yahoo_session()
    league = get_league(session, 5652)
    raw = league.yhandler.get_standings_raw(league.league_id)

    # Write raw snapshot for offline inspection
    out_path = ROOT / "data" / "debug_yahoo_standings.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(f"wrote raw standings snapshot -> {out_path}")
    return raw


def extract_yahoo_points(raw: dict) -> list[dict]:
    """Pull name, points_for, outcome_totals, and raw-precision stats from Yahoo."""
    league_data = raw.get("fantasy_content", {}).get("league", [])
    standings_block = league_data[1].get("standings", [{}])
    raw_teams = standings_block[0].get("teams", {})

    out = []
    for key in sorted(raw_teams.keys()):
        if key == "count":
            continue
        team_entry = raw_teams[key].get("team", [])
        if not team_entry:
            continue

        meta_list = team_entry[0] if isinstance(team_entry[0], list) else []
        name = team_key = ""
        for item in meta_list:
            if isinstance(item, dict):
                if "name" in item:
                    name = item["name"]
                if "team_key" in item:
                    team_key = item["team_key"]

        detail = team_entry[1] if len(team_entry) > 1 else {}

        # Grab the per-stat raw values *as strings* — preserves full precision
        # for rate stats so we can see if ties are real.
        raw_stats: dict[str, str] = {}
        parsed_stats: dict[str, float] = {}
        ts = detail.get("team_stats", {})
        for se in ts.get("stats", []):
            stat = se.get("stat", {})
            sid = str(stat.get("stat_id", ""))
            val = stat.get("value", "")
            if sid in YAHOO_STAT_ID_MAP:
                cat = YAHOO_STAT_ID_MAP[sid]
                raw_stats[cat] = val
                try:
                    parsed_stats[cat] = float(val)
                except (TypeError, ValueError):
                    pass

        # team_standings.points_for is Yahoo's own roto total
        standings_obj = detail.get("team_standings", {})
        # Sometimes team_standings lives at team_entry[2] — check there too
        if not standings_obj and len(team_entry) > 2 and isinstance(team_entry[2], dict):
            standings_obj = team_entry[2].get("team_standings", {})

        points_for = None
        outcome = None
        rank = None
        if standings_obj:
            points_for = standings_obj.get("points_for")
            outcome = standings_obj.get("outcome_totals")
            rank = standings_obj.get("rank")

        out.append({
            "name": name,
            "team_key": team_key,
            "yahoo_rank": rank,
            "yahoo_points_for": points_for,
            "yahoo_outcome_totals": outcome,
            "raw_stats_str": raw_stats,
            "stats": parsed_stats,
        })
    return out


def diff_scoring(yahoo_rows: list[dict]) -> None:
    # Parse into the form score_roto expects
    stats_by_team = {r["name"]: r["stats"] for r in yahoo_rows}
    roto = score_roto(stats_by_team)

    print("\n=== Per-team: Yahoo points_for vs our score_roto total ===")
    print(f"{'Team':30s}  {'YahooPts':>10s}  {'OurTotal':>10s}  {'Diff':>8s}  {'YahooRank':>10s}")
    for r in yahoo_rows:
        name = r["name"]
        our_total = roto[name]["total"]
        yahoo_pts = r["yahoo_points_for"]
        try:
            yahoo_float = float(yahoo_pts)
            diff = our_total - yahoo_float
            diff_str = f"{diff:+.2f}"
        except (TypeError, ValueError):
            diff_str = "n/a"
        print(f"{name:30s}  {str(yahoo_pts):>10s}  {our_total:>10.2f}  {diff_str:>8s}  {str(r['yahoo_rank']):>10s}")

    # Per-category comparison — which category gives us +1 over Yahoo?
    print("\n=== Per-category totals (our computation) for each team ===")
    header = "Team".ljust(30) + "  " + "  ".join(f"{c:>6s}" for c in ALL_CATEGORIES) + "  " + f"{'TOT':>7s}"
    print(header)
    for r in yahoo_rows:
        name = r["name"]
        row = name.ljust(30)
        for cat in ALL_CATEGORIES:
            pts = roto[name].get(f"{cat}_pts", 0.0)
            row += f"  {pts:6.1f}"
        row += f"  {roto[name]['total']:7.2f}"
        print(row)

    # Show raw-precision stats so we can spot display-level ties that aren't
    # real ties in Yahoo's internal computation.
    print("\n=== Raw category values (Yahoo string-level) ===")
    header = "Team".ljust(30) + "  " + "  ".join(f"{c:>8s}" for c in ALL_CATEGORIES)
    print(header)
    for r in yahoo_rows:
        row = r["name"].ljust(30)
        for cat in ALL_CATEGORIES:
            row += f"  {str(r['raw_stats_str'].get(cat, '')):>8s}"
        print(row)


def main() -> int:
    raw = dump_yahoo_standings()
    rows = extract_yahoo_points(raw)
    diff_scoring(rows)

    # Also write the parsed rows for later inspection
    out = ROOT / "data" / "debug_yahoo_standings_parsed.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote parsed rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
