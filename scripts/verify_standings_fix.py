"""Verify that format_standings_for_display now matches Yahoo exactly.

Runs the full live path: fetch Yahoo standings -> parse -> snapshot ->
format_standings_for_display. Prints the displayed total/rank next to
Yahoo's own, so we can confirm they match.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
from fantasy_baseball.lineup.yahoo_roster import fetch_standings
from fantasy_baseball.web.season_data import (
    _standings_to_snapshot, format_standings_for_display,
)


def main() -> int:
    session = get_yahoo_session()
    league = get_league(session, 5652)
    standings = fetch_standings(league)

    snap = _standings_to_snapshot(standings)
    display = format_standings_for_display(snap, "Hart of the Order")

    print(f"{'Rank':<5} {'Team':30s}  {'DisplayedTotal':>14s}  {'YahooPointsFor':>14s}  {'CatSum':>8s}")
    for t in display["teams"]:
        name = t["name"]
        total = t["roto_points"]["total"]
        cat_sum = sum(t["roto_points"].get(f"{c}_pts", 0.0) for c in
                      ["R","HR","RBI","SB","AVG","W","K","SV","ERA","WHIP"])
        yahoo_pf = next(
            (s["points_for"] for s in standings if s["name"] == name),
            None,
        )
        print(f"{t['rank']:<5} {name:30s}  {total:>14.2f}  {str(yahoo_pf):>14s}  {cat_sum:>8.2f}")

    # Assert match
    mismatches = 0
    for t in display["teams"]:
        name = t["name"]
        yahoo_pf = next(
            (s["points_for"] for s in standings if s["name"] == name),
            None,
        )
        if yahoo_pf is None or abs(t["roto_points"]["total"] - float(yahoo_pf)) > 1e-9:
            mismatches += 1
            print(f"  MISMATCH: {name} displayed {t['roto_points']['total']} vs Yahoo {yahoo_pf}")
    print(f"\n{mismatches} total mismatches")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
