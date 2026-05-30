"""One-shot diagnostic: print every stat_id -> name mapping Yahoo returns
for this league's ``stat_categories()`` call, plus the stat_ids actually
present in the current ``team_standings`` response.

Run this once after authenticating with Yahoo to confirm which stat_id
corresponds to AB (at-bats) for the league. The mapping in
``yahoo_roster.YAHOO_OPP_STAT_ID_MAP`` is updated based on the result
(Task 2.2 of the team-YTD projection refactor).

Usage:
    python scripts/discover_yahoo_stat_ids.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
from fantasy_baseball.config import load_config


def main() -> None:
    sc = get_yahoo_session()
    config = load_config(ROOT / "config" / "league.yaml")
    lg = get_league(sc, config.league_id, config.game_code)

    print("=== lg.stat_categories() ===")
    cats = lg.stat_categories()
    for cat in cats:
        sid = cat.get("stat_id")
        display = cat.get("display_name") or cat.get("name", "?")
        name = cat.get("name", "")
        sort_order = cat.get("sort_order", "")
        print(f"  stat_id={sid!s:>4}  display={display:>12}  name={name}  sort={sort_order}")

    print()
    print("=== stat_ids present in team_standings response ===")
    raw = lg.yhandler.get_standings_raw(lg.league_id)
    seen: set[str] = set()
    league_data = raw.get("fantasy_content", {}).get("league", [])
    if len(league_data) >= 2:
        standings_block = league_data[1].get("standings", [{}])
        if standings_block:
            teams = standings_block[0].get("teams", {})
            for key in sorted(teams.keys()):
                if key == "count":
                    continue
                entry = teams[key].get("team", [])
                if len(entry) < 2 or not isinstance(entry[1], dict):
                    continue
                stats = entry[1].get("team_stats", {}).get("stats", [])
                for stat_entry in stats:
                    sid = stat_entry.get("stat", {}).get("stat_id")
                    if sid is not None:
                        seen.add(str(sid))
    print(f"  stat_ids present: {sorted(seen, key=int) if seen else '(none)'}")


if __name__ == "__main__":
    main()
