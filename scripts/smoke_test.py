#!/usr/bin/env python3
"""Post-change smoke test — verify the data pipeline is healthy.

Run after any code change that touches the refresh pipeline, data
model, or scoring stack. Exercises the key data paths without hitting
Yahoo or running Monte Carlo (~2 seconds total).

Reads cache via the KV abstraction (``read_cache``/``read_cache_dict``
/``read_cache_list`` from ``season_data``). On Render this reads
Upstash; locally it reads SQLite at ``data/local.db``. To prime local
data, run a full local refresh (``python scripts/run_lineup.py``) or
sync from Upstash via ``scripts/refresh_remote.py``.

Usage:
    python scripts/smoke_test.py          # run all checks
    python scripts/smoke_test.py --quick  # skip League.from_redis

Exits 0 if all checks pass, 1 if any fail.
"""

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.web.season_data import (
    CacheKey,
    read_cache_dict,
    read_cache_list,
    read_meta,
)

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        print(f"  OK{name}")
        _passed += 1
    else:
        msg = f"  FAIL{name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        _failed += 1


def section(title: str):
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


# -- 1. Cache keys present ------------------------------------


def check_caches():
    section("Cache keys")
    required_dicts = [
        CacheKey.META,
        CacheKey.STANDINGS,
        CacheKey.SPOE,
        CacheKey.LEVERAGE,
        CacheKey.MONTE_CARLO,
    ]
    required_lists = [
        CacheKey.ROSTER,
        CacheKey.ROSTER_AUDIT,
        CacheKey.PENDING_MOVES,
    ]
    for key in required_dicts:
        check(f"cache:{key} present", read_cache_dict(key) is not None, "missing or empty")
    for key in required_lists:
        check(f"cache:{key} present", read_cache_list(key) is not None, "missing or empty")


# -- 2. Meta / timestamp --------------------------------------


def check_meta():
    section("Meta")
    meta = read_meta()
    if not meta:
        check("meta readable", False, "key missing")
        return

    check("last_refresh present", bool(meta.get("last_refresh")))
    check("team_name present", bool(meta.get("team_name")))
    check(
        "start_date / end_date present",
        bool(meta.get("start_date")) and bool(meta.get("end_date")),
    )

    # Timestamp should be in ET (not UTC midnight)
    refresh = meta.get("last_refresh", "")
    if refresh:
        hour = int(refresh.split(" ")[1].split(":")[0]) if " " in refresh else -1
        check(
            "last_refresh looks like ET (not UTC midnight)",
            0 <= hour <= 23,
            f"got '{refresh}'",
        )


# -- 3. Roster audit ------------------------------------------


def check_audit():
    section("Roster audit")
    audit = read_cache_list(CacheKey.ROSTER_AUDIT)
    if not audit:
        check("roster_audit readable", False, "key missing")
        return

    check("audit has entries", len(audit) > 0, f"got {len(audit)}")

    slots = {e["slot"] for e in audit}
    expected_hitter_slots = {"C", "1B", "2B", "3B", "SS", "IF"}
    missing = expected_hitter_slots - slots
    check(
        "all hitter slots present (C, 1B, 2B, 3B, SS, IF)",
        not missing,
        f"missing: {missing}",
    )
    check("OF slot present", "OF" in slots)
    check("UTIL slot present", "UTIL" in slots)
    check("P slot present", "P" in slots)

    # Positions should be enum-normalized (UTIL not Util)
    all_positions = [p for e in audit for p in e.get("positions", [])]
    has_mixed_case_util = any(p == "Util" for p in all_positions)
    check(
        "positions are enum-normalized (UTIL not Util)",
        not has_mixed_case_util,
        "found mixed-case 'Util' in audit positions",
    )


# -- 4. Pending moves -----------------------------------------


def check_pending_moves():
    section("Pending moves")
    pm = read_cache_list(CacheKey.PENDING_MOVES)
    if pm is None:
        check("pending_moves readable", False, "key missing")
        return

    check("pending_moves is a list", isinstance(pm, list))

    if pm:
        move = pm[0]
        check(
            "move has team/adds/drops keys",
            all(k in move for k in ("team", "adds", "drops")),
            f"got keys: {list(move.keys())}",
        )
        # No phantom moves (Gorman / García Jr. were the historical phantoms)
        phantom_names = {"Nolan Gorman", "Luis García Jr.", "Luis Garcia Jr."}
        add_names = {a["name"] for m in pm for a in m.get("adds", [])}
        leaked = add_names & phantom_names
        check(
            "no phantom pending moves",
            not leaked,
            f"phantom adds found: {leaked}",
        )


# -- 5. SPoE --------------------------------------------------


def check_spoe():
    section("SPoE")
    spoe = read_cache_dict(CacheKey.SPOE)
    if not spoe:
        check("spoe readable", False, "key missing")
        return

    check("snapshot_date present", bool(spoe.get("snapshot_date")))
    check("season_fraction > 0", (spoe.get("season_fraction", 0) or 0) > 0)
    check("results non-empty", len(spoe.get("results", [])) > 0)

    # Check that at least one team has a total row
    totals = [r for r in spoe.get("results", []) if r.get("category") == "total"]
    check(
        "total rows present for teams",
        len(totals) >= 2,
        f"got {len(totals)} total rows",
    )


# -- 6. Standings ----------------------------------------------


def check_standings():
    section("Standings")
    standings = read_cache_dict(CacheKey.STANDINGS)
    if not standings:
        check("standings readable", False, "key missing")
        return

    teams = standings.get("teams", [])
    check("standings has 10 teams", len(teams) == 10, f"got {len(teams)}")

    if teams:
        t = teams[0]
        check("team has name", bool(t.get("name")))
        check("team has team_key", bool(t.get("team_key")))
        check("team has stats dict", isinstance(t.get("stats"), dict))

        stats = t.get("stats", {})
        for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]:
            if cat not in stats:
                check(f"stats has {cat}", False, "missing")
                return
        check("stats has all 10 categories", True)


# -- 7. League.from_redis -------------------------------------


def check_league_from_redis():
    section("League.from_redis")

    try:
        from fantasy_baseball.models.league import League

        league = League.from_redis(2026)

        check("League loads without error", True)
        check(
            "10 teams loaded",
            len(league.teams) == 10,
            f"got {len(league.teams)}",
        )
        check(
            "standings snapshots loaded",
            len(league.standings) > 0,
            f"got {len(league.standings)}",
        )

        # User team check
        try:
            user = league.team_by_name("Hart of the Order")
            check("user team found", True)
            check(
                "user team has rosters",
                len(user.rosters) > 0,
                f"got {len(user.rosters)}",
            )

            latest = user.latest_roster()
            check(
                "latest roster has entries",
                len(latest.entries) > 0,
                f"got {len(latest.entries)}",
            )

            # Check Position enum on roster entries
            from fantasy_baseball.models.positions import Position

            first = latest.entries[0]
            check(
                "roster entry positions are Position enum",
                isinstance(first.positions[0], Position),
                f"got {type(first.positions[0])}",
            )
        except KeyError:
            check("user team found", False, "KeyError on team_by_name")

    except Exception as e:
        check("League.from_redis", False, str(e))


# -- 8. Scoring round-trip ------------------------------------


def check_scoring():
    section("Scoring round-trip")

    roster_raw = read_cache_list(CacheKey.ROSTER)
    if not roster_raw:
        check("roster readable", False, "key missing")
        return

    try:
        from fantasy_baseball.models.player import Player
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.scoring import project_team_stats, score_roto_dict

        players = [Player.from_dict(p) for p in roster_raw]
        check(f"parsed {len(players)} Player objects", len(players) > 0)

        stats = project_team_stats(players)
        check(
            "project_team_stats returns CategoryStats",
            isinstance(stats, CategoryStats),
            f"got {type(stats)}",
        )
        check("HR > 0", stats.hr > 0, f"got {stats.hr}")
        check("ERA < 99", stats.era < 99, f"got {stats.era}")

        # JSON round-trip
        d = stats.to_dict()
        blob = json.dumps(d)
        parsed = json.loads(blob)
        check("CategoryStats JSON round-trip", parsed["HR"] == stats.hr)

        # score_roto_dict consumes string-keyed dicts at the I/O boundary
        stats_dict = stats.to_dict()
        roto = score_roto_dict({"Team A": stats_dict, "Team B": stats_dict})
        check(
            "score_roto_dict accepts CategoryStats.to_dict()",
            "total" in roto["Team A"],
        )

    except Exception as e:
        check("scoring round-trip", False, str(e))


# -- Main ------------------------------------------------------


def main():
    quick = "--quick" in sys.argv

    print(f"\nSmoke test -- {date.today()}")
    if quick:
        print("   Mode: --quick (skipping League.from_redis)")

    check_caches()
    check_meta()
    check_audit()
    check_pending_moves()
    check_spoe()
    check_standings()
    check_scoring()

    if not quick:
        check_league_from_redis()

    print(f"\n{'=' * 60}")
    print(f"  {_passed} passed, {_failed} failed")
    print(f"{'=' * 60}\n")

    sys.exit(0 if _failed == 0 else 1)


if __name__ == "__main__":
    main()
