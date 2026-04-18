"""Freeze the preseason Monte Carlo baseline for the current season.

Fetches every team's Opening-Day roster from Yahoo, matches against
preseason projections from Redis, runs run_monte_carlo twice
(base + with_management) at 1000 iterations each, and writes the
result to Redis under ``preseason_baseline:{season_year}``.

Run this once per season, after the draft completes. The refresh
pipeline reads this artifact on every refresh instead of re-running
the MCs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct invocation.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season-year",
        type=int,
        default=None,
        help="Override season_year from config/league.yaml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing baseline in Redis.",
    )
    args = parser.parse_args(argv)

    import pandas as pd

    from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
    from fantasy_baseball.config import load_config
    from fantasy_baseball.data.projections import match_roster_to_projections
    from fantasy_baseball.data.redis_store import (
        get_blended_projections,
        get_default_client,
        get_preseason_baseline,
        set_preseason_baseline,
    )
    from fantasy_baseball.lineup.yahoo_roster import fetch_roster
    from fantasy_baseball.simulation import run_monte_carlo
    from fantasy_baseball.utils.name_utils import normalize_name

    config = load_config(_PROJECT_ROOT / "config" / "league.yaml")
    season_year = args.season_year or config.season_year

    client = get_default_client()
    if client is None:
        raise RuntimeError(
            "Redis client not configured: set UPSTASH_REDIS_REST_URL / "
            "UPSTASH_REDIS_REST_TOKEN in the environment (or .env)."
        )

    existing = get_preseason_baseline(client, season_year)
    if existing and not args.force:
        frozen_at = existing.get("meta", {}).get("frozen_at", "?")
        print(
            f"Preseason baseline for {season_year} already frozen at "
            f"{frozen_at}. Re-run with --force to overwrite."
        )
        sys.exit(1)

    print("Authenticating with Yahoo...")
    sc = get_yahoo_session()
    league = get_league(sc, config.league_id, config.game_code)

    print(f"Fetching Opening-Day rosters (day={config.season_start})...")
    team_rosters_raw: dict[str, list[dict]] = {}
    for team_key, team_info in league.teams().items():
        tname = team_info.get("name", team_key)
        team_rosters_raw[tname] = fetch_roster(league, team_key, day=config.season_start)
        print(f"  {tname}: {len(team_rosters_raw[tname])} players")

    print("Loading preseason projections from Redis...")
    hitter_rows = get_blended_projections(client, "hitters") or []
    pitcher_rows = get_blended_projections(client, "pitchers") or []
    if not hitter_rows or not pitcher_rows:
        raise RuntimeError(
            "Preseason projections not found in Redis "
            "(blended_projections:hitters / blended_projections:pitchers). "
            "Run `python scripts/build_db.py` first."
        )
    hitters_proj = pd.DataFrame(hitter_rows)
    pitchers_proj = pd.DataFrame(pitcher_rows)
    hitters_proj["_name_norm"] = hitters_proj["name"].apply(normalize_name)
    pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)

    print("Matching rosters to projections...")
    team_rosters: dict[str, list] = {}
    for tname, raw in team_rosters_raw.items():
        team_rosters[tname] = match_roster_to_projections(
            raw,
            hitters_proj,
            pitchers_proj,
            context=f"preseason_baseline:{tname}",
        )

    h_slots = sum(v for k, v in config.roster_slots.items() if k not in ("P", "BN", "IL", "DL"))
    p_slots = config.roster_slots.get("P", 9)

    print("Running base Monte Carlo (1000 iterations)...")
    base = run_monte_carlo(
        team_rosters,
        h_slots,
        p_slots,
        config.team_name,
        n_iterations=1000,
        use_management=False,
    )
    print("Running with-management Monte Carlo (1000 iterations)...")
    with_mgmt = run_monte_carlo(
        team_rosters,
        h_slots,
        p_slots,
        config.team_name,
        n_iterations=1000,
        use_management=True,
    )

    payload = {
        "base": base,
        "with_management": with_mgmt,
        "meta": {
            "frozen_at": _dt.datetime.now(_dt.UTC).isoformat(),
            "season_year": season_year,
            "roster_date": config.season_start,
            "projections_source": "blended",
        },
    }
    set_preseason_baseline(client, season_year, payload)
    print(f"Wrote preseason_baseline:{season_year} ({len(team_rosters)} teams).")


if __name__ == "__main__":
    main()
