"""MLB Stats API game log fetching → Redis aggregates.

Builds per-player counting-stat totals in memory during a parallel
per-player fetch, then writes two Redis keys:

- ``game_log_totals:hitters``  → {mlbam_id: {"name": str, "pa": int, ...}}
- ``game_log_totals:pitchers`` → {mlbam_id: {"name": str, "ip": float, ...}}
- ``season_progress``          → {"games_elapsed": int, "total": 162, "as_of": "YYYY-MM-DD"}

``games_elapsed`` counts distinct calendar dates encountered across every
player's game log — i.e. how many MLB game-days have been played.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as _date

_HITTER_STATS = ("pa", "ab", "h", "r", "hr", "rbi", "sb")
_PITCHER_STATS = ("ip", "k", "er", "bb", "h_allowed", "w", "sv")


def _empty_hitter_total() -> dict:
    total: dict = {s: 0 for s in _HITTER_STATS}
    total["name"] = ""
    return total


def _empty_pitcher_total() -> dict:
    total: dict = {s: 0 for s in _PITCHER_STATS}
    total["name"] = ""
    return total


def fetch_game_log_totals(season: int, progress_cb=None) -> tuple[dict, dict, int]:
    """Fetch all MLB game logs for *season*, accumulate totals per player,
    write Redis keys, and return (hitters_totals, pitchers_totals, games_elapsed).

    hitters_totals / pitchers_totals are keyed by mlbam_id (str) and each
    entry includes a ``"name"`` field alongside the counting stats so that
    downstream readers can normalize by name without a separate id→name map.
    games_elapsed = number of distinct calendar dates seen across all players.
    """
    import statsapi

    from fantasy_baseball.analysis.game_logs import fetch_player_game_log
    from fantasy_baseball.data.kv_store import get_kv
    from fantasy_baseball.data.redis_store import (
        set_game_log_totals,
        set_season_progress,
    )
    from fantasy_baseball.models.player import PlayerType

    # Get all MLB teams
    if progress_cb:
        progress_cb("Fetching MLB team rosters...")
    teams_data = statsapi.get("teams", {"sportId": 1, "season": season})
    teams_list = teams_data.get("teams", [])

    # Build player list from rosters
    players = []
    seen_ids = set()
    for team in teams_list:
        team_id = team["id"]
        team_abbrev = team.get("abbreviation", "")
        try:
            roster_data = statsapi.get(
                "team_roster",
                {"teamId": team_id, "rosterType": "fullSeason", "season": season},
            )
        except Exception:
            try:
                roster_data = statsapi.get(
                    "team_roster",
                    {"teamId": team_id, "rosterType": "active", "season": season},
                )
            except Exception:
                continue

        for entry in roster_data.get("roster", []):
            person = entry.get("person", {})
            mlbam_id = person.get("id")
            if not mlbam_id or mlbam_id in seen_ids:
                continue
            seen_ids.add(mlbam_id)

            pos_type = entry.get("position", {}).get("type", "")
            player_type = PlayerType.PITCHER if pos_type == "Pitcher" else PlayerType.HITTER

            players.append(
                {
                    "mlbam_id": mlbam_id,
                    "name": person.get("fullName", ""),
                    "team": team_abbrev,
                    "player_type": player_type,
                }
            )

    if progress_cb:
        progress_cb(f"Found {len(players)} MLB players, fetching game logs...")

    # Full rebuild each call: the Redis totals blob has no MAX(date) lookup,
    # so we always re-fetch the full season from the MLB API and accumulate
    # fresh. Trades bandwidth for correctness (no double-counting risk).
    def _fetch_one(player):
        mid = player["mlbam_id"]
        group = "hitting" if player["player_type"] == PlayerType.HITTER else "pitching"
        try:
            games = fetch_player_game_log(mid, season, group)
        except Exception:
            return (player, [])
        return (player, games or [])

    hitters_totals: dict[str, dict] = {}
    pitchers_totals: dict[str, dict] = {}
    distinct_dates: set[str] = set()
    done_count = 0

    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = [pool.submit(_fetch_one, p) for p in players]
        for future in as_completed(futures):
            player, games = future.result()
            done_count += 1

            if games:
                mid = str(player["mlbam_id"])
                pt = player["player_type"]
                if pt == PlayerType.HITTER:
                    agg = hitters_totals.setdefault(mid, _empty_hitter_total())
                    agg["name"] = player["name"]
                    for g in games:
                        distinct_dates.add(g["date"])
                        for stat in _HITTER_STATS:
                            agg[stat] = (agg[stat] or 0) + (g.get(stat) or 0)
                else:
                    agg = pitchers_totals.setdefault(mid, _empty_pitcher_total())
                    agg["name"] = player["name"]
                    for g in games:
                        distinct_dates.add(g["date"])
                        for stat in _PITCHER_STATS:
                            agg[stat] = (agg[stat] or 0) + (g.get(stat) or 0)

            if done_count % 50 == 0 and progress_cb:
                progress_cb(f"Game logs: {done_count}/{len(players)} players...")

    client = get_kv()
    set_game_log_totals(client, "hitters", hitters_totals)
    set_game_log_totals(client, "pitchers", pitchers_totals)
    set_season_progress(
        client,
        games_elapsed=len(distinct_dates),
        total=162,
        as_of=_date.today().isoformat(),
    )
    return hitters_totals, pitchers_totals, len(distinct_dates)
