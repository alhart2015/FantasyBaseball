"""Fetch the list of hitters with ≥min_pa PA in a given season.

Wraps the MLB Stats API ``/stats/leaders?leaderCategories=plateAppearances``
endpoint via the ``statsapi`` package (MLB-StatsAPI). Returns one row per
qualifying hitter with player_id, name, team, and PA.
"""

from __future__ import annotations

from typing import Any

import statsapi


def parse_leader_row(row: dict[str, Any]) -> dict[str, Any]:
    """Extract player_id, name, team, pa from one /stats/leaders entry."""
    person = row.get("person", {})
    team = row.get("team", {})
    return {
        "player_id": int(person["id"]),
        "name": person["fullName"],
        "team": team.get("abbreviation"),
        "pa": int(row["value"]),
    }


def fetch_qualified_hitters(
    season: int, min_pa: int = 150, limit: int = 1000
) -> list[dict[str, Any]]:
    """Return all hitters with PA >= min_pa for the given season.

    Each result dict has keys: player_id, name, team, pa.
    """
    response = statsapi.get(
        "stats_leaders",
        params={
            "leaderCategories": "plateAppearances",
            "season": season,
            "statGroup": "hitting",
            "limit": limit,
        },
    )
    leaders_groups = response.get("leagueLeaders", [])
    rows: list[dict[str, Any]] = []
    for group in leaders_groups:
        for leader in group.get("leaders", []):
            parsed = parse_leader_row(leader)
            if parsed["pa"] >= min_pa:
                rows.append(parsed)
    return rows
