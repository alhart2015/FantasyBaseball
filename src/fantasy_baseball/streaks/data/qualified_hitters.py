"""Fetch the list of hitters with ≥min_pa PA in a given season.

Wraps the MLB Stats API ``/stats/leaders?leaderCategories=plateAppearances``
endpoint via the ``statsapi`` package (MLB-StatsAPI). Returns one
:class:`QualifiedHitter` per qualifying hitter.
"""

from __future__ import annotations

from typing import Any

import statsapi

from fantasy_baseball.streaks.models import QualifiedHitter


def parse_leader_row(row: dict[str, Any]) -> QualifiedHitter:
    """Build a :class:`QualifiedHitter` from one /stats/leaders entry."""
    person = row.get("person", {})
    team = row.get("team", {})
    return QualifiedHitter(
        player_id=int(person["id"]),
        name=person["fullName"],
        team=team.get("abbreviation"),
        pa=int(row["value"]),
    )


def fetch_qualified_hitters(
    season: int, min_pa: int = 150, limit: int = 1000
) -> list[QualifiedHitter]:
    """Return all hitters with PA >= min_pa for the given season."""
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
    rows: list[QualifiedHitter] = []
    for group in leaders_groups:
        for leader in group.get("leaders", []):
            parsed = parse_leader_row(leader)
            if parsed.pa >= min_pa:
                rows.append(parsed)
    return rows
