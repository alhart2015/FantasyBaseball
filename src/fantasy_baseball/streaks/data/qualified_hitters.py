"""Fetch the list of hitters with ≥min_pa PA in a given season.

Wraps the MLB Stats API ``/stats?stats=season&group=hitting&playerPool=All``
endpoint via the ``statsapi`` package (MLB-StatsAPI). Returns one
:class:`QualifiedHitter` per qualifying hitter.

We use the ``/stats`` endpoint with ``playerPool=All`` rather than the
``/stats/leaders`` endpoint because leaders silently caps at 100 results
regardless of the ``limit`` parameter — the bottom of the cap sits at
~560 PA, well above our 150 PA cutoff. ``playerPool=All`` returns every
player who took a single PA in the season (~750/year), which we then
filter client-side to ≥min_pa.
"""

from __future__ import annotations

from typing import Any

import statsapi

from fantasy_baseball.streaks.models import QualifiedHitter


def parse_season_split(split: dict[str, Any]) -> QualifiedHitter:
    """Build a :class:`QualifiedHitter` from one /stats season split.

    The MLB Stats API season-stats response shape is
    ``stats[0].splits[]``; each split has ``player`` (id, fullName),
    ``team`` (abbreviation), and ``stat.plateAppearances``.
    """
    person = split.get("player", {})
    team = split.get("team", {})
    return QualifiedHitter(
        player_id=int(person["id"]),
        name=person["fullName"],
        team=team.get("abbreviation"),
        pa=int(split["stat"].get("plateAppearances", 0)),
    )


def fetch_qualified_hitters(
    season: int, min_pa: int = 150, limit: int = 5000
) -> list[QualifiedHitter]:
    """Return all hitters with PA >= min_pa for the given season."""
    response = statsapi.get(
        "stats",
        params={
            "stats": "season",
            "group": "hitting",
            "season": season,
            "sportId": 1,
            "playerPool": "All",
            "limit": limit,
        },
    )
    stat_groups = response.get("stats", [])
    rows: list[QualifiedHitter] = []
    for group in stat_groups:
        for split in group.get("splits", []):
            parsed = parse_season_split(split)
            if parsed.pa >= min_pa:
                rows.append(parsed)
    return rows
