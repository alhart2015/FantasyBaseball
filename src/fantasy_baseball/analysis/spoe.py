"""Current season-to-date Standings Points Over Expected (SPoE) — luck quantification.

Walks the weekly_rosters history and compares accumulated expected
stats (from preseason projections, scaled by ownership days) against
live standings. Runs on every refresh.

History (removed 2026-04-10): SPoE was originally a weekly metric that
computed per-week projected vs actual roto points. It required a pile
of support (compute_spoe weekly loop, prorate_spoe for partial weeks,
project_team_week, aggregate_game_logs_before, get_standings_for_date,
spoe_results/spoe_components SQLite tables). All of that is gone.
The current design is simpler: walk the roster history, scale preseason
projections by days owned, compare to live standings once per refresh.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any

import pandas as pd

from fantasy_baseball.models.league import League
from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES,
    HITTING_COUNTING,
    PITCHING_COUNTING,
)
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip
from fantasy_baseball.utils.time_utils import local_today

HITTER_COMPONENTS = HITTING_COUNTING
PITCHER_COMPONENTS = PITCHING_COUNTING
ALL_COMPONENTS = HITTER_COMPONENTS + PITCHER_COMPONENTS


def load_rosters_for_date(conn, snapshot_date: str) -> dict[str, list[dict]]:
    """Load all team rosters for a given snapshot date from weekly_rosters."""
    rows = conn.execute(
        "SELECT team, player_name, positions "
        "FROM weekly_rosters "
        "WHERE snapshot_date = ?",
        (snapshot_date,),
    ).fetchall()

    rosters: dict[str, list[dict]] = {}
    for row in rows:
        team = row["team"]
        positions_str = row["positions"] or ""
        positions = [p.strip() for p in positions_str.split(",")] if positions_str else []
        rosters.setdefault(team, []).append(
            {"name": row["player_name"], "positions": positions}
        )
    return rosters


def get_week_dates(conn, season_year: int) -> list[str]:
    """Return distinct snapshot_dates from weekly_rosters for the season, sorted.

    Each returned date is the Tuesday-of-scoring-week label the refresh
    writes via append_roster_snapshot.
    """
    rows = conn.execute(
        "SELECT DISTINCT snapshot_date FROM weekly_rosters "
        "WHERE snapshot_date >= ? ORDER BY snapshot_date",
        (f"{season_year}-",),
    ).fetchall()
    return [r["snapshot_date"] for r in rows]


def build_preseason_lookup(
    hitters_df: pd.DataFrame, pitchers_df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Build a {normalized_name: {stats..., player_type}} lookup for SPoE.

    Takes the preseason blend DataFrames from `get_blended_projections`
    and produces a flat lookup so spoe can resolve per-player preseason
    stats by name without pandas filtering per row.
    """
    lookup: dict[str, dict[str, Any]] = {}
    for df, ptype, cols in [
        (hitters_df, "hitter", HITTER_COMPONENTS),
        (pitchers_df, "pitcher", PITCHER_COMPONENTS),
    ]:
        if df is None or df.empty:
            continue
        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            name = row_dict.get("name")
            if not name:
                continue
            entry: dict[str, Any] = {"player_type": ptype}
            for c in cols:
                entry[c] = float(row_dict.get(c, 0) or 0)
            lookup[normalize_name(name)] = entry
    return lookup


def _components_to_stats(components: dict[str, float]) -> dict[str, float]:
    """Convert accumulated counting components to roto category stats.

    Rate stats are computed from the accumulated component totals so
    AVG = h/ab, ERA = 9*er/ip, WHIP = (bb + h_allowed)/ip even when
    the components themselves have been scaled by an arbitrary fraction.
    """
    return {
        "R": components["r"],
        "HR": components["hr"],
        "RBI": components["rbi"],
        "SB": components["sb"],
        "AVG": calculate_avg(components["h"], components["ab"]),
        "W": components["w"],
        "K": components["k"],
        "SV": components["sv"],
        "ERA": calculate_era(components["er"], components["ip"]),
        "WHIP": calculate_whip(components["bb"], components["h_allowed"], components["ip"]),
    }


def _empty_components() -> dict[str, float]:
    return {c: 0.0 for c in ALL_COMPONENTS}


def compute_current_spoe(
    league: League,
    standings: list[dict],
    preseason_lookup: dict[str, dict[str, Any]],
    season_start: str,
    season_end: str,
    today: date | None = None,
) -> dict:
    """Compute current season-to-date SPoE.

    Iterates league.teams and calls Team.ownership_periods() on each
    to get already-clipped (entry, period_start, period_end) tuples.
    Each player's preseason projection is scaled by
    days_covered / total_season_days and added to the owning team's
    accumulated expected components. Final expected stats are compared
    to live standings via score_roto.

    Args:
        league: Loaded League object. SPoE iterates league.teams and
            calls Team.ownership_periods() on each to get the
            (entry, period_start, period_end) tuples it needs.
        standings: list of team dicts from cache:standings — each dict
            has ``name`` and ``stats`` keys.
        preseason_lookup: output of build_preseason_lookup.
        season_start: "YYYY-MM-DD" season start date.
        season_end: "YYYY-MM-DD" season end date.
        today: Date to treat as "now". Defaults to the user's local
            date (see :func:`fantasy_baseball.utils.time_utils.local_today`).
            Used by tests to pin the walk.

    Returns:
        Dict with ``snapshot_date`` (today), ``season_fraction``, and
        ``results`` list. Shape matches the previous weekly cache:spoe
        so the luck page template doesn't need to change.
    """
    today = today or local_today()
    start = date.fromisoformat(season_start)
    end = date.fromisoformat(season_end)
    total_days = (end - start).days
    if total_days <= 0:
        return {
            "snapshot_date": today.isoformat(),
            "season_fraction": 0.0,
            "results": [],
        }
    days_elapsed = max(0, (today - start).days)
    season_fraction = min(1.0, days_elapsed / total_days)

    team_components: dict[str, dict[str, float]] = defaultdict(_empty_components)

    for team_obj in league.teams:
        comps = team_components[team_obj.name]
        periods = team_obj.ownership_periods(
            season_start=start,
            season_end=end,
            today=today,
        )
        for entry, period_start, period_end in periods:
            days_covered = (period_end - period_start).days
            if days_covered <= 0:
                continue
            fraction = days_covered / total_days

            preseason = preseason_lookup.get(normalize_name(entry.name))
            if preseason is None:
                continue
            ptype = preseason.get("player_type")
            relevant = (
                HITTER_COMPONENTS if ptype == "hitter"
                else PITCHER_COMPONENTS
            )
            for c in relevant:
                comps[c] += preseason.get(c, 0.0) * fraction

    actual_stats: dict[str, dict[str, float]] = {}
    for t in standings:
        actual_stats[t["name"]] = dict(t["stats"])
        # Ensure every team that appears in standings shows up in the
        # results, even if its roster walk contributed zero (empty roster
        # or all players missing from preseason lookup). Without this,
        # score_roto would rank a reduced team set and the results would
        # omit teams that deserve a row.
        _ = team_components[t["name"]]

    expected_stats = {team: _components_to_stats(comps) for team, comps in team_components.items()}

    common = set(expected_stats) & set(actual_stats)
    if len(common) < 2:
        return {
            "snapshot_date": today.isoformat(),
            "season_fraction": season_fraction,
            "results": [],
        }

    projected_roto = score_roto({t: expected_stats[t] for t in common})
    actual_roto = score_roto({t: actual_stats[t] for t in common})

    results: list[dict] = []
    for team in sorted(common):
        total_spoe = 0.0
        for cat in ALL_CATEGORIES:
            proj_pts = projected_roto[team].get(f"{cat}_pts", 0)
            act_pts = actual_roto[team].get(f"{cat}_pts", 0)
            spoe = act_pts - proj_pts
            total_spoe += spoe
            results.append({
                "team": team,
                "category": cat,
                "projected_stat": expected_stats[team][cat],
                "actual_stat": actual_stats[team][cat],
                "projected_pts": proj_pts,
                "actual_pts": act_pts,
                "spoe": spoe,
            })
        results.append({
            "team": team,
            "category": "total",
            "projected_stat": None,
            "actual_stat": None,
            "projected_pts": projected_roto[team]["total"],
            "actual_pts": actual_roto[team]["total"],
            "spoe": total_spoe,
        })

    return {
        "snapshot_date": today.isoformat(),
        "season_fraction": season_fraction,
        "results": results,
    }
