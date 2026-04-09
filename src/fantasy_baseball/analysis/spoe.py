"""Standings Points Over Expected (SPOE) — luck quantification."""

from __future__ import annotations

from datetime import date

import pandas as pd

from fantasy_baseball.data.db import (
    get_completed_spoe_weeks,
    load_spoe_components,
    save_spoe_components,
    save_spoe_results,
)
from fantasy_baseball.data.projections import match_roster_to_projections
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES,
    HITTING_COUNTING,
    PITCHING_COUNTING,
)
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

HITTER_COMPONENTS = HITTING_COUNTING
PITCHER_COMPONENTS = PITCHING_COUNTING
ALL_COMPONENTS = HITTER_COMPONENTS + PITCHER_COMPONENTS


def load_rosters_for_date(conn, snapshot_date: str) -> dict[str, list[dict]]:
    """Load all team rosters for a given snapshot date.

    Args:
        conn: SQLite connection.
        snapshot_date: Date string in YYYY-MM-DD format.

    Returns:
        Dict mapping team name to list of player dicts with keys:
        - "name": str
        - "positions": list[str]
    """
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
        player = {"name": row["player_name"], "positions": positions}
        rosters.setdefault(team, []).append(player)

    return rosters


def load_projections_for_date(
    conn, year: int, target_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find the best ROS blended projections for a target date.

    Queries ros_blended_projections for the MAX snapshot_date <= target_date.
    Falls back to blended_projections (preseason) if no ROS data exists.

    Args:
        conn: SQLite connection.
        year: Season year.
        target_date: Date string in YYYY-MM-DD format.

    Returns:
        Tuple of (hitters_df, pitchers_df) DataFrames with a _name_norm column.
    """
    row = conn.execute(
        "SELECT MAX(snapshot_date) as best_date "
        "FROM ros_blended_projections "
        "WHERE year = ? AND snapshot_date <= ?",
        (year, target_date),
    ).fetchone()

    best_date = row["best_date"] if row else None

    if best_date is not None:
        rows = conn.execute(
            "SELECT * FROM ros_blended_projections "
            "WHERE year = ? AND snapshot_date = ?",
            (year, best_date),
        ).fetchall()
        df = pd.DataFrame([dict(r) for r in rows])
    else:
        rows = conn.execute(
            "SELECT * FROM blended_projections WHERE year = ?",
            (year,),
        ).fetchall()
        df = pd.DataFrame([dict(r) for r in rows])

    if df.empty:
        empty = pd.DataFrame()
        return empty, empty

    df["_name_norm"] = df["name"].apply(normalize_name)

    hitters_df = df[df["player_type"] == "hitter"].reset_index(drop=True)
    pitchers_df = df[df["player_type"] == "pitcher"].reset_index(drop=True)

    return hitters_df, pitchers_df


def project_team_week(roster, game_log_totals, days_remaining):
    """Project one week of component stats for a team's roster.

    For each player, subtracts actual stats from full-season projection
    to get remaining-season stats, then scales to one week.  Players
    without ROS projections contribute nothing.

    Args:
        roster: list of Player objects with .ros populated
        game_log_totals: {normalized_name: {stat: value}} from game logs
        days_remaining: days from this week's Monday to season end

    Returns:
        dict of component stats for the team for this week.
    """
    weekly_fraction = 7 / days_remaining if days_remaining > 0 else 0
    team_components = {c: 0.0 for c in ALL_COMPONENTS}

    for player in roster:
        if player.ros is None:
            continue

        name_norm = normalize_name(player.name)
        actuals = game_log_totals.get(name_norm, {})

        if player.player_type == PlayerType.HITTER:
            component_keys = HITTER_COMPONENTS
        else:
            component_keys = PITCHER_COMPONENTS

        for key in component_keys:
            projected = getattr(player.ros, key, 0) or 0
            actual = actuals.get(key, 0)
            remaining = max(0, projected - actual)
            team_components[key] += remaining * weekly_fraction

    return team_components


def aggregate_game_logs_before(
    conn, season: int, before_date: str
) -> dict[str, dict[str, float]]:
    """Sum game log stats for each player before a given date.

    Args:
        conn: SQLite connection.
        season: Season year.
        before_date: Exclusive upper bound date string in YYYY-MM-DD format.

    Returns:
        Dict mapping normalized player name to stat totals. Stat keys are
        lowercase: h, ab, r, hr, rbi, sb, ip, k, er, bb, h_allowed, w, sv.
    """
    stat_cols = ALL_COMPONENTS
    select_cols = ", ".join(f"SUM({col}) as {col}" for col in stat_cols)

    rows = conn.execute(
        f"SELECT name, {select_cols} "
        "FROM game_logs "
        "WHERE season = ? AND date < ? "
        "GROUP BY name",
        (season, before_date),
    ).fetchall()

    result: dict[str, dict[str, float]] = {}
    for row in rows:
        name_norm = normalize_name(row["name"])
        stats = {col: float(row[col] or 0) for col in stat_cols}
        result[name_norm] = stats

    return result


def components_to_roto_stats(components: dict[str, float]) -> dict[str, float]:
    """Convert accumulated component stats to the roto stat dict score_roto expects."""
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


def get_week_dates(conn, season_year: int) -> list[str]:
    """Get all distinct roster snapshot dates for the season, sorted."""
    rows = conn.execute(
        "SELECT DISTINCT snapshot_date FROM weekly_rosters "
        "WHERE snapshot_date >= ? ORDER BY snapshot_date",
        (f"{season_year}-",),
    ).fetchall()
    return [r["snapshot_date"] for r in rows]


def get_standings_for_date(
    conn, season_year: int, snapshot_date: str
) -> dict[str, dict[str, float]]:
    """Load actual standings. Returns {team: {R: val, HR: val, ...}}."""
    rows = conn.execute(
        "SELECT team, r, hr, rbi, sb, avg, w, k, sv, era, whip "
        "FROM standings WHERE year = ? AND snapshot_date = ?",
        (season_year, snapshot_date),
    ).fetchall()
    return {
        r["team"]: {
            "R": r["r"],
            "HR": r["hr"],
            "RBI": r["rbi"],
            "SB": r["sb"],
            "AVG": r["avg"],
            "W": r["w"],
            "K": r["k"],
            "SV": r["sv"],
            "ERA": r["era"],
            "WHIP": r["whip"],
        }
        for r in rows
    }


def compute_spoe(conn, config) -> None:
    """Compute SPOE for all weeks with available data.

    Completed weeks are skipped. The last (current) week is always
    recomputed. Results stored in spoe_results and spoe_components tables.
    """
    week_dates = get_week_dates(conn, config.season_year)
    if not week_dates:
        return

    completed = get_completed_spoe_weeks(conn, config.season_year)
    season_end = date.fromisoformat(config.season_end)
    current_week = week_dates[-1]

    # Resume from last completed week
    team_components: dict[str, dict[str, float]] = {}
    start_idx = 0
    for prev_date in reversed(week_dates):
        if prev_date in completed and prev_date != current_week:
            team_components = load_spoe_components(
                conn, config.season_year, prev_date
            )
            start_idx = week_dates.index(prev_date) + 1
            break

    for i in range(start_idx, len(week_dates)):
        snapshot_date = week_dates[i]
        if snapshot_date in completed and snapshot_date != current_week:
            continue

        monday = date.fromisoformat(snapshot_date)
        days_remaining = (season_end - monday).days
        if days_remaining <= 0:
            continue

        rosters = load_rosters_for_date(conn, snapshot_date)
        if not rosters:
            continue

        hitters_proj, pitchers_proj = load_projections_for_date(
            conn, config.season_year, snapshot_date
        )
        if hitters_proj.empty and pitchers_proj.empty:
            continue

        game_log_totals = aggregate_game_logs_before(
            conn, config.season_year, snapshot_date
        )

        actual_stats = get_standings_for_date(
            conn, config.season_year, snapshot_date
        )
        if not actual_stats:
            continue

        for team_name, roster_dicts in rosters.items():
            matched = match_roster_to_projections(
                roster_dicts, hitters_proj, pitchers_proj
            )
            weekly = project_team_week(matched, game_log_totals, days_remaining)

            if team_name not in team_components:
                team_components[team_name] = {c: 0.0 for c in ALL_COMPONENTS}
            for comp in ALL_COMPONENTS:
                team_components[team_name][comp] += weekly[comp]

        projected_stats = {
            team: components_to_roto_stats(comps)
            for team, comps in team_components.items()
            if team in actual_stats
        }

        common_teams = set(projected_stats) & set(actual_stats)
        if len(common_teams) < 2:
            continue

        proj_for_scoring = {t: projected_stats[t] for t in common_teams}
        actual_for_scoring = {t: actual_stats[t] for t in common_teams}

        projected_roto = score_roto(proj_for_scoring)
        actual_roto = score_roto(actual_for_scoring)

        results = []
        for team in common_teams:
            total_spoe = 0.0
            for cat in ALL_CATEGORIES:
                proj_pts = projected_roto[team].get(f"{cat}_pts", 0)
                act_pts = actual_roto[team].get(f"{cat}_pts", 0)
                spoe = act_pts - proj_pts
                total_spoe += spoe
                results.append(
                    {
                        "team": team,
                        "category": cat,
                        "projected_stat": projected_stats[team][cat],
                        "actual_stat": actual_stats[team][cat],
                        "projected_pts": proj_pts,
                        "actual_pts": act_pts,
                        "spoe": spoe,
                    }
                )
            results.append(
                {
                    "team": team,
                    "category": "total",
                    "projected_stat": None,
                    "actual_stat": None,
                    "projected_pts": projected_roto[team]["total"],
                    "actual_pts": actual_roto[team]["total"],
                    "spoe": total_spoe,
                }
            )

        save_spoe_results(conn, config.season_year, snapshot_date, results,
                          commit=False)
        save_spoe_components(conn, config.season_year, snapshot_date,
                             team_components)  # commits both


def prorate_spoe(
    current_components: dict[str, dict[str, float]],
    previous_components: dict[str, dict[str, float]],
    actual_stats: dict[str, dict[str, float]],
    days_played: int,
) -> list[dict]:
    """Re-score SPOE with the current week's projection prorated.

    The DB stores full 7-day weekly projections. When the current week is
    incomplete, this function scales the current week's contribution by
    ``days_played / 7`` so projected stats match the time period covered
    by actual standings.

    Args:
        current_components: Accumulated components through the current week
            (full 7-day projection). {team: {component: value}}
        previous_components: Accumulated components through the end of the
            previous week. {team: {component: value}}  Missing teams
            are treated as zero-accumulated (week 1 behavior).
        actual_stats: Current standings. {team: {R: val, HR: val, ...}}
        days_played: Days elapsed in the current week (0-7). Pinned to
            the refresh date, not wall-clock time.

    Returns:
        List of result dicts matching spoe_results schema (team, category,
        projected_stat, actual_stat, projected_pts, actual_pts, spoe).
    """
    fraction = max(0, min(days_played, 7)) / 7

    prorated_components: dict[str, dict[str, float]] = {}
    common_teams = set(current_components) & set(actual_stats)

    for team in common_teams:
        prev = previous_components.get(team, {})
        curr = current_components[team]
        prorated_components[team] = {}
        for comp in ALL_COMPONENTS:
            prev_val = prev.get(comp, 0.0)
            curr_week_val = curr.get(comp, 0.0) - prev_val
            prorated_components[team][comp] = prev_val + curr_week_val * fraction

    projected_stats = {
        team: components_to_roto_stats(comps)
        for team, comps in prorated_components.items()
    }

    if len(common_teams) < 2:
        return []

    proj_for_scoring = {t: projected_stats[t] for t in common_teams}
    actual_for_scoring = {t: actual_stats[t] for t in common_teams}

    projected_roto = score_roto(proj_for_scoring)
    actual_roto = score_roto(actual_for_scoring)

    results = []
    for team in common_teams:
        total_spoe = 0.0
        for cat in ALL_CATEGORIES:
            proj_pts = projected_roto[team].get(f"{cat}_pts", 0)
            act_pts = actual_roto[team].get(f"{cat}_pts", 0)
            spoe = act_pts - proj_pts
            total_spoe += spoe
            results.append({
                "team": team,
                "category": cat,
                "projected_stat": projected_stats[team][cat],
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

    return results
