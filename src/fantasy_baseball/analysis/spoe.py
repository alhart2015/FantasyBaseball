"""Standings Points Over Expected (SPOE) — luck quantification."""

from __future__ import annotations

from datetime import date

import pandas as pd

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import ALL_CATEGORIES, RATE_STATS
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

# Components tracked for accumulation
HITTER_COMPONENTS = ["r", "hr", "rbi", "sb", "h", "ab"]
PITCHER_COMPONENTS = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]
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
    # Find the best ROS snapshot date on or before target_date
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
        # Fall back to preseason blended projections
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
    stat_cols = ["h", "ab", "r", "hr", "rbi", "sb", "ip", "k", "er", "bb", "h_allowed", "w", "sv"]
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
