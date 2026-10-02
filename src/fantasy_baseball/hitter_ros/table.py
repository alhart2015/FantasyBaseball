"""Training table: one row per (hitter, season, as-of date) -> inputs + rest-of-season answer (#403).

As-of dates are the season's first game date (week 0 = preseason: nothing played yet)
and every 7 days after it. A row exists for each hitter with at least one lineup
appearance on or after the as-of date that season.

Everything is stored as **counts**, not rates, so the model code can choose its own
rates, smoothing and weights without rebuilding the table:

* ``std_*``  season to date: games strictly **before** the as-of date.
* ``p1_*``   the whole previous season.
* ``p3_*``   the previous three seasons combined.
* ``ros_*``  the answer: games **on or after** the as-of date. Never an input.

Count families: box-score batting line (``pa``, ``hr``, ``r``, ``rbi``, ``sb`` ...,
from ``lineups``) and pitch-level stats (swings, whiffs, chases, exit velo and launch
angle sums, barrels, batted-ball types, pulled air balls, xwOBA sums ..., from
``pitches``). Sums of squares (``ev_sq_sum``, ``la_sq_sum``) let the model rebuild a
standard deviation. Pitch counts use regular-season games only.

Context columns: sprint speed for the two previous seasons (the current season's
leaderboard is end-of-season, so it would leak), the hitter's team going forward (the
team of his first game on or after the date) and that team's runs, PA and games before
the date and last season.

Two inputs come from on or after the date, both known on the date itself: ``team_id``
(the roster) and ``age`` (Savant's season age, ``age_bat``, constant within a season).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd

from fantasy_baseball.analysis.game_logs import FULL_HITTER_FIELDS
from fantasy_baseball.keepers.savant import (
    CONTACT_DESCRIPTIONS,
    SWING_DESCRIPTIONS,
    WHIFF_DESCRIPTIONS,
)
from fantasy_baseball.pitch_data.store import connect

AS_OF_STEP_DAYS = 7

BOX_COUNTS = (*FULL_HITTER_FIELDS.values(), "games", "starts", "spot_sum")
TARGET_COUNTS = (
    "pa",
    "ab",
    "h",
    "b2",
    "b3",
    "hr",
    "r",
    "rbi",
    "sb",
    "cs",
    "bb",
    "k",
    "hbp",
    "sf",
    "games",
)

# Spray angle in degrees from home plate; negative = toward left field. Savant's
# hit-coordinate origin (125.42, 198.27) is home plate.
_SPRAY = "degrees(atan((hc_x - 125.42) / (198.27 - hc_y)))"
_PULL_DEG = 15


def _in(values: Iterable[str]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in sorted(values)) + ")"


_SWING = f"description IN {_in(SWING_DESCRIPTIONS)}"
_WHIFF = f"description IN {_in(WHIFF_DESCRIPTIONS)}"
_CONTACT = f"description IN {_in(CONTACT_DESCRIPTIONS)}"
_ZONE = "zone BETWEEN 1 AND 9"
_CHASE = "zone BETWEEN 11 AND 14"
_BIP = "type = 'X'"
_AIR = "bb_type IN ('fly_ball', 'line_drive')"
_PULLED = f"((stand = 'R' AND {_SPRAY} < -{_PULL_DEG}) OR (stand = 'L' AND {_SPRAY} > {_PULL_DEG}))"
_OPPO = f"((stand = 'R' AND {_SPRAY} > {_PULL_DEG}) OR (stand = 'L' AND {_SPRAY} < -{_PULL_DEG}))"

# name -> SQL aggregate over one batter's pitches
PITCH_AGGS: dict[str, str] = {
    "pitches": "count(*)",
    "swings": f"count(*) FILTER (WHERE {_SWING})",
    "whiffs": f"count(*) FILTER (WHERE {_WHIFF})",
    "zone_pitches": f"count(*) FILTER (WHERE {_ZONE})",
    "zone_swings": f"count(*) FILTER (WHERE {_ZONE} AND {_SWING})",
    "zone_contacts": f"count(*) FILTER (WHERE {_ZONE} AND {_CONTACT})",
    "chase_pitches": f"count(*) FILTER (WHERE {_CHASE})",
    "chase_swings": f"count(*) FILTER (WHERE {_CHASE} AND {_SWING})",
    "chase_contacts": f"count(*) FILTER (WHERE {_CHASE} AND {_CONTACT})",
    "first_pitch_swings": f"count(*) FILTER (WHERE balls = 0 AND strikes = 0 AND {_SWING})",
    "first_pitches": "count(*) FILTER (WHERE balls = 0 AND strikes = 0)",
    "pitches_vs_lhp": "count(*) FILTER (WHERE p_throws = 'L')",
    "pitches_as_lhb": "count(*) FILTER (WHERE stand = 'L')",
    "bip": f"count(*) FILTER (WHERE {_BIP})",
    "ev_n": f"count(launch_speed) FILTER (WHERE {_BIP})",
    "ev_sum": f"sum(launch_speed) FILTER (WHERE {_BIP})",
    "ev_sq_sum": f"sum(launch_speed * launch_speed) FILTER (WHERE {_BIP})",
    "la_n": f"count(launch_angle) FILTER (WHERE {_BIP})",
    "la_sum": f"sum(launch_angle) FILTER (WHERE {_BIP})",
    "la_sq_sum": f"sum(launch_angle * launch_angle) FILTER (WHERE {_BIP})",
    "ev95": f"count(*) FILTER (WHERE {_BIP} AND launch_speed >= 95)",
    "ev100": f"count(*) FILTER (WHERE {_BIP} AND launch_speed >= 100)",
    "ev105": f"count(*) FILTER (WHERE {_BIP} AND launch_speed >= 105)",
    "barrels": f"count(*) FILTER (WHERE {_BIP} AND launch_speed_angle = 6)",
    "solid": f"count(*) FILTER (WHERE {_BIP} AND launch_speed_angle = 5)",
    "sweet_spot": f"count(*) FILTER (WHERE {_BIP} AND launch_angle BETWEEN 8 AND 32)",
    "gb": f"count(*) FILTER (WHERE {_BIP} AND bb_type = 'ground_ball')",
    "ld": f"count(*) FILTER (WHERE {_BIP} AND bb_type = 'line_drive')",
    "fb": f"count(*) FILTER (WHERE {_BIP} AND bb_type = 'fly_ball')",
    "pu": f"count(*) FILTER (WHERE {_BIP} AND bb_type = 'popup')",
    "spray_n": f"count(*) FILTER (WHERE {_BIP} AND hc_x IS NOT NULL AND hc_y IS NOT NULL)",
    "pulled": f"count(*) FILTER (WHERE {_BIP} AND {_PULLED})",
    "oppo": f"count(*) FILTER (WHERE {_BIP} AND {_OPPO})",
    "pulled_air": f"count(*) FILTER (WHERE {_BIP} AND {_AIR} AND {_PULLED})",
    "xwoba_n": f"count(estimated_woba_using_speedangle) FILTER (WHERE {_BIP})",
    "xwoba_sum": f"sum(estimated_woba_using_speedangle) FILTER (WHERE {_BIP})",
    "xba_sum": f"sum(estimated_ba_using_speedangle) FILTER (WHERE {_BIP})",
    "bat_speed_n": "count(bat_speed)",
    "bat_speed_sum": "sum(bat_speed)",
}
PITCH_COUNTS = tuple(PITCH_AGGS)
TEAM_COUNTS = ("team_r", "team_pa", "team_games")


def _sums(cols: Iterable[str], src: str, prefix: str) -> str:
    return ", ".join(f"coalesce(sum({src}.{c}), 0) AS {prefix}{c}" for c in cols)


def _stage(conn: duckdb.DuckDBPyConnection) -> None:
    """Daily per-player sums, season bounds and the as-of grid, as temp tables."""
    box_sums = ", ".join(f"sum({c}) AS {c}" for c in FULL_HITTER_FIELDS.values())
    conn.execute(
        f"""
        CREATE TEMP TABLE box_daily AS
        SELECT player_id, CAST(game_date AS DATE) AS game_date,
               year(CAST(game_date AS DATE)) AS season,
               {box_sums},
               count(DISTINCT game_pk) AS games,
               count(*) FILTER (WHERE sub_index = 0) AS starts,
               coalesce(sum(lineup_spot) FILTER (WHERE sub_index = 0), 0) AS spot_sum,
               arg_max(team_id, game_pk) AS team_id,
               bool_or(position IS DISTINCT FROM 'P') AS batted_as_hitter
        FROM lineups GROUP BY player_id, CAST(game_date AS DATE)
        """
    )
    pitch_sums = ", ".join(f"{agg} AS {name}" for name, agg in PITCH_AGGS.items())
    conn.execute(
        f"""
        CREATE TEMP TABLE pitch_daily AS
        SELECT batter AS player_id, CAST(game_date AS DATE) AS game_date,
               CAST(season AS INTEGER) AS season, {pitch_sums}
        FROM pitches WHERE game_type = 'R'
        GROUP BY batter, CAST(game_date AS DATE), season
        """
    )
    conn.execute(
        """
        CREATE TEMP TABLE season_age AS
        SELECT batter AS player_id, CAST(season AS INTEGER) AS season, max(age_bat) AS age
        FROM pitches WHERE game_type = 'R' GROUP BY 1, 2
        """
    )
    conn.execute(
        """
        CREATE TEMP TABLE team_daily AS
        SELECT team_id, CAST(game_date AS DATE) AS game_date,
               year(CAST(game_date AS DATE)) AS season,
               sum(r) AS team_r, sum(pa) AS team_pa, count(DISTINCT game_pk) AS team_games
        FROM lineups GROUP BY 1, 2
        """
    )
    conn.execute(
        """
        CREATE TEMP TABLE seasons AS
        SELECT season, min(game_date) AS first_date, max(game_date) AS last_date
        FROM box_daily GROUP BY season
        """
    )
    conn.execute(
        f"""
        CREATE TEMP TABLE grid AS
        SELECT s.season, s.first_date, s.last_date,
               CAST(s.first_date + CAST(g.k * {AS_OF_STEP_DAYS} AS INTEGER) AS DATE) AS as_of,
               g.k AS week
        FROM seasons s,
             LATERAL (SELECT unnest(range(0, (s.last_date - s.first_date) // {AS_OF_STEP_DAYS} + 1)) AS k) g
        """
    )


def _season_totals(conn: duckdb.DuckDBPyConnection, daily: str, cols: Iterable[str]) -> str:
    name = f"{daily}_season"
    sums = ", ".join(f"sum({c}) AS {c}" for c in cols)
    conn.execute(
        f"CREATE TEMP TABLE {name} AS SELECT player_id, season, {sums} FROM {daily} GROUP BY 1, 2"
    )
    return name


def build_table(root: Path) -> pd.DataFrame:
    """Build the full training table from the pitch-data store at ``root``."""
    conn = connect(root)
    _stage(conn)
    box_season = _season_totals(conn, "box_daily", BOX_COUNTS)
    pitch_season = _season_totals(conn, "pitch_daily", PITCH_COUNTS)

    conn.execute(
        """
        CREATE TEMP TABLE pop AS
        SELECT DISTINCT b.player_id, g.season, g.week, g.as_of, g.first_date, g.last_date
        FROM grid g JOIN box_daily b ON b.season = g.season AND b.game_date >= g.as_of
        WHERE (b.player_id, b.season) IN (
            SELECT player_id, season FROM box_daily GROUP BY 1, 2 HAVING bool_or(batted_as_hitter)
        )
        """
    )

    def window(source: str, cols: Iterable[str], prefix: str, cond: str) -> str:
        """Per-pop-row sums of ``source`` over the rows matching ``cond``."""
        return f"""
            SELECT pop.player_id, pop.season, pop.week, {_sums(cols, "d", prefix)}
            FROM pop LEFT JOIN {source} d ON d.player_id = pop.player_id AND {cond}
            GROUP BY 1, 2, 3
        """

    std = "d.season = pop.season AND d.game_date < pop.as_of"
    ros = "d.season = pop.season AND d.game_date >= pop.as_of"
    p1 = "d.season = pop.season - 1"
    p3 = "d.season BETWEEN pop.season - 3 AND pop.season - 1"
    parts = {
        "std_box": window("box_daily", BOX_COUNTS, "std_", std),
        "std_pitch": window("pitch_daily", PITCH_COUNTS, "std_", std),
        "p1_box": window(box_season, BOX_COUNTS, "p1_", p1),
        "p1_pitch": window(pitch_season, PITCH_COUNTS, "p1_", p1),
        "p3_box": window(box_season, BOX_COUNTS, "p3_", p3),
        "p3_pitch": window(pitch_season, PITCH_COUNTS, "p3_", p3),
        "ros": window("box_daily", TARGET_COUNTS, "ros_", ros),
    }
    for name, sql in parts.items():
        conn.execute(f"CREATE TEMP TABLE {name} AS {sql}")

    # The hitter's team going forward: the team of his first game on or after the date.
    # That is the roster as known on the date (it catches offseason and deadline moves);
    # the team's stats below still use only games before the date.
    conn.execute(
        """
        CREATE TEMP TABLE cur_team AS
        SELECT pop.player_id, pop.season, pop.week, b.team_id
        FROM pop ASOF LEFT JOIN box_daily b
          ON b.player_id = pop.player_id AND pop.as_of <= b.game_date
        """
    )
    team_std = _sums(TEAM_COUNTS, "t", "std_")
    team_p1 = _sums(TEAM_COUNTS, "t1", "p1_")
    conn.execute(
        f"""
        CREATE TEMP TABLE team_ctx AS
        SELECT pop.player_id, pop.season, pop.week, ct.team_id, {team_std}
        FROM pop JOIN cur_team ct USING (player_id, season, week)
        LEFT JOIN team_daily t
          ON t.team_id = ct.team_id AND t.season = pop.season AND t.game_date < pop.as_of
        GROUP BY 1, 2, 3, 4
        """
    )
    conn.execute(
        f"""
        CREATE TEMP TABLE team_ctx_p1 AS
        SELECT pop.player_id, pop.season, pop.week, {team_p1}
        FROM pop JOIN cur_team ct USING (player_id, season, week)
        LEFT JOIN team_daily t1 ON t1.team_id = ct.team_id AND t1.season = pop.season - 1
        GROUP BY 1, 2, 3
        """
    )

    sprint = """
        SELECT CAST(player_id AS BIGINT) AS player_id, CAST(season AS INTEGER) AS season,
               sprint_speed, competitive_runs
        FROM sprint_speed
    """
    has_sprint = "sprint_speed" in {
        r[0] for r in conn.execute("SELECT view_name FROM duckdb_views()").fetchall()
    }
    sprint_cols = (
        """
        s1.sprint_speed AS p1_sprint_speed, s1.competitive_runs AS p1_sprint_runs,
        s2.sprint_speed AS p2_sprint_speed, s2.competitive_runs AS p2_sprint_runs,
        """
        if has_sprint
        else ""
    )
    sprint_joins = (
        f"""
        LEFT JOIN ({sprint}) s1 ON s1.player_id = pop.player_id AND s1.season = pop.season - 1
        LEFT JOIN ({sprint}) s2 ON s2.player_id = pop.player_id AND s2.season = pop.season - 2
        """
        if has_sprint
        else ""
    )

    joined = ", ".join(f"{p}.* EXCLUDE (player_id, season, week)" for p in parts)
    df = conn.execute(
        f"""
        SELECT pop.player_id, pop.season, pop.week, pop.as_of,
               (pop.last_date - pop.as_of + 1) / (pop.last_date - pop.first_date + 1)
                   AS frac_season_left,
               a.age, {sprint_cols}
               tc.team_id, tc.std_team_r, tc.std_team_pa, tc.std_team_games,
               tp.p1_team_r, tp.p1_team_pa, tp.p1_team_games,
               {joined}
        FROM pop
        {" ".join(f"JOIN {p} USING (player_id, season, week)" for p in parts)}
        JOIN team_ctx tc USING (player_id, season, week)
        JOIN team_ctx_p1 tp USING (player_id, season, week)
        LEFT JOIN season_age a ON a.player_id = pop.player_id AND a.season = pop.season
        {sprint_joins}
        ORDER BY pop.season, pop.week, pop.player_id
        """
    ).df()
    return df
