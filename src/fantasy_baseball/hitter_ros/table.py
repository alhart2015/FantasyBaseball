"""Training table: one row per (hitter, season, as-of date) -> inputs + rest-of-season answer (#403).

As-of dates are the season's first scheduled date (week 0 = preseason: nothing played
yet) and every 7 days after it. A row exists for each hitter with at least one lineup
appearance on or after the as-of date that season. A hitter-season is one where he
started a game at a position other than P, or had a non-pitching lineup role and threw
no pitches that season -- so a pitcher who pinch-runs is not a hitter row.

Season bounds come from the stored schedule (``schedule/YYYY.parquet``), not from the
games on disk. ``season_complete`` is False while a season is still being played; its
``ros_*`` answers then cover only the games played so far, so **train only on rows with
season_complete**. Its inputs are still right for predicting.

Everything is stored as **counts**, not rates, so the model code can choose its own
rates, smoothing and weights without rebuilding the table:

* ``std_*``  season to date: games strictly **before** the as-of date.
* ``p1_*``   the whole previous season.
* ``p3_*``   the previous three seasons combined.
* ``car_*``  every earlier season in the store.
* ``ros_*``  the answer: games **on or after** the as-of date. Never an input.

Count families: box-score batting line (``pa``, ``hr``, ``r``, ``rbi``, ``sb`` ...,
from ``lineups``) and pitch-level stats (swings, whiffs, chases, exit velo and launch
angle sums, barrels, batted-ball types, pulled air balls, xwOBA sums ..., from
``pitches``). Sums of squares (``ev_sq_sum``, ``la_sq_sum``) let the model rebuild a
standard deviation. Pitch counts use regular-season games only.

The store starts in 2015, so earlier history reads as zero. ``p1_in_store``,
``p3_seasons_in_store`` and ``car_seasons_in_store`` say how much of each window the
store actually covers, so the model can tell "no history" from "rookie".

Context columns: sprint speed for the two previous seasons, NULL when unknown (the
current season's leaderboard is end-of-season, so it would leak), the hitter's team
going forward (the team of his first game on or after the date) and that team's runs,
PA and games before the date and last season.

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
# hit-coordinate origin (125.42, 198.27) is home plate. atan2, not atan of the ratio:
# a ball fielded behind home's y origin (hc_y > 198.27) would otherwise flip sides.
_SPRAY = "degrees(atan2(hc_x - 125.42, 198.27 - hc_y))"
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
    "xba_n": f"count(estimated_ba_using_speedangle) FILTER (WHERE {_BIP})",
    "xba_sum": f"sum(estimated_ba_using_speedangle) FILTER (WHERE {_BIP})",
    "bat_speed_n": "count(bat_speed)",
    "bat_speed_sum": "sum(bat_speed)",
}
PITCH_COUNTS = tuple(PITCH_AGGS)
TEAM_COUNTS = ("team_r", "team_pa", "team_games")

_REQUIRED_VIEWS = ("pitches", "lineups", "schedule")


def _sums(cols: Iterable[str], src: str, prefix: str) -> str:
    return ", ".join(f"coalesce(sum({src}.{c}), 0) AS {prefix}{c}" for c in cols)


def _stage(conn: duckdb.DuckDBPyConnection, *, has_sprint: bool) -> None:
    """Daily per-player sums, hitter-seasons, season bounds and the as-of grid."""
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
               bool_or(sub_index = 0 AND position IS DISTINCT FROM 'P') AS started_off_mound,
               bool_or(position IS DISTINCT FROM 'P') AS non_pitcher_role
        FROM lineups GROUP BY player_id, CAST(game_date AS DATE)
        """
    )
    pitch_sums = ", ".join(f"{agg} AS {name}" for name, agg in PITCH_AGGS.items())
    conn.execute(
        f"""
        CREATE TEMP TABLE pitch_daily AS
        SELECT batter AS player_id, CAST(game_date AS DATE) AS game_date,
               CAST(season AS INTEGER) AS season, {pitch_sums}, max(age_bat) AS age_bat
        FROM pitches WHERE game_type = 'R'
        GROUP BY batter, CAST(game_date AS DATE), season
        """
    )
    conn.execute(
        """
        CREATE TEMP TABLE season_age AS
        SELECT player_id, season, max(age_bat) AS age FROM pitch_daily GROUP BY 1, 2
        """
    )
    conn.execute(
        """
        CREATE TEMP TABLE pitcher_seasons AS
        SELECT DISTINCT pitcher AS player_id, CAST(season AS INTEGER) AS season
        FROM pitches WHERE game_type = 'R'
        """
    )
    conn.execute(
        """
        CREATE TEMP TABLE hitter_seasons AS
        SELECT b.player_id, b.season FROM box_daily b
        LEFT JOIN pitcher_seasons p USING (player_id, season)
        GROUP BY b.player_id, b.season
        HAVING bool_or(b.started_off_mound)
            OR (bool_or(b.non_pitcher_role) AND count(p.player_id) = 0)
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
        SELECT CAST(s.season AS INTEGER) AS season,
               CAST(s.first_date AS DATE) AS first_date,
               CAST(s.last_date AS DATE) AS last_date,
               coalesce(max(b.game_date) >= CAST(s.last_date AS DATE), false) AS season_complete
        FROM schedule s LEFT JOIN box_daily b ON b.season = s.season
        GROUP BY 1, 2, 3
        """
    )
    conn.execute(
        f"""
        CREATE TEMP TABLE grid AS
        SELECT s.season, s.first_date, s.last_date, s.season_complete,
               CAST(s.first_date + CAST(g.k * {AS_OF_STEP_DAYS} AS INTEGER) AS DATE) AS as_of,
               g.k AS week
        FROM seasons s,
             LATERAL (SELECT unnest(range(0, (s.last_date - s.first_date) // {AS_OF_STEP_DAYS} + 1)) AS k) g
        """
    )
    # Sprint speed: always the same columns, NULL when unknown or not fetched.
    sprint_src = (
        """SELECT CAST(player_id AS BIGINT) AS player_id, CAST(season AS INTEGER) AS season,
                  sprint_speed, competitive_runs FROM sprint_speed"""
        if has_sprint
        else """SELECT CAST(NULL AS BIGINT) AS player_id, CAST(NULL AS INTEGER) AS season,
                       CAST(NULL AS DOUBLE) AS sprint_speed,
                       CAST(NULL AS BIGINT) AS competitive_runs WHERE false"""
    )
    conn.execute(f"CREATE TEMP TABLE sprint AS {sprint_src}")


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
    try:
        views = {r[0] for r in conn.execute("SELECT view_name FROM duckdb_views()").fetchall()}
        missing = [v for v in _REQUIRED_VIEWS if v not in views]
        if missing:
            raise FileNotFoundError(
                f"pitch-data store {root} has no {', '.join(missing)} files; "
                "fill it with scripts/fetch_pitch_data.py"
            )
        return _build(conn, has_sprint="sprint_speed" in views)
    finally:
        conn.close()


def _build(conn: duckdb.DuckDBPyConnection, *, has_sprint: bool) -> pd.DataFrame:
    _stage(conn, has_sprint=has_sprint)
    box_season = _season_totals(conn, "box_daily", BOX_COUNTS)
    pitch_season = _season_totals(conn, "pitch_daily", PITCH_COUNTS)

    conn.execute(
        """
        CREATE TEMP TABLE pop AS
        SELECT DISTINCT b.player_id, g.season, g.week, g.as_of, g.first_date, g.last_date,
               g.season_complete
        FROM grid g JOIN box_daily b ON b.season = g.season AND b.game_date >= g.as_of
        WHERE (b.player_id, b.season) IN (SELECT player_id, season FROM hitter_seasons)
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
    car = "d.season < pop.season"
    parts = {
        "std_box": window("box_daily", BOX_COUNTS, "std_", std),
        "std_pitch": window("pitch_daily", PITCH_COUNTS, "std_", std),
        "p1_box": window(box_season, BOX_COUNTS, "p1_", p1),
        "p1_pitch": window(pitch_season, PITCH_COUNTS, "p1_", p1),
        "p3_box": window(box_season, BOX_COUNTS, "p3_", p3),
        "p3_pitch": window(pitch_season, PITCH_COUNTS, "p3_", p3),
        "car_box": window(box_season, BOX_COUNTS, "car_", car),
        "car_pitch": window(pitch_season, PITCH_COUNTS, "car_", car),
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

    first_store_season = "(SELECT min(season) FROM box_daily)"
    joined = ", ".join(f"{p}.* EXCLUDE (player_id, season, week)" for p in parts)
    df = conn.execute(
        f"""
        SELECT pop.player_id, pop.season, pop.week, pop.as_of, pop.season_complete,
               (pop.last_date - pop.as_of + 1) / (pop.last_date - pop.first_date + 1)
                   AS frac_season_left,
               pop.season - 1 >= {first_store_season} AS p1_in_store,
               least(3, pop.season - {first_store_season}) AS p3_seasons_in_store,
               pop.season - {first_store_season} AS car_seasons_in_store,
               a.age,
               s1.sprint_speed AS p1_sprint_speed, s1.competitive_runs AS p1_sprint_runs,
               s2.sprint_speed AS p2_sprint_speed, s2.competitive_runs AS p2_sprint_runs,
               tc.team_id, tc.std_team_r, tc.std_team_pa, tc.std_team_games,
               tp.p1_team_r, tp.p1_team_pa, tp.p1_team_games,
               {joined}
        FROM pop
        {" ".join(f"JOIN {p} USING (player_id, season, week)" for p in parts)}
        JOIN team_ctx tc USING (player_id, season, week)
        JOIN team_ctx_p1 tp USING (player_id, season, week)
        LEFT JOIN season_age a ON a.player_id = pop.player_id AND a.season = pop.season
        LEFT JOIN sprint s1 ON s1.player_id = pop.player_id AND s1.season = pop.season - 1
        LEFT JOIN sprint s2 ON s2.player_id = pop.player_id AND s2.season = pop.season - 2
        ORDER BY pop.season, pop.week, pop.player_id
        """
    ).df()
    return df
