"""Pitcher training table: one row per (pitcher, season, as-of date) -> inputs +
rest-of-season answer (#465). The pitcher twin of ``hitter_ros/table.py``.

As-of dates are the season's first scheduled date (week 0 = preseason: nothing pitched
yet) and every 7 days after it. A row exists for each pitcher with at least one
appearance on or after the as-of date in one of his pitcher-seasons. Until a season is
complete, its week 0 also has a row for every pitcher of the three seasons before,
pitched this season or not: the rows to project it from (next season, once its schedule
is stored). A row with no game after its date has his last team; its answers are empty.

A **pitcher-season** is one where he started a game on the mound, or never started a
game at another position: a position player who mops up is not a pitcher row, and a
two-way starter (Ohtani) is.

Season bounds come from the stored schedule. ``season_complete`` is False while a season
is still being played; its ``ros_*`` answers then cover only the games so far, so
**train only on rows with season_complete**.

Each line is dated as in ``stats.load_games``: a suspended game on the day it was
finished. Everything is stored as **counts**:

* ``std_*``  season to date: games strictly **before** the as-of date.
* ``l14_*``, ``l30_*``  the last 14 and 30 days before the date, same season.
* ``p1_*``   the whole previous season.
* ``p3_*``   the previous three seasons combined.
* ``car_*``  every earlier season in the store.
* ``ros_*``  the answer: games **on or after** the as-of date. Never an input.

Counts: the box-score pitching line (``FULL_PITCHER_FIELDS``), ``g`` (games pitched)
and ``rel_outs`` / ``rel_bf`` (the outs and batters in games he didn't start), so
starter and relief volume can be told apart. And Statcast from the pitcher's side
(``sc_*``, ``PITCH_AGGS``, regular season only): velocity, spin, extension, arm angle
and movement overall and per pitch group (fastball, cutter, breaking, offspeed), pitch
mix, swings, whiffs, called strikes, zone and chase, first-pitch strikes, contact
quality against (exit velo, launch angle, hard hit, barrels, batted-ball types), wOBA
and xwOBA against and run value (which of these exist before 2015: see ``PITCH_AGGS``).
Pitches are dated by their game's date in the pitching lines, so both halves of a row
see the same games. ``p1_in_store``, ``p3_seasons_in_store`` and
``car_seasons_in_store`` say how much of each window the box scores cover (from 2008),
``sc_p1_in_store`` and the other ``sc_`` ones how much Statcast's batted-ball tracking
covers (from 2015), so the model can tell "no history" from "rookie".

Context: the pitcher's team going forward (the team of his first game on or after the
date; the roster as known on the date) and that team's runs scored, games, saves, save
chances, outs and earned runs before the date and last season (``team_*``: run support
for W, the bullpen's save chances for SV). League totals per window (``lg_*``) over
pitcher-seasons. ``age``: Savant's season age (``age_pit``; NULL with no pitch files),
the last one plus the seasons since when a season isn't complete and he hasn't pitched
yet.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd

from fantasy_baseball.analysis.game_logs import FULL_PITCHER_FIELDS
from fantasy_baseball.hitter_ros.statcast_sql import CONTACT_SQL, SWING_SQL, WHIFF_SQL, sql_in
from fantasy_baseball.pitch_data.store import connect
from fantasy_baseball.pitcher_ros.stats import GAME_DATE_SQL

AS_OF_STEP_DAYS = 7
LINE_COUNTS = tuple(FULL_PITCHER_FIELDS.values())
BOX_COUNTS = (*LINE_COUNTS, "g", "rel_outs", "rel_bf")
RECENT_WINDOWS = {"l14": 14, "l30": 30}
TARGET_COUNTS = (
    "g",
    "gs",
    "outs",
    "bf",
    "k",
    "bb",
    "hbp",
    "h",
    "hr",
    "r",
    "er",
    "w",
    "l",
    "sv",
    "svo",
    "hld",
    "bs",
    "rel_outs",
    "rel_bf",
)
# Pitch tracking, from the pitcher's side: every count is ``sc_``-prefixed so it can't
# collide with a box-score count of the same name (``pitches``, ``sb``). Averages are
# stored as a sum and its count (``*_sum`` / ``*_n``), spreads with a sum of squares.
# The store's pitches start in 2008 (PITCHf/x: velocity, movement, zone, run value);
# spin, batted-ball tracking, wOBA and xwOBA start in 2015, arm angle in 2020. Before
# that their ``_n`` (and counts like ``sc_ev_n``, ``sc_woba_n``) are 0: not measured.
FASTBALLS = ("FF", "SI", "FA")
CUTTERS = ("FC",)
BREAKING = ("SL", "ST", "CU", "KC", "SV", "CS")
OFFSPEED = ("CH", "FS", "FO", "SC")
_GROUPS = {"fb": FASTBALLS, "fc": CUTTERS, "brk": BREAKING, "off": OFFSPEED}
_ZONE = "zone BETWEEN 1 AND 9"
_CHASE = "zone BETWEEN 11 AND 14"
_BIP = "type = 'X'"
# The pitch that ended a plate appearance (Savant puts wOBA's denominator there).
_PA_END = "woba_denom = 1"


def _mean_parts(name: str, expr: str, where: str | None = None) -> dict[str, str]:
    """``{name}_n``, ``{name}_sum`` and ``{name}_sq_sum`` aggregates of ``expr``."""
    f = f" FILTER (WHERE {where})" if where else ""
    return {
        f"{name}_n": f"count({expr}){f}",
        f"{name}_sum": f"sum({expr}){f}",
        f"{name}_sq_sum": f"sum(({expr}) * ({expr})){f}",
    }


def _group_aggs() -> dict[str, str]:
    """Per pitch group: how often, swings and whiffs, velocity, spin and movement."""
    out: dict[str, str] = {}
    for g, types in _GROUPS.items():
        is_g = f"pitch_type IN {sql_in(types)}"
        out[f"sc_{g}_pitches"] = f"count(*) FILTER (WHERE {is_g})"
        out[f"sc_{g}_swings"] = f"count(*) FILTER (WHERE {is_g} AND {SWING_SQL})"
        out[f"sc_{g}_whiffs"] = f"count(*) FILTER (WHERE {is_g} AND {WHIFF_SQL})"
        out |= _mean_parts(f"sc_{g}_velo", "release_speed", is_g)
        out |= _mean_parts(f"sc_{g}_spin", "release_spin_rate", is_g)
        # Movement in inches (pfx_* are feet): induced vertical break, and horizontal
        # break by size, so a lefty's and a righty's compare.
        out |= _mean_parts(f"sc_{g}_ivb", "12 * pfx_z", is_g)
        out |= _mean_parts(f"sc_{g}_hbreak", "12 * abs(pfx_x)", is_g)
    return out


# name -> SQL aggregate over one pitcher's pitches
PITCH_AGGS: dict[str, str] = {
    "sc_pitches": "count(*)",
    **_mean_parts("sc_velo", "release_speed"),
    **_mean_parts("sc_ext", "release_extension"),
    **_mean_parts("sc_arm_angle", "arm_angle"),
    **_group_aggs(),
    "sc_swings": f"count(*) FILTER (WHERE {SWING_SQL})",
    "sc_whiffs": f"count(*) FILTER (WHERE {WHIFF_SQL})",
    "sc_called_strikes": "count(*) FILTER (WHERE description = 'called_strike')",
    "sc_zone_pitches": f"count(*) FILTER (WHERE {_ZONE})",
    "sc_zone_swings": f"count(*) FILTER (WHERE {_ZONE} AND {SWING_SQL})",
    "sc_zone_contacts": f"count(*) FILTER (WHERE {_ZONE} AND {CONTACT_SQL})",
    "sc_chase_pitches": f"count(*) FILTER (WHERE {_CHASE})",
    "sc_chase_swings": f"count(*) FILTER (WHERE {_CHASE} AND {SWING_SQL})",
    "sc_first_pitches": "count(*) FILTER (WHERE balls = 0 AND strikes = 0)",
    "sc_first_strikes": (
        "count(*) FILTER (WHERE balls = 0 AND strikes = 0 AND type IN ('S', 'X'))"
    ),
    "sc_pitches_vs_lhb": "count(*) FILTER (WHERE stand = 'L')",
    "sc_bip": f"count(*) FILTER (WHERE {_BIP})",
    **_mean_parts("sc_ev", "launch_speed", _BIP),
    **_mean_parts("sc_la", "launch_angle", _BIP),
    "sc_hard_hit": f"count(*) FILTER (WHERE {_BIP} AND launch_speed >= 95)",
    "sc_barrels": f"count(*) FILTER (WHERE {_BIP} AND launch_speed_angle = 6)",
    "sc_gb": f"count(*) FILTER (WHERE {_BIP} AND bb_type = 'ground_ball')",
    "sc_ld": f"count(*) FILTER (WHERE {_BIP} AND bb_type = 'line_drive')",
    "sc_fb": f"count(*) FILTER (WHERE {_BIP} AND bb_type = 'fly_ball')",
    "sc_pu": f"count(*) FILTER (WHERE {_BIP} AND bb_type = 'popup')",
    # wOBA and xwOBA against, Savant's way, per plate appearance (``woba_denom``): a
    # ball in play counts its expected wOBA, everything else (K, BB, HBP) its actual one.
    "sc_woba_n": f"count(*) FILTER (WHERE {_PA_END})",
    "sc_woba_sum": f"coalesce(sum(woba_value) FILTER (WHERE {_PA_END}), 0)",
    "sc_xwoba_n": (
        f"count(coalesce(estimated_woba_using_speedangle, woba_value)) FILTER (WHERE {_PA_END})"
    ),
    "sc_xwoba_sum": (
        f"sum(coalesce(estimated_woba_using_speedangle, woba_value)) FILTER (WHERE {_PA_END})"
    ),
    # Run value of every pitch, from the pitcher's side (positive = good for him).
    "sc_rv_n": "count(delta_pitcher_run_exp)",
    "sc_rv_sum": "sum(delta_pitcher_run_exp)",
}
PITCH_COUNTS = tuple(PITCH_AGGS)
TEAM_COUNTS = ("team_r", "team_games", "team_sv", "team_svo", "team_outs", "team_er")
LEAGUE_COUNTS = ("bf", "outs", "k", "bb", "hbp", "h", "hr", "r", "er", "w", "sv")

_REQUIRED_VIEWS = ("pitching", "lineups", "schedule")


def _sums(cols: Iterable[str], src: str, prefix: str) -> str:
    return ", ".join(f"coalesce(sum({src}.{c}), 0) AS {prefix}{c}" for c in cols)


def _stage(conn: duckdb.DuckDBPyConnection, *, has_pitches: bool) -> None:
    """Daily per-pitcher sums, pitcher-seasons, team and league days, season bounds and
    the as-of grid."""
    line_sums = ", ".join(f"sum({c}) AS {c}" for c in LINE_COUNTS)
    conn.execute(
        f"""
        CREATE TEMP TABLE box_daily AS
        SELECT player_id, {GAME_DATE_SQL} AS game_date, year({GAME_DATE_SQL}) AS season,
               {line_sums}, count(*) AS g,
               coalesce(sum(outs) FILTER (WHERE gs = 0), 0) AS rel_outs,
               coalesce(sum(bf) FILTER (WHERE gs = 0), 0) AS rel_bf,
               arg_max(team_id, game_pk) AS team_id
        FROM pitching GROUP BY 1, 2, 3
        """
    )
    conn.execute(
        """
        CREATE TEMP TABLE pitcher_seasons AS
        WITH off_mound AS (
            SELECT DISTINCT player_id, year(CAST(game_date AS DATE)) AS season
            FROM lineups WHERE sub_index = 0 AND position IS DISTINCT FROM 'P'
        )
        SELECT b.player_id, b.season FROM box_daily b
        LEFT JOIN off_mound o USING (player_id, season)
        GROUP BY 1, 2
        HAVING sum(b.gs) > 0 OR count(o.player_id) = 0
        """
    )
    # One date per game, the pitching lines' (a suspended game's resume date), so the
    # team's runs line up with its pitching.
    conn.execute(
        f"""
        CREATE TEMP TABLE team_daily AS
        WITH game_dates AS (
            SELECT game_pk, max({GAME_DATE_SQL}) AS game_date FROM pitching GROUP BY 1
        ), batting AS (
            SELECT l.team_id, g.game_date, sum(l.r) AS team_r,
                   count(DISTINCT l.game_pk) AS team_games
            FROM lineups l JOIN game_dates g USING (game_pk) GROUP BY 1, 2
        ), pitching_days AS (
            SELECT team_id, {GAME_DATE_SQL} AS game_date, sum(sv) AS team_sv,
                   sum(svo) AS team_svo, sum(outs) AS team_outs, sum(er) AS team_er
            FROM pitching GROUP BY 1, 2
        )
        SELECT team_id, game_date, year(game_date) AS season,
               coalesce(b.team_r, 0) AS team_r, coalesce(b.team_games, 0) AS team_games,
               coalesce(p.team_sv, 0) AS team_sv, coalesce(p.team_svo, 0) AS team_svo,
               coalesce(p.team_outs, 0) AS team_outs, coalesce(p.team_er, 0) AS team_er
        FROM batting b FULL OUTER JOIN pitching_days p USING (team_id, game_date)
        """
    )
    league_sums = ", ".join(f"sum(b.{c}) AS {c}" for c in LEAGUE_COUNTS)
    conn.execute(
        f"""
        CREATE TEMP TABLE league_daily AS
        SELECT b.season, b.game_date, {league_sums}
        FROM box_daily b JOIN pitcher_seasons ps USING (player_id, season)
        GROUP BY 1, 2
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
    if has_pitches:
        # Dated like the box-score lines, by the game's date in the pitching lines (a
        # suspended game's resume date), so both halves of a row see the same games.
        pitch_sums = ", ".join(f"{agg} AS {name}" for name, agg in PITCH_AGGS.items())
        conn.execute(
            f"""
            CREATE TEMP TABLE pitch_daily AS
            WITH game_dates AS (
                SELECT game_pk, max({GAME_DATE_SQL}) AS game_date FROM pitching GROUP BY 1
            ), dated AS (
                SELECT p.*, coalesce(g.game_date, CAST(p.game_date AS DATE)) AS line_date
                FROM pitches p LEFT JOIN game_dates g USING (game_pk)
                WHERE p.game_type = 'R'
            )
            SELECT pitcher AS player_id, line_date AS game_date,
                   year(line_date) AS season, {pitch_sums}
            FROM dated GROUP BY 1, 2, 3
            """
        )
    else:
        cols = ", ".join(f"CAST(NULL AS DOUBLE) AS {c}" for c in PITCH_COUNTS)
        conn.execute(
            "CREATE TEMP TABLE pitch_daily AS SELECT CAST(NULL AS BIGINT) AS player_id, "
            f"CAST(NULL AS DATE) AS game_date, CAST(NULL AS INTEGER) AS season, {cols} "
            "WHERE false"
        )
    age_src = (
        """SELECT pitcher AS player_id, CAST(season AS INTEGER) AS season,
                  max(age_pit) AS age
           FROM pitches WHERE game_type = 'R' GROUP BY 1, 2"""
        if has_pitches
        else """SELECT CAST(NULL AS BIGINT) AS player_id, CAST(NULL AS INTEGER) AS season,
                       CAST(NULL AS DOUBLE) AS age WHERE false"""
    )
    conn.execute(f"CREATE TEMP TABLE season_age AS {age_src}")


def _season_totals(conn: duckdb.DuckDBPyConnection, daily: str, cols: Iterable[str]) -> str:
    name = f"{daily}_season"
    sums = ", ".join(f"sum({c}) AS {c}" for c in cols)
    conn.execute(
        f"CREATE TEMP TABLE {name} AS SELECT player_id, season, {sums} FROM {daily} GROUP BY 1, 2"
    )
    return name


def build_table(root: Path) -> pd.DataFrame:
    """Build the full pitcher training table from the pitch-data store at ``root``."""
    conn = connect(root)
    try:
        views = {r[0] for r in conn.execute("SELECT view_name FROM duckdb_views()").fetchall()}
        missing = [v for v in _REQUIRED_VIEWS if v not in views]
        if missing:
            raise FileNotFoundError(
                f"pitch-data store {root} has no {', '.join(missing)} files; "
                "fill it with scripts/fetch_pitch_data.py"
            )
        return _build(conn, has_pitches="pitches" in views)
    finally:
        conn.close()


def _build(conn: duckdb.DuckDBPyConnection, *, has_pitches: bool) -> pd.DataFrame:
    _stage(conn, has_pitches=has_pitches)
    box_season = _season_totals(conn, "box_daily", BOX_COUNTS)
    pitch_season = _season_totals(conn, "pitch_daily", PITCH_COUNTS)
    conn.execute(
        "CREATE TEMP TABLE team_daily_season AS SELECT team_id, season, "
        + ", ".join(f"sum({c}) AS {c}" for c in TEAM_COUNTS)
        + " FROM team_daily GROUP BY 1, 2"
    )
    conn.execute(
        "CREATE TEMP TABLE league_season AS SELECT season, "
        + ", ".join(f"sum({c}) AS {c}" for c in LEAGUE_COUNTS)
        + " FROM league_daily GROUP BY 1"
    )

    conn.execute(
        """
        CREATE TEMP TABLE pop AS
        SELECT DISTINCT b.player_id, g.season, g.week, g.as_of, g.first_date, g.last_date,
               g.season_complete
        FROM grid g JOIN box_daily b ON b.season = g.season AND b.game_date >= g.as_of
        WHERE (b.player_id, b.season) IN (SELECT player_id, season FROM pitcher_seasons)
        UNION
        -- A season not complete: week 0 for every pitcher of the three seasons before.
        SELECT DISTINCT ps.player_id, g.season, g.week, g.as_of, g.first_date, g.last_date,
               g.season_complete
        FROM grid g JOIN pitcher_seasons ps ON ps.season BETWEEN g.season - 3 AND g.season - 1
        WHERE g.week = 0 AND NOT g.season_complete
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
        **{
            f"{w}_{kind}": window(source, cols, f"{w}_", cond)
            for kind, daily, season_src, cols in (
                ("box", "box_daily", box_season, BOX_COUNTS),
                ("sc", "pitch_daily", pitch_season, PITCH_COUNTS),
            )
            for w, source, cond in (
                ("std", daily, std),
                ("p1", season_src, p1),
                ("p3", season_src, p3),
                ("car", season_src, car),
                *(
                    (w, daily, f"{std} AND d.game_date >= pop.as_of - {days}")
                    for w, days in RECENT_WINDOWS.items()
                ),
            )
        },
        "ros": window("box_daily", TARGET_COUNTS, "ros_", ros),
    }
    for name, sql in parts.items():
        conn.execute(f"CREATE TEMP TABLE {name} AS {sql}")

    # The team going forward: of his first game on or after the date, else his last.
    conn.execute(
        """
        CREATE TEMP TABLE cur_team AS
        SELECT pop.player_id, pop.season, pop.week,
               CASE WHEN b.game_date IS NULL THEN last.team_id ELSE b.team_id END AS team_id
        FROM pop
        ASOF LEFT JOIN box_daily b
          ON b.player_id = pop.player_id AND pop.as_of <= b.game_date
        ASOF LEFT JOIN box_daily last
          ON last.player_id = pop.player_id AND pop.as_of > last.game_date
        """
    )
    conn.execute(
        f"""
        CREATE TEMP TABLE team_ctx AS
        SELECT pop.player_id, pop.season, pop.week, ct.team_id, {_sums(TEAM_COUNTS, "t", "std_")}
        FROM pop JOIN cur_team ct USING (player_id, season, week)
        LEFT JOIN team_daily t
          ON t.team_id = ct.team_id AND t.season = pop.season AND t.game_date < pop.as_of
        GROUP BY 1, 2, 3, 4
        """
    )
    conn.execute(
        f"""
        CREATE TEMP TABLE team_ctx_p1 AS
        SELECT pop.player_id, pop.season, pop.week, {_sums(TEAM_COUNTS, "t", "p1_")}
        FROM pop JOIN cur_team ct USING (player_id, season, week)
        LEFT JOIN team_daily_season t ON t.team_id = ct.team_id AND t.season = pop.season - 1
        GROUP BY 1, 2, 3
        """
    )

    def league(prefix: str, cond: str) -> str:
        return f"""
            SELECT s.season, {_sums(LEAGUE_COUNTS, "l", f"lg_{prefix}_")}
            FROM (SELECT DISTINCT season FROM pop) s
            LEFT JOIN league_season l ON {cond}
            GROUP BY 1
        """

    conn.execute(f"CREATE TEMP TABLE league_p1 AS {league('p1', 'l.season = s.season - 1')}")
    conn.execute(
        "CREATE TEMP TABLE league_p3 AS "
        + league("p3", "l.season BETWEEN s.season - 3 AND s.season - 1")
    )
    conn.execute(f"CREATE TEMP TABLE league_car AS {league('car', 'l.season < s.season')}")
    conn.execute(
        f"""
        CREATE TEMP TABLE league_std AS
        SELECT pop.player_id, pop.season, pop.week, {_sums(LEAGUE_COUNTS, "l", "lg_std_")}
        FROM pop LEFT JOIN league_daily l
          ON l.season = pop.season AND l.game_date < pop.as_of
        GROUP BY 1, 2, 3
        """
    )

    first_store_season = "(SELECT min(season) FROM box_daily)"
    # Statcast's batted-ball tracking starts later (2015); NULL with no pitch files.
    first_sc_season = "(SELECT min(season) FROM pitch_daily WHERE sc_ev_n > 0)"
    joined = ", ".join(f"{p}.* EXCLUDE (player_id, season, week)" for p in parts)
    df = conn.execute(
        f"""
        SELECT pop.player_id, pop.season, pop.week, pop.as_of, pop.season_complete,
               (pop.last_date - pop.as_of + 1) / (pop.last_date - pop.first_date + 1)
                   AS frac_season_left,
               pop.season - 1 >= {first_store_season} AS p1_in_store,
               least(3, pop.season - {first_store_season}) AS p3_seasons_in_store,
               pop.season - {first_store_season} AS car_seasons_in_store,
               pop.season - 1 >= {first_sc_season} AS sc_p1_in_store,
               greatest(0, least(3, pop.season - {first_sc_season})) AS sc_p3_seasons_in_store,
               greatest(0, pop.season - {first_sc_season}) AS sc_car_seasons_in_store,
               CASE WHEN a.age IS NULL AND NOT pop.season_complete
                    THEN a1.age + pop.season - a1.season ELSE a.age END AS age,
               tc.team_id, tc.* EXCLUDE (player_id, season, week, team_id),
               tp.* EXCLUDE (player_id, season, week),
               {joined},
               ls.* EXCLUDE (player_id, season, week),
               p1l.* EXCLUDE (season), p3l.* EXCLUDE (season), carl.* EXCLUDE (season)
        FROM pop
        {" ".join(f"JOIN {p} USING (player_id, season, week)" for p in parts)}
        JOIN team_ctx tc USING (player_id, season, week)
        JOIN team_ctx_p1 tp USING (player_id, season, week)
        JOIN league_std ls USING (player_id, season, week)
        JOIN league_p1 p1l ON p1l.season = pop.season
        JOIN league_p3 p3l ON p3l.season = pop.season
        JOIN league_car carl ON carl.season = pop.season
        LEFT JOIN season_age a ON a.player_id = pop.player_id AND a.season = pop.season
        ASOF LEFT JOIN season_age a1 ON a1.player_id = pop.player_id AND pop.season > a1.season
        ORDER BY pop.season, pop.week, pop.player_id
        """
    ).df()
    return df
