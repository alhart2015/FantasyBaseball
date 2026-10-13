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
* ``milb_p1_*``, ``milb_p3_*``  minor-league pitching (``MILB_COUNTS``), the previous
  season and the previous three, every level summed with each line's league-expected
  counts beside it, so a rate can be read against his leagues' (``_stage_milb``). ``milb_p1_in_store`` and
  ``milb_p3_seasons_in_store`` say how much of each window the store covers (there was
  no minor-league season in 2020), so "no minor-league line" can be told from "not
  stored".
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
``car_seasons_in_store`` say how much of each window the box scores cover (from 2008);
the ``pitchtrack_`` ones the same for pitch tracking (velocity, movement, whiffs: from
the store's first pitch season, 2008) and the ``battedtrack_`` ones for Statcast's
batted-ball tracking (exit velo, xwOBA: 2015), all no coverage without pitch files, so
the model can tell "no history" or "not measured" from "rookie".

Context: the pitcher's team going forward (``ros_table.stage_current_team``: the team
of his next game when it is at most ``TEAM_LOOKAHEAD_DAYS`` off, else his last team, so
a team he joins later while hurt or in the minors isn't read from the future) and that
team's runs scored, games, saves, save
chances, outs, earned runs and wins before the date and last season (``team_*``: run
support for W, the bullpen's save chances for SV), and its games left on the stored
schedule (``team_games_left``). For the saves model (#474), answers: the team's saves
and games from the date on (``ros_team_sv``, ``ros_team_games``) and his saves for that
team (``ros_sv_for_team``). League totals per window (``lg_*``) over
pitcher-seasons. ``age``: Savant's season age (``age_pit``; NULL with no pitch files),
the last one plus the seasons since when a season isn't complete and he hasn't pitched
yet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from fantasy_baseball.analysis.game_logs import FULL_PITCHER_FIELDS
from fantasy_baseball.hitter_ros.milb_features import LEVEL_RANK
from fantasy_baseball.hitter_ros.milb_grade import EXCLUDED_LEAGUES, MIN_LEAGUE_PA
from fantasy_baseball.hitter_ros.statcast_sql import CONTACT_SQL, SWING_SQL, WHIFF_SQL, sql_in
from fantasy_baseball.pitch_data.store import connect
from fantasy_baseball.pitcher_ros.stats import GAME_DATE_SQL
from fantasy_baseball.ros_table import (
    AGE_JOINS,
    AGE_SELECT,
    CAR,
    P1,
    P3,
    ROS,
    STD,
    coverage_sql,
    history_sql,
    league_window_sql,
    season_totals,
    stage_current_team,
    stage_grid,
    stage_population,
    sums,
    window_sql,
)

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
    "sc_woba_sum": f"sum(woba_value) FILTER (WHERE {_PA_END})",
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
TEAM_COUNTS = ("team_r", "team_games", "team_sv", "team_svo", "team_outs", "team_er", "team_w")
LEAGUE_COUNTS = ("bf", "outs", "k", "bb", "hbp", "h", "hr", "r", "er", "w", "sv")

# Minor-league pitching (#465). The line's counts, then its level's expected counts
# (``x*``: the level's rate that season times his batters faced, or outs for ER), the
# level rank (1 = AAA .. 6 = Rookie) and age gap to the level's mean (each times
# batters faced), and the batters faced the age gap covers (lines with an age).
MILB_LINE = {
    "bf": "battersFaced",
    "k": "strikeOuts",
    "bb": "baseOnBalls",
    "hr": "homeRuns",
    "h": "hits",
    "er": "earnedRuns",
    "outs": "outs",
    "g": "gamesPitched",
    "gs": "gamesStarted",
}
MILB_EXPECTED = ("k", "bb", "hr", "h", "er")
MILB_COUNTS = (
    *MILB_LINE,
    *(f"x{s}" for s in MILB_EXPECTED),
    "level_bf",
    "age_bf",
    "age_gap_bf",
)
# A league smaller than this that season (a few stray lines filed under the wrong
# league) is read against its level's pooled rates instead; the hitters' threshold, a
# batter faced being a plate appearance from the other side.
MILB_MIN_LEAGUE_BF = MIN_LEAGUE_PA
# Lines of these position types are position players pitching, left out like their
# major-league mop-up outings (the table's pitcher-seasons).
MILB_NOT_PITCHERS = ("Infielder", "Outfielder", "Catcher", "Hitter")
MILB_WINDOWS = {"p1": P1, "p3": P3}

_REQUIRED_VIEWS = ("pitching", "lineups", "schedule")
logger = logging.getLogger(__name__)
# The team going forward is the team of his next game only when it is at most this many
# days off (ros_table.stage_current_team).
TEAM_LOOKAHEAD_DAYS = 14


def _stage(conn: duckdb.DuckDBPyConnection, *, has_pitches: bool) -> None:
    """Daily per-pitcher sums, pitcher-seasons, team and league days, season bounds and
    the as-of grid."""
    # One date per game, the pitching lines' (a suspended game's resume date): the
    # team's runs and the pitches are dated by it, so every part of a row sees the same
    # games.
    conn.execute(
        f"CREATE TEMP TABLE game_dates AS SELECT game_pk, max({GAME_DATE_SQL}) AS game_date "
        "FROM pitching GROUP BY 1"
    )
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
    conn.execute(
        f"""
        CREATE TEMP TABLE team_daily AS
        WITH batting AS (
            SELECT l.team_id, g.game_date, sum(l.r) AS team_r,
                   count(DISTINCT l.game_pk) AS team_games
            FROM lineups l JOIN game_dates g USING (game_pk) GROUP BY 1, 2
        ), pitching_days AS (
            SELECT team_id, {GAME_DATE_SQL} AS game_date, sum(sv) AS team_sv,
                   sum(svo) AS team_svo, sum(outs) AS team_outs, sum(er) AS team_er,
                   sum(w) AS team_w
            FROM pitching GROUP BY 1, 2
        )
        SELECT team_id, game_date, year(game_date) AS season,
               coalesce(b.team_r, 0) AS team_r, coalesce(b.team_games, 0) AS team_games,
               coalesce(p.team_sv, 0) AS team_sv, coalesce(p.team_svo, 0) AS team_svo,
               coalesce(p.team_outs, 0) AS team_outs, coalesce(p.team_er, 0) AS team_er,
               coalesce(p.team_w, 0) AS team_w
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
    stage_grid(conn, "box_daily")
    if has_pitches:
        pitch_sums = ", ".join(f"{agg} AS {name}" for name, agg in PITCH_AGGS.items())
        conn.execute(
            f"""
            CREATE TEMP TABLE pitch_daily AS
            WITH dated AS (
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


def _stage_milb(conn: duckdb.DuckDBPyConnection, *, has_milb: bool) -> None:
    """``milb_player_season``: per player-season minor-league pitching totals
    (``MILB_COUNTS``), every level summed; empty with no minor-league pitching files.
    Read from the weekly lines, whose league is the window's club, so a pitcher who
    changed leagues at a level has each stretch read against its own league (and his
    ``EXCLUDED_LEAGUES`` stretches left out, not his whole level). Expected counts use
    each league's rates that season (its level's below ``MILB_MIN_LEAGUE_BF``); ages come
    from the season lines, and the age gap is to the level's mean. Position players
    pitching (``MILB_NOT_PITCHERS``) are left out. ``milb_cover``: per season, whether
    last season's minor-league lines are stored and how many of the last three are."""
    if not has_milb:
        cols = ", ".join(f"CAST(NULL AS DOUBLE) AS {c}" for c in MILB_COUNTS)
        conn.execute(
            "CREATE TEMP TABLE milb_player_season AS SELECT CAST(NULL AS BIGINT) AS player_id, "
            f"CAST(NULL AS INTEGER) AS season, {cols} WHERE false"
        )
    else:
        line = ", ".join(f'sum(w."stat.{v}") AS {k}' for k, v in MILB_LINE.items())
        excluded = ", ".join(str(i) for i in EXCLUDED_LEAGUES)
        conn.execute(
            f"""
            CREATE TEMP TABLE milb_lines AS
            WITH age AS (
                SELECT CAST(season AS INTEGER) AS season,
                       CAST("player.id" AS BIGINT) AS player_id,
                       CAST(sport_id AS INTEGER) AS sport_id,
                       max(CAST("stat.age" AS DOUBLE)) AS age
                FROM milb_pitching_season GROUP BY ALL
            )
            SELECT CAST(w."player.id" AS BIGINT) AS player_id,
                   CAST(w.season AS INTEGER) AS season, CAST(w.sport_id AS INTEGER) AS sport_id,
                   coalesce(w."league.id", 0) AS league_id, {line}, any_value(age.age) AS age
            FROM milb_pitching_weekly w
            LEFT JOIN age ON age.season = w.season AND age.player_id = w."player.id"
                         AND age.sport_id = w.sport_id
            WHERE coalesce(w."position.type", '') NOT IN {sql_in(MILB_NOT_PITCHERS)}
              AND coalesce(w."league.id", 0) NOT IN ({excluded})
            GROUP BY 1, 2, 3, 4
            HAVING sum(w."stat.battersFaced") > 0
            """
        )

        def den(stat: str) -> str:
            return "outs" if stat == "er" else "bf"

        totals = ", ".join(f"sum({c}) AS {c}" for c in ("bf", "outs", *MILB_EXPECTED))

        def pick(c: str) -> str:
            return f"CASE WHEN g.bf >= {MILB_MIN_LEAGUE_BF} THEN g.{c} ELSE v.{c} END"

        rates = ", ".join(
            f"({pick(s)}) / nullif({pick(den(s))}, 0) AS lg_{s}" for s in MILB_EXPECTED
        )
        expected = ", ".join(f"sum(r.lg_{s} * m.{den(s)}) AS x{s}" for s in MILB_EXPECTED)
        rank = " ".join(f"WHEN {s} THEN {r}" for s, r in LEVEL_RANK.items())
        conn.execute(
            f"""
            CREATE TEMP TABLE milb_player_season AS
            WITH lv AS (SELECT season, sport_id, {totals} FROM milb_lines GROUP BY 1, 2),
            lg AS (SELECT season, sport_id, league_id, {totals} FROM milb_lines GROUP BY 1, 2, 3),
            ref AS (
                SELECT g.season, g.sport_id, g.league_id, {rates}
                FROM lg g JOIN lv v USING (season, sport_id)
            ),
            ages AS (
                SELECT season, sport_id,
                       sum(age * bf) / nullif(sum(bf) FILTER (WHERE age IS NOT NULL), 0)
                           AS lg_age
                FROM milb_lines GROUP BY 1, 2
            )
            SELECT m.player_id, m.season, {", ".join(f"sum(m.{c}) AS {c}" for c in MILB_LINE)},
                   {expected},
                   sum((CASE m.sport_id {rank} END) * m.bf) AS level_bf,
                   coalesce(sum(m.bf) FILTER (WHERE m.age IS NOT NULL), 0) AS age_bf,
                   coalesce(sum((m.age - a.lg_age) * m.bf), 0) AS age_gap_bf
            FROM milb_lines m
            JOIN ref r USING (season, sport_id, league_id)
            JOIN ages a USING (season, sport_id)
            GROUP BY 1, 2
            """
        )
    conn.execute(
        """
        CREATE TEMP TABLE milb_cover AS
        WITH stored AS (SELECT DISTINCT season FROM milb_player_season)
        SELECT g.season,
               coalesce(bool_or(s.season = g.season - 1), false) AS milb_p1_in_store,
               count(s.season) AS milb_p3_seasons_in_store
        FROM (SELECT DISTINCT season FROM grid) g
        LEFT JOIN stored s ON s.season BETWEEN g.season - 3 AND g.season - 1
        GROUP BY 1
        """
    )


def _has_milb(views: set[str], root: Path) -> bool:
    """Whether the store has minor-league pitching lines (season and weekly). Without
    them every ``milb_*`` input is blank, so say so: the net would train without them."""
    if {"milb_pitching_season", "milb_pitching_weekly"} <= views:
        return True
    logger.warning(
        "pitch-data store %s has no minor-league pitching lines; the milb_* inputs will be "
        "blank. Fill them with scripts/fetch_pitch_data.py --only milb_pitching",
        root,
    )
    return False


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
        return _build(
            conn,
            has_pitches="pitches" in views,
            has_games="games" in views,
            has_milb=_has_milb(views, root),
        )
    finally:
        conn.close()


def _build(
    conn: duckdb.DuckDBPyConnection,
    *,
    has_pitches: bool,
    has_games: bool = False,
    has_milb: bool = False,
) -> pd.DataFrame:
    _stage(conn, has_pitches=has_pitches)
    _stage_milb(conn, has_milb=has_milb)
    box_season = season_totals(conn, "box_daily", BOX_COUNTS)
    pitch_season = season_totals(conn, "pitch_daily", PITCH_COUNTS)
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

    stage_population(conn, "box_daily", "pitcher_seasons")
    parts = {
        **{
            f"{w}_{kind}": sql(source, cols, f"{w}_", cond)
            for kind, daily, season_src, cols in (
                ("box", "box_daily", box_season, BOX_COUNTS),
                ("sc", "pitch_daily", pitch_season, PITCH_COUNTS),
            )
            for w, sql, source, cond in (
                ("std", window_sql, daily, STD),
                ("p1", history_sql, season_src, P1),
                ("p3", history_sql, season_src, P3),
                ("car", history_sql, season_src, CAR),
                *(
                    (w, window_sql, daily, f"{STD} AND d.game_date >= pop.as_of - {days}")
                    for w, days in RECENT_WINDOWS.items()
                ),
            )
        },
        **{
            f"{w}_milb": history_sql("milb_player_season", MILB_COUNTS, f"milb_{w}_", cond)
            for w, cond in MILB_WINDOWS.items()
        },
        "ros": window_sql("box_daily", TARGET_COUNTS, "ros_", ROS),
    }
    for name, sql in parts.items():
        conn.execute(f"CREATE TEMP TABLE {name} AS {sql}")

    stage_current_team(conn, "box_daily", TEAM_LOOKAHEAD_DAYS)
    conn.execute(
        f"""
        CREATE TEMP TABLE team_ctx AS
        SELECT pop.player_id, pop.season, pop.week, ct.team_id, {sums(TEAM_COUNTS, "t", "std_")}
        FROM pop JOIN cur_team ct USING (player_id, season, week)
        LEFT JOIN team_daily t
          ON t.team_id = ct.team_id AND t.season = pop.season AND t.game_date < pop.as_of
        GROUP BY 1, 2, 3, 4
        """
    )
    conn.execute(
        f"""
        CREATE TEMP TABLE team_ctx_p1 AS
        SELECT pop.player_id, pop.season, pop.week, {sums(TEAM_COUNTS, "t", "p1_")}
        FROM pop JOIN cur_team ct USING (player_id, season, week)
        LEFT JOIN team_daily_season t ON t.team_id = ct.team_id AND t.season = pop.season - 1
        GROUP BY 1, 2, 3
        """
    )
    # Answers for the saves model (#474): his team's saves and games from the date on,
    # and his own saves for that team (a pitcher traded later saves for another).
    conn.execute(
        """
        CREATE TEMP TABLE team_ros AS
        SELECT pop.player_id, pop.season, pop.week,
               coalesce(sum(t.team_sv), 0) AS ros_team_sv,
               coalesce(sum(t.team_games), 0) AS ros_team_games
        FROM pop JOIN cur_team ct USING (player_id, season, week)
        LEFT JOIN team_daily t
          ON t.team_id = ct.team_id AND t.season = pop.season AND t.game_date >= pop.as_of
        GROUP BY 1, 2, 3
        """
    )
    conn.execute(
        """
        CREATE TEMP TABLE sv_for_team AS
        SELECT pop.player_id, pop.season, pop.week, coalesce(sum(b.sv), 0) AS ros_sv_for_team
        FROM pop JOIN cur_team ct USING (player_id, season, week)
        LEFT JOIN box_daily b
          ON b.player_id = pop.player_id AND b.season = pop.season
         AND b.game_date >= pop.as_of AND b.team_id = ct.team_id
        GROUP BY 1, 2, 3
        """
    )
    # The team's games left on the season's schedule (the stored one: every game listed
    # on or after the date, played or not), known on the date. NULL without schedule
    # files or a team.
    sched = (
        """SELECT pop.player_id, pop.season, pop.week,
                  count(g.game_pk) FILTER (WHERE ct.team_id IS NOT NULL) AS team_games_left
           FROM pop JOIN cur_team ct USING (player_id, season, week)
           LEFT JOIN games g ON g.season = pop.season AND g.game_date >= pop.as_of
            AND ct.team_id IN (g.home_team_id, g.away_team_id)
           GROUP BY 1, 2, 3"""
        if has_games
        else """SELECT player_id, season, week, CAST(NULL AS BIGINT) AS team_games_left
                FROM pop"""
    )
    conn.execute(f"CREATE TEMP TABLE team_sched AS {sched}")

    for w, cond in (
        ("p1", "l.season = s.season - 1"),
        ("p3", "l.season BETWEEN s.season - 3 AND s.season - 1"),
        ("car", "l.season < s.season"),
    ):
        sql = league_window_sql("league_season", LEAGUE_COUNTS, f"lg_{w}_", cond)
        conn.execute(f"CREATE TEMP TABLE league_{w} AS {sql}")
    # This season before the date: the same for every row of a grid date.
    conn.execute(
        f"""
        CREATE TEMP TABLE league_std AS
        SELECT g.season, g.week, {sums(LEAGUE_COUNTS, "l", "lg_std_")}
        FROM grid g LEFT JOIN league_daily l ON l.season = g.season AND l.game_date < g.as_of
        GROUP BY 1, 2
        """
    )

    # Box scores from 2008; pitch tracking (any pitch) and Statcast's batted-ball
    # tracking (2015) each from their own first season; NULL with no pitch files.
    first_box = "(SELECT min(season) FROM box_daily)"
    first_pitch = "(SELECT min(season) FROM pitch_daily)"
    first_batted = "(SELECT min(season) FROM pitch_daily WHERE sc_ev_n > 0)"
    joined = ", ".join(f"{p}.* EXCLUDE (player_id, season, week)" for p in parts)
    df = conn.execute(
        f"""
        SELECT pop.player_id, pop.season, pop.week, pop.as_of, pop.season_complete,
               (pop.last_date - pop.as_of + 1) / (pop.last_date - pop.first_date + 1)
                   AS frac_season_left,
               {coverage_sql(first_box)},
               {coverage_sql(first_pitch, "pitchtrack_")},
               {coverage_sql(first_batted, "battedtrack_")},
               mc.milb_p1_in_store, mc.milb_p3_seasons_in_store,
               {AGE_SELECT},
               tc.team_id, tc.* EXCLUDE (player_id, season, week, team_id),
               tp.* EXCLUDE (player_id, season, week),
               ts.team_games_left,
               {joined},
               tr.ros_team_sv, tr.ros_team_games, sft.ros_sv_for_team,
               ls.* EXCLUDE (season, week),
               p1l.* EXCLUDE (season), p3l.* EXCLUDE (season), carl.* EXCLUDE (season)
        FROM pop
        {" ".join(f"JOIN {p} USING (player_id, season, week)" for p in parts)}
        JOIN team_ctx tc USING (player_id, season, week)
        JOIN team_ctx_p1 tp USING (player_id, season, week)
        JOIN team_sched ts USING (player_id, season, week)
        JOIN team_ros tr USING (player_id, season, week)
        JOIN sv_for_team sft USING (player_id, season, week)
        JOIN league_std ls USING (season, week)
        JOIN league_p1 p1l ON p1l.season = pop.season
        JOIN league_p3 p3l ON p3l.season = pop.season
        JOIN league_car carl ON carl.season = pop.season
        JOIN milb_cover mc ON mc.season = pop.season
        {AGE_JOINS}
        ORDER BY pop.season, pop.week, pop.player_id
        """
    ).df()
    return df
