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
from typing import Any

import duckdb
import pandas as pd

from fantasy_baseball.analysis.game_logs import FULL_HITTER_FIELDS
from fantasy_baseball.hitter_ros.features import ALL_ANSWER_COUNTS, FIXED_ZONE, zone_options
from fantasy_baseball.hitter_ros.statcast_sql import CONTACT_SQL, SPRAY_SQL, SWING_SQL, WHIFF_SQL
from fantasy_baseball.pitch_data.store import connect

AS_OF_STEP_DAYS = 7

# Starts at the positions that say most about running (#413): catchers almost never
# run; shortstops and center fielders are usually fast; a DH is usually slow.
START_POSITIONS = ("C", "SS", "CF", "DH")
BOX_COUNTS = (
    *FULL_HITTER_FIELDS.values(),
    "games",
    "starts",
    "spot_sum",
    *(f"starts_{p.lower()}" for p in START_POSITIONS),
)
# Recent-form windows (#419): days before the as-of date, same season.
RECENT_WINDOWS = {"l7": 7, "l14": 14}
# Short-horizon answers (#419): his next N PA from the as-of date, same season only.
HORIZONS = (25, 100, 250)
# Their counts: the answers' COUNTS, plus K for AVG's pieces (#433) and CS and steal
# opportunities for SB's (#413).
HORIZON_COUNTS = ALL_ANSWER_COUNTS
# Steal opportunities (#413), from the runners on base at the first pitch of each PA:
# on 1B with 2B open, and on 2B with 3B open.
STEAL_COUNTS = ("steal_opp2", "steal_opp3")
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

_PULL_DEG = 15
_ZONE = "zone BETWEEN 1 AND 9"
_CHASE = "zone BETWEEN 11 AND 14"
# features.FIXED_ZONE (#433): outside it, with a known location, is a chase pitch.
# The table records the box it counted with (build_options) next to it.
_FZONE = f"abs(plate_x) <= {FIXED_ZONE[0]} AND plate_z BETWEEN {FIXED_ZONE[1]} AND {FIXED_ZONE[2]}"
_FCHASE = f"plate_x IS NOT NULL AND plate_z IS NOT NULL AND NOT ({_FZONE})"
_BIP = "type = 'X'"
_AIR = "bb_type IN ('fly_ball', 'line_drive')"
_PULLED = (
    f"((stand = 'R' AND {SPRAY_SQL} < -{_PULL_DEG}) OR (stand = 'L' AND {SPRAY_SQL} > {_PULL_DEG}))"
)
_OPPO = (
    f"((stand = 'R' AND {SPRAY_SQL} > {_PULL_DEG}) OR (stand = 'L' AND {SPRAY_SQL} < -{_PULL_DEG}))"
)

# name -> SQL aggregate over one batter's pitches
PITCH_AGGS: dict[str, str] = {
    "pitches": "count(*)",
    "swings": f"count(*) FILTER (WHERE {SWING_SQL})",
    "whiffs": f"count(*) FILTER (WHERE {WHIFF_SQL})",
    "zone_pitches": f"count(*) FILTER (WHERE {_ZONE})",
    "zone_swings": f"count(*) FILTER (WHERE {_ZONE} AND {SWING_SQL})",
    "zone_contacts": f"count(*) FILTER (WHERE {_ZONE} AND {CONTACT_SQL})",
    "chase_pitches": f"count(*) FILTER (WHERE {_CHASE})",
    "chase_swings": f"count(*) FILTER (WHERE {_CHASE} AND {SWING_SQL})",
    "chase_contacts": f"count(*) FILTER (WHERE {_CHASE} AND {CONTACT_SQL})",
    "fzone_pitches": f"count(*) FILTER (WHERE {_FZONE})",
    "fzone_swings": f"count(*) FILTER (WHERE {_FZONE} AND {SWING_SQL})",
    "fzone_contacts": f"count(*) FILTER (WHERE {_FZONE} AND {CONTACT_SQL})",
    "fchase_pitches": f"count(*) FILTER (WHERE {_FCHASE})",
    "fchase_swings": f"count(*) FILTER (WHERE {_FCHASE} AND {SWING_SQL})",
    "fchase_contacts": f"count(*) FILTER (WHERE {_FCHASE} AND {CONTACT_SQL})",
    "first_pitch_swings": f"count(*) FILTER (WHERE balls = 0 AND strikes = 0 AND {SWING_SQL})",
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
TEAM_COUNTS = ("team_r", "team_pa", "team_games", "team_sb", "team_cs")
# League totals per window (#421), for league rates and player-vs-league ratios: the
# box-score counts, plus steal opportunities for SB's pieces (#413).
LEAGUE_BOX_COUNTS = ("pa", "ab", "h", "b2", "b3", "hr", "r", "rbi", "sb", "cs", "bb", "k", "hbp")
LEAGUE_COUNTS = (*LEAGUE_BOX_COUNTS, *STEAL_COUNTS)

_REQUIRED_VIEWS = ("pitches", "lineups", "schedule")


def _sums(cols: Iterable[str], src: str, prefix: str) -> str:
    return ", ".join(f"coalesce(sum({src}.{c}), 0) AS {prefix}{c}" for c in cols)


def _stage(conn: duckdb.DuckDBPyConnection, *, has_sprint: bool) -> None:
    """Daily per-player sums, hitter-seasons, season bounds and the as-of grid."""
    box_sums = ", ".join(f"sum({c}) AS {c}" for c in FULL_HITTER_FIELDS.values())
    position_starts = ", ".join(
        f"count(*) FILTER (WHERE sub_index = 0 AND position = '{p}') AS starts_{p.lower()}"
        for p in START_POSITIONS
    )
    conn.execute(
        f"""
        CREATE TEMP TABLE box_daily AS
        SELECT player_id, CAST(game_date AS DATE) AS game_date,
               year(CAST(game_date AS DATE)) AS season,
               {box_sums},
               count(DISTINCT game_pk) AS games,
               count(*) FILTER (WHERE sub_index = 0) AS starts,
               coalesce(sum(lineup_spot) FILTER (WHERE sub_index = 0), 0) AS spot_sum,
               {position_starts},
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
        CREATE TEMP TABLE steal_daily AS
        SELECT runner AS player_id, game_date, season,
               sum(opp2) AS steal_opp2, sum(opp3) AS steal_opp3
        FROM (
            SELECT on_1b AS runner, CAST(game_date AS DATE) AS game_date,
                   CAST(season AS INTEGER) AS season, 1 AS opp2, 0 AS opp3
            FROM pitches
            WHERE game_type = 'R' AND pitch_number = 1 AND on_1b IS NOT NULL AND on_2b IS NULL
            UNION ALL
            SELECT on_2b, CAST(game_date AS DATE), CAST(season AS INTEGER), 0, 1
            FROM pitches
            WHERE game_type = 'R' AND pitch_number = 1 AND on_2b IS NOT NULL AND on_3b IS NULL
        )
        GROUP BY 1, 2, 3
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
        SELECT l.team_id, CAST(l.game_date AS DATE) AS game_date,
               year(CAST(l.game_date AS DATE)) AS season,
               coalesce(sum(l.r) FILTER (WHERE hs.player_id IS NOT NULL), 0) AS team_r,
               coalesce(sum(l.pa) FILTER (WHERE hs.player_id IS NOT NULL), 0) AS team_pa,
               coalesce(sum(l.sb) FILTER (WHERE hs.player_id IS NOT NULL), 0) AS team_sb,
               coalesce(sum(l.cs) FILTER (WHERE hs.player_id IS NOT NULL), 0) AS team_cs,
               count(DISTINCT l.game_pk) AS team_games
        FROM lineups l
        LEFT JOIN hitter_seasons hs
          ON hs.player_id = l.player_id AND hs.season = year(CAST(l.game_date AS DATE))
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
    # Sprint speed: always the same columns, NULL when unknown or not fetched.
    sprint_src = (
        """SELECT CAST(player_id AS BIGINT) AS player_id, CAST(season AS INTEGER) AS season,
                  sprint_speed, competitive_runs, CAST(hp_to_1b AS DOUBLE) AS hp_to_1b,
                  CAST(bolts AS DOUBLE) AS bolts FROM sprint_speed"""
        if has_sprint
        else """SELECT CAST(NULL AS BIGINT) AS player_id, CAST(NULL AS INTEGER) AS season,
                       CAST(NULL AS DOUBLE) AS sprint_speed,
                       CAST(NULL AS BIGINT) AS competitive_runs,
                       CAST(NULL AS DOUBLE) AS hp_to_1b, CAST(NULL AS DOUBLE) AS bolts
                WHERE false"""
    )
    conn.execute(f"CREATE TEMP TABLE sprint AS {sprint_src}")


def _league_context(conn: duckdb.DuckDBPyConnection) -> None:
    """League totals for each row's windows (#421): this season before the date, last
    season, the last three seasons and every earlier season in the store. Counts only;
    ``features.py`` turns them into league rates and player-vs-league ratios."""
    sums = ", ".join(f"sum({c}) AS {c}" for c in LEAGUE_COUNTS)
    # Hitter-seasons only (same rule as the table's population). Before the universal DH,
    # pitchers took ~3% of PA, which lowered league rates (about -1.3 R, -0.5 HR per 600
    # PA, -4 points of AVG) and made them jump in 2020 and 2022 when pitchers stopped
    # batting. Filtering on the season rather than the game's position keeps a position
    # player's at-bats on a day he also mopped up on the mound (listed as P).
    by_row = ", ".join(f"sum(l.{c}) AS {c}" for c in LEAGUE_BOX_COUNTS)
    steal_by_day = ", ".join(f"sum(d.{c}) AS {c}" for c in STEAL_COUNTS)
    steal_cols = ", ".join(f"coalesce(st.{c}, 0) AS {c}" for c in STEAL_COUNTS)
    conn.execute(
        f"""
        CREATE TEMP TABLE league_daily AS
        WITH box AS (
            SELECT year(CAST(l.game_date AS DATE)) AS season,
                   CAST(l.game_date AS DATE) AS game_date, {by_row}
            FROM lineups l
            JOIN hitter_seasons hs
              ON hs.player_id = l.player_id AND hs.season = year(CAST(l.game_date AS DATE))
            GROUP BY 1, 2
        ), steal AS (
            SELECT d.season, d.game_date, {steal_by_day}
            FROM steal_daily d
            JOIN hitter_seasons hs ON hs.player_id = d.player_id AND hs.season = d.season
            GROUP BY 1, 2
        )
        SELECT box.*, {steal_cols}
        FROM box LEFT JOIN steal st USING (season, game_date)
        """
    )
    running = ", ".join(
        f"sum({c}) OVER (PARTITION BY season ORDER BY game_date) AS {c}" for c in LEAGUE_COUNTS
    )
    conn.execute(
        f"CREATE TEMP TABLE league_cum AS SELECT season, game_date, {running} FROM league_daily"
    )
    conn.execute(
        f"CREATE TEMP TABLE league_season AS SELECT season, {sums} FROM league_daily GROUP BY 1"
    )

    def window(prefix: str, cond: str) -> str:
        cols = ", ".join(f"coalesce(sum(l.{c}), 0) AS lg_{prefix}_{c}" for c in LEAGUE_COUNTS)
        return f"""
            SELECT s.season, {cols}
            FROM (SELECT DISTINCT season FROM pop) s
            LEFT JOIN league_season l ON {cond}
            GROUP BY 1
        """

    conn.execute(f"CREATE TEMP TABLE league_p1 AS {window('p1', 'l.season = s.season - 1')}")
    conn.execute(
        "CREATE TEMP TABLE league_p3 AS "
        + window("p3", "l.season BETWEEN s.season - 3 AND s.season - 1")
    )
    conn.execute(f"CREATE TEMP TABLE league_car AS {window('car', 'l.season < s.season')}")
    std_cols = ", ".join(f"coalesce(c.{c}, 0) AS lg_std_{c}" for c in LEAGUE_COUNTS)
    # League running totals through the latest game day strictly before the date.
    conn.execute(
        f"""
        CREATE TEMP TABLE league_ctx AS
        SELECT pop.player_id, pop.season, pop.week, {std_cols},
               p1.* EXCLUDE (season), p3.* EXCLUDE (season), car.* EXCLUDE (season)
        FROM pop
        ASOF LEFT JOIN league_cum c ON c.season = pop.season AND pop.as_of > c.game_date
        JOIN league_p1 p1 ON p1.season = pop.season
        JOIN league_p3 p3 ON p3.season = pop.season
        JOIN league_car car ON car.season = pop.season
        """
    )


def _horizon_answers(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Answers for the next N PA (#419): ``ros_n{N}_*`` counts over his games from the
    as-of date until his PA reach N, within the same season. NULL when he doesn't reach
    N before the season ends (no answer for that horizon). Games end mid-horizon, so a
    window holds N to N + a few PA; rates use its actual PA. Same-day games are one step,
    as in ``box_daily``. Answers only: the ``ros_`` prefix keeps the leakage tests on them.

    Built from running totals per player-season: the window ends on the first date whose
    running PA reaches (PA before the as-of date) + N, and its counts are the running
    totals there minus those before the date (``std_*``).
    """
    running = ", ".join(
        f"sum({c}) OVER (PARTITION BY player_id, season ORDER BY game_date) AS cum_{c}"
        for c in HORIZON_COUNTS
    )
    # Only dates with a PA: there the running PA strictly increases, so the ASOF match
    # below is the one first date reaching the goal. A 0-PA date (pinch-running, a
    # defensive sub) ties the date before it, and could otherwise be matched instead,
    # adding its runs and steals from after the window. Its counts still land in the
    # running totals of the next date with a PA.
    steal_cols = ", ".join(f"coalesce(st.{c}, 0) AS {c}" for c in STEAL_COUNTS)
    conn.execute(
        f"CREATE TEMP TABLE box_cum AS SELECT * FROM ("
        f"SELECT player_id, season, game_date, pa, {running} FROM ("
        f"SELECT b.*, {steal_cols} FROM box_daily b"
        f" LEFT JOIN steal_daily st USING (player_id, season, game_date))"
        ") WHERE coalesce(pa, 0) > 0"
    )
    names = []
    for n in HORIZONS:
        name = f"horizon_{n}"
        # Counts before the date: box-score ones from std_box, steal opportunities from
        # std_steal.
        counts = ", ".join(
            f"CASE WHEN b.cum_pa IS NULL THEN NULL "
            f"ELSE b.cum_{c} - {'ss' if c in STEAL_COUNTS else 's'}.std_{c} END AS ros_n{n}_{c}"
            for c in HORIZON_COUNTS
        )
        conn.execute(
            f"""
            CREATE TEMP TABLE {name} AS
            WITH goal AS (
                SELECT pop.player_id, pop.season, pop.week, s.std_pa + {n} AS goal_pa
                FROM pop JOIN std_box s USING (player_id, season, week)
            )
            SELECT g.player_id, g.season, g.week, {counts}
            FROM goal g
            JOIN std_box s USING (player_id, season, week)
            JOIN std_steal ss USING (player_id, season, week)
            ASOF LEFT JOIN box_cum b
              ON b.player_id = g.player_id AND b.season = g.season AND g.goal_pa <= b.cum_pa
            """
        )
        names.append(name)
    return names


def _season_totals(conn: duckdb.DuckDBPyConnection, daily: str, cols: Iterable[str]) -> str:
    name = f"{daily}_season"
    sums = ", ".join(f"sum({c}) AS {c}" for c in cols)
    conn.execute(
        f"CREATE TEMP TABLE {name} AS SELECT player_id, season, {sums} FROM {daily} GROUP BY 1, 2"
    )
    return name


def build_options() -> dict[str, Any]:
    """The settings a table is built with; saved next to it as ``<table>.json``. The
    fixed-box counts (fzone_ / fchase_) depend on features.FIXED_ZONE."""
    return {"fixed_zone": zone_options("fixed")["fixed_zone"]}


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
    steal_season = _season_totals(conn, "steal_daily", STEAL_COUNTS)

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
        "std_steal": window("steal_daily", STEAL_COUNTS, "std_", std),
        "p1_steal": window(steal_season, STEAL_COUNTS, "p1_", p1),
        "p3_steal": window(steal_season, STEAL_COUNTS, "p3_", p3),
        "car_steal": window(steal_season, STEAL_COUNTS, "car_", car),
        "ros": window("box_daily", TARGET_COUNTS, "ros_", ros),
        # SB's pieces (#413): steal opportunities over the answer window.
        "ros_steal": window("steal_daily", STEAL_COUNTS, "ros_", ros),
        # Recent form (#419): the last 7 and 14 days before the date, this season only.
        **{
            f"{w}_{kind}": window(
                src, cols, f"{w}_", f"{std} AND d.game_date >= pop.as_of - {days}"
            )
            for w, days in RECENT_WINDOWS.items()
            for kind, src, cols in (
                ("box", "box_daily", BOX_COUNTS),
                ("pitch", "pitch_daily", PITCH_COUNTS),
            )
        },
    }
    for name, sql in parts.items():
        conn.execute(f"CREATE TEMP TABLE {name} AS {sql}")
    horizon_tables = _horizon_answers(conn)

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

    _league_context(conn)

    first_store_season = "(SELECT min(season) FROM box_daily)"
    tables = [*parts, *horizon_tables]
    joined = ", ".join(f"{p}.* EXCLUDE (player_id, season, week)" for p in tables)
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
               s1.hp_to_1b AS p1_hp_to_1b, s1.bolts AS p1_bolts,
               s2.hp_to_1b AS p2_hp_to_1b, s2.bolts AS p2_bolts,
               tc.team_id, tc.* EXCLUDE (player_id, season, week, team_id),
               tp.* EXCLUDE (player_id, season, week),
               {joined},
               lg.* EXCLUDE (player_id, season, week)
        FROM pop
        {" ".join(f"JOIN {p} USING (player_id, season, week)" for p in tables)}
        JOIN team_ctx tc USING (player_id, season, week)
        JOIN team_ctx_p1 tp USING (player_id, season, week)
        JOIN league_ctx lg USING (player_id, season, week)
        LEFT JOIN season_age a ON a.player_id = pop.player_id AND a.season = pop.season
        LEFT JOIN sprint s1 ON s1.player_id = pop.player_id AND s1.season = pop.season - 1
        LEFT JOIN sprint s2 ON s2.player_id = pop.player_id AND s2.season = pop.season - 2
        ORDER BY pop.season, pop.week, pop.player_id
        """
    ).df()
    return df
