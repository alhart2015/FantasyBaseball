"""Building blocks shared by the hitter and pitcher training tables
(``hitter_ros/table.py``, ``pitcher_ros/table.py``): the weekly as-of grid, who gets a
row, the window sums, the team going forward and how much history the store covers.

Each builder stages its own per-player daily table (``player_id``, ``game_date``,
``season``, ``team_id`` and its counts) and a members table (``player_id``, ``season``:
the seasons that make him a hitter or a pitcher), then calls these in order:
:func:`stage_grid`, :func:`stage_population`, the window SQL, :func:`stage_current_team`.
The SQL reads the temp tables they create: ``seasons``, ``grid``, ``pop``, ``cur_team``.
"""

from __future__ import annotations

from collections.abc import Iterable

import duckdb

AS_OF_STEP_DAYS = 7

# Window conditions over a source row ``d`` for a ``pop`` row.
STD = "d.season = pop.season AND d.game_date < pop.as_of"
ROS = "d.season = pop.season AND d.game_date >= pop.as_of"
P1 = "d.season = pop.season - 1"
P3 = "d.season BETWEEN pop.season - 3 AND pop.season - 1"
CAR = "d.season < pop.season"


def sums(cols: Iterable[str], src: str, prefix: str) -> str:
    return ", ".join(f"coalesce(sum({src}.{c}), 0) AS {prefix}{c}" for c in cols)


def season_totals(conn: duckdb.DuckDBPyConnection, daily: str, cols: Iterable[str]) -> str:
    """Per player-season totals of ``daily``'s ``cols``, as temp table ``{daily}_season``."""
    name = f"{daily}_season"
    col_sums = ", ".join(f"sum({c}) AS {c}" for c in cols)
    conn.execute(
        f"CREATE TEMP TABLE {name} AS SELECT player_id, season, {col_sums} "
        f"FROM {daily} GROUP BY 1, 2"
    )
    return name


def stage_grid(conn: duckdb.DuckDBPyConnection, played: str) -> None:
    """``seasons`` (bounds from the stored schedule; complete once ``played`` reaches the
    scheduled last date) and ``grid`` (week 0 = the first scheduled date, then every
    ``AS_OF_STEP_DAYS``)."""
    conn.execute(
        f"""
        CREATE TEMP TABLE seasons AS
        SELECT CAST(s.season AS INTEGER) AS season,
               CAST(s.first_date AS DATE) AS first_date,
               CAST(s.last_date AS DATE) AS last_date,
               coalesce(max(b.game_date) >= CAST(s.last_date AS DATE), false) AS season_complete
        FROM schedule s LEFT JOIN {played} b ON b.season = s.season
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


def stage_population(conn: duckdb.DuckDBPyConnection, played: str, members: str) -> None:
    """``pop``: a row per grid date for every member with a game in ``played`` on or
    after it that season; and, while a season isn't complete, week 0 for every member of
    the three seasons before (the rows to project it from)."""
    conn.execute(
        f"""
        CREATE TEMP TABLE pop AS
        SELECT DISTINCT b.player_id, g.season, g.week, g.as_of, g.first_date, g.last_date,
               g.season_complete
        FROM grid g JOIN {played} b ON b.season = g.season AND b.game_date >= g.as_of
        WHERE (b.player_id, b.season) IN (SELECT player_id, season FROM {members})
        UNION
        -- A season not complete: week 0 for every member of the three seasons before.
        SELECT DISTINCT m.player_id, g.season, g.week, g.as_of, g.first_date, g.last_date,
               g.season_complete
        FROM grid g JOIN {members} m ON m.season BETWEEN g.season - 3 AND g.season - 1
        WHERE g.week = 0 AND NOT g.season_complete
        """
    )


def window_sql(source: str, cols: Iterable[str], prefix: str, cond: str) -> str:
    """Per-``pop``-row sums of ``source`` (as ``d``) over the rows matching ``cond``."""
    return f"""
        SELECT pop.player_id, pop.season, pop.week, {sums(cols, "d", prefix)}
        FROM pop LEFT JOIN {source} d ON d.player_id = pop.player_id AND {cond}
        GROUP BY 1, 2, 3
    """


def history_sql(source: str, cols: Iterable[str], prefix: str, cond: str) -> str:
    """:func:`window_sql` for a window that depends only on the season (``P1``, ``P3``,
    ``CAR``): summed once per player-season, then copied to each of its weeks."""
    return f"""
        WITH h AS (
            SELECT pop.player_id, pop.season, {sums(cols, "d", prefix)}
            FROM (SELECT DISTINCT player_id, season FROM pop) pop
            LEFT JOIN {source} d ON d.player_id = pop.player_id AND {cond}
            GROUP BY 1, 2
        )
        SELECT pop.player_id, pop.season, pop.week, h.* EXCLUDE (player_id, season)
        FROM pop JOIN h USING (player_id, season)
    """


def stage_current_team(
    conn: duckdb.DuckDBPyConnection, played: str, max_days_ahead: int | None = None
) -> None:
    """``cur_team``: the team going forward, the team of his first game in ``played`` on
    or after the date -- the roster as known then. With ``max_days_ahead``, only when
    that game is at most that many days off: a later one may be with a team he joins
    after the date (a trade or signing while he is hurt or in the minors). Otherwise his
    last team before the date (any season), or NULL with none."""
    near = (
        "b.game_date IS NOT NULL"
        if max_days_ahead is None
        else f"b.game_date IS NOT NULL AND b.game_date <= pop.as_of + {max_days_ahead}"
    )
    conn.execute(
        f"""
        CREATE TEMP TABLE cur_team AS
        SELECT pop.player_id, pop.season, pop.week,
               CASE WHEN {near} THEN b.team_id ELSE last.team_id END AS team_id
        FROM pop
        ASOF LEFT JOIN {played} b
          ON b.player_id = pop.player_id AND pop.as_of <= b.game_date
        ASOF LEFT JOIN {played} last
          ON last.player_id = pop.player_id AND pop.as_of > last.game_date
        """
    )


def league_window_sql(source: str, cols: Iterable[str], prefix: str, cond: str) -> str:
    """League sums of ``source`` (per-season totals, as ``l``) for each season in
    ``pop`` (as ``s``) over the seasons matching ``cond``."""
    return f"""
        SELECT s.season, {sums(cols, "l", prefix)}
        FROM (SELECT DISTINCT season FROM pop) s
        LEFT JOIN {source} l ON {cond}
        GROUP BY 1
    """


def coverage_sql(first_season: str, prefix: str = "") -> str:
    """Select columns saying how much of each history window the store covers, from
    ``first_season`` (SQL for its first season; NULL = no data at all, which reads as
    no coverage): ``{prefix}p1_in_store``, ``{prefix}p3_seasons_in_store`` (0-3) and
    ``{prefix}car_seasons_in_store``."""
    f = first_season
    return f"""
        coalesce(pop.season - 1 >= {f}, false) AS {prefix}p1_in_store,
        CASE WHEN {f} IS NULL THEN 0 ELSE greatest(0, least(3, pop.season - {f})) END
            AS {prefix}p3_seasons_in_store,
        CASE WHEN {f} IS NULL THEN 0 ELSE greatest(0, pop.season - {f}) END
            AS {prefix}car_seasons_in_store"""


# Season age from ``season_age`` (player_id, season, age): this season's, or -- when it
# isn't complete and he hasn't played yet -- his last one plus the seasons since.
AGE_SELECT = """CASE WHEN a.age IS NULL AND NOT pop.season_complete
                    THEN a1.age + pop.season - a1.season ELSE a.age END AS age"""
AGE_JOINS = """LEFT JOIN season_age a ON a.player_id = pop.player_id AND a.season = pop.season
        ASOF LEFT JOIN season_age a1 ON a1.player_id = pop.player_id AND pop.season > a1.season"""
