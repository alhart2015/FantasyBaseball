"""DuckDB schema for the streaks analysis project.

All DDL is `CREATE TABLE IF NOT EXISTS` so init_schema is idempotent and
safe to call on every connection open.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

DEFAULT_DB_PATH = Path("data/streaks/streaks.duckdb")

_SCHEMA_DDL = [
    """
    CREATE TABLE IF NOT EXISTS hitter_games (
        player_id INTEGER NOT NULL,
        name VARCHAR NOT NULL,
        team VARCHAR,
        season INTEGER NOT NULL,
        date DATE NOT NULL,
        pa INTEGER NOT NULL,
        ab INTEGER NOT NULL,
        h INTEGER NOT NULL,
        hr INTEGER NOT NULL,
        r INTEGER NOT NULL,
        rbi INTEGER NOT NULL,
        sb INTEGER NOT NULL,
        bb INTEGER NOT NULL,
        k INTEGER NOT NULL,
        PRIMARY KEY (player_id, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hitter_statcast_pa (
        player_id INTEGER NOT NULL,
        date DATE NOT NULL,
        pa_index INTEGER NOT NULL,
        event VARCHAR,
        launch_speed DOUBLE,
        launch_angle DOUBLE,
        estimated_woba_using_speedangle DOUBLE,
        barrel BOOLEAN,
        PRIMARY KEY (player_id, date, pa_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hitter_windows (
        player_id INTEGER NOT NULL,
        window_end DATE NOT NULL,
        window_days INTEGER NOT NULL,
        pa INTEGER NOT NULL,
        hr INTEGER NOT NULL,
        r INTEGER NOT NULL,
        rbi INTEGER NOT NULL,
        sb INTEGER NOT NULL,
        avg DOUBLE,
        babip DOUBLE,
        k_pct DOUBLE,
        bb_pct DOUBLE,
        iso DOUBLE,
        ev_avg DOUBLE,
        barrel_pct DOUBLE,
        xwoba_avg DOUBLE,
        pt_bucket VARCHAR NOT NULL,
        PRIMARY KEY (player_id, window_end, window_days)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS thresholds (
        season_set VARCHAR NOT NULL,
        category VARCHAR NOT NULL,
        window_days INTEGER NOT NULL,
        pt_bucket VARCHAR NOT NULL,
        p10 DOUBLE NOT NULL,
        p90 DOUBLE NOT NULL,
        PRIMARY KEY (season_set, category, window_days, pt_bucket)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hitter_streak_labels (
        player_id INTEGER NOT NULL,
        window_end DATE NOT NULL,
        window_days INTEGER NOT NULL,
        category VARCHAR NOT NULL,
        label VARCHAR NOT NULL,
        PRIMARY KEY (player_id, window_end, window_days, category)
    )
    """,
]


def get_connection(path: Path | str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open (or create) the streaks DuckDB at *path* and return the connection.

    Parent directory is created if missing. Schema is initialized on every open.
    """
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    init_schema(conn)
    return conn


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all streaks tables if they don't already exist."""
    for ddl in _SCHEMA_DDL:
        conn.execute(ddl)
