"""Idempotent loaders for the streaks DuckDB tables."""

from __future__ import annotations

from datetime import date
from typing import Any

import duckdb

_HITTER_GAME_COLS = (
    "player_id",
    "name",
    "team",
    "season",
    "date",
    "pa",
    "ab",
    "h",
    "hr",
    "r",
    "rbi",
    "sb",
    "bb",
    "k",
)


def upsert_hitter_games(conn: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> None:
    """Insert or replace rows in `hitter_games` keyed by (player_id, date).

    Empty input is a no-op. DuckDB's `INSERT OR REPLACE` handles PK
    collisions atomically.
    """
    if not rows:
        return
    placeholders = ", ".join(["?"] * len(_HITTER_GAME_COLS))
    sql = (
        f"INSERT OR REPLACE INTO hitter_games ({', '.join(_HITTER_GAME_COLS)}) "
        f"VALUES ({placeholders})"
    )
    conn.executemany(sql, [tuple(r[c] for c in _HITTER_GAME_COLS) for r in rows])


_STATCAST_COLS = (
    "player_id",
    "date",
    "pa_index",
    "event",
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
    "barrel",
)


def upsert_statcast_pa(conn: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> None:
    """Insert or replace rows in `hitter_statcast_pa` keyed by (player_id, date, pa_index)."""
    if not rows:
        return
    placeholders = ", ".join(["?"] * len(_STATCAST_COLS))
    sql = (
        f"INSERT OR REPLACE INTO hitter_statcast_pa ({', '.join(_STATCAST_COLS)}) "
        f"VALUES ({placeholders})"
    )
    conn.executemany(sql, [tuple(r[c] for c in _STATCAST_COLS) for r in rows])


def existing_player_seasons(
    conn: duckdb.DuckDBPyConnection,
) -> set[tuple[int, int]]:
    """Return distinct (player_id, season) pairs already loaded in hitter_games.

    Used by fetch orchestration to skip player-seasons we've already pulled.
    """
    rows = conn.execute("SELECT DISTINCT player_id, season FROM hitter_games").fetchall()
    return {(int(r[0]), int(r[1])) for r in rows}


def existing_statcast_dates(conn: duckdb.DuckDBPyConnection) -> set[date]:
    """Return distinct calendar dates already loaded in hitter_statcast_pa.

    Used by Statcast fetch to skip date ranges we've already pulled.
    """
    rows = conn.execute("SELECT DISTINCT date FROM hitter_statcast_pa").fetchall()
    return {r[0] for r in rows}
