"""Idempotent loaders for the streaks DuckDB tables."""

from __future__ import annotations

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
