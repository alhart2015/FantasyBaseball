"""Idempotent loaders for the streaks DuckDB tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from datetime import date
from operator import attrgetter

import duckdb

from fantasy_baseball.streaks.models import HitterGame, HitterStatcastPA

# Column lists derive from dataclass field order — see streaks/models.py.
# `attrgetter(*cols)` builds a C-level row→tuple extractor that's ~10x faster
# than dataclasses.astuple (which deepcopies every field) at the row counts
# Statcast pulls produce.
_HITTER_GAME_COLS = tuple(f.name for f in fields(HitterGame))
_STATCAST_COLS = tuple(f.name for f in fields(HitterStatcastPA))
_hitter_game_row = attrgetter(*_HITTER_GAME_COLS)
_statcast_row = attrgetter(*_STATCAST_COLS)


def upsert_hitter_games(
    conn: duckdb.DuckDBPyConnection, rows: Sequence[HitterGame]
) -> None:
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
    conn.executemany(sql, [_hitter_game_row(r) for r in rows])


def upsert_statcast_pa(
    conn: duckdb.DuckDBPyConnection, rows: Sequence[HitterStatcastPA]
) -> None:
    """Insert or replace rows in `hitter_statcast_pa` keyed by (player_id, date, pa_index)."""
    if not rows:
        return
    placeholders = ", ".join(["?"] * len(_STATCAST_COLS))
    sql = (
        f"INSERT OR REPLACE INTO hitter_statcast_pa ({', '.join(_STATCAST_COLS)}) "
        f"VALUES ({placeholders})"
    )
    conn.executemany(sql, [_statcast_row(r) for r in rows])


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
