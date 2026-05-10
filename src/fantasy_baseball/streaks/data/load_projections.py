"""Idempotent loader for ``hitter_projection_rates``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from operator import attrgetter

import duckdb

from fantasy_baseball.streaks.models import HitterProjectionRate

_PROJECTION_RATE_COLS = tuple(f.name for f in fields(HitterProjectionRate))
_projection_rate_row = attrgetter(*_PROJECTION_RATE_COLS)


def upsert_projection_rates(
    conn: duckdb.DuckDBPyConnection, rows: Sequence[HitterProjectionRate]
) -> None:
    """Insert or replace rows in `hitter_projection_rates` keyed by (player_id, season)."""
    if not rows:
        return
    placeholders = ", ".join(["?"] * len(_PROJECTION_RATE_COLS))
    sql = (
        f"INSERT OR REPLACE INTO hitter_projection_rates ({', '.join(_PROJECTION_RATE_COLS)}) "
        f"VALUES ({placeholders})"
    )
    conn.executemany(sql, [_projection_rate_row(r) for r in rows])
