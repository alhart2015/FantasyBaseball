"""Idempotent loaders for the streaks DuckDB tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from datetime import date
from operator import attrgetter
from typing import Any

import duckdb
import pandas as pd

from fantasy_baseball.streaks.data.schema import table_primary_key
from fantasy_baseball.streaks.models import HitterGame, HitterStatcastPA

# Column lists derive from dataclass field order -- see streaks/models.py.
_HITTER_GAME_COLS = tuple(f.name for f in fields(HitterGame))
_STATCAST_COLS = tuple(f.name for f in fields(HitterStatcastPA))


def _bulk_upsert(
    conn: duckdb.DuckDBPyConnection,
    *,
    table: str,
    columns: tuple[str, ...],
    rows: Sequence[Any],
) -> None:
    """INSERT OR REPLACE ``rows`` into ``table`` via one bulk DataFrame scan.

    ``columns`` is the dataclass field order (which matches the row tuple
    ``attrgetter`` extracts); the table's primary key is read from the live
    schema so the dedupe below can't drift from the DDL.

    DuckDB is columnar: a row-by-row ``executemany`` of INSERT OR REPLACE costs
    ~4ms/row (per-row PK-conflict handling), so seeding a season of Statcast PAs
    took tens of seconds. A single ``INSERT OR REPLACE ... SELECT`` from a
    registered DataFrame is ~400x faster and byte-identical.

    Two semantics are preserved from the old ``executemany`` path so callers
    see no behavior change:

    - **Last-wins on within-batch duplicate keys.** ``executemany`` applied rows
      sequentially, so a later row replaced an earlier one sharing a PK; a bulk
      INSERT keeps the *first* match instead, so we
      ``drop_duplicates(..., keep="last")`` on the PK up front to match.
    - **None -> SQL NULL.** Building the frame with ``dtype=object`` keeps Python
      ``None`` as-is (no float ``NaN`` coercion), so nullable VARCHAR / INTEGER /
      DOUBLE columns round-trip as NULL rather than ``NaN``.
    """
    if not rows:
        return
    # attrgetter(*columns) is a C-level row->tuple extractor, ~10x faster than
    # dataclasses.astuple (which deepcopies every field) when materializing the
    # frame; the column order matches `columns` so the SELECT lines up.
    extractor = attrgetter(*columns)
    df = pd.DataFrame([extractor(r) for r in rows], columns=list(columns), dtype=object)
    pk = table_primary_key(table)
    if pk:  # guard a future PK-less table: drop_duplicates(subset=[]) raises
        df = df.drop_duplicates(subset=list(pk), keep="last")
    collist = ", ".join(columns)
    conn.register("_bulk_upsert_df", df)
    try:
        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({collist}) SELECT {collist} FROM _bulk_upsert_df"
        )
    finally:
        conn.unregister("_bulk_upsert_df")


def upsert_hitter_games(conn: duckdb.DuckDBPyConnection, rows: Sequence[HitterGame]) -> None:
    """Insert or replace rows in `hitter_games` keyed by (player_id, game_pk).

    Empty input is a no-op. DuckDB's `INSERT OR REPLACE` handles PK collisions
    atomically; within-batch duplicate keys resolve last-wins.
    """
    _bulk_upsert(conn, table="hitter_games", columns=_HITTER_GAME_COLS, rows=rows)


def upsert_statcast_pa(conn: duckdb.DuckDBPyConnection, rows: Sequence[HitterStatcastPA]) -> None:
    """Insert or replace rows in `hitter_statcast_pa` keyed by (player_id, date, pa_index).

    Empty input is a no-op; within-batch duplicate keys resolve last-wins.
    """
    _bulk_upsert(conn, table="hitter_statcast_pa", columns=_STATCAST_COLS, rows=rows)


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
