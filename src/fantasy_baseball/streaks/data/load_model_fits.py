"""Idempotent loader for ``model_fits``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from operator import attrgetter

import duckdb

from fantasy_baseball.streaks.models import ModelFit

_MODEL_FIT_COLS = tuple(f.name for f in fields(ModelFit))
_model_fit_row = attrgetter(*_MODEL_FIT_COLS)


def upsert_model_fits(
    conn: duckdb.DuckDBPyConnection, rows: Sequence[ModelFit]
) -> None:
    """Insert or replace rows in `model_fits` keyed by model_id."""
    if not rows:
        return
    placeholders = ", ".join(["?"] * len(_MODEL_FIT_COLS))
    sql = (
        f"INSERT OR REPLACE INTO model_fits ({', '.join(_MODEL_FIT_COLS)}) "
        f"VALUES ({placeholders})"
    )
    conn.executemany(sql, [_model_fit_row(r) for r in rows])
