"""Per-PA Statcast fetch via pybaseball.

Pulls pitch-level data in 7-day chunks (pybaseball's recommended size to
avoid Baseball Savant timeouts), filters to terminal-PA rows (where
``events`` is non-null), and assigns a per-(player, date) PA index for
the (player_id, date, pa_index) primary key.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

import pandas as pd
from pybaseball import statcast


def chunk_date_range(start: date, end: date, days: int = 7) -> Iterator[tuple[date, date]]:
    """Yield (chunk_start, chunk_end) tuples covering [start, end] in *days*-long chunks.

    Final chunk is shorter if the range doesn't divide evenly.
    """
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=days - 1), end)
        yield (current, chunk_end)
        current = chunk_end + timedelta(days=1)


def filter_terminal_pa(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where the pitch ended a plate appearance (events non-null)."""
    return df[df["events"].notna()].reset_index(drop=True)


def _val_or_none(v: Any) -> Any:
    """Convert pandas/numpy NaN to None; pass everything else through unchanged."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def pitches_to_pa_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a Statcast pitch DataFrame to upsert-ready PA rows.

    Filters to terminal PAs, assigns pa_index per (batter, game_date), and
    converts NaN values to None.
    """
    df = filter_terminal_pa(df)
    if df.empty:
        return []
    df = df.sort_values(["batter", "game_date"]).reset_index(drop=True)
    df["pa_index"] = df.groupby(["batter", "game_date"]).cumcount() + 1

    rows: list[dict[str, Any]] = []
    has_barrel = "barrel" in df.columns
    for r in df.itertuples(index=False):
        rows.append(
            {
                "player_id": int(r.batter),
                "date": pd.to_datetime(r.game_date).date(),
                "pa_index": int(r.pa_index),
                "event": _val_or_none(r.events),
                "launch_speed": _val_or_none(getattr(r, "launch_speed", None)),
                "launch_angle": _val_or_none(getattr(r, "launch_angle", None)),
                "estimated_woba_using_speedangle": _val_or_none(
                    getattr(r, "estimated_woba_using_speedangle", None)
                ),
                "barrel": (bool(r.barrel) if has_barrel and not pd.isna(r.barrel) else None),
            }
        )
    return rows


def fetch_statcast_pa_for_date_range(
    start: date, end: date, chunk_days: int = 7
) -> list[dict[str, Any]]:
    """Fetch and parse all per-PA Statcast rows in [start, end].

    Chunks the date range to avoid Baseball Savant timeouts. Returns
    upsert-ready dicts keyed for `hitter_statcast_pa`.
    """
    all_rows: list[dict[str, Any]] = []
    for chunk_start, chunk_end in chunk_date_range(start, end, chunk_days):
        df = statcast(start_dt=chunk_start.isoformat(), end_dt=chunk_end.isoformat())
        all_rows.extend(pitches_to_pa_rows(df))
    return all_rows
