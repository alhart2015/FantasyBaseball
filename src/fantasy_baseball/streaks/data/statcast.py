"""Per-PA Statcast fetch via pybaseball.

Pulls pitch-level data in 7-day chunks (pybaseball's recommended size to
avoid Baseball Savant timeouts), filters to terminal-PA rows (where
``events`` is non-null), and assigns a per-(player, date) PA index for
the (player_id, date, pa_index) primary key.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

import pandas as pd
from pybaseball import statcast

from fantasy_baseball.streaks.models import HitterStatcastPA


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


def _na_to_none(v: Any) -> Any:
    """Convert pandas/numpy missing values to None and unbox numpy scalars.

    DuckDB's Python binder doesn't auto-coerce ``numpy.int64`` / ``numpy.float64``
    via the array-protocol path used by ``executemany``, so we call ``.item()``
    on any object that has it (numpy scalars do; native Python types don't).
    """
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def pitches_to_pa_rows(df: pd.DataFrame) -> list[HitterStatcastPA]:
    """Convert a Statcast pitch DataFrame to a list of :class:`HitterStatcastPA`.

    Filters to terminal PAs, sorts by ``[batter, game_date, at_bat_number]``
    (so ``pa_index`` is chronologically stable across re-fetches), assigns
    ``pa_index`` per (batter, game_date), and converts NaN/NaT/pd.NA values
    to None.
    """
    df = filter_terminal_pa(df)
    if df.empty:
        return []
    sort_cols = ["batter", "game_date"]
    if "at_bat_number" in df.columns:
        sort_cols.append("at_bat_number")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["pa_index"] = df.groupby(["batter", "game_date"]).cumcount() + 1

    rows: list[HitterStatcastPA] = []
    has_barrel = "barrel" in df.columns
    has_at_bat_number = "at_bat_number" in df.columns
    has_bb_type = "bb_type" in df.columns
    has_xba = "estimated_ba_using_speedangle" in df.columns
    has_distance = "hit_distance_sc" in df.columns
    for r in df.itertuples(index=False):
        rows.append(
            HitterStatcastPA(
                player_id=int(r.batter),
                date=pd.to_datetime(r.game_date).date(),
                pa_index=int(r.pa_index),
                event=_na_to_none(r.events),
                launch_speed=_na_to_none(getattr(r, "launch_speed", None)),
                launch_angle=_na_to_none(getattr(r, "launch_angle", None)),
                estimated_woba_using_speedangle=_na_to_none(
                    getattr(r, "estimated_woba_using_speedangle", None)
                ),
                barrel=(bool(r.barrel) if has_barrel and not pd.isna(r.barrel) else None),
                at_bat_number=(
                    _na_to_none(getattr(r, "at_bat_number", None)) if has_at_bat_number else None
                ),
                bb_type=(_na_to_none(getattr(r, "bb_type", None)) if has_bb_type else None),
                estimated_ba_using_speedangle=(
                    _na_to_none(getattr(r, "estimated_ba_using_speedangle", None))
                    if has_xba
                    else None
                ),
                hit_distance_sc=(
                    _na_to_none(getattr(r, "hit_distance_sc", None)) if has_distance else None
                ),
            )
        )
    return rows


def fetch_statcast_pa_for_date_range(
    start: date, end: date, chunk_days: int = 7
) -> list[HitterStatcastPA]:
    """Fetch and parse all per-PA Statcast rows in [start, end].

    Chunks the date range to avoid Baseball Savant timeouts.
    """
    all_rows: list[HitterStatcastPA] = []
    for chunk_start, chunk_end in chunk_date_range(start, end, chunk_days):
        df = statcast(start_dt=chunk_start.isoformat(), end_dt=chunk_end.isoformat())
        all_rows.extend(pitches_to_pa_rows(df))
    return all_rows
