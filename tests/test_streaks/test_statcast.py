"""Tests for the per-PA Statcast fetcher."""

from datetime import date
from unittest.mock import patch

import pandas as pd

from fantasy_baseball.streaks.data.statcast import (
    chunk_date_range,
    filter_terminal_pa,
    pitches_to_pa_rows,
)


def test_chunk_date_range_produces_seven_day_chunks():
    chunks = list(chunk_date_range(date(2024, 4, 1), date(2024, 4, 21), days=7))
    assert chunks == [
        (date(2024, 4, 1), date(2024, 4, 7)),
        (date(2024, 4, 8), date(2024, 4, 14)),
        (date(2024, 4, 15), date(2024, 4, 21)),
    ]


def test_chunk_date_range_handles_partial_final_chunk():
    chunks = list(chunk_date_range(date(2024, 4, 1), date(2024, 4, 10), days=7))
    assert chunks == [
        (date(2024, 4, 1), date(2024, 4, 7)),
        (date(2024, 4, 8), date(2024, 4, 10)),
    ]


def test_filter_terminal_pa_keeps_only_rows_with_events():
    df = pd.DataFrame(
        {
            "events": [None, "single", None, "strikeout"],
            "batter": [1, 1, 1, 1],
            "game_date": ["2024-04-01"] * 4,
        }
    )
    out = filter_terminal_pa(df)
    assert list(out["events"]) == ["single", "strikeout"]


def test_pitches_to_pa_rows_assigns_pa_index_per_player_per_date():
    df = pd.DataFrame(
        {
            "events": ["single", "double", "strikeout", "home_run"],
            "batter": [660271, 660271, 545361, 660271],
            "game_date": ["2024-04-01", "2024-04-01", "2024-04-01", "2024-04-02"],
            "launch_speed": [95.0, 102.0, None, 110.0],
            "launch_angle": [10.0, 25.0, None, 28.0],
            "estimated_woba_using_speedangle": [0.4, 0.7, 0.0, 0.95],
            "barrel": [0, 1, 0, 1],
        }
    )
    rows = pitches_to_pa_rows(df)

    trout_rows = sorted(
        (r for r in rows if r.player_id == 660271),
        key=lambda r: (r.date, r.pa_index),
    )
    other_rows = sorted(
        (r for r in rows if r.player_id == 545361),
        key=lambda r: (r.date, r.pa_index),
    )

    # Trout 4/1: 2 PAs, indices 1 and 2
    assert trout_rows[0].date == date(2024, 4, 1)
    assert trout_rows[0].pa_index == 1
    assert trout_rows[0].event == "single"
    assert trout_rows[1].pa_index == 2
    assert trout_rows[1].event == "double"
    assert trout_rows[1].barrel is True
    # Trout 4/2: 1 PA, index 1
    assert trout_rows[2].date == date(2024, 4, 2)
    assert trout_rows[2].pa_index == 1
    # Other player 4/1: 1 PA, index 1, NaN launch_speed → None
    assert other_rows[0].date == date(2024, 4, 1)
    assert other_rows[0].pa_index == 1
    assert other_rows[0].launch_speed is None


def test_pitches_to_pa_rows_handles_missing_barrel_column():
    df = pd.DataFrame(
        {
            "events": ["single"],
            "batter": [660271],
            "game_date": ["2024-04-01"],
            "launch_speed": [95.0],
            "launch_angle": [10.0],
            "estimated_woba_using_speedangle": [0.4],
        }
    )
    rows = pitches_to_pa_rows(df)
    assert rows[0].barrel is None


def test_pitches_to_pa_rows_unboxes_numpy_scalars():
    """DuckDB's executemany binder rejects numpy.int64/float64 — values must be native Python."""
    df = pd.DataFrame(
        {
            "events": ["single"],
            "batter": [660271],
            "game_date": ["2024-04-01"],
            "launch_speed": [95.0],
            "launch_angle": [10.0],
            "estimated_woba_using_speedangle": [0.4],
            "barrel": [1],
        }
    )
    row = pitches_to_pa_rows(df)[0]
    # Type, not just value — numpy.float64 is a subclass of float so a
    # naive isinstance(row.launch_speed, float) check would pass even on
    # the broken pre-fix path. type() exact-match catches it.
    assert type(row.launch_speed) is float
    assert type(row.launch_angle) is float
    assert type(row.estimated_woba_using_speedangle) is float
    assert type(row.player_id) is int
    assert type(row.pa_index) is int
    assert type(row.barrel) is bool


def test_fetch_statcast_pa_for_date_range_concatenates_chunks():
    from fantasy_baseball.streaks.data.statcast import fetch_statcast_pa_for_date_range

    chunk_a = pd.DataFrame(
        {
            "events": ["single"],
            "batter": [660271],
            "game_date": ["2024-04-01"],
            "launch_speed": [95.0],
            "launch_angle": [10.0],
            "estimated_woba_using_speedangle": [0.4],
            "barrel": [0],
        }
    )
    chunk_b = pd.DataFrame(
        {
            "events": ["home_run"],
            "batter": [660271],
            "game_date": ["2024-04-08"],
            "launch_speed": [110.0],
            "launch_angle": [28.0],
            "estimated_woba_using_speedangle": [0.95],
            "barrel": [1],
        }
    )
    with patch(
        "fantasy_baseball.streaks.data.statcast.statcast",
        side_effect=[chunk_a, chunk_b],
    ):
        rows = fetch_statcast_pa_for_date_range(date(2024, 4, 1), date(2024, 4, 14))
    assert len(rows) == 2
    assert {r.event for r in rows} == {"single", "home_run"}
