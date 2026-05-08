"""Tests for the streaks rolling-window aggregator.

Synthetic data only — populates an in-memory DuckDB with a small set of
hitter_games rows and asserts rolling sums match expected values.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from fantasy_baseball.streaks.data.load import upsert_hitter_games, upsert_statcast_pa
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.models import HitterGame, HitterStatcastPA
from fantasy_baseball.streaks.windows import (
    _add_rate_stats,
    _add_statcast_peripherals,
    _compute_rolling_sums,
)


def _g(player_id: int, day: int, **kwargs: int) -> HitterGame:
    """Build a HitterGame with sensible defaults; override fields via kwargs."""
    defaults: dict[str, object] = dict(
        player_id=player_id,
        game_pk=player_id * 1000 + day,
        name="X",
        team="ABC",
        season=2025,
        date=date(2025, 4, day),
        pa=4,
        ab=4,
        h=1,
        hr=0,
        r=0,
        rbi=0,
        sb=0,
        bb=0,
        k=1,
        b2=0,
        b3=0,
        sf=0,
        hbp=0,
        ibb=0,
        cs=0,
        gidp=0,
        sh=0,
        ci=0,
        is_home=True,
    )
    defaults.update(kwargs)
    return HitterGame(**defaults)  # type: ignore[arg-type]


def test_rolling_sums_3_day_window_aggregates_played_and_off_days() -> None:
    """Player plays 3 games on 4/1, 4/2, 4/4. Window ending 4/4 (3-day) covers 4/2, 4/3, 4/4."""
    conn = get_connection(":memory:")
    upsert_hitter_games(
        conn,
        [
            _g(1, 1, pa=4, ab=4, h=1, hr=0, k=1),
            _g(1, 2, pa=5, ab=4, h=2, hr=1, k=1),
            _g(1, 4, pa=4, ab=3, h=1, hr=0, bb=1, k=0),
        ],
    )
    df = _compute_rolling_sums(conn, window_days=3)
    row = df[(df["player_id"] == 1) & (df["window_end"] == pd.Timestamp("2025-04-04"))].iloc[0]
    # 4/2 + 4/3 (off, zeros) + 4/4
    assert row["pa"] == 9
    assert row["hr"] == 1
    assert row["h"] == 3


def test_rolling_sums_emits_row_for_off_day_inside_active_range() -> None:
    """Player plays 4/1 and 4/3. Window ending 4/2 (off-day) covers only 4/1 (PA=4)."""
    conn = get_connection(":memory:")
    upsert_hitter_games(
        conn,
        [
            _g(2, 1, pa=4),
            _g(2, 3, pa=5),
        ],
    )
    df = _compute_rolling_sums(conn, window_days=3)
    mask = (df["player_id"] == 2) & (df["window_end"] == pd.Timestamp("2025-04-02"))
    assert mask.any()
    assert df[mask].iloc[0]["pa"] == 4


def test_rolling_sums_does_not_emit_before_first_played_or_after_last() -> None:
    conn = get_connection(":memory:")
    upsert_hitter_games(conn, [_g(3, 5, pa=4), _g(3, 7, pa=4)])
    df = _compute_rolling_sums(conn, window_days=3)
    pdates = set(df[df["player_id"] == 3]["window_end"].dt.date)
    assert pdates == {date(2025, 4, 5), date(2025, 4, 6), date(2025, 4, 7)}


def test_rolling_sums_window_days_7_and_14() -> None:
    """Same data, both windows return; 14-day sum == 7-day sum here (only 7 days of data)."""
    conn = get_connection(":memory:")
    upsert_hitter_games(conn, [_g(4, d, pa=4, ab=4, h=1) for d in range(1, 8)])
    df7 = _compute_rolling_sums(conn, window_days=7)
    df14 = _compute_rolling_sums(conn, window_days=14)
    end = pd.Timestamp("2025-04-07")
    p7 = df7[(df7["player_id"] == 4) & (df7["window_end"] == end)].iloc[0]
    p14 = df14[(df14["player_id"] == 4) & (df14["window_end"] == end)].iloc[0]
    assert p7["pa"] == 28
    assert p14["pa"] == 28


def test_add_rate_stats_computes_avg_babip_iso_k_bb() -> None:
    sums = pd.DataFrame(
        [
            {
                "player_id": 1,
                "window_end": pd.Timestamp("2025-04-07"),
                "window_days": 7,
                "pa": 30,
                "ab": 26,
                "h": 8,
                "hr": 2,
                "r": 5,
                "rbi": 6,
                "sb": 1,
                "bb": 3,
                "k": 6,
                "b2": 2,
                "b3": 0,
                "sf": 1,
                "hbp": 0,
            }
        ]
    )
    out = _add_rate_stats(sums)
    row = out.iloc[0]
    # avg = 8/26
    assert math.isclose(row["avg"], 8 / 26)
    # babip = (h - hr) / (ab - k - hr + sf) = 6 / (26 - 6 - 2 + 1) = 6/19
    assert math.isclose(row["babip"], 6 / 19)
    # iso = (b2 + 2*b3 + 3*hr) / ab = (2 + 0 + 6) / 26
    assert math.isclose(row["iso"], 8 / 26)
    # k_pct, bb_pct
    assert math.isclose(row["k_pct"], 6 / 30)
    assert math.isclose(row["bb_pct"], 3 / 30)


def test_add_rate_stats_handles_zero_denominators() -> None:
    sums = pd.DataFrame(
        [
            {
                "player_id": 1,
                "window_end": pd.Timestamp("2025-04-07"),
                "window_days": 7,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "hr": 0,
                "r": 0,
                "rbi": 0,
                "sb": 0,
                "bb": 0,
                "k": 0,
                "b2": 0,
                "b3": 0,
                "sf": 0,
                "hbp": 0,
            }
        ]
    )
    out = _add_rate_stats(sums)
    row = out.iloc[0]
    # All denominators zero -> NaN
    for col in ("avg", "babip", "iso", "k_pct", "bb_pct"):
        assert pd.isna(row[col])


def test_add_statcast_peripherals_aggregates_per_window() -> None:
    """Player has 3 PAs on 4/1, 2 PAs on 4/3. Window ending 4/3 (3-day) averages all 5."""
    conn = get_connection(":memory:")
    upsert_statcast_pa(
        conn,
        [
            HitterStatcastPA(
                player_id=1,
                date=date(2025, 4, 1),
                pa_index=i,
                event="single",
                launch_speed=100.0,
                launch_angle=10.0,
                estimated_woba_using_speedangle=0.8,
                barrel=False,
                at_bat_number=i,
                bb_type="line_drive",
                estimated_ba_using_speedangle=0.6,
                hit_distance_sc=200.0,
            )
            for i in (1, 2, 3)
        ]
        + [
            # NOTE: only pa_index=1 has barrel=True so the assertion below
            # ("1 of 5 barrels = 0.2") is consistent with the fixture. The
            # plan snippet had ``barrel=True`` for both 4/3 PAs which would
            # produce 0.4, not 0.2 — fixed here to match the asserted spec.
            HitterStatcastPA(
                player_id=1,
                date=date(2025, 4, 3),
                pa_index=i,
                event="strikeout",
                launch_speed=None,
                launch_angle=None,
                estimated_woba_using_speedangle=0.0,
                barrel=(i == 1),
                at_bat_number=i,
                bb_type=None,
                estimated_ba_using_speedangle=0.0,
                hit_distance_sc=None,
            )
            for i in (1, 2)
        ],
    )
    sums = pd.DataFrame(
        [
            {
                "player_id": 1,
                "window_end": pd.Timestamp("2025-04-03"),
                "window_days": 3,
                "pa": 5,
            }
        ]
    )
    out = _add_statcast_peripherals(conn, sums)
    row = out.iloc[0]
    # ev_avg averages only the non-null launch_speeds (the 3 from 4/1) -> 100.0
    assert math.isclose(row["ev_avg"], 100.0)
    # barrel_pct: 1 of 5 barrels = 0.2
    assert math.isclose(row["barrel_pct"], 0.2)
    # xwoba_avg: (0.8*3 + 0.0*2) / 5 = 0.48
    assert math.isclose(row["xwoba_avg"], 0.48)


def test_add_statcast_peripherals_returns_nan_for_window_with_no_statcast_data() -> None:
    conn = get_connection(":memory:")
    sums = pd.DataFrame(
        [
            {
                "player_id": 99,
                "window_end": pd.Timestamp("2025-04-03"),
                "window_days": 3,
                "pa": 5,
            }
        ]
    )
    out = _add_statcast_peripherals(conn, sums)
    row = out.iloc[0]
    assert pd.isna(row["ev_avg"])
    assert pd.isna(row["barrel_pct"])
    assert pd.isna(row["xwoba_avg"])
