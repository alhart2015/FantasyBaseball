"""Tests for the streaks rolling-window aggregator.

Synthetic data only — populates an in-memory DuckDB with a small set of
hitter_games rows and asserts rolling sums match expected values.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from fantasy_baseball.streaks.data.load import upsert_hitter_games
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.models import HitterGame
from fantasy_baseball.streaks.windows import _compute_rolling_sums


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
