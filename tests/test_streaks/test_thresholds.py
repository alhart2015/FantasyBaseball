"""Tests for threshold calibration."""

from __future__ import annotations

from datetime import date, timedelta

from fantasy_baseball.streaks.data.load import upsert_hitter_games
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.models import HitterGame
from fantasy_baseball.streaks.thresholds import compute_thresholds
from fantasy_baseball.streaks.windows import compute_windows

# Use April 1, 2025 as the base; ``day`` is treated as a 1-indexed offset
# so callers can pass values past 30 without hitting month-boundary errors.
_BASE = date(2025, 4, 1)


def _g(pid: int, day: int, **kwargs: int) -> HitterGame:
    defaults = dict(
        player_id=pid,
        game_pk=pid * 1000 + day,
        name="X",
        team="ABC",
        season=2025,
        date=_BASE + timedelta(days=day - 1),
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
    return HitterGame(**defaults)


def test_compute_thresholds_writes_p10_p90_per_strata() -> None:
    """Set up two qualifying players, build windows, compute thresholds, verify shape."""
    conn = get_connection(":memory:")
    games: list[HitterGame] = []
    # Player 1: 200 PA over 50 days, 5 HR (hot)
    for d in range(1, 51):
        games.append(_g(1, d, pa=4, ab=4, h=1, hr=(1 if d % 10 == 0 else 0)))
    # Player 2: 200 PA over 50 days, 0 HR (cold)
    for d in range(1, 51):
        games.append(_g(2, d, pa=4, ab=4, h=1, hr=0))
    upsert_hitter_games(conn, games)
    compute_windows(conn)

    n = compute_thresholds(conn, season_set="2025", qualifying_pa=150)
    assert n > 0

    rows = conn.execute(
        "SELECT category, window_days, pt_bucket, p10, p90 "
        "FROM thresholds WHERE season_set = '2025' "
        "ORDER BY category, window_days, pt_bucket"
    ).fetchall()
    # 5 categories x 3 window_days x up to 3 buckets = up to 45 rows; here
    # we should have at least one row per (category, window_days) seen in data.
    cats = {r[0] for r in rows}
    assert cats == {"hr", "r", "rbi", "sb", "avg"}
    for r in rows:
        assert r[3] <= r[4], f"p10 > p90 for {r}"


def test_compute_thresholds_excludes_unqualified_player_seasons() -> None:
    """Player with <150 PA in a season is dropped from calibration entirely."""
    conn = get_connection(":memory:")
    # Player 1: ~28 PA total, season=2025 (won't qualify at 150 cutoff)
    games = [_g(1, d, pa=4, ab=4, h=1) for d in range(1, 8)]
    upsert_hitter_games(conn, games)
    compute_windows(conn)

    n = compute_thresholds(conn, season_set="2025", qualifying_pa=150)
    # No qualifying rows -> no thresholds written.
    assert n == 0


def test_compute_thresholds_is_idempotent() -> None:
    conn = get_connection(":memory:")
    games = [
        _g(pid, d, pa=4, ab=4, h=1, hr=(1 if pid == 1 and d % 5 == 0 else 0))
        for pid in (1, 2, 3)
        for d in range(1, 51)
    ]
    upsert_hitter_games(conn, games)
    compute_windows(conn)
    n1 = compute_thresholds(conn, season_set="2025", qualifying_pa=150)
    n2 = compute_thresholds(conn, season_set="2025", qualifying_pa=150)
    assert n1 == n2
    total = conn.execute("SELECT COUNT(*) FROM thresholds WHERE season_set = '2025'").fetchone()[0]
    assert total == n1
