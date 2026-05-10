"""Tests for continuation-rate computation."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from fantasy_baseball.streaks.analysis.continuation import (
    compute_continuation_rates,
)
from fantasy_baseball.streaks.data.load import upsert_hitter_games
from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.labels import apply_labels
from fantasy_baseball.streaks.models import HitterGame, HitterProjectionRate
from fantasy_baseball.streaks.thresholds import compute_thresholds
from fantasy_baseball.streaks.windows import compute_windows


def _seed_full_pipeline(conn) -> None:
    """Set up a synthetic 2025 season with hot/cold-tilted players, then run
    the full label-application pipeline so continuation has data to chew on."""
    base = date(2025, 4, 1)
    games: list[HitterGame] = []
    # 8 players, 60 days each. Even-pid (2,4,6,8) are "high-rate"; odd are "low".
    for pid in range(1, 9):
        for d in range(1, 61):
            high = pid % 2 == 0
            hr = 1 if (high and d % 6 == 0) else 0
            sb = 1 if (high and d % 5 == 0) else 0
            r_val = 2 if high else 1
            rbi = 2 if high else 1
            games.append(
                HitterGame(
                    player_id=pid,
                    game_pk=pid * 1000 + d,
                    name=f"P{pid}",
                    team="ABC",
                    season=2025,
                    date=base + timedelta(days=d - 1),
                    pa=4,
                    ab=4,
                    h=1 if high else 0,
                    hr=hr,
                    r=r_val,
                    rbi=rbi,
                    sb=sb,
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
            )
    upsert_hitter_games(conn, games)
    upsert_projection_rates(
        conn,
        [
            HitterProjectionRate(
                player_id=pid,
                season=2025,
                hr_per_pa=0.05 if pid % 2 == 0 else 0.005,
                sb_per_pa=0.04 if pid % 2 == 0 else 0.004,
                r_per_pa=None,
                rbi_per_pa=None,
                avg=None,
                n_systems=2,
            )
            for pid in range(1, 9)
        ],
    )
    compute_windows(conn)
    compute_thresholds(conn, season_set="2025", qualifying_pa=150)
    apply_labels(conn, season_set="2025")


def test_continuation_writes_at_least_one_row_per_present_stratum() -> None:
    conn = get_connection(":memory:")
    _seed_full_pipeline(conn)
    n = compute_continuation_rates(conn, season_set="2025")
    assert n > 0
    methods = {
        r[0] for r in conn.execute("SELECT DISTINCT cold_method FROM continuation_rates").fetchall()
    }
    # Dense cats produce 'empirical' rows; sparse cats produce both poisson methods.
    assert "empirical" in methods
    assert "poisson_p10" in methods
    assert "poisson_p20" in methods


def test_continuation_lift_equals_p_continued_minus_p_baserate() -> None:
    conn = get_connection(":memory:")
    _seed_full_pipeline(conn)
    compute_continuation_rates(conn, season_set="2025")
    rows = conn.execute("SELECT p_continued, p_baserate, lift FROM continuation_rates").fetchall()
    assert rows
    for p_cont, p_base, lift in rows:
        assert lift == pytest.approx(p_cont - p_base, abs=1e-9)


def test_continuation_p_baserate_is_constant_within_category_window_bucket_direction() -> None:
    """The base rate is a property of the unconditioned population in a stratum;
    different `strength_bucket` rows in the same (cat, win, bucket, dir, method)
    must agree on `p_baserate`."""
    conn = get_connection(":memory:")
    _seed_full_pipeline(conn)
    compute_continuation_rates(conn, season_set="2025")
    rows = conn.execute(
        """
        SELECT category, window_days, pt_bucket, direction, cold_method,
               COUNT(DISTINCT p_baserate) AS distinct_baserates
        FROM continuation_rates
        GROUP BY category, window_days, pt_bucket, direction, cold_method
        """
    ).fetchall()
    for *_, distinct in rows:
        assert distinct == 1


def test_continuation_idempotent() -> None:
    conn = get_connection(":memory:")
    _seed_full_pipeline(conn)
    n1 = compute_continuation_rates(conn, season_set="2025")
    n2 = compute_continuation_rates(conn, season_set="2025")
    assert n1 == n2
    total = conn.execute("SELECT COUNT(*) FROM continuation_rates").fetchone()[0]
    assert total == n1
