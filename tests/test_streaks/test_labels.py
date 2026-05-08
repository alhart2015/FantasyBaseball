"""Tests for label application."""

from __future__ import annotations

from datetime import date, timedelta

from fantasy_baseball.streaks.data.load import upsert_hitter_games
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.labels import apply_labels
from fantasy_baseball.streaks.models import HitterGame
from fantasy_baseball.streaks.thresholds import compute_thresholds
from fantasy_baseball.streaks.windows import compute_windows

# Plan's _g helper used date(2025, 4, day) which raises for day > 30. Same fix
# we made in test_thresholds.py: use a base date + timedelta so day=50 -> May 20.
_BASE = date(2025, 4, 1)


def _seed_population(conn) -> None:
    """Synthetic dataset: spread of HR rates so percentile thresholds are non-degenerate."""
    games: list[HitterGame] = []
    # 5 players, 50 days each, varying HR rates 0-4 per 10 days.
    for pid in range(1, 6):
        for d in range(1, 51):
            hr = 1 if (d % (12 - 2 * pid) == 0) else 0
            games.append(
                HitterGame(
                    player_id=pid,
                    game_pk=pid * 1000 + d,
                    name=f"P{pid}",
                    team="ABC",
                    season=2025,
                    date=_BASE + timedelta(days=d - 1),
                    pa=4,
                    ab=4,
                    h=1,
                    hr=hr,
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
            )
    upsert_hitter_games(conn, games)
    compute_windows(conn)
    compute_thresholds(conn, season_set="2025", qualifying_pa=150)


def test_apply_labels_writes_rows_per_category() -> None:
    conn = get_connection(":memory:")
    _seed_population(conn)
    n = apply_labels(conn, season_set="2025")
    assert n > 0
    cats = {
        r[0] for r in conn.execute("SELECT DISTINCT category FROM hitter_streak_labels").fetchall()
    }
    assert cats == {"hr", "r", "rbi", "sb", "avg"}


def test_apply_labels_classifies_hot_above_p90_cold_below_p10() -> None:
    conn = get_connection(":memory:")
    _seed_population(conn)
    apply_labels(conn, season_set="2025")
    # Pull one threshold row + a window row matching it; confirm the label
    # respects the inequality.
    threshold = conn.execute(
        "SELECT category, window_days, pt_bucket, p10, p90 "
        "FROM thresholds WHERE season_set = '2025' AND category = 'hr' LIMIT 1"
    ).fetchone()
    cat, win, bucket, p10, p90 = threshold
    sample = conn.execute(
        "SELECT player_id, window_end, hr FROM hitter_windows "
        "WHERE window_days = ? AND pt_bucket = ? LIMIT 5",
        [win, bucket],
    ).fetchall()
    for pid, end, hr in sample:
        label = conn.execute(
            "SELECT label FROM hitter_streak_labels "
            "WHERE player_id = ? AND window_end = ? AND window_days = ? AND category = ?",
            [pid, end, win, cat],
        ).fetchone()[0]
        if hr >= p90:
            assert label == "hot", f"hr={hr} >= p90={p90} but label={label}"
        elif hr <= p10:
            assert label == "cold", f"hr={hr} <= p10={p10} but label={label}"
        else:
            assert label == "neutral"


def test_apply_labels_is_idempotent() -> None:
    conn = get_connection(":memory:")
    _seed_population(conn)
    n1 = apply_labels(conn, season_set="2025")
    n2 = apply_labels(conn, season_set="2025")
    assert n1 == n2
    total = conn.execute("SELECT COUNT(*) FROM hitter_streak_labels").fetchone()[0]
    assert total == n1


def test_apply_labels_skips_windows_without_matching_thresholds() -> None:
    """A window in a bucket with no threshold row gets no labels (not an error)."""
    conn = get_connection(":memory:")
    _seed_population(conn)
    # Manually delete one bucket's thresholds and re-apply.
    conn.execute("DELETE FROM thresholds WHERE season_set='2025' AND pt_bucket='high'")
    apply_labels(conn, season_set="2025")
    # No labels should reference a 'high'-bucket window.
    cnt = conn.execute(
        """
        SELECT COUNT(*) FROM hitter_streak_labels l
        JOIN hitter_windows w
          ON w.player_id = l.player_id
         AND w.window_end = l.window_end
         AND w.window_days = l.window_days
        WHERE w.pt_bucket = 'high'
        """
    ).fetchone()[0]
    assert cnt == 0
