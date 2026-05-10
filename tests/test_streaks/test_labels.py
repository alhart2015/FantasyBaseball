"""Tests for label application."""

from __future__ import annotations

from datetime import date, timedelta

from fantasy_baseball.streaks.data.load import upsert_hitter_games
from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.labels import apply_labels
from fantasy_baseball.streaks.models import HitterGame, HitterProjectionRate
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
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT category FROM hitter_streak_labels "
            "WHERE category IN ('r', 'rbi', 'avg') AND cold_method='empirical'"
        ).fetchall()
    }
    assert cats == {"r", "rbi", "avg"}


def test_apply_labels_classifies_hot_above_p90_cold_below_p10() -> None:
    conn = get_connection(":memory:")
    _seed_population(conn)
    apply_labels(conn, season_set="2025")
    threshold = conn.execute(
        "SELECT category, window_days, pt_bucket, p10, p90 "
        "FROM thresholds WHERE season_set = '2025' AND category = 'r' LIMIT 1"
    ).fetchone()
    cat, win, bucket, p10, p90 = threshold
    sample = conn.execute(
        "SELECT player_id, window_end, r FROM hitter_windows "
        "WHERE window_days = ? AND pt_bucket = ? LIMIT 5",
        [win, bucket],
    ).fetchall()
    for pid, end, r_val in sample:
        label = conn.execute(
            "SELECT label FROM hitter_streak_labels "
            "WHERE player_id = ? AND window_end = ? AND window_days = ? "
            "AND category = ? AND cold_method = 'empirical'",
            [pid, end, win, cat],
        ).fetchone()[0]
        if r_val >= p90:
            assert label == "hot", f"r={r_val} >= p90={p90} but label={label}"
        elif r_val <= p10:
            assert label == "cold", f"r={r_val} <= p10={p10} but label={label}"
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


def _seed_population_with_projections(conn) -> None:
    """Same skeleton as `_seed_population`, but also writes per-season projection rates.

    Player IDs 1-5; pid 1 is a low-rate hitter (0.005 HR/PA) and pid 5 a high-
    rate one (0.10 HR/PA). The intermediate players span the rate space.
    """
    _seed_population(conn)  # writes games + windows + thresholds
    proj_rows = [
        HitterProjectionRate(
            player_id=pid,
            season=2025,
            hr_per_pa=hr,
            sb_per_pa=sb,
            r_per_pa=None,
            rbi_per_pa=None,
            avg=None,
            n_systems=2,
        )
        for pid, hr, sb in [
            (1, 0.005, 0.005),
            (2, 0.020, 0.015),
            (3, 0.050, 0.030),
            (4, 0.075, 0.050),
            (5, 0.100, 0.080),
        ]
    ]
    upsert_projection_rates(conn, proj_rows)


def test_sparse_labels_emit_two_rows_per_window_one_per_method() -> None:
    conn = get_connection(":memory:")
    _seed_population_with_projections(conn)
    apply_labels(conn, season_set="2025")
    methods = {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT cold_method FROM hitter_streak_labels WHERE category = 'hr'"
        ).fetchall()
    }
    assert methods == {"poisson_p10", "poisson_p20"}


def test_sparse_labels_low_rate_player_never_cold() -> None:
    """Player 1 has hr_per_pa=0.005 -> expected ~ 0.05 in 10-PA windows; Poisson
    p10 collapses to 0 (window < 0 impossible). Cold should never fire for them."""
    conn = get_connection(":memory:")
    _seed_population_with_projections(conn)
    apply_labels(conn, season_set="2025")
    n_cold = conn.execute(
        "SELECT COUNT(*) FROM hitter_streak_labels "
        "WHERE player_id = 1 AND category = 'hr' AND label = 'cold'"
    ).fetchone()[0]
    assert n_cold == 0


def test_sparse_labels_high_rate_player_can_be_cold_at_zero() -> None:
    """A player with high projected HR rate going zero HR over a 14-day window
    should be labeled cold under poisson_p10. We build a custom fixture: 60
    days of high-PA games for pid 100, but the first 20 days are zero-HR (so
    several 14d windows ending in those days are zero-HR while still meeting
    the 'high' PT bucket / qualifying-PA thresholds).
    """
    conn = get_connection(":memory:")
    games: list[HitterGame] = []
    for d in range(1, 61):
        # Days 1-20: zero HR. Days 21+: 1 HR every other day.
        hr = 1 if (d > 20 and d % 2 == 0) else 0
        games.append(
            HitterGame(
                player_id=100,
                game_pk=100_000 + d,
                name="ColdSlugger",
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
    upsert_projection_rates(
        conn,
        [
            HitterProjectionRate(
                player_id=100,
                season=2025,
                hr_per_pa=0.10,
                sb_per_pa=0.0,
                r_per_pa=None,
                rbi_per_pa=None,
                avg=None,
                n_systems=2,
            )
        ],
    )
    compute_windows(conn)
    compute_thresholds(conn, season_set="2025", qualifying_pa=150)
    apply_labels(conn, season_set="2025")

    # Days 14-20 each anchor a 14-day window covering only the zero-HR period.
    # pid 100's hr_per_pa=0.10 x ~50 PA window = expected ~5 HR; P(X<=1) << 0.10
    # so Poisson p10 -> cold for those zero-HR windows.
    cold_rows = conn.execute(
        """
        SELECT COUNT(*) FROM hitter_streak_labels l
        JOIN hitter_windows w
          ON w.player_id = l.player_id
         AND w.window_end = l.window_end
         AND w.window_days = l.window_days
        WHERE l.player_id = 100
          AND l.category = 'hr'
          AND l.cold_method = 'poisson_p10'
          AND l.label = 'cold'
          AND w.window_days = 14
        """
    ).fetchone()[0]
    assert cold_rows >= 1, (
        "Expected at least one cold-HR poisson_p10 label for the zero-HR 14d "
        "stretch; got 0 — is the sparse-cat Poisson cold path firing at all?"
    )


def test_sparse_labels_unprojected_player_skipped() -> None:
    conn = get_connection(":memory:")
    _seed_population(conn)
    # No projection rates loaded — the INNER JOIN drops every sparse-cat row.
    apply_labels(conn, season_set="2025")
    n_sparse = conn.execute(
        "SELECT COUNT(*) FROM hitter_streak_labels WHERE category IN ('hr', 'sb')"
    ).fetchone()[0]
    assert n_sparse == 0
    # Dense cats still write rows.
    n_dense = conn.execute(
        "SELECT COUNT(*) FROM hitter_streak_labels WHERE category IN ('r', 'rbi', 'avg')"
    ).fetchone()[0]
    assert n_dense > 0


def test_sparse_p20_widens_cold_net_vs_p10() -> None:
    """Poisson p20 has a larger ppf at every expected value -> at least as
    many (often strictly more) cold labels as p10."""
    conn = get_connection(":memory:")
    _seed_population_with_projections(conn)
    apply_labels(conn, season_set="2025")
    n_p10 = conn.execute(
        "SELECT COUNT(*) FROM hitter_streak_labels "
        "WHERE category IN ('hr', 'sb') AND cold_method='poisson_p10' AND label='cold'"
    ).fetchone()[0]
    n_p20 = conn.execute(
        "SELECT COUNT(*) FROM hitter_streak_labels "
        "WHERE category IN ('hr', 'sb') AND cold_method='poisson_p20' AND label='cold'"
    ).fetchone()[0]
    assert n_p20 >= n_p10
