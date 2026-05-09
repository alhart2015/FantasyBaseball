from __future__ import annotations

from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.models import HitterProjectionRate


def _row(pid: int, season: int, hr_pa: float = 0.05, sb_pa: float = 0.02, n: int = 2):
    return HitterProjectionRate(
        player_id=pid, season=season, hr_per_pa=hr_pa, sb_per_pa=sb_pa, n_systems=n
    )


def test_upsert_projection_rates_inserts_rows() -> None:
    conn = get_connection(":memory:")
    upsert_projection_rates(conn, [_row(1, 2024), _row(2, 2024)])
    n = conn.execute("SELECT COUNT(*) FROM hitter_projection_rates").fetchone()[0]
    assert n == 2


def test_upsert_projection_rates_replaces_on_pk_collision() -> None:
    conn = get_connection(":memory:")
    upsert_projection_rates(conn, [_row(1, 2024, hr_pa=0.05)])
    upsert_projection_rates(conn, [_row(1, 2024, hr_pa=0.10)])
    rate = conn.execute(
        "SELECT hr_per_pa FROM hitter_projection_rates WHERE player_id=1 AND season=2024"
    ).fetchone()[0]
    assert rate == 0.10


def test_upsert_projection_rates_empty_input_is_noop() -> None:
    conn = get_connection(":memory:")
    upsert_projection_rates(conn, [])
    n = conn.execute("SELECT COUNT(*) FROM hitter_projection_rates").fetchone()[0]
    assert n == 0
