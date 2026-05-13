"""Integration tests for the /streaks route."""

from __future__ import annotations

from datetime import date

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.streaks.dashboard import serialize_report
from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
from fantasy_baseball.streaks.reports.sunday import (
    DriverLine,
    Report,
    ReportRow,
)
from fantasy_baseball.web.season_app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


@pytest.fixture
def kv_isolation(tmp_path, monkeypatch):
    """Per-test isolated SQLite KV — mirrors test_season_routes."""
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "test.db"))
    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


def _seed_streak_cache() -> None:
    """Seed CacheKey.STREAK_SCORES with a serialized Report payload.

    Uses lowercase category/label values to match the Literal types
    in fantasy_baseball.streaks.models.
    """
    from fantasy_baseball.web.season_data import write_cache

    score = PlayerCategoryScore(
        player_id=1,
        category="hr",
        label="hot",
        probability=0.6,
        drivers=(Driver(feature="barrel_pct", z_score=1.0),),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="Test Player",
        positions=("OF",),
        player_id=1,
        composite=1,
        scores={"hr": score},
        max_probability=0.6,
    )
    dl = DriverLine(
        player_name="Test Player",
        category="hr",
        label="hot",
        probability=0.6,
        drivers=(Driver(feature="barrel_pct", z_score=1.0),),
    )
    rpt = Report(
        report_date=date(2026, 5, 11),
        window_end=date(2026, 5, 10),
        team_name="Hart of the Order",
        league_id=5652,
        season_set_train="2023-2025",
        roster_rows=(row,),
        fa_rows=(),
        driver_lines=(dl,),
        skipped=(),
    )
    write_cache(CacheKey.STREAK_SCORES, serialize_report(rpt))


def test_streaks_route_with_seeded_cache(client, kv_isolation) -> None:
    _seed_streak_cache()
    resp = client.get("/streaks")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "Your Roster" in body
    assert "Top Free Agent Signals" in body
    assert "Drivers" in body
    assert "Test Player" in body


def test_streaks_route_empty_state(client, kv_isolation) -> None:
    resp = client.get("/streaks")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "No streak data yet" in body
