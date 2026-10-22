"""Integration tests for the /luck route (analysis.luck's payload)."""

from __future__ import annotations

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.web.season_app import create_app
from fantasy_baseball.web.season_data import write_cache

CATS = ("R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "test.db"))
    kv_store._reset_singleton()
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        with c.session_transaction() as sess:
            sess["authenticated"] = True
        yield c
    kv_store._reset_singleton()


def _team(name: str, luck: float) -> dict:
    return {
        "team": name,
        "points": 60.0 + luck,
        "expected_points": 60.0,
        "luck_points": luck,
        "categories": dict.fromkeys(CATS, luck / 10),
        "pa_vs_proj": -54.0,
        "ip_vs_proj": 12.0,
        "perf_sgp": luck / 2,
        "pt_sgp": -1.0,
    }


def _payload() -> dict:
    return {
        "as_of": "2026-09-28",
        "standings_date": "2026-09-28",
        "teams": [_team("Alpha", 4.0), _team("Beta", -4.0)],
        "players": [
            {
                "team": "Alpha",
                "name": "Otto Lopez",
                "kind": "hitter",
                "acquired": "2026-04-21",
                "proj_as_of": "2026-03-30",
                "stale": True,
                "perf_sgp": 2.8,
                "pt_sgp": 1.9,
                "total_sgp": 4.6,
                "volume": 595.0,
                "pt_got": 600.0,
                "pt_proj": 560.0,
                "actual": {"R": 76, "HR": 8, "RBI": 60, "SB": 30, "AVG": 0.300},
                "expected": {"R": 68, "HR": 12, "RBI": 64, "SB": 17, "AVG": 0.267},
            }
        ],
        "unresolved": [],
    }


def test_luck_page_renders_teams_players_and_a_quiet_stale_marker(client) -> None:
    write_cache(CacheKey.LUCK, _payload())
    resp = client.get("/luck")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Alpha" in html and "Beta" in html and "Otto Lopez" in html
    assert "+4.0" in html and "-4.0" in html  # team luck points, signed
    assert "AVG 0.300/0.267" in html  # actual / expected
    # The stale projection is a dim date with a tooltip, nothing more.
    assert 'class="luck-stale" title="projection from 2026-03-30"' in html


def test_luck_page_without_data_says_so_briefly(client) -> None:
    resp = client.get("/luck")
    assert resp.status_code == 200
    assert "No luck data yet. Run a refresh." in resp.get_data(as_text=True)
