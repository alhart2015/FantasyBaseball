"""Tests for scripts/freeze_preseason_baseline.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _mk_config(monkeypatch):
    from fantasy_baseball.config import LeagueConfig

    return LeagueConfig(
        league_id=123,
        num_teams=2,
        game_code="mlb",
        team_name="Team 01",
        draft_position=1,
        keepers=[],
        roster_slots={
            "OF": 3,
            "P": 9,
            "BN": 3,
            "IL": 2,
            "Util": 1,
            "C": 1,
            "1B": 1,
            "2B": 1,
            "3B": 1,
            "SS": 1,
        },
        projection_systems=["atc"],
        projection_weights={"atc": 1.0},
        sgp_overrides={},
        teams={1: "Team 01", 2: "Team 02"},
        strategy="no_punt_opp",
        scoring_mode="var",
        season_year=2026,
        season_start="2026-03-27",
        season_end="2026-09-28",
    )


def _fake_hitter(name, pid):
    return {
        "name": name,
        "positions": ["OF"],
        "selected_position": "OF",
        "player_id": pid,
        "status": "",
    }


def _fake_pitcher(name, pid):
    return {
        "name": name,
        "positions": ["SP"],
        "selected_position": "P",
        "player_id": pid,
        "status": "",
    }


def _fake_projection_row(name, player_type):
    if player_type == "hitter":
        return {
            "name": name,
            "player_type": "hitter",
            "team": "NYY",
            "pa": 600,
            "ab": 540,
            "h": 145,
            "r": 85,
            "hr": 25,
            "rbi": 80,
            "sb": 10,
            "avg": 0.269,
        }
    return {
        "name": name,
        "player_type": "pitcher",
        "team": "NYY",
        "ip": 180,
        "er": 65,
        "bb": 50,
        "h_allowed": 160,
        "w": 12,
        "k": 190,
        "sv": 0,
        "era": 3.25,
        "whip": 1.17,
    }


@pytest.fixture
def patched_script_env(fake_redis, monkeypatch):
    from fantasy_baseball.data import redis_store

    # Seed preseason projections in Redis
    redis_store.set_blended_projections(
        fake_redis,
        "hitters",
        [_fake_projection_row("H1", "hitter"), _fake_projection_row("H2", "hitter")],
    )
    redis_store.set_blended_projections(
        fake_redis,
        "pitchers",
        [_fake_projection_row("P1", "pitcher"), _fake_projection_row("P2", "pitcher")],
    )

    league_mock = MagicMock()
    league_mock.teams.return_value = {
        "t.1": {"name": "Team 01"},
        "t.2": {"name": "Team 02"},
    }

    def _fetch_roster(league, team_key, day=None):
        assert day == "2026-03-27"
        if team_key == "t.1":
            return [_fake_hitter("H1", "1"), _fake_pitcher("P1", "2")]
        return [_fake_hitter("H2", "3"), _fake_pitcher("P2", "4")]

    def _scaled_mc(
        team_rosters,
        h_slots,
        p_slots,
        user_team_name,
        n_iterations=1000,
        use_management=False,
        progress_cb=None,
    ):
        return {
            "team_results": {t: {"median_pts": 70.0} for t in team_rosters},
            "category_risk": {},
            "_used_management": use_management,
        }

    patches = [
        patch("fantasy_baseball.config.load_config", return_value=_mk_config(monkeypatch)),
        patch("fantasy_baseball.auth.yahoo_auth.get_yahoo_session", return_value=MagicMock()),
        patch("fantasy_baseball.auth.yahoo_auth.get_league", return_value=league_mock),
        patch("fantasy_baseball.lineup.yahoo_roster.fetch_roster", side_effect=_fetch_roster),
        patch("fantasy_baseball.data.redis_store.get_default_client", return_value=fake_redis),
        patch("fantasy_baseball.simulation.run_monte_carlo", side_effect=_scaled_mc),
    ]
    for p in patches:
        p.start()
    yield fake_redis
    for p in patches:
        p.stop()


def test_script_writes_baseline_to_redis(patched_script_env):
    from freeze_preseason_baseline import main

    main([])

    from fantasy_baseball.data import redis_store

    baseline = redis_store.get_preseason_baseline(patched_script_env, 2026)
    assert baseline is not None
    assert "base" in baseline and "with_management" in baseline
    assert baseline["base"]["_used_management"] is False
    assert baseline["with_management"]["_used_management"] is True
    assert baseline["meta"]["season_year"] == 2026
    assert baseline["meta"]["roster_date"] == "2026-03-27"
    assert "frozen_at" in baseline["meta"]


def test_script_refuses_to_overwrite_without_force(patched_script_env, capsys):
    from freeze_preseason_baseline import main

    main([])  # first write
    with pytest.raises(SystemExit) as excinfo:
        main([])  # second without --force
    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    assert "already frozen" in captured.out.lower() or "--force" in captured.out.lower()


def test_script_overwrites_with_force(patched_script_env):
    from freeze_preseason_baseline import main

    main([])
    main(["--force"])  # should not raise

    from fantasy_baseball.data import redis_store

    baseline = redis_store.get_preseason_baseline(patched_script_env, 2026)
    assert baseline is not None
