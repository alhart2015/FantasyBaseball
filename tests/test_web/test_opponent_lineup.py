import pytest
from unittest.mock import patch
from fantasy_baseball.web.season_data import format_standings_for_display, get_teams_list
from fantasy_baseball.web.season_app import create_app


def _sample_standings():
    """Minimal 3-team standings for tests."""
    teams = [
        ("Hart of the Order", "469.l.5652.t.3",
         {"R": 300, "HR": 90, "RBI": 290, "SB": 50, "AVG": 0.270,
          "W": 35, "K": 600, "SV": 25, "ERA": 3.50, "WHIP": 1.18}),
        ("Springfield Isotopes", "469.l.5652.t.8",
         {"R": 310, "HR": 85, "RBI": 295, "SB": 40, "AVG": 0.265,
          "W": 38, "K": 580, "SV": 30, "ERA": 3.40, "WHIP": 1.15}),
        ("SkeleThor", "469.l.5652.t.5",
         {"R": 280, "HR": 95, "RBI": 280, "SB": 55, "AVG": 0.260,
          "W": 30, "K": 620, "SV": 20, "ERA": 3.60, "WHIP": 1.22}),
    ]
    return [{"name": n, "team_key": tk, "rank": i + 1, "stats": s}
            for i, (n, tk, s) in enumerate(teams)]


class TestGetTeamsList:
    def test_returns_all_teams(self):
        standings = _sample_standings()
        result = get_teams_list(standings, "Hart of the Order")
        assert len(result["teams"]) == 3

    def test_marks_user_team(self):
        standings = _sample_standings()
        result = get_teams_list(standings, "Hart of the Order")
        hart = next(t for t in result["teams"] if t["name"] == "Hart of the Order")
        assert hart["is_user"] is True
        iso = next(t for t in result["teams"] if t["name"] == "Springfield Isotopes")
        assert iso["is_user"] is False

    def test_includes_team_key_and_rank(self):
        standings = _sample_standings()
        result = get_teams_list(standings, "Hart of the Order")
        iso = next(t for t in result["teams"] if t["name"] == "Springfield Isotopes")
        assert iso["team_key"] == "469.l.5652.t.8"
        assert "rank" in iso

    def test_user_team_key_set(self):
        standings = _sample_standings()
        result = get_teams_list(standings, "Hart of the Order")
        assert result["user_team_key"] == "469.l.5652.t.3"

    def test_empty_standings(self):
        result = get_teams_list([], "Hart of the Order")
        assert result["teams"] == []
        assert result["user_team_key"] is None


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestApiTeams:
    def test_returns_teams_from_standings_cache(self, client):
        with patch("fantasy_baseball.web.season_routes.read_cache") as mock_rc, \
             patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
            mock_rc.side_effect = lambda k: _sample_standings() if k == "standings" else None
            mock_cfg.return_value.team_name = "Hart of the Order"
            resp = client.get("/api/teams")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["teams"]) == 3
        assert data["user_team_key"] == "469.l.5652.t.3"

    def test_returns_empty_without_standings(self, client):
        with patch("fantasy_baseball.web.season_routes.read_cache", return_value=None):
            resp = client.get("/api/teams")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["teams"] == []


class TestStandingsTeamKey:
    def test_team_key_present_in_display_data(self):
        result = format_standings_for_display(
            _sample_standings(), "Hart of the Order"
        )
        for team in result["teams"]:
            assert "team_key" in team, f"Missing team_key for {team['name']}"

    def test_team_key_values_correct(self):
        result = format_standings_for_display(
            _sample_standings(), "Hart of the Order"
        )
        isotopes = next(t for t in result["teams"] if t["name"] == "Springfield Isotopes")
        assert isotopes["team_key"] == "469.l.5652.t.8"


import pandas as pd
from unittest.mock import MagicMock
from fantasy_baseball.web.season_data import build_opponent_lineup


def _sample_projections():
    """Minimal blended projections DataFrames."""
    from fantasy_baseball.utils.name_utils import normalize_name
    hitters = pd.DataFrame([
        {"name": "Salvador Perez", "fg_id": "1", "player_type": "hitter",
         "pa": 550, "ab": 500, "h": 130, "r": 60, "hr": 25, "rbi": 80,
         "sb": 3, "avg": 0.260, "adp": 80},
    ])
    pitchers = pd.DataFrame([
        {"name": "Corbin Burnes", "fg_id": "2", "player_type": "pitcher",
         "w": 14, "k": 200, "sv": 0, "ip": 190, "er": 55, "bb": 40,
         "h_allowed": 155, "era": 2.60, "whip": 1.03, "adp": 15},
    ])
    hitters["_name_norm"] = hitters["name"].apply(normalize_name)
    pitchers["_name_norm"] = pitchers["name"].apply(normalize_name)
    return hitters, pitchers


class TestBuildOpponentLineup:
    def test_returns_hitters_and_pitchers(self):
        roster = [
            {"name": "Salvador Perez", "positions": ["C", "Util"],
             "selected_position": "C", "player_id": "100", "status": ""},
            {"name": "Corbin Burnes", "positions": ["SP"],
             "selected_position": "SP", "player_id": "200", "status": ""},
        ]
        hitters_proj, pitchers_proj = _sample_projections()
        standings = _sample_standings()
        user_leverage = {"R": 0.1, "HR": 0.1, "RBI": 0.1, "SB": 0.1,
                         "AVG": 0.1, "W": 0.1, "K": 0.1, "SV": 0.1,
                         "ERA": 0.1, "WHIP": 0.1}

        result = build_opponent_lineup(
            roster=roster,
            opponent_name="Springfield Isotopes",
            standings=standings,
            hitters_proj=hitters_proj,
            pitchers_proj=pitchers_proj,
            ros_hitters=pd.DataFrame(),
            ros_pitchers=pd.DataFrame(),
            user_leverage=user_leverage,
            season_year=2026,
        )

        assert len(result["hitters"]) == 1
        assert len(result["pitchers"]) == 1
        assert result["hitters"][0]["name"] == "Salvador Perez"
        assert result["pitchers"][0]["name"] == "Corbin Burnes"

    def test_dual_wsgp_columns(self):
        roster = [
            {"name": "Salvador Perez", "positions": ["C", "Util"],
             "selected_position": "C", "player_id": "100", "status": ""},
        ]
        hitters_proj, pitchers_proj = _sample_projections()
        standings = _sample_standings()
        user_leverage = {"R": 0.2, "HR": 0.2, "RBI": 0.2, "SB": 0.1,
                         "AVG": 0.1, "W": 0.05, "K": 0.05, "SV": 0.05,
                         "ERA": 0.025, "WHIP": 0.025}

        result = build_opponent_lineup(
            roster=roster,
            opponent_name="Springfield Isotopes",
            standings=standings,
            hitters_proj=hitters_proj,
            pitchers_proj=pitchers_proj,
            ros_hitters=pd.DataFrame(),
            ros_pitchers=pd.DataFrame(),
            user_leverage=user_leverage,
            season_year=2026,
        )

        perez = result["hitters"][0]
        assert "wsgp_them" in perez
        assert "wsgp_you" in perez
        assert isinstance(perez["wsgp_them"], float)
        assert isinstance(perez["wsgp_you"], float)
