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
