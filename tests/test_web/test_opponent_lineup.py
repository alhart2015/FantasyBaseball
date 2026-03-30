import pytest
from unittest.mock import patch
from fantasy_baseball.web.season_data import format_standings_for_display, get_teams_list, _opponent_cache, clear_opponent_cache
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


class TestOpponentCache:
    def test_clear_opponent_cache(self):
        _opponent_cache["test_key"] = {"data": {}, "fetched_at": 0}
        clear_opponent_cache()
        assert _opponent_cache == {}


class TestApiOpponentLineup:
    def test_requires_auth(self, client):
        resp = client.get("/api/opponent/469.l.5652.t.8/lineup")
        # Should redirect to login (302) or return 401
        assert resp.status_code in (302, 401)

    def test_returns_404_without_standings(self, client):
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        with patch("fantasy_baseball.web.season_routes.read_cache", return_value=None):
            resp = client.get("/api/opponent/469.l.5652.t.8/lineup")
        assert resp.status_code == 404

    def test_returns_lineup_data(self, client):
        with client.session_transaction() as sess:
            sess["authenticated"] = True

        hitters_proj, pitchers_proj = _sample_projections()

        def mock_cache(key):
            if key == "standings":
                return _sample_standings()
            return None

        with patch("fantasy_baseball.web.season_routes.read_cache", side_effect=mock_cache), \
             patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg, \
             patch("fantasy_baseball.web.season_data.build_opponent_lineup") as mock_build, \
             patch("fantasy_baseball.web.season_routes._get_yahoo_league_cached") as mock_league, \
             patch("fantasy_baseball.web.season_routes._get_projections_cached") as mock_proj:
            mock_cfg.return_value.team_name = "Hart of the Order"
            mock_cfg.return_value.season_year = 2026
            mock_build.return_value = {
                "hitters": [{"name": "Salvador Perez", "wsgp_them": 1.8, "wsgp_you": 2.1}],
                "pitchers": [],
            }
            mock_league.return_value = (MagicMock(), "469.l.5652.t.3")
            mock_proj.return_value = (hitters_proj, pitchers_proj, pd.DataFrame(), pd.DataFrame())

            resp = client.get("/api/opponent/469.l.5652.t.8/lineup")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["team_name"] == "Springfield Isotopes"
        assert len(data["hitters"]) == 1


class TestStandingsLinks:
    def test_team_names_are_links(self, client):
        standings = _sample_standings()
        with patch("fantasy_baseball.web.season_routes.read_cache") as mock_rc, \
             patch("fantasy_baseball.web.season_routes.read_meta") as mock_rm, \
             patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
            mock_rc.side_effect = lambda k: standings if k == "standings" else {}
            mock_rm.return_value = {"last_refresh": "9:00 AM", "week": "1"}
            mock_cfg.return_value.team_name = "Hart of the Order"
            resp = client.get("/standings")
        html = resp.data.decode()
        assert '/lineup?team=469.l.5652.t.8' in html  # Springfield Isotopes link
        assert '/lineup?team=469.l.5652.t.3' in html  # Hart of the Order link
