from datetime import date
from unittest.mock import patch

import pytest

from fantasy_baseball.models.standings import (
    CategoryStats,
    Standings,
    StandingsEntry,
)
from fantasy_baseball.web.season_app import create_app
from fantasy_baseball.web.season_data import (
    CacheKey,
    _opponent_cache,
    clear_opponent_cache,
    format_standings_for_display,
    get_teams_list,
)


def _sample_teams() -> list[dict]:
    """Raw ``{name, team_key, rank, stats}`` rows used in multiple tests."""
    teams = [
        (
            "Hart of the Order",
            "469.l.5652.t.3",
            {
                "R": 300,
                "HR": 90,
                "RBI": 290,
                "SB": 50,
                "AVG": 0.270,
                "W": 35,
                "K": 600,
                "SV": 25,
                "ERA": 3.50,
                "WHIP": 1.18,
            },
        ),
        (
            "Springfield Isotopes",
            "469.l.5652.t.8",
            {
                "R": 310,
                "HR": 85,
                "RBI": 295,
                "SB": 40,
                "AVG": 0.265,
                "W": 38,
                "K": 580,
                "SV": 30,
                "ERA": 3.40,
                "WHIP": 1.15,
            },
        ),
        (
            "SkeleThor",
            "469.l.5652.t.5",
            {
                "R": 280,
                "HR": 95,
                "RBI": 280,
                "SB": 55,
                "AVG": 0.260,
                "W": 30,
                "K": 620,
                "SV": 20,
                "ERA": 3.60,
                "WHIP": 1.22,
            },
        ),
    ]
    return [
        {"name": n, "team_key": tk, "rank": i + 1, "stats": s} for i, (n, tk, s) in enumerate(teams)
    ]


def _sample_standings() -> dict:
    """Canonical ``Standings.to_json()`` shape for cache fixtures."""
    return {
        "effective_date": "2026-04-01",
        "teams": _sample_teams(),
    }


def _standings_from_raw(raw: list[dict] | dict) -> Standings:
    rows = raw["teams"] if isinstance(raw, dict) else raw
    return Standings(
        effective_date=date(2026, 4, 1),
        entries=[
            StandingsEntry(
                team_name=t["name"],
                team_key=t.get("team_key", ""),
                rank=t.get("rank", 0),
                stats=CategoryStats.from_dict(t.get("stats", {})),
                yahoo_points_for=t.get("points_for"),
            )
            for t in rows
        ],
    )


class TestGetTeamsList:
    def test_returns_all_teams(self):
        result = get_teams_list(_standings_from_raw(_sample_standings()), "Hart of the Order")
        assert len(result["teams"]) == 3

    def test_marks_user_team(self):
        result = get_teams_list(_standings_from_raw(_sample_standings()), "Hart of the Order")
        hart = next(t for t in result["teams"] if t["name"] == "Hart of the Order")
        assert hart["is_user"] is True
        iso = next(t for t in result["teams"] if t["name"] == "Springfield Isotopes")
        assert iso["is_user"] is False

    def test_includes_team_key_and_rank(self):
        result = get_teams_list(_standings_from_raw(_sample_standings()), "Hart of the Order")
        iso = next(t for t in result["teams"] if t["name"] == "Springfield Isotopes")
        assert iso["team_key"] == "469.l.5652.t.8"
        assert "rank" in iso

    def test_user_team_key_set(self):
        result = get_teams_list(_standings_from_raw(_sample_standings()), "Hart of the Order")
        assert result["user_team_key"] == "469.l.5652.t.3"

    def test_empty_standings(self):
        result = get_teams_list(
            Standings(effective_date=date(2026, 4, 1), entries=[]), "Hart of the Order"
        )
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
        with (
            patch("fantasy_baseball.web.season_routes.read_cache") as mock_rc,
            patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg,
        ):
            mock_rc.side_effect = lambda k: _sample_standings() if k == CacheKey.STANDINGS else None
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
            _standings_from_raw(_sample_standings()), "Hart of the Order"
        )
        for team in result["teams"]:
            assert "team_key" in team, f"Missing team_key for {team['name']}"

    def test_team_key_values_correct(self):
        result = format_standings_for_display(
            _standings_from_raw(_sample_standings()), "Hart of the Order"
        )
        isotopes = next(t for t in result["teams"] if t["name"] == "Springfield Isotopes")
        assert isotopes["team_key"] == "469.l.5652.t.8"


from unittest.mock import MagicMock

import pandas as pd

from fantasy_baseball.web.season_data import build_opponent_lineup


def _sample_projections():
    """Minimal blended projections DataFrames."""
    from fantasy_baseball.utils.name_utils import normalize_name

    hitters = pd.DataFrame(
        [
            {
                "name": "Salvador Perez",
                "fg_id": "1",
                "player_type": "hitter",
                "pa": 550,
                "ab": 500,
                "h": 130,
                "r": 60,
                "hr": 25,
                "rbi": 80,
                "sb": 3,
                "avg": 0.260,
                "adp": 80,
            },
        ]
    )
    pitchers = pd.DataFrame(
        [
            {
                "name": "Corbin Burnes",
                "fg_id": "2",
                "player_type": "pitcher",
                "w": 14,
                "k": 200,
                "sv": 0,
                "ip": 190,
                "er": 55,
                "bb": 40,
                "h_allowed": 155,
                "era": 2.60,
                "whip": 1.03,
                "adp": 15,
            },
        ]
    )
    hitters["_name_norm"] = hitters["name"].apply(normalize_name)
    pitchers["_name_norm"] = pitchers["name"].apply(normalize_name)
    return hitters, pitchers


class TestBuildOpponentLineup:
    def test_returns_hitters_and_pitchers(self):
        roster = [
            {
                "name": "Salvador Perez",
                "positions": ["C", "Util"],
                "selected_position": "C",
                "player_id": "100",
                "status": "",
            },
            {
                "name": "Corbin Burnes",
                "positions": ["SP"],
                "selected_position": "SP",
                "player_id": "200",
                "status": "",
            },
        ]
        hitters_proj, pitchers_proj = _sample_projections()

        result = build_opponent_lineup(
            roster=roster,
            opponent_name="Springfield Isotopes",
            hitters_proj=hitters_proj,
            pitchers_proj=pitchers_proj,
            rest_of_season_hitters=pd.DataFrame(),
            rest_of_season_pitchers=pd.DataFrame(),
            season_year=2026,
        )

        assert len(result["hitters"]) == 1
        assert len(result["pitchers"]) == 1
        assert result["hitters"][0]["name"] == "Salvador Perez"
        assert result["pitchers"][0]["name"] == "Corbin Burnes"

    def test_sgp_column(self):
        roster = [
            {
                "name": "Salvador Perez",
                "positions": ["C", "Util"],
                "selected_position": "C",
                "player_id": "100",
                "status": "",
            },
        ]
        hitters_proj, pitchers_proj = _sample_projections()

        result = build_opponent_lineup(
            roster=roster,
            opponent_name="Springfield Isotopes",
            hitters_proj=hitters_proj,
            pitchers_proj=pitchers_proj,
            rest_of_season_hitters=pd.DataFrame(),
            rest_of_season_pitchers=pd.DataFrame(),
            season_year=2026,
        )

        perez = result["hitters"][0]
        assert "sgp" in perez
        assert isinstance(perez["sgp"], float)

    def test_pace_key_not_stats_key(self):
        """build_opponent_lineup must write pace data under 'pace', not 'stats'.

        Commit ad72b0d renamed the cache key from 'stats' to 'pace' and updated
        the JS template to read p.pace. Any re-introduction of the old 'stats'
        key causes opponent pace highlighting to render blank with no colors.
        """
        roster = [
            {
                "name": "Salvador Perez",
                "positions": ["C", "Util"],
                "selected_position": "C",
                "player_id": "100",
                "status": "",
            },
            {
                "name": "Corbin Burnes",
                "positions": ["SP"],
                "selected_position": "SP",
                "player_id": "200",
                "status": "",
            },
        ]
        hitters_proj, pitchers_proj = _sample_projections()

        result = build_opponent_lineup(
            roster=roster,
            opponent_name="Springfield Isotopes",
            hitters_proj=hitters_proj,
            pitchers_proj=pitchers_proj,
            rest_of_season_hitters=pd.DataFrame(),
            rest_of_season_pitchers=pd.DataFrame(),
            season_year=2026,
        )

        hitter = result["hitters"][0]
        pitcher = result["pitchers"][0]

        # Pace data must be under "pace", never under the legacy "stats" key.
        assert "pace" in hitter, "hitter missing 'pace' key"
        assert "stats" not in hitter, "hitter has legacy 'stats' key — JS will render blank"
        assert isinstance(hitter["pace"], dict), "hitter['pace'] must be a dict"
        assert "R" in hitter["pace"], "hitter pace must include 'R'"
        assert "HR" in hitter["pace"], "hitter pace must include 'HR'"

        assert "pace" in pitcher, "pitcher missing 'pace' key"
        assert "stats" not in pitcher, "pitcher has legacy 'stats' key — JS will render blank"
        assert isinstance(pitcher["pace"], dict), "pitcher['pace'] must be a dict"
        assert "W" in pitcher["pace"], "pitcher pace must include 'W'"
        assert "K" in pitcher["pace"], "pitcher pace must include 'K'"


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
            if key == CacheKey.STANDINGS:
                return _sample_standings()
            return None

        with (
            patch("fantasy_baseball.web.season_routes.read_cache", side_effect=mock_cache),
            patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg,
            patch("fantasy_baseball.web.season_data.build_opponent_lineup") as mock_build,
            patch("fantasy_baseball.web.season_routes._load_yahoo_league") as mock_league,
            patch("fantasy_baseball.web.season_routes._load_projections") as mock_proj,
        ):
            mock_cfg.return_value.team_name = "Hart of the Order"
            mock_cfg.return_value.season_year = 2026
            mock_build.return_value = {
                "hitters": [{"name": "Salvador Perez", "sgp": 3.5}],
                "pitchers": [],
                "hitter_totals": {},
                "pitcher_totals": {},
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
        with (
            patch("fantasy_baseball.web.season_routes.read_cache") as mock_rc,
            patch("fantasy_baseball.web.season_routes.read_meta") as mock_rm,
            patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg,
        ):
            mock_rc.side_effect = lambda k: standings if k == CacheKey.STANDINGS else {}
            mock_rm.return_value = {"last_refresh": "9:00 AM", "week": "1"}
            mock_cfg.return_value.team_name = "Hart of the Order"
            resp = client.get("/standings")
        html = resp.data.decode()
        assert "/lineup?team=469.l.5652.t.8" in html  # Springfield Isotopes link
        assert "/lineup?team=469.l.5652.t.3" in html  # Hart of the Order link
