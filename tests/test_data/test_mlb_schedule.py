import json
from unittest.mock import patch

from fantasy_baseball.data.mlb_schedule import (
    MLB_TO_FANGRAPHS_ABBREV,
    fetch_week_schedule,
    get_week_schedule,
    load_schedule_cache,
    normalize_team_abbrev,
    save_schedule_cache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_game(away_name, home_name, game_date, game_type="R", away_pitcher="", home_pitcher=""):
    return {
        "away_name": away_name,
        "home_name": home_name,
        "game_date": game_date,
        "game_type": game_type,
        "away_probable_pitcher": away_pitcher,
        "home_probable_pitcher": home_pitcher,
    }


def _mock_teams_response():
    return {
        "teams": [
            {"name": "New York Yankees", "teamName": "Yankees", "abbreviation": "NYY"},
            {"name": "Boston Red Sox", "teamName": "Red Sox", "abbreviation": "BOS"},
            {"name": "Arizona Diamondbacks", "teamName": "Diamondbacks", "abbreviation": "AZ"},
            {"name": "Kansas City Royals", "teamName": "Royals", "abbreviation": "KC"},
            {"name": "Tampa Bay Rays", "teamName": "Rays", "abbreviation": "TB"},
            {"name": "Chicago White Sox", "teamName": "White Sox", "abbreviation": "CWS"},
            {"name": "San Diego Padres", "teamName": "Padres", "abbreviation": "SD"},
            {"name": "San Francisco Giants", "teamName": "Giants", "abbreviation": "SF"},
            {"name": "Washington Nationals", "teamName": "Nationals", "abbreviation": "WSH"},
        ],
    }


# ---------------------------------------------------------------------------
# TestFetchWeekScheduleLookback
# ---------------------------------------------------------------------------


class TestFetchWeekScheduleLookback:
    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_lookback_extends_start_date(self, mock_api):
        # Teams call (for _build_team_name_map)
        mock_api.get.return_value = _mock_teams_response()
        # Schedule call returns games across both lookback and window
        mock_api.schedule.return_value = [
            _make_game(
                "New York Yankees", "Boston Red Sox", "2026-04-25", away_pitcher="Gerrit Cole"
            ),
            _make_game(
                "New York Yankees", "Boston Red Sox", "2026-05-05", away_pitcher="Carlos Rodon"
            ),
        ]

        result = fetch_week_schedule("2026-05-05", "2026-05-11", lookback_days=14)

        # statsapi.schedule called with start = 2026-05-05 - 14d = 2026-04-21
        mock_api.schedule.assert_called_once_with(start_date="2026-04-21", end_date="2026-05-11")
        # Both past and current games appear in probable_pitchers
        assert len(result["probable_pitchers"]) == 2
        # The returned start_date/end_date still reflect the scoring week
        assert result["start_date"] == "2026-05-05"
        assert result["end_date"] == "2026-05-11"

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_default_lookback_is_zero(self, mock_api):
        mock_api.get.return_value = _mock_teams_response()
        mock_api.schedule.return_value = []
        fetch_week_schedule("2026-05-05", "2026-05-11")
        mock_api.schedule.assert_called_once_with(start_date="2026-05-05", end_date="2026-05-11")

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_game_number_captured(self, mock_api):
        mock_api.get.return_value = _mock_teams_response()
        mock_api.schedule.return_value = [
            {
                **_make_game(
                    "New York Yankees",
                    "Boston Red Sox",
                    "2026-05-05",
                    away_pitcher="A",
                    home_pitcher="B",
                ),
                "game_num": 1,
            },
            {
                **_make_game(
                    "New York Yankees",
                    "Boston Red Sox",
                    "2026-05-05",
                    away_pitcher="C",
                    home_pitcher="D",
                ),
                "game_num": 2,
            },
        ]
        result = fetch_week_schedule("2026-05-05", "2026-05-11")
        pps = result["probable_pitchers"]
        assert len(pps) == 2
        assert pps[0]["game_number"] == 1
        assert pps[1]["game_number"] == 2

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_game_number_defaults_to_one(self, mock_api):
        # statsapi may omit game_num for non-doubleheader games
        mock_api.get.return_value = _mock_teams_response()
        mock_api.schedule.return_value = [
            _make_game("New York Yankees", "Boston Red Sox", "2026-05-05"),
        ]
        result = fetch_week_schedule("2026-05-05", "2026-05-11")
        assert result["probable_pitchers"][0]["game_number"] == 1


# ---------------------------------------------------------------------------
# TestNormalizeTeamAbbrev
# ---------------------------------------------------------------------------


class TestNormalizeTeamAbbrev:
    def test_passthrough_nyy(self):
        assert normalize_team_abbrev("NYY") == "NYY"

    def test_passthrough_bos(self):
        assert normalize_team_abbrev("BOS") == "BOS"

    def test_passthrough_lad(self):
        assert normalize_team_abbrev("LAD") == "LAD"

    def test_az_converts_to_ari(self):
        assert normalize_team_abbrev("AZ") == "ARI"

    def test_cws_converts_to_chw(self):
        assert normalize_team_abbrev("CWS") == "CHW"

    def test_kc_converts_to_kcr(self):
        assert normalize_team_abbrev("KC") == "KCR"

    def test_sd_converts_to_sdp(self):
        assert normalize_team_abbrev("SD") == "SDP"

    def test_sf_converts_to_sfg(self):
        assert normalize_team_abbrev("SF") == "SFG"

    def test_tb_converts_to_tbr(self):
        assert normalize_team_abbrev("TB") == "TBR"

    def test_wsh_converts_to_wsn(self):
        assert normalize_team_abbrev("WSH") == "WSN"

    def test_mapping_has_exactly_seven_entries(self):
        assert len(MLB_TO_FANGRAPHS_ABBREV) == 7


# ---------------------------------------------------------------------------
# TestFetchWeekSchedule
# ---------------------------------------------------------------------------


class TestFetchWeekSchedule:
    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_counts_games_per_team_with_fangraphs_abbrevs(self, mock_statsapi):
        mock_statsapi.schedule.return_value = [
            _make_game("Kansas City Royals", "New York Yankees", "2026-04-07"),
            _make_game("Kansas City Royals", "New York Yankees", "2026-04-08"),
            _make_game("Boston Red Sox", "New York Yankees", "2026-04-09"),
        ]
        mock_statsapi.get.return_value = _mock_teams_response()

        result = fetch_week_schedule("2026-04-07", "2026-04-13")

        gpt = result["games_per_team"]
        assert gpt["KCR"] == 2
        assert gpt["NYY"] == 3
        assert gpt["BOS"] == 1

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_filters_out_non_regular_season_games(self, mock_statsapi):
        mock_statsapi.schedule.return_value = [
            _make_game("New York Yankees", "Boston Red Sox", "2026-04-07", game_type="R"),
            _make_game("New York Yankees", "Boston Red Sox", "2026-04-06", game_type="S"),
            _make_game("New York Yankees", "Boston Red Sox", "2026-04-05", game_type="E"),
        ]
        mock_statsapi.get.return_value = _mock_teams_response()

        result = fetch_week_schedule("2026-04-07", "2026-04-13")

        gpt = result["games_per_team"]
        assert gpt.get("NYY", 0) == 1
        assert gpt.get("BOS", 0) == 1

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_counts_doubleheader_games_separately(self, mock_statsapi):
        mock_statsapi.schedule.return_value = [
            _make_game("Boston Red Sox", "New York Yankees", "2026-04-07"),
            _make_game("Boston Red Sox", "New York Yankees", "2026-04-07"),
        ]
        mock_statsapi.get.return_value = _mock_teams_response()

        result = fetch_week_schedule("2026-04-07", "2026-04-13")

        gpt = result["games_per_team"]
        assert gpt["BOS"] == 2
        assert gpt["NYY"] == 2


# ---------------------------------------------------------------------------
# TestProbablePitchers
# ---------------------------------------------------------------------------


class TestProbablePitchers:
    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_extracts_pitcher_names_team_abbrevs_and_dates(self, mock_statsapi):
        mock_statsapi.schedule.return_value = [
            _make_game(
                "New York Yankees",
                "Boston Red Sox",
                "2026-04-07",
                away_pitcher="Gerrit Cole",
                home_pitcher="Chris Sale",
            ),
        ]
        mock_statsapi.get.return_value = _mock_teams_response()

        result = fetch_week_schedule("2026-04-07", "2026-04-13")
        pitchers = result["probable_pitchers"]

        assert len(pitchers) == 1
        entry = pitchers[0]
        assert entry["away_team"] == "NYY"
        assert entry["home_team"] == "BOS"
        assert entry["away_pitcher"] == "Gerrit Cole"
        assert entry["home_pitcher"] == "Chris Sale"
        assert entry["date"] == "2026-04-07"

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_empty_pitcher_strings_become_tbd(self, mock_statsapi):
        mock_statsapi.schedule.return_value = [
            _make_game(
                "New York Yankees", "Boston Red Sox", "2026-04-07", away_pitcher="", home_pitcher=""
            ),
        ]
        mock_statsapi.get.return_value = _mock_teams_response()

        result = fetch_week_schedule("2026-04-07", "2026-04-13")
        entry = result["probable_pitchers"][0]

        assert entry["away_pitcher"] == "TBD"
        assert entry["home_pitcher"] == "TBD"

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_metadata_fields_present(self, mock_statsapi):
        mock_statsapi.schedule.return_value = []
        mock_statsapi.get.return_value = _mock_teams_response()

        result = fetch_week_schedule("2026-04-07", "2026-04-13")

        assert "start_date" in result
        assert result["start_date"] == "2026-04-07"
        assert "end_date" in result
        assert result["end_date"] == "2026-04-13"
        assert "fetched_at" in result
        assert "team_abbrev_map" in result
        assert isinstance(result["team_abbrev_map"], dict)

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_team_abbrev_map_uses_fangraphs_abbrevs(self, mock_statsapi):
        mock_statsapi.schedule.return_value = []
        mock_statsapi.get.return_value = _mock_teams_response()

        result = fetch_week_schedule("2026-04-07", "2026-04-13")
        tam = result["team_abbrev_map"]

        # Full name lookup
        assert tam["New York Yankees"] == "NYY"
        # Short name lookup
        assert tam["Yankees"] == "NYY"
        # Divergent abbreviation normalized
        assert tam["Kansas City Royals"] == "KCR"
        assert tam["Royals"] == "KCR"


# ---------------------------------------------------------------------------
# TestScheduleCache
# ---------------------------------------------------------------------------


class TestScheduleCache:
    def test_save_and_load_roundtrip(self, tmp_path):
        data = {
            "games_per_team": {"NYY": 7, "BOS": 6},
            "probable_pitchers": [],
            "team_abbrev_map": {},
            "start_date": "2026-04-07",
            "end_date": "2026-04-13",
            "fetched_at": "2026-04-07T10:00:00",
        }
        cache_path = tmp_path / "schedule.json"
        save_schedule_cache(data, cache_path)
        loaded = load_schedule_cache(cache_path)
        assert loaded == data

    def test_load_missing_file_returns_none(self, tmp_path):
        result = load_schedule_cache(tmp_path / "nonexistent.json")
        assert result is None


# ---------------------------------------------------------------------------
# TestGetWeekSchedule
# ---------------------------------------------------------------------------


class TestGetWeekSchedule:
    @patch("fantasy_baseball.data.mlb_schedule.fetch_week_schedule")
    def test_live_fetch_success_caches_result(self, mock_fetch, tmp_path):
        expected = {
            "games_per_team": {"NYY": 7},
            "probable_pitchers": [],
            "team_abbrev_map": {},
            "start_date": "2026-04-07",
            "end_date": "2026-04-13",
            "fetched_at": "2026-04-07T10:00:00",
        }
        mock_fetch.return_value = expected
        cache_path = tmp_path / "schedule.json"

        result = get_week_schedule("2026-04-07", "2026-04-13", cache_path)

        assert result == expected
        assert cache_path.exists()
        cached = json.loads(cache_path.read_text())
        assert cached == expected

    @patch("fantasy_baseball.data.mlb_schedule.fetch_week_schedule")
    def test_falls_back_to_cache_on_api_failure(self, mock_fetch, tmp_path):
        cached_data = {
            "games_per_team": {"BOS": 6},
            "probable_pitchers": [],
            "team_abbrev_map": {},
            "start_date": "2026-04-07",
            "end_date": "2026-04-13",
            "fetched_at": "2026-04-06T10:00:00",
        }
        cache_path = tmp_path / "schedule.json"
        cache_path.write_text(json.dumps(cached_data))

        mock_fetch.side_effect = Exception("API down")

        result = get_week_schedule("2026-04-07", "2026-04-13", cache_path)

        assert result == cached_data

    @patch("fantasy_baseball.data.mlb_schedule.fetch_week_schedule")
    def test_ignores_stale_cache_with_wrong_dates_returns_none(self, mock_fetch, tmp_path):
        stale_data = {
            "games_per_team": {"LAD": 5},
            "probable_pitchers": [],
            "team_abbrev_map": {},
            "start_date": "2026-03-31",
            "end_date": "2026-04-06",
            "fetched_at": "2026-03-31T10:00:00",
        }
        cache_path = tmp_path / "schedule.json"
        cache_path.write_text(json.dumps(stale_data))

        mock_fetch.side_effect = Exception("API down")

        result = get_week_schedule("2026-04-07", "2026-04-13", cache_path)

        assert result is None

    @patch("fantasy_baseball.data.mlb_schedule.fetch_week_schedule")
    def test_returns_none_with_no_cache_and_api_failure(self, mock_fetch, tmp_path):
        mock_fetch.side_effect = Exception("API down")
        cache_path = tmp_path / "schedule.json"

        result = get_week_schedule("2026-04-07", "2026-04-13", cache_path)

        assert result is None


# ---------------------------------------------------------------------------
# TestGetWeekScheduleLookback
# ---------------------------------------------------------------------------


class TestGetWeekScheduleLookback:
    @patch("fantasy_baseball.data.mlb_schedule.fetch_week_schedule")
    def test_passes_lookback_through(self, mock_fetch, tmp_path):
        mock_fetch.return_value = {
            "games_per_team": {},
            "probable_pitchers": [],
            "team_abbrev_map": {},
            "start_date": "2026-05-05",
            "end_date": "2026-05-11",
            "lookback_days": 14,
            "fetched_at": "2026-05-05T08:00:00",
        }
        cache_path = tmp_path / "schedule.json"

        result = get_week_schedule("2026-05-05", "2026-05-11", cache_path, lookback_days=14)

        mock_fetch.assert_called_once_with("2026-05-05", "2026-05-11", lookback_days=14)
        assert result["lookback_days"] == 14

    @patch(
        "fantasy_baseball.data.mlb_schedule.fetch_week_schedule",
        side_effect=RuntimeError("api down"),
    )
    def test_cache_match_includes_lookback(self, _mock_fetch, tmp_path):
        cache_path = tmp_path / "schedule.json"
        # Cache from a prior 0-lookback fetch
        save_schedule_cache(
            {
                "games_per_team": {},
                "probable_pitchers": [],
                "team_abbrev_map": {},
                "start_date": "2026-05-05",
                "end_date": "2026-05-11",
                "lookback_days": 0,
                "fetched_at": "2026-05-04T08:00:00",
            },
            cache_path,
        )

        # Caller now wants 14-day lookback - cache must NOT match.
        result = get_week_schedule("2026-05-05", "2026-05-11", cache_path, lookback_days=14)
        assert result is None
