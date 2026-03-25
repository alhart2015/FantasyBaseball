import pytest
from fantasy_baseball.lineup.yahoo_roster import (
    parse_roster,
    parse_standings_raw,
)


def _make_mock_roster_player(name, positions, selected_position):
    return {
        "name": name,
        "eligible_positions": positions,
        "selected_position": selected_position,
        "player_id": "12345",
    }


class TestParseRoster:
    def test_extracts_player_info(self):
        raw = [
            _make_mock_roster_player("Juan Soto", ["OF", "Util"], "OF"),
            _make_mock_roster_player("Gerrit Cole", ["SP"], "SP"),
        ]
        roster = parse_roster(raw)
        assert len(roster) == 2
        assert roster[0]["name"] == "Juan Soto"
        assert roster[0]["positions"] == ["OF", "Util"]
        assert roster[0]["selected_position"] == "OF"

    def test_empty_roster(self):
        assert parse_roster([]) == []


def _make_raw_standings(teams_data):
    """Build a raw Yahoo standings JSON from simplified team data."""
    teams = {}
    for i, td in enumerate(teams_data):
        meta = [
            {"team_key": td.get("team_key", f"469.l.5652.t.{i+1}")},
            {"name": td.get("name", f"Team {i+1}")},
        ]
        detail = {}
        if "rank" in td or "stats" in td:
            detail["team_standings"] = {"rank": td.get("rank", 0)}
        if "stats" in td:
            detail["team_stats"] = {
                "coverage_type": "season",
                "stats": [
                    {"stat": {"stat_id": sid, "value": str(val)}}
                    for sid, val in td["stats"].items()
                ],
            }
        teams[str(i)] = {"team": [meta, detail]}
    teams["count"] = len(teams_data)
    return {
        "fantasy_content": {
            "league": [
                {"league_id": "5652"},
                {"standings": [{"teams": teams}]},
            ]
        }
    }


class TestParseStandings:
    def test_extracts_team_stats(self):
        raw = _make_raw_standings([{
            "name": "Hart of the Order",
            "team_key": "469.l.5652.t.4",
            "rank": 3,
            "stats": {"60": 450, "7": 120},
        }])
        standings = parse_standings_raw(raw, stat_id_map={"60": "R", "7": "HR"})
        assert len(standings) == 1
        assert standings[0]["name"] == "Hart of the Order"
        assert standings[0]["rank"] == 3
        assert standings[0]["stats"]["R"] == 450.0
        assert standings[0]["stats"]["HR"] == 120.0

    def test_empty_standings(self):
        raw = {"fantasy_content": {"league": [{}, {"standings": [{"teams": {"count": 0}}]}]}}
        assert parse_standings_raw(raw, stat_id_map={}) == []

    def test_empty_stat_values_skipped(self):
        """Pre-season: stat values are empty strings, should produce empty stats dict."""
        raw = _make_raw_standings([{
            "name": "Team A",
            "rank": 1,
            "stats": {"60": "", "7": ""},
        }])
        standings = parse_standings_raw(raw, stat_id_map={"60": "R", "7": "HR"})
        assert standings[0]["stats"] == {}
