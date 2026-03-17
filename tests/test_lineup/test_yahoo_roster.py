import pytest
from fantasy_baseball.lineup.yahoo_roster import (
    parse_roster,
    parse_standings,
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


class TestParseStandings:
    def test_extracts_team_stats(self):
        raw = {
            "teams": [
                {
                    "name": "Hart of the Order",
                    "team_key": "469.l.5652.t.4",
                    "team_standings": {"rank": 3},
                    "team_stats": {
                        "stats": [
                            {"stat": {"stat_id": "60", "value": "450"}},  # R
                            {"stat": {"stat_id": "7", "value": "120"}},   # HR
                        ]
                    },
                },
            ]
        }
        standings = parse_standings(raw, stat_id_map={"60": "R", "7": "HR"})
        assert len(standings) == 1
        assert standings[0]["name"] == "Hart of the Order"
        assert standings[0]["rank"] == 3
        assert standings[0]["stats"]["R"] == 450.0
        assert standings[0]["stats"]["HR"] == 120.0

    def test_empty_standings(self):
        raw = {"teams": []}
        assert parse_standings(raw, stat_id_map={}) == []
