import pytest
from fantasy_baseball.lineup.yahoo_roster import (
    parse_injuries_raw,
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

    def test_status_included_when_present(self):
        raw = [{"name": "Zack Wheeler", "eligible_positions": ["P", "IL"],
                "selected_position": "IL", "player_id": "9124", "status": "IL15"}]
        roster = parse_roster(raw)
        assert roster[0]["status"] == "IL15"

    def test_status_empty_when_healthy(self):
        raw = [_make_mock_roster_player("Juan Soto", ["OF"], "OF")]
        roster = parse_roster(raw)
        assert roster[0]["status"] == ""

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

    def test_extracts_points_for(self):
        """Yahoo's authoritative roto total must be pulled off team_standings.

        Guards against regressing the fix for the standings-mismatch bug
        (where display-rounded ties made our score_roto diverge from Yahoo
        by ±0.5). points_for is the ground truth for the live standings
        page.
        """
        raw = _make_raw_standings([{
            "name": "Spacemen",
            "team_key": "469.l.5652.t.7",
            "rank": 1,
            "stats": {"7": 136},
        }])
        # Inject points_for into team_standings — the helper doesn't set it.
        raw["fantasy_content"]["league"][1]["standings"][0]["teams"]["0"]["team"][1][
            "team_standings"
        ]["points_for"] = "74.5"
        standings = parse_standings_raw(raw, stat_id_map={"7": "R"})
        assert standings[0]["points_for"] == 74.5

    def test_points_for_absent_is_none(self):
        """Pre-season / projected standings have no points_for — must be None."""
        raw = _make_raw_standings([{
            "name": "Team A",
            "rank": 1,
            "stats": {"7": 10},
        }])
        standings = parse_standings_raw(raw, stat_id_map={"7": "R"})
        assert standings[0]["points_for"] is None


def _make_raw_roster_players(players_data):
    """Build raw Yahoo roster JSON from simplified player dicts.

    Each entry: {name, status?, status_full?, injury_note?, player_id?,
                 positions?, selected_position?}
    """
    players = {}
    for i, pd in enumerate(players_data):
        meta = [
            {"name": {"full": pd["name"], "first": "F", "last": "L"}},
            {"player_id": pd.get("player_id", str(10000 + i))},
        ]
        if "status" in pd:
            status_entry = {"status": pd["status"]}
            if "status_full" in pd:
                status_entry["status_full"] = pd["status_full"]
            meta.append(status_entry)
        if "injury_note" in pd:
            meta.append({"injury_note": pd["injury_note"]})
        if "positions" in pd:
            meta.append({
                "eligible_positions": [
                    {"position": p} for p in pd["positions"]
                ]
            })
        sel_pos = pd.get("selected_position", "BN")
        position_data = {
            "selected_position": [
                {"coverage_type": "date", "date": "2026-03-26"},
                {"position": sel_pos},
            ]
        }
        players[str(i)] = {"player": [[*meta], position_data]}
    players["count"] = len(players_data)
    return {
        "fantasy_content": {
            "team": [
                {"team_key": "469.l.5652.t.4"},
                {"roster": {"0": {"players": players}}},
            ]
        }
    }


class TestParseInjuries:
    def test_returns_only_injured_players(self):
        raw = _make_raw_roster_players([
            {"name": "Juan Soto"},
            {"name": "Zack Wheeler", "status": "IL15",
             "status_full": "15-Day Injured List",
             "injury_note": "Shoulder", "selected_position": "IL",
             "positions": ["P", "IL"]},
            {"name": "Logan Webb"},
        ])
        injuries = parse_injuries_raw(raw)
        assert len(injuries) == 1
        assert injuries[0]["name"] == "Zack Wheeler"

    def test_extracts_all_injury_fields(self):
        raw = _make_raw_roster_players([
            {"name": "Spencer Strider", "status": "IL15",
             "status_full": "15-Day Injured List",
             "injury_note": "Oblique", "selected_position": "IL",
             "positions": ["P", "IL"], "player_id": "12281"},
        ])
        injuries = parse_injuries_raw(raw)
        assert injuries[0]["status"] == "IL15"
        assert injuries[0]["status_full"] == "15-Day Injured List"
        assert injuries[0]["injury_note"] == "Oblique"
        assert injuries[0]["selected_position"] == "IL"
        assert "P" in injuries[0]["positions"]

    def test_dtd_player_included(self):
        raw = _make_raw_roster_players([
            {"name": "Byron Buxton", "status": "DTD",
             "injury_note": "Hip", "selected_position": "OF"},
        ])
        injuries = parse_injuries_raw(raw)
        assert len(injuries) == 1
        assert injuries[0]["status"] == "DTD"

    def test_il_eligible_not_in_il_slot(self):
        raw = _make_raw_roster_players([
            {"name": "Josh Hader", "status": "IL15",
             "injury_note": "Biceps", "selected_position": "BN",
             "positions": ["P", "IL"]},
        ])
        injuries = parse_injuries_raw(raw)
        assert injuries[0]["selected_position"] == "BN"
        assert injuries[0]["status"] == "IL15"

    def test_empty_roster(self):
        raw = {"fantasy_content": {"team": [{}]}}
        assert parse_injuries_raw(raw) == []

    def test_no_injuries(self):
        raw = _make_raw_roster_players([
            {"name": "Juan Soto"},
            {"name": "Julio Rodriguez"},
        ])
        assert parse_injuries_raw(raw) == []
