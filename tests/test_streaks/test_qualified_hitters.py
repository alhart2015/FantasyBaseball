"""Tests for the ≥150 PA qualified hitters fetch."""

from unittest.mock import patch

from fantasy_baseball.streaks.data.qualified_hitters import (
    fetch_qualified_hitters,
    parse_leader_row,
)


def test_parse_leader_row_extracts_id_name_team_pa():
    row = {
        "person": {"id": 660271, "fullName": "Mike Trout"},
        "team": {"abbreviation": "LAA"},
        "value": "162",
    }
    parsed = parse_leader_row(row)
    assert parsed == {
        "player_id": 660271,
        "name": "Mike Trout",
        "team": "LAA",
        "pa": 162,
    }


def test_parse_leader_row_handles_missing_team():
    row = {
        "person": {"id": 545361, "fullName": "Free Agent"},
        "team": {},
        "value": "150",
    }
    parsed = parse_leader_row(row)
    assert parsed["team"] is None


def test_fetch_qualified_hitters_filters_below_min_pa():
    fake_response = {
        "leagueLeaders": [
            {
                "leaders": [
                    {
                        "person": {"id": 1, "fullName": "Above Cutoff"},
                        "team": {"abbreviation": "NYY"},
                        "value": "151",
                    },
                    {
                        "person": {"id": 2, "fullName": "Right At Cutoff"},
                        "team": {"abbreviation": "BOS"},
                        "value": "150",
                    },
                    {
                        "person": {"id": 3, "fullName": "Below Cutoff"},
                        "team": {"abbreviation": "TBR"},
                        "value": "149",
                    },
                ]
            }
        ]
    }
    with patch(
        "fantasy_baseball.streaks.data.qualified_hitters.statsapi.get",
        return_value=fake_response,
    ):
        result = fetch_qualified_hitters(season=2024, min_pa=150)
    ids = {r["player_id"] for r in result}
    assert ids == {1, 2}  # 3 is below cutoff


def test_fetch_qualified_hitters_passes_correct_params():
    with patch(
        "fantasy_baseball.streaks.data.qualified_hitters.statsapi.get",
        return_value={"leagueLeaders": [{"leaders": []}]},
    ) as mock:
        fetch_qualified_hitters(season=2024)
    args, kwargs = mock.call_args
    assert args[0] == "stats_leaders"
    assert kwargs["params"]["season"] == 2024
    assert kwargs["params"]["leaderCategories"] == "plateAppearances"
    assert kwargs["params"]["statGroup"] == "hitting"
