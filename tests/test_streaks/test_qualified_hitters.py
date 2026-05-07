"""Tests for the ≥150 PA qualified hitters fetch."""

from unittest.mock import patch

from fantasy_baseball.streaks.data.qualified_hitters import (
    fetch_qualified_hitters,
    parse_season_split,
)
from fantasy_baseball.streaks.models import QualifiedHitter


def _split(player_id, name, team_abbrev, pa):
    return {
        "player": {"id": player_id, "fullName": name},
        "team": {"abbreviation": team_abbrev} if team_abbrev else {},
        "stat": {"plateAppearances": pa},
    }


def test_parse_season_split_extracts_id_name_team_pa():
    parsed = parse_season_split(_split(660271, "Mike Trout", "LAA", 162))
    assert parsed == QualifiedHitter(player_id=660271, name="Mike Trout", team="LAA", pa=162)


def test_parse_season_split_handles_missing_team():
    parsed = parse_season_split(_split(545361, "Free Agent", None, 150))
    assert parsed.team is None


def test_fetch_qualified_hitters_filters_below_min_pa():
    fake_response = {
        "stats": [
            {
                "splits": [
                    _split(1, "Above Cutoff", "NYY", 151),
                    _split(2, "Right At Cutoff", "BOS", 150),
                    _split(3, "Below Cutoff", "TBR", 149),
                ]
            }
        ]
    }
    with patch(
        "fantasy_baseball.streaks.data.qualified_hitters.statsapi.get",
        return_value=fake_response,
    ):
        result = fetch_qualified_hitters(season=2024, min_pa=150)
    ids = {r.player_id for r in result}
    assert ids == {1, 2}  # 3 is below cutoff


def test_fetch_qualified_hitters_passes_correct_params():
    with patch(
        "fantasy_baseball.streaks.data.qualified_hitters.statsapi.get",
        return_value={"stats": [{"splits": []}]},
    ) as mock:
        fetch_qualified_hitters(season=2024)
    args, kwargs = mock.call_args
    assert args[0] == "stats"
    assert kwargs["params"]["stats"] == "season"
    assert kwargs["params"]["group"] == "hitting"
    assert kwargs["params"]["season"] == 2024
    assert kwargs["params"]["sportId"] == 1
    assert kwargs["params"]["playerPool"] == "All"
