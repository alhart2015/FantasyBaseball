import pytest
import pandas as pd
from unittest.mock import patch
from fantasy_baseball.lineup.matchups import (
    normalize_team_batting_stats,
    calculate_matchup_factors,
    adjust_pitcher_projection,
    fetch_team_batting_stats,
)


def test_normalize_team_batting_stats():
    """Given raw MLB API team stat dicts, produce {abbrev: {ops, k_pct}}."""
    raw = [
        {"team_id": 147, "team_name": "New York Yankees", "abbreviation": "NYY",
         "ops": ".787", "strikeouts": 1463, "plate_appearances": 6235},
        {"team_id": 119, "team_name": "Los Angeles Dodgers", "abbreviation": "LAD",
         "ops": ".820", "strikeouts": 1300, "plate_appearances": 6100},
    ]
    result = normalize_team_batting_stats(raw)
    assert "NYY" in result
    assert "LAD" in result
    assert abs(result["NYY"]["ops"] - 0.787) < 0.001
    assert abs(result["NYY"]["k_pct"] - 1463 / 6235) < 0.001
    assert abs(result["LAD"]["ops"] - 0.820) < 0.001


SAMPLE_STATS = {
    "NYY": {"ops": 0.750, "k_pct": 0.230},
    "COL": {"ops": 0.650, "k_pct": 0.260},
    "LAD": {"ops": 0.800, "k_pct": 0.200},
}


def test_league_average_team_gets_factor_of_one():
    same = {"A": {"ops": 0.750, "k_pct": 0.230}, "B": {"ops": 0.750, "k_pct": 0.230}}
    factors = calculate_matchup_factors(same)
    assert abs(factors["A"]["era_whip_factor"] - 1.0) < 0.001
    assert abs(factors["A"]["k_factor"] - 1.0) < 0.001


def test_weak_offense_lowers_era_factor():
    factors = calculate_matchup_factors(SAMPLE_STATS)
    assert factors["COL"]["era_whip_factor"] < 1.0
    assert factors["LAD"]["era_whip_factor"] > 1.0


def test_high_k_team_raises_k_factor():
    factors = calculate_matchup_factors(SAMPLE_STATS)
    assert factors["COL"]["k_factor"] > 1.0
    assert factors["LAD"]["k_factor"] < 1.0


def test_dampening_limits_adjustment():
    extreme = {
        "GOOD": {"ops": 0.900, "k_pct": 0.230},
        "BAD": {"ops": 0.600, "k_pct": 0.230},
    }
    factors = calculate_matchup_factors(extreme, dampening=0.5)
    assert factors["GOOD"]["era_whip_factor"] < 1.15
    assert factors["GOOD"]["era_whip_factor"] > 1.05


def _make_pitcher(name, team, era, whip, k, w, sv, ip):
    return pd.Series({
        "name": name, "team": team, "player_type": "pitcher",
        "era": era, "whip": whip, "k": k, "w": w, "sv": sv,
        "ip": ip, "er": era * ip / 9, "bb": 40, "h_allowed": 140,
    })


def test_easy_matchup_lowers_era():
    pitcher = _make_pitcher("Ace", "NYY", 3.50, 1.15, 200, 12, 0, 180)
    factors = {"era_whip_factor": 0.90, "k_factor": 1.10}
    adjusted = adjust_pitcher_projection(pitcher, factors)
    assert adjusted["era"] < 3.50
    assert adjusted["whip"] < 1.15
    assert adjusted["k"] > 200


def test_hard_matchup_raises_era():
    pitcher = _make_pitcher("Ace", "NYY", 3.50, 1.15, 200, 12, 0, 180)
    factors = {"era_whip_factor": 1.10, "k_factor": 0.90}
    adjusted = adjust_pitcher_projection(pitcher, factors)
    assert adjusted["era"] > 3.50
    assert adjusted["whip"] > 1.15
    assert adjusted["k"] < 200


def test_neutral_matchup_unchanged():
    pitcher = _make_pitcher("Ace", "NYY", 3.50, 1.15, 200, 12, 0, 180)
    factors = {"era_whip_factor": 1.0, "k_factor": 1.0}
    adjusted = adjust_pitcher_projection(pitcher, factors)
    assert abs(adjusted["era"] - 3.50) < 0.001
    assert abs(adjusted["k"] - 200) < 0.1


def test_wins_and_saves_unchanged():
    pitcher = _make_pitcher("Closer", "NYY", 2.50, 1.00, 60, 3, 35, 65)
    factors = {"era_whip_factor": 1.15, "k_factor": 0.85}
    adjusted = adjust_pitcher_projection(pitcher, factors)
    assert adjusted["w"] == 3
    assert adjusted["sv"] == 35


def test_two_start_blended_factors():
    pitcher = _make_pitcher("Horse", "NYY", 3.50, 1.15, 200, 12, 0, 180)
    matchup_list = [
        {"era_whip_factor": 0.90, "k_factor": 1.10},
        {"era_whip_factor": 1.10, "k_factor": 0.90},
    ]
    adjusted = adjust_pitcher_projection(pitcher, matchup_list)
    assert abs(adjusted["era"] - 3.50) < 0.05
    assert abs(adjusted["k"] - 200) < 1.0


@patch("fantasy_baseball.lineup.matchups.statsapi")
def test_fetch_team_batting_stats(mock_api):
    """fetch_team_batting_stats calls MLB API per team and returns normalized data."""
    mock_api.get.side_effect = [
        # First call: get teams
        {"teams": [
            {"id": 147, "name": "New York Yankees", "abbreviation": "NYY"},
            {"id": 119, "name": "Los Angeles Dodgers", "abbreviation": "LAD"},
        ]},
        # Per-team stat calls
        {"stats": [{"splits": [{"stat": {
            "ops": ".787", "strikeOuts": 1463, "plateAppearances": 6235,
        }}]}]},
        {"stats": [{"splits": [{"stat": {
            "ops": ".820", "strikeOuts": 1300, "plateAppearances": 6100,
        }}]}]},
    ]

    result = fetch_team_batting_stats(season=2025)
    assert "NYY" in result
    assert "LAD" in result
    assert abs(result["NYY"]["ops"] - 0.787) < 0.001
