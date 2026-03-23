import pytest
from fantasy_baseball.lineup.matchups import normalize_team_batting_stats, calculate_matchup_factors


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
