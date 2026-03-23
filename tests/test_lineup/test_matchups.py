import pytest
from fantasy_baseball.lineup.matchups import normalize_team_batting_stats


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
