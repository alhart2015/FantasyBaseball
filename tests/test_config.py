import pytest
from pathlib import Path
from fantasy_baseball.config import load_config, LeagueConfig


@pytest.fixture
def sample_config(tmp_path):
    config_file = tmp_path / "league.yaml"
    config_file.write_text("""
league:
  id: 5652
  num_teams: 10
  game_code: mlb

draft:
  position: 8

keepers:
  - name: "Player A"
    team: "Team 1"
  - name: "Player B"
    team: "Team 2"

roster_slots:
  C: 1
  1B: 1
  2B: 1
  3B: 1
  SS: 1
  IF: 1
  OF: 4
  UTIL: 2
  P: 9
  BN: 2
  IL: 2

projections:
  systems:
    - steamer
    - zips
  weights:
    steamer: 0.6
    zips: 0.4

sgp_denominators:
  HR: 10
""")
    return config_file


def test_load_config_basic(sample_config):
    config = load_config(sample_config)
    assert config.league_id == 5652
    assert config.num_teams == 10
    assert config.draft_position == 8


def test_load_config_keepers(sample_config):
    config = load_config(sample_config)
    assert len(config.keepers) == 2
    assert config.keepers[0]["name"] == "Player A"


def test_load_config_projection_weights(sample_config):
    config = load_config(sample_config)
    assert config.projection_systems == ["steamer", "zips"]
    assert config.projection_weights == {"steamer": 0.6, "zips": 0.4}


def test_load_config_sgp_overrides(sample_config):
    config = load_config(sample_config)
    assert config.sgp_overrides == {"HR": 10}


def test_load_config_roster_slots(sample_config):
    config = load_config(sample_config)
    assert config.roster_slots["C"] == 1
    assert config.roster_slots["OF"] == 4
    assert sum(config.roster_slots.values()) == 25


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/league.yaml"))
