from pathlib import Path

import pytest
import yaml

from fantasy_baseball.config import load_config


@pytest.fixture
def minimal_league_yaml(tmp_path):
    """Factory that writes a minimal valid league.yaml and returns the Path.

    Usage: path = minimal_league_yaml(scoring_mode="var", strategy="default")
    """

    def _make(*, scoring_mode: str, strategy: str, extra: dict | None = None) -> Path:
        cfg = {
            "league": {
                "id": 1,
                "num_teams": 10,
                "game_code": "mlb",
                "team_name": "Test Team",
            },
            "draft": {
                "strategy": strategy,
                "scoring_mode": scoring_mode,
                "position": 1,
            },
            "keepers": [],
            "roster_slots": {
                "C": 1,
                "1B": 1,
                "2B": 1,
                "3B": 1,
                "SS": 1,
                "IF": 1,
                "OF": 4,
                "UTIL": 2,
                "P": 9,
                "BN": 2,
                "IL": 2,
            },
            "projections": {
                "systems": ["steamer"],
                "weights": {"steamer": 1.0},
            },
        }
        if extra:
            cfg.update(extra)
        path = tmp_path / f"league_{scoring_mode}_{strategy}.yaml"
        path.write_text(yaml.dump(cfg))
        return path

    return _make


@pytest.mark.parametrize("mode", ["var", "vona", "deltaroto_immediate", "deltaroto_vopn"])
def test_valid_scoring_modes_accepted(mode, minimal_league_yaml):
    path = minimal_league_yaml(scoring_mode=mode, strategy="default")
    cfg = load_config(path)
    assert cfg.scoring_mode == mode


def test_invalid_scoring_mode_rejected(minimal_league_yaml):
    path = minimal_league_yaml(scoring_mode="bogus", strategy="default")
    with pytest.raises(ValueError, match="Unknown scoring_mode"):
        load_config(path)


@pytest.fixture
def sample_config(tmp_path):
    config_file = tmp_path / "league.yaml"
    config_file.write_text("""
league:
  id: 5652
  num_teams: 10
  game_code: mlb
  team_name: "My Team"

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
    assert config.team_name == "My Team"
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


def test_load_config_sgp_overrides_default_empty(minimal_league_yaml):
    path = minimal_league_yaml(scoring_mode="var", strategy="default")
    config = load_config(path)
    assert config.sgp_overrides == {}


def test_load_config_sgp_overrides_rejects_unknown_category(minimal_league_yaml):
    path = minimal_league_yaml(
        scoring_mode="var", strategy="default", extra={"sgp_denominators": {"HRR": 10}}
    )
    with pytest.raises(ValueError, match="'HRR'"):
        load_config(path)


def test_load_config_sgp_overrides_rejects_negative_value(minimal_league_yaml):
    path = minimal_league_yaml(
        scoring_mode="var", strategy="default", extra={"sgp_denominators": {"HR": -3}}
    )
    with pytest.raises(ValueError, match=r"'HR'.*positive"):
        load_config(path)


def test_load_config_sgp_overrides_rejects_non_numeric_value(minimal_league_yaml):
    path = minimal_league_yaml(
        scoring_mode="var", strategy="default", extra={"sgp_denominators": {"HR": "ten"}}
    )
    with pytest.raises(ValueError, match=r"'HR'.*positive"):
        load_config(path)


def test_load_config_sgp_overrides_rejects_zero_value(minimal_league_yaml):
    path = minimal_league_yaml(
        scoring_mode="var", strategy="default", extra={"sgp_denominators": {"AVG": 0}}
    )
    with pytest.raises(ValueError, match=r"'AVG'.*positive"):
        load_config(path)


def test_load_config_sgp_overrides_rejects_nan_value(minimal_league_yaml):
    # YAML `.nan` parses to float("nan"); NaN slips past `value <= 0`
    # (all NaN comparisons are False) without an explicit isfinite gate.
    path = minimal_league_yaml(
        scoring_mode="var", strategy="default", extra={"sgp_denominators": {"HR": float("nan")}}
    )
    with pytest.raises(ValueError, match=r"'HR'.*positive"):
        load_config(path)


def test_load_config_sgp_overrides_rejects_inf_value(minimal_league_yaml):
    # YAML `.inf` parses to float("inf"), which is numerically positive.
    path = minimal_league_yaml(
        scoring_mode="var", strategy="default", extra={"sgp_denominators": {"HR": float("inf")}}
    )
    with pytest.raises(ValueError, match=r"'HR'.*positive"):
        load_config(path)


def test_load_config_sgp_overrides_rejects_list_form(minimal_league_yaml):
    # A YAML list under sgp_denominators must raise the actionable
    # ValueError, not an AttributeError from .items() on a list.
    path = minimal_league_yaml(
        scoring_mode="var", strategy="default", extra={"sgp_denominators": ["HR", 10]}
    )
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(path)


def test_load_config_sgp_overrides_rejects_scalar_form(minimal_league_yaml):
    path = minimal_league_yaml(
        scoring_mode="var", strategy="default", extra={"sgp_denominators": 10}
    )
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(path)


def test_load_config_roster_slots(sample_config):
    config = load_config(sample_config)
    assert config.roster_slots["C"] == 1
    assert config.roster_slots["OF"] == 4
    assert sum(config.roster_slots.values()) == 25


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/league.yaml"))


def test_load_config_reads_summary_block(tmp_path):
    from fantasy_baseball.config import load_config

    cfg_text = """
league:
  id: 12345
  num_teams: 10
  team_name: "My Team"
summary:
  recipients:
    - "me@example.com"
  from_address: "digest@example.com"
"""
    p = tmp_path / "league.yaml"
    p.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.summary["recipients"] == ["me@example.com"]
    assert cfg.summary["from_address"] == "digest@example.com"


def test_load_config_summary_defaults_empty(tmp_path):
    from fantasy_baseball.config import load_config

    p = tmp_path / "league.yaml"
    p.write_text("league:\n  id: 1\n  num_teams: 10\n  team_name: T\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.summary == {}


class TestKeepersPerTeam:
    """The league's keeper allowance, derived from the configured keeper roster.

    Load-bearing for keeper valuation: ranking teams by their best FIVE when only three
    may be kept counts players nobody can retain, which on 2026-08-22 inverted the
    league ordering on the trajectory board's By-team view.
    """

    @staticmethod
    def _cfg(keepers):
        from fantasy_baseball.config import LeagueConfig

        return LeagueConfig(
            league_id=1,
            num_teams=10,
            game_code="mlb",
            team_name="Mine",
            draft_position=1,
            keepers=keepers,
            roster_slots={},
            projection_systems=[],
            projection_weights={},
        )

    def test_derives_the_count_from_the_configured_keepers(self):
        keepers = [{"name": f"P{i}", "team": t} for t in ("A", "B") for i in range(3)]
        assert self._cfg(keepers).keepers_per_team == 3

    def test_takes_the_max_so_one_team_keeping_fewer_does_not_shrink_the_rule(self):
        keepers = [{"name": "a1", "team": "A"}] + [{"name": f"b{i}", "team": "B"} for i in range(3)]
        assert self._cfg(keepers).keepers_per_team == 3

    def test_falls_back_when_no_keepers_are_configured(self):
        """Pre-season the list is empty and there is nothing to derive from."""
        from fantasy_baseball.config import DEFAULT_KEEPERS_PER_TEAM

        assert self._cfg([]).keepers_per_team == DEFAULT_KEEPERS_PER_TEAM

    def test_ignores_malformed_entries_rather_than_counting_them(self):
        keepers = [{"name": "a", "team": "A"}, {"name": "no team"}, "junk", {"team": ""}]
        assert self._cfg(keepers).keepers_per_team == 1

    def test_the_real_league_config_keeps_three(self):
        """The rule this project actually plays under, read from the shipped config."""
        from pathlib import Path

        from fantasy_baseball.config import load_config

        root = Path(__file__).resolve().parents[1]
        assert load_config(root / "config" / "league.yaml").keepers_per_team == 3
