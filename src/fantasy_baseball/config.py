from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class LeagueConfig:
    league_id: int
    num_teams: int
    game_code: str
    team_name: str
    draft_position: int
    keepers: list[dict]
    roster_slots: dict[str, int]
    projection_systems: list[str]
    projection_weights: dict[str, float]
    sgp_overrides: dict[str, float] = field(default_factory=dict)


def load_config(config_path: Path) -> LeagueConfig:
    """Load league configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    league = raw.get("league", {})
    draft = raw.get("draft", {})
    projections = raw.get("projections", {})

    return LeagueConfig(
        league_id=league.get("id", 0),
        num_teams=league.get("num_teams", 10),
        game_code=league.get("game_code", "mlb"),
        team_name=league.get("team_name", ""),
        draft_position=draft.get("position", 1),
        keepers=raw.get("keepers", []),
        roster_slots=raw.get("roster_slots", {}),
        projection_systems=projections.get("systems", []),
        projection_weights=projections.get("weights", {}),
        sgp_overrides=raw.get("sgp_denominators", {}),
    )
