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
    teams: dict[int, str] = field(default_factory=dict)
    strategy: str = "no_punt_opp"
    scoring_mode: str = "var"
    season_year: int = 2026
    season_start: str = "2026-03-27"
    season_end: str = "2026-09-28"


def load_config(config_path: Path) -> LeagueConfig:
    """Load league configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    league = raw.get("league", {})
    draft = raw.get("draft", {})
    projections = raw.get("projections", {})

    VALID_SCORING_MODES = {"var", "vona"}

    strategy = draft.get("strategy", "no_punt_opp")
    scoring_mode = draft.get("scoring_mode", "var")

    # Import here to avoid circular dependency
    from fantasy_baseball.draft.strategy import STRATEGIES
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            f"Valid strategies: {', '.join(sorted(STRATEGIES))}"
        )
    if scoring_mode not in VALID_SCORING_MODES:
        raise ValueError(
            f"Unknown scoring_mode {scoring_mode!r}. "
            f"Valid modes: {', '.join(sorted(VALID_SCORING_MODES))}"
        )

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
        teams={int(k): v for k, v in draft.get("teams", {}).items()},
        strategy=strategy,
        scoring_mode=scoring_mode,
        season_year=league.get("season_year", 2026),
        season_start=league.get("season_start", "2026-03-27"),
        season_end=league.get("season_end", "2026-09-28"),
    )
