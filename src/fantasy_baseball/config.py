import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fantasy_baseball.utils.constants import Category

log = logging.getLogger(__name__)

#: Used when no keepers are configured yet (pre-season the list is empty).
DEFAULT_KEEPERS_PER_TEAM = 3


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
    summary: dict = field(default_factory=dict)
    teams: dict[int, str] = field(default_factory=dict)
    strategy: str = "no_punt_opp"
    scoring_mode: str = "var"
    season_year: int = 2026
    season_start: str = "2026-03-27"
    season_end: str = "2026-09-28"

    @property
    def keepers_per_team(self) -> int:
        """How many players each team may keep into next season.

        Derived from the ``keepers`` roster rather than stored, because that list is
        already the authority: ``config/league.yaml`` carries one entry per kept player
        per team, so the per-team count IS the rule. Returns the MAX across teams -- a
        team that kept fewer than its allowance would otherwise shrink the league rule
        to its own choice.

        Load-bearing for keeper valuation: ranking teams by their best FIVE when only
        three may be kept counts players nobody can retain, and on 2026-08-22 that
        inverted the league ordering (the depth leader was not the keeper leader).

        TWO WAYS TO GET A WRONG ANSWER, both of which now WARN rather than pass
        silently, because this number goes straight into a headline that asserts a
        league rule ("the best 3 they may keep"):

        1. No entry carries a ``team`` key. Legitimate pre-season -- the list is
           empty and there is nothing to derive from -- but it also happens when
           the schema drifts, and the fallback looks identical either way.
        2. Only ONE team's keepers are listed. ``max`` over a single team returns
           that team's own choice, which is a lower bound on the rule, not the rule.
           A team that kept fewer than its allowance shrinks the league rule to its
           own decision -- the exact failure ``max`` across ten teams prevents.
        """
        from collections import Counter

        counts = Counter(
            k.get("team") for k in self.keepers if isinstance(k, dict) and k.get("team")
        )
        if not counts:
            log.warning(
                "keepers_per_team: no keeper entry in league.yaml carries a 'team' key; "
                "assuming %d per team. Pre-season this is expected; mid-season it means "
                "the keeper list is empty or its schema changed.",
                DEFAULT_KEEPERS_PER_TEAM,
            )
            return DEFAULT_KEEPERS_PER_TEAM
        if len(counts) == 1:
            only_team, only_count = next(iter(counts.items()))
            log.warning(
                "keepers_per_team: league.yaml lists keepers for exactly one team (%r, "
                "%d players), so the league rule is being read off one team's choice. "
                "A team that kept fewer than its allowance would understate it.",
                only_team,
                only_count,
            )
        return max(counts.values())


def _validate_sgp_overrides(raw_overrides: dict) -> dict[str, float]:
    """Validate the ``sgp_denominators`` block from league.yaml.

    Every key must be a valid Category value ("R", "HR", "AVG", ...) and
    every value a positive number. A typo'd category silently ignored is
    this repo's worst failure mode, so fail loudly naming the offender.
    """
    if not raw_overrides:
        return {}
    if not isinstance(raw_overrides, dict):
        raise ValueError(
            "sgp_denominators must be a mapping of category to positive number "
            f"(e.g. 'HR: 10'), got {type(raw_overrides).__name__}: {raw_overrides!r}"
        )
    valid_keys = {c.value for c in Category}
    validated: dict[str, float] = {}
    for key, value in raw_overrides.items():
        if key not in valid_keys:
            raise ValueError(
                f"Unknown sgp_denominators category {key!r}. "
                f"Valid categories: {', '.join(sorted(valid_keys))}"
            )
        # NaN passes `value <= 0` (all NaN comparisons are False) and inf is
        # numerically positive, so both need the explicit isfinite gate.
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"sgp_denominators value for {key!r} must be a positive number, got {value!r}"
            )
        validated[key] = float(value)
    return validated


def load_config(config_path: Path) -> LeagueConfig:
    """Load league configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    league = raw.get("league", {})
    draft = raw.get("draft", {})
    projections = raw.get("projections", {})

    VALID_SCORING_MODES = {"var", "vona", "deltaroto_immediate", "deltaroto_vopn"}

    strategy = draft.get("strategy", "no_punt_opp")
    scoring_mode = draft.get("scoring_mode", "var")

    # Import here to avoid circular dependency
    from fantasy_baseball.draft.strategy import STRATEGIES

    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Valid strategies: {', '.join(sorted(STRATEGIES))}"
        )
    if scoring_mode not in VALID_SCORING_MODES:
        raise ValueError(
            f"Unknown scoring_mode {scoring_mode!r}. "
            f"Valid modes: {', '.join(sorted(VALID_SCORING_MODES))}"
        )

    sgp_overrides = _validate_sgp_overrides(raw.get("sgp_denominators", {}))

    summary = raw.get("summary", {})

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
        sgp_overrides=sgp_overrides,
        summary=summary,
        teams={int(k): v for k, v in draft.get("teams", {}).items()},
        strategy=strategy,
        scoring_mode=scoring_mode,
        season_year=league.get("season_year", 2026),
        season_start=league.get("season_start", "2026-03-27"),
        season_end=league.get("season_end", "2026-09-28"),
    )
