"""deltaRoto — roto-point impact metric for player swaps.

Uses EV-based score_roto, so deltaRoto.total is simply the change in
total expected roto points across all categories. No tuning knobs,
no tie bands, no defensive-comfort heuristic — the Gaussian pairwise
win-probabilities price projection uncertainty and vulnerability
directly into the score.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category

if TYPE_CHECKING:
    from fantasy_baseball.models.player import Player
    from fantasy_baseball.models.standings import ProjectedStandings


@dataclass
class CategoryDelta:
    roto_delta: float


@dataclass
class DeltaRotoResult:
    total: float
    categories: dict[str, CategoryDelta]
    before_total: float
    after_total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 2),
            "before_total": round(self.before_total, 2),
            "after_total": round(self.after_total, 2),
            "categories": {
                cat: {"roto_delta": round(cd.roto_delta, 2)} for cat, cd in self.categories.items()
            },
        }


def score_swap(
    roto_before: dict[str, dict[str, float]],
    roto_after: dict[str, dict[str, float]],
    team_name: str,
) -> DeltaRotoResult:
    """Per-category deltaRoto from before/after ``score_roto`` outputs.

    Total is the change in team total EV roto points. Each category's
    ``roto_delta`` is the change in that category's EV points for the
    user's team. No discounts, no penalties — the EV already reflects
    projection uncertainty, defensive vulnerability, and boundary
    proximity via the sigmoid on pairwise win probabilities.
    """
    categories: dict[str, CategoryDelta] = {}
    for cat in ALL_CATEGORIES:
        rd = roto_after[team_name][f"{cat.value}_pts"] - roto_before[team_name][f"{cat.value}_pts"]
        categories[cat.value] = CategoryDelta(roto_delta=rd)

    return DeltaRotoResult(
        total=roto_after[team_name]["total"] - roto_before[team_name]["total"],
        categories=categories,
        before_total=roto_before[team_name]["total"],
        after_total=roto_after[team_name]["total"],
    )


def compute_delta_roto(
    drop_name: str,
    add_player: Player,
    user_roster: list[Player],
    projected_standings: ProjectedStandings,
    team_name: str,
    *,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
) -> DeltaRotoResult:
    """Compute deltaRoto for dropping one player and adding another.

    When ``team_sds`` is provided, ``score_roto`` uses pairwise Gaussian
    win-probabilities so a swap's impact reflects projection
    uncertainty (ERoto). Pass ``None`` explicitly for exact-rank
    semantics — no default: callers must make the choice so we can't
    silently fall back to integer roto by forgetting the argument.

    Args:
        drop_name: roster player to drop.
        add_player: Player to add.
        user_roster: current roster (used to resolve the dropped player's ROS).
        projected_standings: end-of-season stats for all teams.
        team_name: user's team name.
        team_sds: ``{team: {Category: sd}}`` for EV scoring, or ``None``
            for rank-based. Required keyword — no default.

    Raises:
        ValueError: if drop_name is not found on the roster.
    """
    from fantasy_baseball.scoring import score_roto_dict
    from fantasy_baseball.trades.evaluate import build_swap_standings, find_player_by_name

    dropped = find_player_by_name(drop_name, user_roster)
    if dropped is None:
        raise ValueError(f"Player '{drop_name}' not found on roster")

    all_before, all_after = build_swap_standings(
        dropped, add_player, projected_standings, team_name
    )
    roto_before = score_roto_dict(all_before, team_sds=team_sds)
    roto_after = score_roto_dict(all_after, team_sds=team_sds)

    return score_swap(roto_before, roto_after, team_name)
