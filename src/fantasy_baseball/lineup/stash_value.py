"""Stash board -- rank injured players (owned IL + injured FAs) by their
leverage-aware marginal active value, and allocate the scarce IL slots.

Sibling of ``il_return_planner``: reuses the optimizer and the
double-count-safe deltaRoto band. A candidate's Gain is the band mean of
activating him into the optimized lineup (floored at ~0 when he can't crack
it -- "no harm, no foul"). Cost is the IL-slot allocation cost: 0 when a slot
is open, else the Gain of the weakest owned IL stash he displaces.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from fantasy_baseball.lineup.delta_roto import compute_delta_roto_band
from fantasy_baseball.lineup.optimizer import (
    optimize_hitter_lineup,
    optimize_pitcher_lineup,
)
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.utils.constants import Category

__all__ = ["StashScore", "StashResult", "score_stash_candidates"]


@dataclass
class StashScore:
    """One injured player's stash evaluation."""

    name: str
    player_type: str
    status: str  # IL10 / IL15 / IL60 / DTD / ...
    owned: bool  # already on the user's roster
    gain: float  # marginal active value (deltaRoto band mean), floored at ~0
    cost: float  # deltaRoto sacrificed to roster him (0 if open IL slot)
    stash_value: float  # gain - cost
    band: dict[str, Any]  # {mean, sd, p_positive, verdict}
    recommended_drop: str | None  # who to drop to make room (None if free slot)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StashResult:
    """Ranked stash board."""

    open_il_slots: int
    cutline_rank: int  # = IL capacity; top-N are "hold/grab"
    candidates: list[StashScore] = field(default_factory=list)
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_il_slots": self.open_il_slots,
            "cutline_rank": self.cutline_rank,
            "candidates": [c.to_dict() for c in self.candidates],
            "warning": self.warning,
        }


def _activate(p: Player) -> Player:
    """Copy with IL signals cleared so the optimizer treats the player as
    active-eligible. Identical to il_return_planner._activate."""
    return dataclasses.replace(p, status="", selected_position=None)


def _solve_active(
    pool: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
) -> list[Player]:
    """Optimized active lineup (hitters + pitcher starters) over ``pool``."""
    hitters = [p for p in pool if p.player_type != PlayerType.PITCHER]
    pitchers = [p for p in pool if p.player_type == PlayerType.PITCHER]
    h_assign = optimize_hitter_lineup(
        hitters=hitters,
        full_roster=pool,
        projected_standings=projected_standings,
        team_name=team_name,
        roster_slots=roster_slots,
        team_sds=team_sds,
        fraction_remaining=None,
    )
    p_starters, _bench = optimize_pitcher_lineup(
        pitchers=pitchers,
        full_roster=pool,
        projected_standings=projected_standings,
        team_name=team_name,
        slots=roster_slots.get("P", 9),
        team_sds=team_sds,
        fraction_remaining=None,
    )
    return [a.player for a in h_assign] + [s.player for s in p_starters]


def _counted_pool(roster: list[Player], exclude_name: str | None = None) -> list[Player]:
    """Active + bench bodies (everything not in a true IL slot), optionally
    excluding one player by name."""
    out: list[Player] = []
    for p in roster:
        if p.selected_position in IL_SLOTS:
            continue
        if exclude_name is not None and p.name == exclude_name:
            continue
        out.append(p)
    return out


def _marginal_band(
    candidate: Player,
    *,
    before_active: list[Player],
    roster: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float,
) -> dict[str, Any]:
    """Return the deltaRoto band dict for activating ``candidate``."""
    pool_with = [*_counted_pool(roster, exclude_name=candidate.name), _activate(candidate)]
    after_active = _solve_active(pool_with, roster_slots, projected_standings, team_name, team_sds)
    band = compute_delta_roto_band(
        before_active,
        after_active,
        projected_standings.field_stats(team_name),
        team_name,
        fraction_remaining,
        projected_standings=projected_standings,
        team_sds=team_sds,
    )
    return band.to_dict()


def _marginal_value(
    candidate: Player,
    *,
    before_active: list[Player],
    roster: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float,
) -> float:
    """Gain = band mean of activating ``candidate``. Floored at ~0."""
    band = _marginal_band(
        candidate,
        before_active=before_active,
        roster=roster,
        roster_slots=roster_slots,
        projected_standings=projected_standings,
        team_name=team_name,
        team_sds=team_sds,
        fraction_remaining=fraction_remaining,
    )
    return band["mean"]
