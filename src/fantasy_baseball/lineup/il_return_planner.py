"""IL return planner -- compute the optimal legal roster + transaction plan
when injured-list players are reactivated.

When IL players come off the IL they temporarily push the roster over the
active+bench body-count cap, forcing a drop plus an active/bench reshuffle.
Given the IL players a manager wants to activate, this module computes the
forced drops and returns the top transaction plans ranked by deltaRoto.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS, Position
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.utils.constants import Category

logger = logging.getLogger(__name__)


@dataclass
class Move:
    """A single roster transaction for one player."""

    name: str
    player_type: str
    from_slot: str
    to_slot: str  # active slot label, "BN", or "DROP"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MovePlan:
    """One complete plan: the forced drop(s) plus the resulting move list."""

    drops: list[str]
    moves: list[Move]
    delta_roto: float
    band: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "drops": list(self.drops),
            "moves": [m.to_dict() for m in self.moves],
            "delta_roto": round(self.delta_roto, 2),
            "band": self.band,
        }


@dataclass
class IlReturnPlanResult:
    """All plans for activating a chosen set of IL players."""

    activating: list[str]
    capacity: int
    overflow: int
    plans: list[MovePlan] = field(default_factory=list)
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "activating": list(self.activating),
            "capacity": self.capacity,
            "overflow": self.overflow,
            "plans": [p.to_dict() for p in self.plans],
            "warning": self.warning,
        }


def roster_capacity(roster_slots: dict[str, int]) -> int:
    """Active + bench slot count -- every slot except IL slots.

    IL slots are exempt from Yahoo's active-roster size limit, so they do
    not count toward the body-count cap that forces a drop.
    """
    total = 0
    for key, count in roster_slots.items():
        pos = key if isinstance(key, Position) else Position.parse(key)
        if pos in IL_SLOTS:
            continue
        total += count
    return total


def _counts_against_cap(p: Player) -> bool:
    """True if this body counts against the active+bench cap.

    Slot-based, NOT status-based: a BN+IL-status player (Yahoo lets you
    stash an IL guy on the bench) still counts; only a true IL-slot body
    is exempt. This is why activating an IL-slot player is what forces a
    drop.
    """
    return p.selected_position not in IL_SLOTS


def _activate(p: Player) -> Player:
    """Return a copy with IL signals cleared so the optimizer treats the
    player as active-eligible. ``is_on_il()`` and the optimizers would
    otherwise keep excluding a returning IL player."""
    return dataclasses.replace(p, status="", selected_position=None)


def _build_pool(roster: list[Player], activating_il: list[Player]) -> list[Player]:
    """The set of players competing for active/bench slots after activation.

    = current counted bodies (active + healthy bench + any BN+IL-status
    players) UNION the activating players that were in true IL slots.
    Activating players get their IL signals cleared. Unchecked IL players
    stay parked and are excluded.
    """
    activating_names = {p.name for p in activating_il}
    counted = [p for p in roster if _counts_against_cap(p)]
    counted_names = {p.name for p in counted}
    extra = [p for p in activating_il if p.name not in counted_names]
    pool: list[Player] = []
    for p in counted + extra:
        pool.append(_activate(p) if p.name in activating_names else p)
    return pool


def _solve_lineup(
    pool: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
):
    """Run both optimizers over ``pool``; return
    ``(hitter_assignments, pitcher_starters, pitcher_bench)``.

    Both optimizer calls use ``fraction_remaining=None`` so they skip
    per-starter band computation; the planner computes a plan-level band
    separately.
    """
    hitters = [p for p in pool if p.player_type != PlayerType.PITCHER]
    pitchers = [p for p in pool if p.player_type == PlayerType.PITCHER]
    hitter_assignments = optimize_hitter_lineup(
        hitters=hitters,
        full_roster=pool,
        projected_standings=projected_standings,
        team_name=team_name,
        roster_slots=roster_slots,
        team_sds=team_sds,
        fraction_remaining=None,
    )
    pitcher_starters, pitcher_bench = optimize_pitcher_lineup(
        pitchers=pitchers,
        full_roster=pool,
        projected_standings=projected_standings,
        team_name=team_name,
        slots=roster_slots.get("P", 9),
        team_sds=team_sds,
        fraction_remaining=None,
    )
    return hitter_assignments, pitcher_starters, pitcher_bench
