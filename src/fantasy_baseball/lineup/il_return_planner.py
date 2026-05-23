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
from dataclasses import asdict, dataclass, field
from typing import Any

from fantasy_baseball.models.player import Player
from fantasy_baseball.models.positions import IL_SLOTS, Position

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
