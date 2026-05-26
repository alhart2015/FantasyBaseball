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
