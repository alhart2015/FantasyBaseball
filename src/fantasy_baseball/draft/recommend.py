from __future__ import annotations

from dataclasses import dataclass, field

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.positions import Position


@dataclass
class RankedPick:
    """One ranked draft candidate, uniform across every scoring mode.

    ``score`` is the active mode's primary metric. ``metrics`` carries every
    mode-native metric (deltaRoto modes populate both ``immediate_delta`` and
    ``value_of_picking_now`` so the dashboard can toggle between them).
    """

    player_id: str
    name: str
    positions: list[Position]
    player_type: PlayerType
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    per_category: dict[str, float] = field(default_factory=dict)
    note: str = ""
    need_flag: bool = False

    def position_strings(self) -> list[str]:
        """Position codes as plain strings (for JSON / display)."""
        return [p.value if isinstance(p, Position) else str(p) for p in self.positions]
