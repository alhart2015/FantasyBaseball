from __future__ import annotations

from dataclasses import dataclass, field

from fantasy_baseball.draft.eroto_recs import RecRow
from fantasy_baseball.draft.recommender import Recommendation
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


def from_recommendation(rec: Recommendation, *, player_id: str) -> RankedPick:
    """Adapt a VAR/VONA ``Recommendation`` into a ``RankedPick``.

    ``Recommendation`` carries no player_id, so callers pass it in (the
    board lookup already has it).
    """
    return RankedPick(
        player_id=player_id,
        name=rec.name,
        positions=list(rec.positions),
        player_type=rec.player_type,
        score=rec.var,
        metrics={"var": rec.var},
        note=rec.note,
        need_flag=rec.need_flag,
    )


_DELTAROTO_METRICS = ("immediate_delta", "value_of_picking_now")


def from_recrow(row: RecRow, *, metric: str, player_type: PlayerType) -> RankedPick:
    """Adapt a deltaRoto ``RecRow`` into a ``RankedPick``.

    ``metric`` selects which native metric becomes ``score``; both are kept
    in ``metrics`` so the dashboard can display/toggle both.
    """
    if metric not in _DELTAROTO_METRICS:
        raise ValueError(f"metric must be one of {_DELTAROTO_METRICS}, got {metric!r}")
    metrics = {
        "immediate_delta": row.immediate_delta,
        "value_of_picking_now": row.value_of_picking_now,
    }
    return RankedPick(
        player_id=row.player_id,
        name=row.name,
        positions=[Position.parse(p) for p in row.positions],
        player_type=player_type,
        score=metrics[metric],
        metrics=metrics,
        per_category=dict(row.per_category),
    )
