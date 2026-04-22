"""Roster capacity + fill state for the draft recommender.

Bundles the two dicts that were previously passed around as a pair
(``filled_positions`` + ``roster_slots``) into a single dataclass with
Position-keyed maps and the derived queries the recommender needs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from fantasy_baseball.models.positions import (
    BENCH_SLOTS,
    IL_SLOTS,
    Position,
)
from fantasy_baseball.utils.positions import can_fill_slot


@dataclass(frozen=True)
class RosterState:
    """Immutable snapshot of filled slots and league capacity.

    Both maps are Position-keyed. Use :meth:`from_dicts` to construct
    from the string-keyed dicts returned by ``get_filled_positions`` or
    read from ``config/league.yaml``.
    """

    filled: Mapping[Position, int]
    capacity: Mapping[Position, int]

    @classmethod
    def from_dicts(
        cls,
        filled: Mapping[str | Position, int],
        capacity: Mapping[str | Position, int],
    ) -> RosterState:
        return cls(
            filled=_coerce(filled),
            capacity=_coerce(capacity),
        )

    def open_slots(
        self,
        *,
        exclude: frozenset[Position] = IL_SLOTS,
    ) -> dict[Position, int]:
        """Slots with remaining capacity, mapped to the remaining count.

        ``exclude`` defaults to IL variants — you don't draft to IL, but
        BN is a valid destination so it stays in by default. Pass
        ``BENCH_SLOTS`` to get only starter slots.
        """
        out: dict[Position, int] = {}
        for pos, total in self.capacity.items():
            if pos in exclude:
                continue
            current = self.filled.get(pos, 0)
            if current < total:
                out[pos] = total - current
        return out

    def unfilled_starter_slots(self) -> set[Position]:
        """Starter slots (not BN/IL) that still have open capacity."""
        return set(self.open_slots(exclude=BENCH_SLOTS).keys())

    def any_slot_open_for(self, positions: Iterable[Position | str]) -> bool:
        """True if the player fits in any open (non-IL) slot."""
        open_ = self.open_slots()
        if not open_:
            return False
        pos_list = list(positions)
        return any(can_fill_slot(pos_list, slot) for slot in open_)


def _coerce(d: Mapping[str | Position, int]) -> dict[Position, int]:
    out: dict[Position, int] = {}
    for k, v in d.items():
        pos = k if isinstance(k, Position) else Position.parse(k)
        out[pos] = int(v)
    return out
