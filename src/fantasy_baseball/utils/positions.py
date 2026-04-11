"""Legacy position helpers — backed by the Position enum in models/.

This module predates the ``Position`` enum and exists so that call
sites using ``HITTER_POSITIONS``, ``PITCHER_POSITIONS``, and
``can_fill_slot`` / ``can_cover_slots`` keep working. New code should
import from ``fantasy_baseball.models.positions`` directly.

The frozensets below are defined in terms of the enum, so containment
checks work for both ``Position`` members and equivalent strings
(``StrEnum`` makes ``Position.OF == "OF"``).
"""

from fantasy_baseball.models.positions import (
    HITTER_ELIGIBLE,
    PITCHER_ELIGIBLE,
    Position,
)

# Exported for backward compatibility with existing callers
HITTER_POSITIONS: frozenset = HITTER_ELIGIBLE
PITCHER_POSITIONS: frozenset = PITCHER_ELIGIBLE

# IF-eligible positions — a subset of HITTER_ELIGIBLE used by can_fill_slot
_IF_ELIGIBLE: frozenset[Position] = frozenset({
    Position.FIRST_BASE, Position.SECOND_BASE,
    Position.THIRD_BASE, Position.SS,
})


def _coerce(p) -> Position | None:
    """Accept a Position, canonical position string, or Yahoo mixed-case
    string, and return the canonical Position. Empty string and ``None``
    return ``None`` so callers that receive a missing position
    (e.g. ``selected_position == ""`` from Yahoo) degrade gracefully
    instead of raising.
    """
    if isinstance(p, Position):
        return p
    if not p:
        return None
    return Position.parse(p)


def can_fill_slot(player_positions, slot) -> bool:
    """Check if a player's eligible positions can fill a roster slot.

    Accepts either ``Position`` enum values or string positions in
    either argument for backward compatibility. An empty/None slot
    returns ``False`` (nothing fills a non-existent slot); empty
    entries in ``player_positions`` are ignored.
    """
    slot_p = _coerce(slot)
    if slot_p is None:
        return False
    eligible = [c for c in (_coerce(p) for p in player_positions)
                if c is not None]

    if slot_p in (Position.BN, Position.IL, Position.IL_PLUS,
                  Position.DL, Position.DL_PLUS):
        return True
    if slot_p is Position.UTIL:
        return any(p in HITTER_ELIGIBLE for p in eligible)
    if slot_p is Position.IF:
        return any(p in _IF_ELIGIBLE for p in eligible)
    if slot_p is Position.OF:
        return Position.OF in eligible
    if slot_p is Position.P:
        return any(p in PITCHER_ELIGIBLE for p in eligible)
    return slot_p in eligible


def can_cover_slots(player_positions_list, roster_slots) -> bool:
    """Check if a group of players can fill all required hitter slots.

    Uses augmenting-path bipartite matching to verify feasibility.
    Only checks hitter slots (C, 1B, 2B, 3B, SS, IF, OF, UTIL) since
    all pitcher slots are interchangeable.

    Accepts ``player_positions_list`` as a list of lists where each
    inner list contains either ``Position`` enum values or strings.
    ``roster_slots`` is the config dict mapping slot names to counts
    (string keys are fine — matches config format).
    """
    skip = {"P", "BN", "IL", "IL+", "DL", "DL+"}
    slots: list[Position] = []
    for pos_key, count in roster_slots.items():
        if pos_key in skip:
            continue
        for _ in range(count):
            slots.append(_coerce(pos_key))

    if not slots:
        return True
    if len(player_positions_list) < len(slots):
        return False

    n_slots = len(slots)
    match_slot = [-1] * n_slots

    def _try_assign(player_idx: int, visited: set[int]) -> bool:
        for slot_idx in range(n_slots):
            if slot_idx in visited:
                continue
            if can_fill_slot(player_positions_list[player_idx], slots[slot_idx]):
                visited.add(slot_idx)
                if match_slot[slot_idx] == -1 or _try_assign(match_slot[slot_idx], visited):
                    match_slot[slot_idx] = player_idx
                    return True
        return False

    matched = 0
    for p_idx in range(len(player_positions_list)):
        if _try_assign(p_idx, set()):
            matched += 1
        if matched >= n_slots:
            return True

    return matched >= n_slots


def is_hitter(positions) -> bool:
    """Check if a player is a hitter based on their eligible positions.

    Empty/None entries in ``positions`` are ignored.
    """
    eligible = [c for c in (_coerce(p) for p in positions) if c is not None]
    return any(p in HITTER_ELIGIBLE for p in eligible)


def is_pitcher(positions) -> bool:
    """Check if a player is a pitcher based on their eligible positions.

    Empty/None entries in ``positions`` are ignored.
    """
    eligible = [c for c in (_coerce(p) for p in positions) if c is not None]
    return any(p in PITCHER_ELIGIBLE for p in eligible)
