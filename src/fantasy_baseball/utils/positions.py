from .constants import IF_ELIGIBLE

HITTER_POSITIONS: set[str] = {"C", "1B", "2B", "3B", "SS", "OF", "DH", "IF"}
PITCHER_POSITIONS: set[str] = {"P", "SP", "RP"}


def can_fill_slot(player_positions: list[str], slot: str) -> bool:
    """Check if a player with given eligible positions can fill a roster slot."""
    if slot in ("BN", "IL"):
        return True
    if slot == "UTIL":
        return any(pos in HITTER_POSITIONS for pos in player_positions)
    if slot == "IF":
        return any(pos in IF_ELIGIBLE for pos in player_positions)
    if slot == "OF":
        return "OF" in player_positions
    if slot == "P":
        return any(pos in PITCHER_POSITIONS for pos in player_positions)
    return slot in player_positions


def can_cover_slots(
    player_positions_list: list[list[str]],
    roster_slots: dict[str, int],
) -> bool:
    """Check if a group of players can fill all required hitter slots.

    Uses augmenting-path bipartite matching to verify feasibility.
    Only checks hitter slots (C, 1B, 2B, 3B, SS, IF, OF, UTIL) since
    all pitcher slots are interchangeable.

    Args:
        player_positions_list: List of eligible-position lists, one per player.
        roster_slots: Config roster slots dict (e.g. {"C": 1, "1B": 1, ...}).

    Returns:
        True if every hitter slot can be filled by some player.
    """
    # Build the list of hitter slots to fill
    skip = {"P", "BN", "IL", "IL+", "DL", "DL+"}
    slots: list[str] = []
    for pos, count in roster_slots.items():
        if pos in skip:
            continue
        for _ in range(count):
            slots.append(pos)

    if not slots:
        return True
    if len(player_positions_list) < len(slots):
        return False

    n_slots = len(slots)
    # match_slot[slot_idx] = player_idx assigned, or -1
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


def is_hitter(positions: list[str]) -> bool:
    """Check if a player is a hitter based on their eligible positions."""
    return any(pos in HITTER_POSITIONS for pos in positions)


def is_pitcher(positions: list[str]) -> bool:
    """Check if a player is a pitcher based on their eligible positions."""
    return any(pos in PITCHER_POSITIONS for pos in positions)
