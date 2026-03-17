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


def is_hitter(positions: list[str]) -> bool:
    """Check if a player is a hitter based on their eligible positions."""
    return any(pos in HITTER_POSITIONS for pos in positions)


def is_pitcher(positions: list[str]) -> bool:
    """Check if a player is a pitcher based on their eligible positions."""
    return any(pos in PITCHER_POSITIONS for pos in positions)
