HITTING_CATEGORIES: list[str] = ["R", "HR", "RBI", "SB", "AVG"]
PITCHING_CATEGORIES: list[str] = ["W", "K", "ERA", "WHIP", "SV"]
ALL_CATEGORIES: list[str] = HITTING_CATEGORIES + PITCHING_CATEGORIES

RATE_STATS: set[str] = {"AVG", "ERA", "WHIP"}
INVERSE_STATS: set[str] = {"ERA", "WHIP"}  # Lower is better

DEFAULT_ROSTER_SLOTS: dict[str, int] = {
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "IF": 1,
    "OF": 4,
    "UTIL": 2,
    "P": 9,
    "BN": 2,
    "IL": 2,
}

# Backward-compatible alias so existing imports keep working.
ROSTER_SLOTS: dict[str, int] = DEFAULT_ROSTER_SLOTS

DEFAULT_NUM_TEAMS: int = 10

# Backward-compatible alias.
NUM_TEAMS: int = DEFAULT_NUM_TEAMS


def compute_starters_per_position(
    roster_slots: dict[str, int] | None = None,
    num_teams: int | None = None,
) -> dict[str, int]:
    """Derive starters-per-position from roster slots and league size.

    ``starters = slots * num_teams`` for every non-bench/IL position.
    """
    if roster_slots is None:
        roster_slots = DEFAULT_ROSTER_SLOTS
    if num_teams is None:
        num_teams = DEFAULT_NUM_TEAMS
    return {
        pos: count * num_teams
        for pos, count in roster_slots.items()
        if pos not in ("BN", "IL")
    }


# Default value kept for backward compatibility.
STARTERS_PER_POSITION: dict[str, int] = compute_starters_per_position()

IF_ELIGIBLE: set[str] = {"1B", "2B", "3B", "SS"}

DEFAULT_SGP_DENOMINATORS: dict[str, float] = {
    "R": 20.0,
    "HR": 9.0,
    "RBI": 20.0,
    "SB": 8.0,
    "AVG": 0.005,
    "W": 3.0,
    "K": 30.0,
    "ERA": 0.15,
    "WHIP": 0.015,
    "SV": 7.0,
}
