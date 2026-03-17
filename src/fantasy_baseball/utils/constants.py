HITTING_CATEGORIES: list[str] = ["R", "HR", "RBI", "SB", "AVG"]
PITCHING_CATEGORIES: list[str] = ["W", "K", "ERA", "WHIP", "SV"]
ALL_CATEGORIES: list[str] = HITTING_CATEGORIES + PITCHING_CATEGORIES

RATE_STATS: set[str] = {"AVG", "ERA", "WHIP"}
INVERSE_STATS: set[str] = {"ERA", "WHIP"}  # Lower is better

ROSTER_SLOTS: dict[str, int] = {
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

STARTERS_PER_POSITION: dict[str, int] = {
    "C": 10,
    "1B": 10,
    "2B": 10,
    "3B": 10,
    "SS": 10,
    "IF": 10,
    "OF": 40,
    "UTIL": 20,
    "P": 90,
}

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

NUM_TEAMS: int = 10
