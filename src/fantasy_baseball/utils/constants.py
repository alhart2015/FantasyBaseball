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

# Minimum projected SV to classify a pitcher as a closer.
CLOSER_SV_THRESHOLD: int = 20

# Stat column lists for counting stats (used by Monte Carlo, stat aggregation)
HITTING_COUNTING: list[str] = ["r", "hr", "rbi", "sb", "h", "ab"]
PITCHING_COUNTING: list[str] = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]

# Monte Carlo simulation parameters (single source of truth)
INJURY_PROB: dict[str, float] = {"pitcher": 0.45, "hitter": 0.18}
INJURY_SEVERITY: dict[str, tuple[float, float]] = {
    "pitcher": (0.20, 0.60),
    "hitter": (0.15, 0.40),
}
STAT_VARIANCE: dict[str, float] = {"hitter": 0.10, "pitcher": 0.18}

# Replacement-level full-season stats for waiver pickups
REPLACEMENT_HITTER: dict[str, int] = {
    "r": 55, "hr": 12, "rbi": 50, "sb": 5, "h": 125, "ab": 500,
}
REPLACEMENT_SP: dict[str, int] = {
    "w": 7, "k": 120, "sv": 0, "ip": 140, "er": 70, "bb": 50, "h_allowed": 139,
}
REPLACEMENT_RP: dict[str, int] = {
    "w": 2, "k": 55, "sv": 5, "ip": 60, "er": 30, "bb": 21, "h_allowed": 60,
}

# Waiver-quality stats for injury backfill blending (10-team league).
# Separate from REPLACEMENT_* (used by Monte Carlo) — these model what
# you'd actually stream from waivers, which is slightly better quality.
WAIVER_SP: dict[str, float] = {
    "w": 7, "k": 120, "sv": 0, "ip": 140, "er": 65, "bb": 48, "h_allowed": 133,
}
WAIVER_RP: dict[str, float] = {
    "w": 2, "k": 55, "sv": 5, "ip": 60, "er": 30, "bb": 21, "h_allowed": 60,
}
WAIVER_HITTER: dict[str, float] = {
    "r": 55, "hr": 12, "rbi": 50, "sb": 5, "h": 150, "ab": 600,
}

# Healthy baselines for backfill blending
HEALTHY_SP_IP: float = 178.0
HEALTHY_CLOSER_IP: float = 60.0
HEALTHY_HITTER_AB: float = 600.0

# Gap thresholds — backfill only applies when gap exceeds these
BACKFILL_SP_THRESHOLD: float = 15.0
BACKFILL_CLOSER_THRESHOLD: float = 10.0
BACKFILL_HITTER_THRESHOLD: float = 50.0

# IP threshold to distinguish starters from middle relievers
STARTER_IP_THRESHOLD: float = 100.0


def safe_float(value) -> float:
    """Coerce None/NaN to 0. Used to guard against bad data in stat pipelines."""
    if value is None:
        return 0.0
    f = float(value)
    return 0.0 if f != f else f  # f != f is the NaN check


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
