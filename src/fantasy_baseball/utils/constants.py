from enum import StrEnum


class Category(StrEnum):
    """Roto scoring category.

    ``StrEnum`` members *are* strings, so existing code that compares
    category values to bare strings (``cat == "HR"``, ``cat in {"ERA",
    "WHIP"}``, dict lookups keyed on ``"R"``) continues to work
    unchanged. New code should prefer the enum members for type safety.
    """

    R = "R"
    HR = "HR"
    RBI = "RBI"
    SB = "SB"
    AVG = "AVG"
    W = "W"
    K = "K"
    ERA = "ERA"
    WHIP = "WHIP"
    SV = "SV"


HITTING_CATEGORIES: list[Category] = [
    Category.R,
    Category.HR,
    Category.RBI,
    Category.SB,
    Category.AVG,
]
PITCHING_CATEGORIES: list[Category] = [
    Category.W,
    Category.K,
    Category.ERA,
    Category.WHIP,
    Category.SV,
]
ALL_CATEGORIES: list[Category] = HITTING_CATEGORIES + PITCHING_CATEGORIES

RATE_STATS: frozenset[Category] = frozenset({Category.AVG, Category.ERA, Category.WHIP})
INVERSE_STATS: frozenset[Category] = frozenset({Category.ERA, Category.WHIP})  # Lower is better

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
        pos: count * num_teams for pos, count in roster_slots.items() if pos not in ("BN", "IL")
    }


# Default value kept for backward compatibility.
STARTERS_PER_POSITION: dict[str, int] = compute_starters_per_position()

IF_ELIGIBLE: set[str] = {"1B", "2B", "3B", "SS"}

# Minimum projected SV to classify a pitcher as a closer.
CLOSER_SV_THRESHOLD: int = 20

# Stat column lists for counting stats (used by Monte Carlo, stat aggregation)
HITTING_COUNTING: list[str] = ["r", "hr", "rbi", "sb", "h", "ab"]
PITCHING_COUNTING: list[str] = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]

# Full projection key lists (counting + rate + opportunity stats, used by pace)
HITTER_PROJ_KEYS: list[str] = ["pa", "r", "hr", "rbi", "sb", "h", "ab", "avg"]
PITCHER_PROJ_KEYS: list[str] = ["ip", "w", "k", "sv", "er", "bb", "h_allowed", "era", "whip"]

# Monte Carlo simulation parameters (single source of truth)
INJURY_PROB: dict[str, float] = {"pitcher": 0.45, "hitter": 0.18}
INJURY_SEVERITY: dict[str, tuple[float, float]] = {
    "pitcher": (0.20, 0.60),
    "hitter": (0.15, 0.40),
}
# Per-stat performance variance (SD of actual/projected ratio residuals).
# Calibrated from Steamer+ZiPS projections vs MLB actuals, 2022-2024.
# Per-PA rates for hitters, per-IP rates for pitchers (isolates
# performance variance from playing time, which the injury model handles).
# Stats with 0.0 variance (ab, ip) are playing-time-only — no perf multiplier.
STAT_VARIANCE: dict[str, float] = {
    # Hitter counting stats
    "r": 0.156,
    "hr": 0.343,
    "rbi": 0.187,
    "sb": 0.715,
    "h": 0.103,
    "ab": 0.0,
    # Pitcher counting stats
    "w": 0.416,
    "k": 0.139,
    "sv": 0.900,
    "ip": 0.0,
    "er": 0.252,
    "bb": 0.257,
    "h_allowed": 0.143,
}

# Empirical correlation matrices for correlated variance draws.
# Calibrated from projection-vs-actual residuals, 2022-2024.
# Column order must match the stat lists below.

# Hitter correlated stats and their correlation matrix.
# Column order: r, hr, rbi, sb, h
# AVG emerges from h/ab, so we correlate h (hits) with the counting stats.
# h-to-counting correlations derived from AVG correlations with R/HR/RBI/SB.
HITTER_CORR_STATS: list[str] = ["r", "hr", "rbi", "sb", "h"]
HITTER_CORRELATION: list[list[float]] = [
    #    r      hr     rbi    sb     h
    [+1.000, +0.653, +0.728, +0.435, +0.463],  # r
    [+0.653, +1.000, +0.760, +0.278, +0.321],  # hr
    [+0.728, +0.760, +1.000, +0.343, +0.466],  # rbi
    [+0.435, +0.278, +0.343, +1.000, +0.290],  # sb
    [+0.463, +0.321, +0.466, +0.290, +1.000],  # h
]

# Pitcher correlated stats and their correlation matrix.
# Column order: w, k, sv, er, bb, h_allowed
# ERA/WHIP emerge from er/bb/h_allowed components, so we correlate those.
# er/bb/h_allowed correlations derived from ERA/WHIP correlations with W/K/SV.
PITCHER_CORR_STATS: list[str] = ["w", "k", "sv", "er", "bb", "h_allowed"]
PITCHER_CORRELATION: list[list[float]] = [
    #    w      k      sv     er     bb     h_a
    [+1.000, +0.366, +0.030, -0.057, -0.105, -0.057],  # w
    [+0.366, +1.000, +0.115, -0.237, -0.317, -0.237],  # k
    [+0.030, +0.115, +1.000, -0.341, -0.345, -0.341],  # sv
    [-0.057, -0.237, -0.341, +1.000, +0.729, +0.729],  # er
    [-0.105, -0.317, -0.345, +0.729, +1.000, +0.729],  # bb
    [-0.057, -0.237, -0.341, +0.729, +0.729, +1.000],  # h_allowed
]

# In-season management adjustment (roto points).
# Calibrated from draft-day projected roto vs actual final standings, 2023-2025.
# Represents the net effect of waiver moves, trades, and streaming over a
# full season. Normalized to sum to zero (roto is zero-sum).
# Each entry is (mean_adjustment, sd). Applied per-simulation as
# N(mean, sd) added to a team's roto total after scoring.
MANAGEMENT_ADJUSTMENT: dict[str, tuple[float, float]] = {
    "Hello Peanuts!": (+17.1, 8.4),
    "Boston Estrellas": (+12.8, 7.8),
    "Springfield Isotopes": (+12.1, 5.1),
    "Hart of the Order": (+11.1, 14.4),
    "Spacemen": (+5.6, 18.6),
    "Send in the Cavalli": (+5.0, 10.0),
    "Work in Progress": (-5.2, 9.9),
    "Jon's Underdogs": (-11.9, 9.7),
    "Tortured Baseball Department": (-13.7, 10.0),
    "SkeleThor": (-33.0, 12.6),
}

# Default adjustment for teams not in the lookup (new managers, other leagues).
MANAGEMENT_ADJUSTMENT_DEFAULT: tuple[float, float] = (0.0, 10.0)

# Replacement-level full-season stats for waiver pickups
REPLACEMENT_HITTER: dict[str, int] = {
    "r": 55,
    "hr": 12,
    "rbi": 50,
    "sb": 5,
    "h": 125,
    "ab": 500,
}
REPLACEMENT_SP: dict[str, int] = {
    "w": 7,
    "k": 120,
    "sv": 0,
    "ip": 140,
    "er": 70,
    "bb": 50,
    "h_allowed": 139,
}
REPLACEMENT_RP: dict[str, int] = {
    "w": 2,
    "k": 55,
    "sv": 5,
    "ip": 60,
    "er": 30,
    "bb": 21,
    "h_allowed": 60,
}

# Waiver-quality stats for injury backfill blending (10-team league).
# Separate from REPLACEMENT_* (used by Monte Carlo) — these model what
# you'd actually stream from waivers, which is slightly better quality.
WAIVER_SP: dict[str, float] = {
    "w": 7,
    "k": 120,
    "sv": 0,
    "ip": 140,
    "er": 65,
    "bb": 48,
    "h_allowed": 133,
}
WAIVER_RP: dict[str, float] = {
    "w": 2,
    "k": 55,
    "sv": 5,
    "ip": 60,
    "er": 30,
    "bb": 21,
    "h_allowed": 60,
}
WAIVER_HITTER: dict[str, float] = {
    "r": 55,
    "hr": 12,
    "rbi": 50,
    "sb": 5,
    "h": 150,
    "ab": 600,
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


# Yahoo player statuses that indicate a player is on the injured list.
IL_STATUSES = frozenset({"IL", "IL+", "IL10", "IL15", "IL60", "DL", "DL+"})

DEFAULT_SGP_DENOMINATORS: dict[Category, float] = {
    Category.R: 20.0,
    Category.HR: 9.0,
    Category.RBI: 20.0,
    Category.SB: 8.0,
    Category.AVG: 0.005,
    Category.W: 3.0,
    Category.K: 30.0,
    Category.ERA: 0.15,
    Category.WHIP: 0.015,
    Category.SV: 7.0,
}
