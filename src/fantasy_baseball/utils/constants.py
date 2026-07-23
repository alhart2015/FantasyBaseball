from enum import Enum


class Category(Enum):
    """Roto scoring category.

    Plain ``Enum`` (not ``StrEnum``): members are not strings. Compare
    against ``Category`` members directly (``cat == Category.HR``), and
    use ``.value`` at I/O boundaries when you need the uppercase
    string form for JSON/Redis.
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


class OpportunityStat(Enum):
    """Non-roto volume stats preserved from Yahoo standings.

    PA, IP are best-effort ingest. AB is enum-only until Yahoo standings
    ingest is wired up; until then it lives on extras only when callers
    supply it explicitly.

    These aren't scoring categories, but they ride alongside
    :class:`CategoryStats` on :class:`StandingsEntry.extras` so
    team-level opportunity totals survive the typed round-trip for
    UI consumers like the lineup pace display (``_compute_team_totals_pace``).
    Same contract as ``Category``: compare members directly, use
    ``.value`` only at JSON/Redis/template boundaries.
    """

    PA = "PA"
    IP = "IP"
    AB = "AB"


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

# Counting categories (R, HR, RBI, SB, W, K, SV) — every roto category that
# isn't a rate. Derived so adding a new ``Category`` member auto-routes to
# either rate or counting without a second edit.
COUNTING_STATS: frozenset[Category] = frozenset(ALL_CATEGORIES) - RATE_STATS

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

DEFAULT_NUM_TEAMS: int = 10

# League-typical full-season team volumes, shared by the SGP scale
# (rate-stat SGP is proportional to 1/team_ip), the simulation's
# YTD-blend fallback, and the trade evaluator's legacy fallback.
# TEAM_IP calibrated 2026-07-05 from this league's live standings
# (cache:standings extras.IP, all 10 teams extrapolated to season end:
# median 1288, mean 1283, range 1010-1451). The prior 1450 matched only
# the single highest-volume team and compressed every pitcher's
# ERA/WHIP SGP by ~11%. Recalibrate from the same source if roster
# slots or league streaming behavior change materially.
DEFAULT_TEAM_AB: int = 5500
DEFAULT_TEAM_IP: int = 1300


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


# Module-level default derived from the league-default roster slots and size.
# Consumed by sgp/replacement.py; callers with a non-default league pass their
# own via compute_starters_per_position(...).
STARTERS_PER_POSITION: dict[str, int] = compute_starters_per_position()

IF_ELIGIBLE: set[str] = {"1B", "2B", "3B", "SS"}

# Minimum projected SV to classify a pitcher as a closer.
CLOSER_SV_THRESHOLD: int = 20

# Sentinel ERA/WHIP assigned to a zero-IP team in a Monte Carlo draw so it ranks
# dead-last in those categories. Load-bearing for roto scoring (must survive into
# score_roto_dict); the distributions builder drops it before KDE so it cannot
# paint a phantom tail. Shared by the MC batch producer
# (simulate_remaining_season_batch) and the distributions consumer so the drop
# matches the fill. NOTE: other zero-IP 99.0 literals exist in scoring.py and
# utils/rate_stats.py; they are not yet wired to this constant.
ZERO_IP_RATE_SENTINEL: float = 99.0

# Stat column lists for counting stats (used by Monte Carlo, stat aggregation)
HITTING_COUNTING: list[str] = ["r", "hr", "rbi", "sb", "h", "ab"]
PITCHING_COUNTING: list[str] = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]

# Full projection key lists (counting + rate + opportunity stats, used by pace)
HITTER_PROJ_KEYS: list[str] = ["pa", "r", "hr", "rbi", "sb", "h", "ab", "avg"]
PITCHER_PROJ_KEYS: list[str] = ["ip", "w", "k", "sv", "er", "bb", "h_allowed", "era", "whip"]

# League at-bats per plate appearance. Stable (~0.88-0.91 across teams/seasons);
# used to convert between PA and AB (refresh pipeline) and as the AB->PA fallback
# in the playing-time curve lookup when a dict carries ``ab`` but not ``pa``.
AB_PER_PA: float = 0.90

# Per-stat Negative-Binomial dispersion r (var = mu + mu^2/r), calibrated from
# 2022-2024 projection-vs-actual residuals conditional on realized playing time
# (see scripts/calibrate_stat_dispersion.py). A value is either a scalar r or a
# list of (mu_upper, r) bands for stats with mean-dependent overdispersion
# (sb/hr/rbi/er, r generally rises with the projected count -- the per-band MLE
# can invert slightly in an adjacent band, e.g. er's 14.701 -> 13.263, which is
# fit noise, not signal); float("inf") == Poisson floor (no overdispersion beyond
# Poisson). sv is fit on the role-stable closer population and collapses to a
# robust scalar given the thin (n=43) data.
STAT_DISPERSION: dict[str, float | list[tuple[float, float]]] = {
    "r": 84.725,
    "hr": [(3.9, 8.580), (8.5, 7.754), (16.2, 12.455), (float("inf"), 23.868)],
    "rbi": [(17.5, 7.207), (36.5, 27.104), (59.3, 34.039), (float("inf"), 45.005)],
    "sb": [(1.1, 0.763), (2.8, 1.876), (6.9, 2.709), (float("inf"), 4.747)],
    "h": float("inf"),
    "w": float("inf"),
    "k": 109.134,
    "sv": 37.757,
    "er": [(13.7, 7.545), (24.6, 14.701), (40.1, 13.263), (float("inf"), 27.425)],
    "bb": 21.645,
    "h_allowed": 81.291,
}

# Closer role-switch mixture curves (SV variance). A K-component mixture keyed on
# projected SV s: component probabilities p(s) and unit-mean shares w(s), each a K-way
# softmax over K-1 free [b0, b1] logit lines (a zero logit is appended). Then
# a_k = w_k/p_k -- mean-1 (sum p*a == 1) and non-negative by construction.
# 3 components, MLE-fit on 2022-2025; (re)generate with scripts/calibrate_closer_mixture.py.
# See src/fantasy_baseball/sgp/closer_mixture.py and the design spec.
SV_ROLE_MIXTURE: dict[str, list[list[float]]] = {
    "p_logits": [[1.1551, -0.0708], [-2.0943, 0.1456]],
    "w_logits": [[-0.7243, -0.0487], [-0.8134, 0.122]],
}

# Playing-time model: realized PA/IP relative to projection, calibrated from
# 2022-2025 Steamer+ZiPS vs actuals on the rosterable population (volume floor
# at the >=90%-MLB-appearance knee for hitters/SP; MLB-appearance required for
# RP). See scripts/calibrate_playing_time.py. Two-sided (a player can beat his
# projection) and volume-scaled, which a single league-wide injury haircut
# cannot represent.
#
# Per (type, role), the curve maps projected volume (PA for hitters, IP for
# pitchers) to (mean_scale, cv_pt): mean_scale is the multiplicative haircut on
# projected counting stats; cv_pt is the SD of actual/projected playing time.
# Both are monotone in volume (more projected volume -> higher, tighter). The
# lookup interpolates between band centers (see utils/playing_time.py); pitcher
# role is SP if projected IP >= STARTER_IP_THRESHOLD else RP (PitcherStats has
# no GS field, so IP is the only role signal available at deployment).
# The realized-PT distribution SHAPE (skew + the volume-dependent ceiling that
# replaced the old flat 2.0 clip) lives in PLAYING_TIME_SHAPE below.
PLAYING_TIME_CURVES: dict[str, list[dict[str, float]]] = {
    "hitters": [
        {"vol": 382.5, "mean_scale": 0.7518, "cv_pt": 0.4181},
        {"vol": 440.9, "mean_scale": 0.8352, "cv_pt": 0.3751},
        {"vol": 498.8, "mean_scale": 0.8352, "cv_pt": 0.3267},
        {"vol": 563.9, "mean_scale": 0.9102, "cv_pt": 0.2556},
        {"vol": 629.0, "mean_scale": 0.9452, "cv_pt": 0.1929},
    ],
    "SP": [
        {"vol": 109.4, "mean_scale": 0.8052, "cv_pt": 0.4650},
        {"vol": 126.6, "mean_scale": 0.8437, "cv_pt": 0.4504},
        {"vol": 147.1, "mean_scale": 0.8544, "cv_pt": 0.3693},
        {"vol": 169.9, "mean_scale": 0.8544, "cv_pt": 0.3002},
    ],
    "RP": [
        {"vol": 48.3, "mean_scale": 0.7579, "cv_pt": 0.4758},
        {"vol": 58.4, "mean_scale": 0.7861, "cv_pt": 0.4172},
        {"vol": 71.5, "mean_scale": 0.7861, "cv_pt": 0.4172},
    ],
}

# Empirical SHAPE of realized/projected playing time, as a standardized-z ladder
# per (role, projected-volume band): z = (ratio - band_mean) / band_sd, so each
# ladder has mean 0, sd 1 and carries only the DISTRIBUTION SHAPE (skew + bounded
# tails). The MC sampler draws u ~ Uniform(0,1), interpolates z at u over
# QUANTILE_LEVELS, then realizes scale = mean_scale + z * cv_pt (with the
# fraction_remaining damping). This replaces the old symmetric-Normal-clipped-at-
# 2.0 draw: hitters/SP are left-skewed with a realistic ceiling that SHRINKS with
# projected volume (an everyday regular tops out near 1.1-1.2x; only low-IP
# relievers reach 2x+ via role changes), which the flat clip could not express.
# ERoto keeps using just mean_scale/cv_pt, so the two consumers stay moment-
# consistent. Regenerate with scripts/calibrate_playing_time.py (2022-2025).
# Band 'vol' values match PLAYING_TIME_CURVES so the curve and shape interpolate
# on the same volume axis.
QUANTILE_LEVELS: list[float] = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
PLAYING_TIME_SHAPE: dict[str, list[dict[str, object]]] = {
    "hitters": [
        {
            "vol": 382.5,
            "z": [-1.798, -1.6923, -1.3884, -0.7808, 0.0398, 0.7343, 1.2931, 1.5539, 2.146],
        },
        {
            "vol": 440.9,
            "z": [-2.2293, -1.7231, -1.4946, -0.7521, 0.1821, 0.7456, 1.2049, 1.409, 1.6584],
        },
        {
            "vol": 498.8,
            "z": [-2.206, -1.8144, -1.5111, -0.7046, 0.153, 0.7413, 1.2512, 1.4049, 1.7774],
        },
        {
            "vol": 563.9,
            "z": [-2.9485, -2.1829, -1.429, -0.4398, 0.2868, 0.7146, 1.0014, 1.0865, 1.449],
        },
        {
            "vol": 629.0,
            "z": [-3.2184, -2.1493, -1.4167, -0.3563, 0.3609, 0.6864, 0.8931, 0.9819, 1.1377],
        },
    ],
    "SP": [
        {
            "vol": 109.4,
            "z": [-1.7314, -1.7314, -1.4046, -0.8592, 0.0478, 0.7935, 1.3232, 1.4729, 1.8569],
        },
        {
            "vol": 126.6,
            "z": [-1.8732, -1.8732, -1.5227, -0.775, 0.1575, 0.88, 1.1819, 1.3084, 1.4966],
        },
        {
            "vol": 147.1,
            "z": [-2.269, -1.816, -1.5314, -0.7766, 0.314, 0.819, 1.0567, 1.2197, 1.3896],
        },
        {
            "vol": 169.9,
            "z": [-2.823, -2.2242, -1.4742, -0.5106, 0.3478, 0.7546, 0.9465, 1.1039, 1.2447],
        },
    ],
    "RP": [
        {
            "vol": 48.3,
            "z": [-1.5004, -1.2803, -1.0941, -0.8487, -0.1491, 0.7231, 1.371, 1.7501, 2.5877],
        },
        {
            "vol": 58.4,
            "z": [-1.8528, -1.5227, -1.3245, -0.8019, 0.1014, 0.6512, 0.9827, 1.3176, 3.1141],
        },
        {
            "vol": 71.5,
            "z": [-1.6466, -1.4737, -1.354, -0.7861, 0.0459, 0.561, 1.2455, 1.9408, 2.5305],
        },
    ],
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

# Replacement-level full-season stats for waiver pickups.
# REPLACEMENT_HITTER/SP/RP are the legacy flat lines, still used by
# analysis/transactions.py for drop-cost. The Monte Carlo injury backfill uses
# REPLACEMENT_BY_POSITION instead: a position-aware floor calibrated from this
# league's OWN free agents (top-3 un-rostered by SGP, averaged over the post-draft
# and current snapshots; see scripts/analyze_replacement_levels.py). It captures
# that a streamed catcher gives ~0 SB while a streamed middle infielder gives ~15,
# which the flat line erased. Regenerate after a season for fresh calibration.
REPLACEMENT_BY_POSITION: dict[str, dict[str, int]] = {
    "C": {"r": 55, "hr": 14, "rbi": 56, "sb": 4, "h": 107, "ab": 423},
    "1B": {"r": 65, "hr": 18, "rbi": 68, "sb": 6, "h": 121, "ab": 498},
    "2B": {"r": 62, "hr": 13, "rbi": 60, "sb": 17, "h": 124, "ab": 508},
    "3B": {"r": 65, "hr": 17, "rbi": 68, "sb": 7, "h": 122, "ab": 496},
    "SS": {"r": 64, "hr": 14, "rbi": 62, "sb": 15, "h": 127, "ab": 520},
    "OF": {"r": 65, "hr": 17, "rbi": 63, "sb": 15, "h": 110, "ab": 451},
    "SP": {"w": 9, "k": 164, "sv": 0, "ip": 164, "er": 77, "bb": 58, "h_allowed": 153},
    "RP": {"w": 6, "k": 96, "sv": 8, "ip": 75, "er": 27, "bb": 31, "h_allowed": 61},
}

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

# IP threshold to distinguish starters from middle relievers
STARTER_IP_THRESHOLD: float = 100.0


def safe_float(value) -> float:
    """Coerce None/NaN to 0. Used to guard against bad data in stat pipelines."""
    if value is None:
        return 0.0
    f = float(value)
    return 0.0 if f != f else f  # f != f is the NaN check


def role_from_ip(ip: float) -> str:
    """Classify a pitcher as 'SP' or 'RP' from projected innings.

    Single source of truth for the IP -> role rule: ``ip >=
    STARTER_IP_THRESHOLD`` is a starter. NaN/None coerce to 0 (RP) via
    ``safe_float`` so callers don't each re-guard bad data.

    Pass FULL-SEASON IP: the threshold is a full-season bar, so a partial
    to-date or rest-of-season IP misclassifies starters as relievers (issue #251).
    """
    return "SP" if safe_float(ip) >= STARTER_IP_THRESHOLD else "RP"


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
