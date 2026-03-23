"""Recency-weighted prediction models for fantasy baseball projections.

Each model function takes (projection, games, cutoff) and returns a dict of
predicted per-PA rates (hitters) or per-IP rates (pitchers).
"""
import math
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DECAY_HALF_LIFE_DAYS = 7
FIXED_BLEND_ACTUAL_WEIGHT = 0.30
FIXED_BLEND_WINDOW_DAYS = 30

HITTER_STAT_KEYS = ["hr_per_pa", "r_per_pa", "rbi_per_pa", "sb_per_pa", "avg"]
PITCHER_STAT_KEYS = ["k_per_ip", "era", "whip", "w_per_gs", "sv_per_g"]

# How many PA/IP of projection a stat is "worth" (reliability constants)
HITTER_RELIABILITY = {
    "hr_per_pa": 200,
    "r_per_pa": 300,
    "rbi_per_pa": 300,
    "sb_per_pa": 300,
    "avg": 400,
}
PITCHER_RELIABILITY = {
    "k_per_ip": 50,
    "era": 120,
    "whip": 80,
    "w_per_gs": 200,
    "sv_per_g": 200,
}

_DECAY_RATE = math.log(2) / DECAY_HALF_LIFE_DAYS


# ---------------------------------------------------------------------------
# Player type detection
# ---------------------------------------------------------------------------

def _is_hitter(projection: dict) -> bool:
    return "hr_per_pa" in projection


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse_date(d) -> date:
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d))


def _filter_games_before(games: list[dict], cutoff) -> list[dict]:
    """Return games with date strictly before cutoff."""
    cutoff_date = _parse_date(cutoff)
    return [g for g in games if _parse_date(g["date"]) < cutoff_date]


def _filter_games_window(games: list[dict], cutoff, window_days: int) -> list[dict]:
    """Return games in the N days before cutoff (date in [cutoff - window_days, cutoff))."""
    cutoff_date = _parse_date(cutoff)
    start_date = cutoff_date - timedelta(days=window_days)
    return [g for g in games if start_date <= _parse_date(g["date"]) < cutoff_date]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_hitter_games(games: list[dict]) -> dict:
    """Sum totals and compute per-PA rates from a list of hitter game log dicts."""
    pa = sum(g["pa"] for g in games)
    ab = sum(g["ab"] for g in games)
    h = sum(g["h"] for g in games)
    hr = sum(g["hr"] for g in games)
    r = sum(g["r"] for g in games)
    rbi = sum(g["rbi"] for g in games)
    sb = sum(g["sb"] for g in games)

    if pa == 0:
        return {
            "pa": 0, "ab": 0, "h": 0, "hr": 0, "r": 0, "rbi": 0, "sb": 0,
            "hr_per_pa": 0, "r_per_pa": 0, "rbi_per_pa": 0, "sb_per_pa": 0, "avg": 0,
        }

    return {
        "pa": pa, "ab": ab, "h": h, "hr": hr, "r": r, "rbi": rbi, "sb": sb,
        "hr_per_pa": hr / pa,
        "r_per_pa": r / pa,
        "rbi_per_pa": rbi / pa,
        "sb_per_pa": sb / pa,
        "avg": h / ab if ab > 0 else 0,
    }


def _aggregate_pitcher_games(games: list[dict]) -> dict:
    """Sum totals and compute per-IP rates from a list of pitcher game log dicts."""
    ip = sum(g["ip"] for g in games)
    k = sum(g["k"] for g in games)
    er = sum(g["er"] for g in games)
    bb = sum(g["bb"] for g in games)
    h_allowed = sum(g["h_allowed"] for g in games)
    w = sum(g["w"] for g in games)
    sv = sum(g["sv"] for g in games)
    gs = sum(g["gs"] for g in games)
    g = sum(g["g"] for g in games)

    if ip == 0:
        return {
            "ip": 0, "k": 0, "er": 0, "bb": 0, "h_allowed": 0, "w": 0,
            "sv": 0, "gs": 0, "g": 0,
            "k_per_ip": 0, "era": 0, "whip": 0, "w_per_gs": 0, "sv_per_g": 0,
        }

    return {
        "ip": ip, "k": k, "er": er, "bb": bb, "h_allowed": h_allowed,
        "w": w, "sv": sv, "gs": gs, "g": g,
        "k_per_ip": k / ip,
        "era": er / ip * 9,
        "whip": (bb + h_allowed) / ip,
        "w_per_gs": w / gs if gs > 0 else 0,
        "sv_per_g": sv / g if g > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Model 1: Preseason — return projection unchanged
# ---------------------------------------------------------------------------

def predict_preseason(projection: dict, games: list[dict], cutoff) -> dict:
    """Return projection rates unchanged, ignoring all game log data."""
    if _is_hitter(projection):
        return {k: projection[k] for k in HITTER_STAT_KEYS}
    return {k: projection[k] for k in PITCHER_STAT_KEYS}


# ---------------------------------------------------------------------------
# Model 2: Season-to-date — pure actuals, ignore projection
# ---------------------------------------------------------------------------

def predict_season_to_date(projection: dict, games: list[dict], cutoff) -> dict:
    """Return rates computed purely from games before cutoff. Returns zeros if no games."""
    filtered = _filter_games_before(games, cutoff)
    if _is_hitter(projection):
        agg = _aggregate_hitter_games(filtered)
        return {k: agg[k] for k in HITTER_STAT_KEYS}
    else:
        agg = _aggregate_pitcher_games(filtered)
        return {k: agg[k] for k in PITCHER_STAT_KEYS}


# ---------------------------------------------------------------------------
# Model 3: Fixed blend — 30% last-30-days actuals + 70% projection
# ---------------------------------------------------------------------------

def predict_fixed_blend(projection: dict, games: list[dict], cutoff) -> dict:
    """Blend 30% of last-30-day rates with 70% projection. Falls back to projection if no games."""
    window_games = _filter_games_window(games, cutoff, FIXED_BLEND_WINDOW_DAYS)

    if _is_hitter(projection):
        stat_keys = HITTER_STAT_KEYS
    else:
        stat_keys = PITCHER_STAT_KEYS

    if not window_games:
        return {k: projection[k] for k in stat_keys}

    if _is_hitter(projection):
        agg = _aggregate_hitter_games(window_games)
    else:
        agg = _aggregate_pitcher_games(window_games)

    w_actual = FIXED_BLEND_ACTUAL_WEIGHT
    w_proj = 1.0 - w_actual
    return {
        k: w_actual * agg[k] + w_proj * projection[k]
        for k in stat_keys
    }


# ---------------------------------------------------------------------------
# Model 4: Reliability blend — weight actual vs projection by sample size
# ---------------------------------------------------------------------------

def predict_reliability_blend(projection: dict, games: list[dict], cutoff) -> dict:
    """Blend actuals with projection using reliability-weighted actual weight.

    actual_weight = total_PA / (total_PA + reliability_constant)  per stat.
    Uses all games before cutoff.
    """
    filtered = _filter_games_before(games, cutoff)

    if _is_hitter(projection):
        agg = _aggregate_hitter_games(filtered)
        reliability = HITTER_RELIABILITY
        stat_keys = HITTER_STAT_KEYS
        sample_size = agg["pa"]
    else:
        agg = _aggregate_pitcher_games(filtered)
        reliability = PITCHER_RELIABILITY
        stat_keys = PITCHER_STAT_KEYS
        sample_size = agg["ip"]

    result = {}
    for k in stat_keys:
        rel_const = reliability[k]
        actual_weight = sample_size / (sample_size + rel_const)
        proj_weight = 1.0 - actual_weight
        result[k] = actual_weight * agg[k] + proj_weight * projection[k]
    return result


# ---------------------------------------------------------------------------
# Model 5: Exponential decay — recent games weighted more heavily
# ---------------------------------------------------------------------------

def predict_exponential_decay(projection: dict, games: list[dict], cutoff) -> dict:
    """Weight each game by exp(-decay_rate * days_ago), blend with projection via reliability.

    decay_rate = ln(2) / DECAY_HALF_LIFE_DAYS
    """
    filtered = _filter_games_before(games, cutoff)
    cutoff_date = _parse_date(cutoff)

    if _is_hitter(projection):
        return _decay_hitter(projection, filtered, cutoff_date)
    else:
        return _decay_pitcher(projection, filtered, cutoff_date)


def _decay_hitter(projection: dict, games: list[dict], cutoff_date: date) -> dict:
    """Compute exponential-decay weighted hitter rates, blended with projection.

    Decay weights determine which games' rates contribute most to the estimate.
    Reliability blending uses total unweighted PA so that actual sample size —
    not the exponential weight sum — governs how much we trust the actuals.
    """
    if not games:
        # No games: fall back to pure projection blended at weight 0
        return {k: projection[k] for k in HITTER_STAT_KEYS}

    # Weighted numerators and denominator (weighted PA) for rate estimation
    w_pa = 0.0
    w_ab = 0.0
    w_hr = 0.0
    w_r = 0.0
    w_rbi = 0.0
    w_sb = 0.0
    w_h = 0.0
    total_pa = 0
    total_ab = 0

    for g in games:
        days_ago = (cutoff_date - _parse_date(g["date"])).days
        weight = math.exp(-_DECAY_RATE * days_ago)
        w_pa += weight * g["pa"]
        w_ab += weight * g["ab"]
        w_hr += weight * g["hr"]
        w_r += weight * g["r"]
        w_rbi += weight * g["rbi"]
        w_sb += weight * g["sb"]
        w_h += weight * g["h"]
        total_pa += g["pa"]
        total_ab += g["ab"]

    if w_pa == 0:
        return {k: projection[k] for k in HITTER_STAT_KEYS}

    actual_rates = {
        "hr_per_pa": w_hr / w_pa,
        "r_per_pa": w_r / w_pa,
        "rbi_per_pa": w_rbi / w_pa,
        "sb_per_pa": w_sb / w_pa,
        "avg": w_h / w_ab if w_ab > 0 else 0,
    }

    # Blend with projection using reliability constants and total unweighted PA
    # so that actual sample size governs trust in actuals, not the weight sum.
    result = {}
    for k in HITTER_STAT_KEYS:
        rel_const = HITTER_RELIABILITY[k]
        actual_weight = total_pa / (total_pa + rel_const)
        result[k] = actual_weight * actual_rates[k] + (1.0 - actual_weight) * projection[k]
    return result


def _decay_pitcher(projection: dict, games: list[dict], cutoff_date: date) -> dict:
    """Compute exponential-decay weighted pitcher rates, blended with projection.

    Decay weights determine which games' rates contribute most to the estimate.
    Reliability blending uses total unweighted IP so that actual sample size —
    not the exponential weight sum — governs how much we trust the actuals.
    """
    if not games:
        return {k: projection[k] for k in PITCHER_STAT_KEYS}

    w_ip = 0.0
    w_k = 0.0
    w_er = 0.0
    w_bb = 0.0
    w_h_allowed = 0.0
    w_w = 0.0
    w_sv = 0.0
    w_gs = 0.0
    w_g = 0.0
    total_ip = 0.0
    total_gs = 0
    total_g = 0

    for g in games:
        days_ago = (cutoff_date - _parse_date(g["date"])).days
        weight = math.exp(-_DECAY_RATE * days_ago)
        w_ip += weight * g["ip"]
        w_k += weight * g["k"]
        w_er += weight * g["er"]
        w_bb += weight * g["bb"]
        w_h_allowed += weight * g["h_allowed"]
        w_w += weight * g["w"]
        w_sv += weight * g["sv"]
        w_gs += weight * g["gs"]
        w_g += weight * g["g"]
        total_ip += g["ip"]
        total_gs += g["gs"]
        total_g += g["g"]

    if w_ip == 0:
        return {k: projection[k] for k in PITCHER_STAT_KEYS}

    actual_rates = {
        "k_per_ip": w_k / w_ip,
        "era": w_er / w_ip * 9,
        "whip": (w_bb + w_h_allowed) / w_ip,
        "w_per_gs": w_w / w_gs if w_gs > 0 else 0,
        "sv_per_g": w_sv / w_g if w_g > 0 else 0,
    }

    result = {}
    for k in PITCHER_STAT_KEYS:
        rel_const = PITCHER_RELIABILITY[k]
        actual_weight = total_ip / (total_ip + rel_const)
        result[k] = actual_weight * actual_rates[k] + (1.0 - actual_weight) * projection[k]
    return result
