"""Compute player performance vs projection pace with z-score color coding."""

from fantasy_baseball.utils.constants import INVERSE_STATS, STAT_VARIANCE

# Roto categories by player type
HITTER_COUNTING = ["r", "hr", "rbi", "sb"]
PITCHER_COUNTING = ["w", "k", "sv"]

# Rate stat -> component stat for variance lookup
RATE_COMPONENT = {"avg": "h", "era": "er", "whip": "h_allowed"}

# Sample size thresholds
# Hitters: < 10 PA = all neutral, 10-29 PA = counting colored / rates neutral, >= 30 PA = all colored
HITTER_MIN_COUNTING = 10   # PA threshold for counting stats to be colored
HITTER_MIN_RATES = 30      # PA threshold for rate stats to be colored

# Pitchers: < 5 IP = all neutral, 5-9 IP = counting colored / rates neutral, >= 10 IP = all colored
PITCHER_MIN_COUNTING = 5   # IP threshold for counting stats to be colored
PITCHER_MIN_RATES = 10     # IP threshold for rate stats to be colored

# Z-score thresholds for color coding
Z_BRIGHT = 2.0   # >= this: stat-hot-2 / stat-cold-2 (bright green/red)
Z_LIGHT = 1.0    # >= this: stat-hot-1 / stat-cold-1 (light green/red)

# Minimum absolute difference (actual vs expected) for counting stats to be
# colored.  Prevents e.g. 1 RBI vs 0.2 expected from showing bright green.
COUNTING_MIN_ABS_DIFF = 1.0


def _z_to_color(z: float) -> str:
    """Map z-score to CSS color class."""
    if z > Z_BRIGHT:
        return "stat-hot-2"
    if z > Z_LIGHT:
        return "stat-hot-1"
    if z < -Z_BRIGHT:
        return "stat-cold-2"
    if z < -Z_LIGHT:
        return "stat-cold-1"
    return "stat-neutral"


def compute_player_pace(
    actual_stats: dict,
    projected_stats: dict,
    player_type: str,
) -> dict:
    """Compute z-scores and color classes for each roto stat.

    Args:
        actual_stats: Season-to-date from game_logs (lowercase keys).
        projected_stats: Full-season from blended_projections (lowercase keys).
        player_type: "hitter" or "pitcher".

    Returns:
        Dict with UPPERCASE display keys, each containing:
        {"actual", "expected", "z_score", "color_class", "projection"}
    """
    result = {}

    if player_type == "hitter":
        opp_key = "pa"
        counting = HITTER_COUNTING
        min_counting = HITTER_MIN_COUNTING
        min_rates = HITTER_MIN_RATES
    else:
        opp_key = "ip"
        counting = PITCHER_COUNTING
        min_counting = PITCHER_MIN_COUNTING
        min_rates = PITCHER_MIN_RATES

    actual_opp = actual_stats.get(opp_key, 0) or 0
    proj_opp = projected_stats.get(opp_key, 0) or 0

    # Opportunity column (PA or IP) — always neutral
    result[opp_key.upper()] = {
        "actual": actual_opp if player_type == "hitter" else actual_stats.get("ip", 0),
        "color_class": "stat-neutral",
    }

    # Counting stats — suppress color below min_counting threshold
    counting_colored = actual_opp >= min_counting

    for stat in counting:
        actual = actual_stats.get(stat, 0) or 0
        proj = projected_stats.get(stat, 0) or 0

        if proj_opp > 0 and proj > 0:
            expected = proj * (actual_opp / proj_opp)
        else:
            expected = 0.0

        if expected > 0 and counting_colored:
            ratio = actual / expected
            variance = STAT_VARIANCE.get(stat, 0.0)
            z = (ratio - 1.0) / variance if variance > 0 else 0.0
        else:
            z = 0.0

        display_key = stat.upper()
        result[display_key] = {
            "actual": actual,
            "expected": round(expected, 1),
            "z_score": round(z, 2),
            "color_class": _z_to_color(z) if abs(actual - expected) >= COUNTING_MIN_ABS_DIFF else "stat-neutral",
            "projection": round(proj),
        }

    # Rate stats — always computed, but color suppressed below min_rates threshold
    rates_colored = actual_opp >= min_rates

    if player_type == "hitter":
        actual_h = actual_stats.get("h", 0) or 0
        actual_ab = actual_stats.get("ab", 0) or 0
        proj_avg = projected_stats.get("avg", 0.0) or 0.0

        actual_avg = round(actual_h / actual_ab, 3) if actual_ab > 0 else 0.0

        if proj_avg > 0 and actual_ab > 0 and rates_colored:
            variance = STAT_VARIANCE.get("h", 0.0)
            z = (actual_avg - proj_avg) / (variance * proj_avg) if variance > 0 else 0.0
        else:
            z = 0.0

        result["AVG"] = {
            "actual": actual_avg,
            "expected": proj_avg,
            "z_score": round(z, 2),
            "color_class": _z_to_color(z),
            "projection": proj_avg,
        }

    else:  # pitcher
        actual_ip = actual_stats.get("ip", 0) or 0
        actual_er = actual_stats.get("er", 0) or 0
        actual_bb = actual_stats.get("bb", 0) or 0
        actual_ha = actual_stats.get("h_allowed", 0) or 0
        proj_era = projected_stats.get("era", 0.0) or 0.0
        proj_whip = projected_stats.get("whip", 0.0) or 0.0

        # ERA
        actual_era = round(actual_er * 9 / actual_ip, 2) if actual_ip > 0 else 0.0
        if proj_era > 0 and rates_colored:
            variance = STAT_VARIANCE.get("er", 0.0)
            z = (actual_era - proj_era) / (variance * proj_era) if variance > 0 else 0.0
            z = -z  # inverse stat: lower is better
        else:
            z = 0.0

        result["ERA"] = {
            "actual": actual_era,
            "expected": proj_era,
            "z_score": round(z, 2),
            "color_class": _z_to_color(z),
            "projection": proj_era,
        }

        # WHIP
        actual_whip = round((actual_bb + actual_ha) / actual_ip, 2) if actual_ip > 0 else 0.0
        if proj_whip > 0 and rates_colored:
            variance = STAT_VARIANCE.get("h_allowed", 0.0)
            z = (actual_whip - proj_whip) / (variance * proj_whip) if variance > 0 else 0.0
            z = -z  # inverse stat: lower is better
        else:
            z = 0.0

        result["WHIP"] = {
            "actual": actual_whip,
            "expected": proj_whip,
            "z_score": round(z, 2),
            "color_class": _z_to_color(z),
            "projection": proj_whip,
        }

    return result
