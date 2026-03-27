"""Compute player performance vs projection pace with z-score color coding."""

from fantasy_baseball.utils.constants import INVERSE_STATS, STAT_VARIANCE

# Roto categories by player type
HITTER_COUNTING = ["r", "hr", "rbi", "sb"]
PITCHER_COUNTING = ["w", "k", "sv"]

# Rate stat -> component stat for variance lookup
RATE_COMPONENT = {"avg": "h", "era": "er", "whip": "h_allowed"}


def _z_to_color(z: float) -> str:
    """Map z-score to CSS color class."""
    if z > 1.0:
        return "stat-hot-2"
    if z > 0.5:
        return "stat-hot-1"
    if z < -1.0:
        return "stat-cold-2"
    if z < -0.5:
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
        rate_stats = {"avg": ("h", "ab")}
    else:
        opp_key = "ip"
        counting = PITCHER_COUNTING
        rate_stats = {
            "era": ("er",),
            "whip": ("bb", "h_allowed"),
        }

    actual_opp = actual_stats.get(opp_key, 0) or 0
    proj_opp = projected_stats.get(opp_key, 0) or 0

    # Opportunity column (PA or IP) — always neutral
    result[opp_key.upper()] = {
        "actual": actual_opp if player_type == "hitter" else actual_stats.get("ip", 0),
        "color_class": "stat-neutral",
    }

    # Counting stats
    for stat in counting:
        actual = actual_stats.get(stat, 0) or 0
        proj = projected_stats.get(stat, 0) or 0

        if proj_opp > 0 and proj > 0:
            expected = proj * (actual_opp / proj_opp)
        else:
            expected = 0.0

        if expected > 0:
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
            "color_class": _z_to_color(z),
            "projection": proj,
        }

    return result
