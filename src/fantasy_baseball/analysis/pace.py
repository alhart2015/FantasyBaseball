"""Compute player performance vs projection pace with z-score color coding.

DISPLAY ONLY. The output of this module is used for hot/cold color
highlighting on the lineup page and nowhere else. It is NOT a projection
and must NOT be fed into roster decisions, trade evaluation, waiver
scoring, or projected standings. Those all rely on the raw ROS
projections from the `ros_blended_projections` SQLite table.

History: recency blending used to run over roster players and overwrite
their ROS stats with reliability-weighted rates from game logs. That
produced two sources of truth (blended for the user team, raw for
opponents) and caused the Arozarena/Suarez bug on the player
comparison page. It has been removed - pace highlighting is now the
only legitimate use of in-season game logs for display.
"""

import math
from typing import Any

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.player_value import (
    REPLACEMENT_AVG,
    REPLACEMENT_ERA,
    REPLACEMENT_WHIP,
    calculate_counting_sgp,
    calculate_hitting_rate_sgp,
    calculate_pitching_rate_sgp,
)
from fantasy_baseball.utils.constants import (
    DEFAULT_TEAM_AB,
    DEFAULT_TEAM_IP,
    HITTER_PROJ_KEYS,
    INVERSE_STATS,
    PITCHER_PROJ_KEYS,
    Category,
)
from fantasy_baseball.utils.dispersion import negbin_perf_cv
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

# Roto categories by player type
HITTER_COUNTING = ["r", "hr", "rbi", "sb"]
PITCHER_COUNTING = ["w", "k", "sv"]

# Sample size thresholds
# Hitters: < 10 PA = all neutral, 10-29 PA = counting colored / rates neutral, >= 30 PA = all colored
HITTER_MIN_COUNTING = 10  # PA threshold for counting stats to be colored
HITTER_MIN_RATES = 30  # PA threshold for rate stats to be colored

# Pitchers: < 5 IP = all neutral, 5-9 IP = counting colored / rates neutral, >= 10 IP = all colored
PITCHER_MIN_COUNTING = 5  # IP threshold for counting stats to be colored
PITCHER_MIN_RATES = 10  # IP threshold for rate stats to be colored

# Z-score thresholds for color coding
Z_BRIGHT = 2.0  # >= this: stat-hot-2 / stat-cold-2 (bright green/red)
Z_LIGHT = 1.0  # >= this: stat-hot-1 / stat-cold-1 (light green/red)

# Minimum qualified players in a pool before percentile bucketing is meaningful.
MIN_POOL_SIZE = 6

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


def _prorated_expected(proj: float, actual_opp: float, proj_opp: float) -> float:
    """Projected counting stat scaled to actual playing time so far.

    Returns 0.0 when the projection has no opportunity (proj_opp <= 0) or
    projects zero of the stat, matching the guard compute_player_pace uses.
    """
    if proj_opp > 0 and proj > 0:
        return proj * (actual_opp / proj_opp)
    return 0.0


def compute_player_pace(
    actual_stats: dict[str, Any],
    projected_stats: dict[str, Any],
    player_type: str,
    rest_of_season_stats: dict[str, Any] | None = None,
    sgp_denoms: dict[Any, float] | None = None,
) -> dict[str, Any]:
    """Compute z-scores and color classes for each roto stat.

    Args:
        actual_stats: Season-to-date from game_logs (lowercase keys).
        projected_stats: Full-season from blended_projections (lowercase keys).
        player_type: "hitter" or "pitcher".
        rest_of_season_stats: Optional ROS projection dict (lowercase keys) for deviation calc.
        sgp_denoms: Optional SGP denominator dict (UPPERCASE keys) for deviation calc.

    Returns:
        Dict with UPPERCASE display keys, each containing:
        {"actual", "expected", "z_score", "color_class", "projection",
         "rest_of_season_deviation_sgp"}
    """
    result: dict[str, Any] = {}

    if player_type == PlayerType.HITTER:
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

    def _rest_of_season_deviation(cat: str) -> float:
        """Compute SGP deviation: (ros - preseason) / denom, positive = good."""
        if not rest_of_season_stats or not sgp_denoms:
            return 0.0
        rest_of_season_key = cat.lower()
        rest_of_season_val = rest_of_season_stats.get(rest_of_season_key)
        pre_val = projected_stats.get(rest_of_season_key)
        try:
            cat_enum = Category(cat)
        except ValueError:
            return 0.0
        # sgp_denoms may be keyed by Category (library callers) or by the
        # uppercase string form (tests, external callers) — check both.
        denom = sgp_denoms.get(cat_enum)
        if denom is None:
            denom = sgp_denoms.get(cat)
        if rest_of_season_val is None or pre_val is None or not denom:
            return 0.0
        dev = (rest_of_season_val - pre_val) / denom
        if cat_enum in INVERSE_STATS:
            dev = -dev
        return float(round(dev, 2))

    # Opportunity column (PA or IP) — always neutral
    result[opp_key.upper()] = {
        "actual": actual_opp if player_type == PlayerType.HITTER else actual_stats.get("ip", 0),
        "color_class": "stat-neutral",
    }

    # Counting stats — suppress color below min_counting threshold
    counting_colored = actual_opp >= min_counting

    for stat in counting:
        actual = actual_stats.get(stat, 0) or 0
        proj = projected_stats.get(stat, 0) or 0

        expected = _prorated_expected(proj, actual_opp, proj_opp)

        if expected > 0 and counting_colored:
            ratio = actual / expected
            cv = float(negbin_perf_cv(stat, expected))  # expected > 0 guaranteed above
            z = (ratio - 1.0) / cv if cv > 0 else 0.0
        else:
            z = 0.0

        display_key = stat.upper()
        result[display_key] = {
            "actual": actual,
            "expected": round(expected, 1),
            "z_score": round(z, 2),
            "color_class": _z_to_color(z)
            if abs(actual - expected) >= COUNTING_MIN_ABS_DIFF
            else "stat-neutral",
            "projection": round(proj),
            "rest_of_season_deviation_sgp": _rest_of_season_deviation(display_key),
        }

    # Rate stats — always computed, but color suppressed below min_rates threshold
    rates_colored = actual_opp >= min_rates

    if player_type == PlayerType.HITTER:
        actual_h = actual_stats.get("h", 0) or 0
        actual_ab = actual_stats.get("ab", 0) or 0
        proj_avg = projected_stats.get("avg", 0.0) or 0.0

        actual_avg = round(calculate_avg(actual_h, actual_ab, default=0.0), 3)

        if proj_avg > 0 and actual_ab > 0 and rates_colored:
            # Binomial sampling SD: SD(hits/AB) = sqrt(p*(1-p)/AB).
            # This is the distribution of a sample mean of a Bernoulli trial
            # with probability proj_avg over actual_ab at-bats.
            sd = math.sqrt(proj_avg * (1.0 - proj_avg) / actual_ab)
            z = (actual_avg - proj_avg) / sd if sd > 0 else 0.0
        else:
            z = 0.0

        result["AVG"] = {
            "actual": actual_avg,
            "expected": proj_avg,
            "z_score": round(z, 2),
            "color_class": _z_to_color(z),
            "projection": proj_avg,
            "rest_of_season_deviation_sgp": _rest_of_season_deviation("AVG"),
        }

    else:  # pitcher
        actual_ip = actual_stats.get("ip", 0) or 0
        actual_er = actual_stats.get("er", 0) or 0
        actual_bb = actual_stats.get("bb", 0) or 0
        actual_ha = actual_stats.get("h_allowed", 0) or 0
        proj_era = projected_stats.get("era", 0.0) or 0.0
        proj_whip = projected_stats.get("whip", 0.0) or 0.0

        # ERA
        actual_era = round(calculate_era(actual_er, actual_ip, default=0.0), 2)
        if proj_era > 0 and actual_ip > 0 and rates_colored:
            # Poisson sampling SD on ER count: Var(ER) ≈ λ = proj_era*ip/9.
            # Dividing by ip/9 to convert count SD back to ERA units gives
            # SD(ERA) = sqrt(proj_era * 9 / ip).
            sd = math.sqrt(proj_era * 9.0 / actual_ip)
            z = (actual_era - proj_era) / sd if sd > 0 else 0.0
            z = -z  # inverse stat: lower is better
        else:
            z = 0.0

        result["ERA"] = {
            "actual": actual_era,
            "expected": proj_era,
            "z_score": round(z, 2),
            "color_class": _z_to_color(z),
            "projection": proj_era,
            "rest_of_season_deviation_sgp": _rest_of_season_deviation("ERA"),
        }

        # WHIP
        actual_whip = round(calculate_whip(actual_bb, actual_ha, actual_ip, default=0.0), 2)
        if proj_whip > 0 and actual_ip > 0 and rates_colored:
            # Poisson sampling SD on baserunners: Var(BB+H) ≈ λ = proj_whip*ip.
            # SD(WHIP) = sqrt(proj_whip / ip).
            sd = math.sqrt(proj_whip / actual_ip)
            z = (actual_whip - proj_whip) / sd if sd > 0 else 0.0
            z = -z  # inverse stat: lower is better
        else:
            z = 0.0

        result["WHIP"] = {
            "actual": actual_whip,
            "expected": proj_whip,
            "z_score": round(z, 2),
            "color_class": _z_to_color(z),
            "projection": proj_whip,
            "rest_of_season_deviation_sgp": _rest_of_season_deviation("WHIP"),
        }

    return result


def attach_pace_to_roster(
    players: list[Any],
    hitter_logs: dict[str, dict[str, Any]],
    pitcher_logs: dict[str, dict[str, Any]],
    preseason_lookup: dict[str, Any],
    sgp_denoms: dict[Any, float],
) -> None:
    """Attach a ``pace`` attribute to every player in ``players``.

    For each player, picks the right log dict (hitter_logs vs pitcher_logs)
    by player_type, builds projected stats from ``preseason_lookup`` (zero-
    filled if no preseason entry), pulls current ROS stats from the player
    if present, and calls ``compute_player_pace``. Mutates each player.
    """
    for player in players:
        norm = normalize_name(player.name)
        if player.player_type == PlayerType.HITTER:
            actuals = hitter_logs.get(norm, {})
            ros_keys = ["r", "hr", "rbi", "sb", "avg"]
            proj_keys = HITTER_PROJ_KEYS
        else:
            actuals = pitcher_logs.get(norm, {})
            ros_keys = ["w", "k", "sv", "era", "whip"]
            proj_keys = PITCHER_PROJ_KEYS
        pre_player = preseason_lookup.get(norm)
        if pre_player and pre_player.rest_of_season:
            projected = {k: getattr(pre_player.rest_of_season, k, 0) for k in proj_keys}
        else:
            projected = {k: 0 for k in proj_keys}
        # Pace deviation compares preseason expectations to the current
        # full-season expectation (=ROS-remaining + YTD-actuals), so this
        # reads full_season_projection rather than rest_of_season. The
        # local variable names retain "ros" terminology for diff
        # minimization; semantically the values are full-season totals.
        ros_dict = (
            {k: getattr(player.full_season_projection, k, 0) for k in ros_keys}
            if player.full_season_projection
            else None
        )
        player.pace = compute_player_pace(
            actuals,
            projected,
            player.player_type,
            rest_of_season_stats=ros_dict,
            sgp_denoms=sgp_denoms,
        )


def compute_overall_pace(
    sgp_summary: dict[str, Any] | None,
    cutpoints: list[float] | None,
) -> dict[str, Any]:
    """Bucket a player's cached SGP deviation against its pool cutpoints.

    ``sgp_summary`` is one entry from the ``PACE_DEVIATIONS`` deviations map
    ({"sgp_dev", "actual_sgp", "expected_sgp"}); ``cutpoints`` is
    ``[q16, q33, q66, q83]`` for the player's type (or None). Renders neutral
    when the deviation is undefined, cutpoints are missing, or the pool was
    too small (cutpoints None). The tooltip values pass through unchanged.
    """
    summary = sgp_summary if sgp_summary is not None else {}
    dev = summary.get("sgp_dev")
    result = {
        "color_class": "stat-neutral",
        "sgp_dev": dev,
        "actual_sgp": summary.get("actual_sgp"),
        "expected_sgp": summary.get("expected_sgp"),
    }
    if dev is None or not cutpoints:
        return result

    q16, q33, q66, q83 = cutpoints
    if dev >= q83:
        result["color_class"] = "stat-hot-2"
    elif dev >= q66:
        result["color_class"] = "stat-hot-1"
    elif dev >= q33:
        result["color_class"] = "stat-neutral"
    elif dev >= q16:
        result["color_class"] = "stat-cold-1"
    else:
        result["color_class"] = "stat-cold-2"
    return result


def compute_sgp_deviation(
    actual_stats: dict[str, Any],
    projected_stats: dict[str, Any],
    player_type: str,
    denoms: dict[Category, float],
) -> dict[str, Any]:
    """SGP delivered vs preseason-expected, prorated to actual playing time.

    Returns {"sgp_dev", "actual_sgp", "expected_sgp"} in roto-point (SGP)
    units. ``sgp_dev`` is None when the player has no games above the counting
    gate or no projection. Per-category value comes from ``sgp.player_value``'s
    ``calculate_*_sgp`` helpers (counting, hitting-rate, pitching-rate), scored
    over the player's actual playing time so the metric is rate-fair. The
    replacement baseline cancels in the delta but is kept in the returned
    ``actual_sgp`` / ``expected_sgp`` so the tooltip reads as value-over-
    replacement. Sample-size gates mirror ``compute_player_pace``.
    """
    none_result = {"sgp_dev": None, "actual_sgp": None, "expected_sgp": None}
    if not projected_stats:
        return none_result

    if player_type == PlayerType.HITTER:
        opp_key, counting = "pa", HITTER_COUNTING
        min_counting, min_rates = HITTER_MIN_COUNTING, HITTER_MIN_RATES
    else:
        opp_key, counting = "ip", PITCHER_COUNTING
        min_counting, min_rates = PITCHER_MIN_COUNTING, PITCHER_MIN_RATES

    actual_opp = actual_stats.get(opp_key, 0) or 0
    proj_opp = projected_stats.get(opp_key, 0) or 0
    # A degenerate preseason line (0 or NaN projected PA/IP -- the phantom-
    # projection trap in this codebase) has no basis for an "expected" value;
    # crediting full actuals against ~0 expected would inflate sgp_dev and
    # pollute the leaguewide cutpoints, so exclude the player. ``not proj_opp > 0``
    # rejects 0, negatives, and NaN (NaN > 0 is False).
    if actual_opp < min_counting or not proj_opp > 0:
        return none_result

    actual_sgp = 0.0
    expected_sgp = 0.0

    for stat in counting:
        denom = denoms.get(Category(stat.upper()))
        if not denom:
            continue
        actual = actual_stats.get(stat, 0) or 0
        expected = _prorated_expected(projected_stats.get(stat, 0) or 0, actual_opp, proj_opp)
        actual_sgp += calculate_counting_sgp(actual, denom)
        expected_sgp += calculate_counting_sgp(expected, denom)

    if actual_opp >= min_rates:
        if player_type == PlayerType.HITTER:
            actual_ab = actual_stats.get("ab", 0) or 0
            proj_avg = projected_stats.get("avg", 0.0) or 0.0
            denom = denoms.get(Category.AVG)
            # proj_avg > 0 mirrors compute_player_pace: a 0/None/NaN projected
            # rate has no valid baseline and would be scored against .000.
            if denom and actual_ab > 0 and proj_avg > 0:
                actual_avg = calculate_avg(actual_stats.get("h", 0) or 0, actual_ab, default=0.0)
                actual_sgp += calculate_hitting_rate_sgp(
                    actual_avg, actual_ab, REPLACEMENT_AVG, denom, DEFAULT_TEAM_AB
                )
                expected_sgp += calculate_hitting_rate_sgp(
                    proj_avg, actual_ab, REPLACEMENT_AVG, denom, DEFAULT_TEAM_AB
                )
        else:
            actual_ip = actual_stats.get("ip", 0) or 0
            if actual_ip > 0:
                era_denom = denoms.get(Category.ERA)
                proj_era = projected_stats.get("era", 0.0) or 0.0
                if era_denom and proj_era > 0:
                    actual_era = calculate_era(
                        actual_stats.get("er", 0) or 0, actual_ip, default=0.0
                    )
                    actual_sgp += calculate_pitching_rate_sgp(
                        actual_era, actual_ip, REPLACEMENT_ERA, era_denom, DEFAULT_TEAM_IP, 9
                    )
                    expected_sgp += calculate_pitching_rate_sgp(
                        proj_era, actual_ip, REPLACEMENT_ERA, era_denom, DEFAULT_TEAM_IP, 9
                    )
                whip_denom = denoms.get(Category.WHIP)
                proj_whip = projected_stats.get("whip", 0.0) or 0.0
                if whip_denom and proj_whip > 0:
                    actual_whip = calculate_whip(
                        actual_stats.get("bb", 0) or 0,
                        actual_stats.get("h_allowed", 0) or 0,
                        actual_ip,
                        default=0.0,
                    )
                    actual_sgp += calculate_pitching_rate_sgp(
                        actual_whip, actual_ip, REPLACEMENT_WHIP, whip_denom, DEFAULT_TEAM_IP, 1
                    )
                    expected_sgp += calculate_pitching_rate_sgp(
                        proj_whip, actual_ip, REPLACEMENT_WHIP, whip_denom, DEFAULT_TEAM_IP, 1
                    )

    sgp_dev = actual_sgp - expected_sgp
    if math.isnan(sgp_dev):
        # Belt-and-suspenders: a NaN would sort incorrectly into the leaguewide
        # cutpoints and paint the player bright red. The projection guards above
        # should prevent it; exclude defensively if one slips through.
        return none_result
    return {
        "sgp_dev": round(sgp_dev, 3),
        "actual_sgp": round(actual_sgp, 3),
        "expected_sgp": round(expected_sgp, 3),
    }


def compute_pace_cutpoints(devs: list[float]) -> list[float] | None:
    """Return [q16, q33, q66, q83] nearest-rank cutpoints for a pool of
    SGP deviations, or None when the pool is smaller than MIN_POOL_SIZE.

    Nearest-rank index = round(q * (n - 1)) over the ascending-sorted list.
    """
    if len(devs) < MIN_POOL_SIZE:
        return None
    ordered = sorted(devs)
    n = len(ordered)
    return [ordered[round(q * (n - 1))] for q in (1 / 6, 1 / 3, 2 / 3, 5 / 6)]


def pace_dev_key(name: str, player_type: str) -> str:
    """Key for the ``PACE_DEVIATIONS`` deviations map.

    ``normalize_name(name)::player_type`` -- the same normalized-name
    convention the game-log and ranking lookups use, so the pipeline writer
    and the two display readers share one format and cannot drift.
    """
    return f"{normalize_name(name)}::{player_type}"


def build_pace_deviation_payload(
    players: list[Any],
    hitter_logs: dict[str, dict[str, Any]],
    pitcher_logs: dict[str, dict[str, Any]],
    denoms: dict[Category, float],
) -> dict[str, Any]:
    """Compute the leaguewide SGP-deviation map + per-type cutpoints.

    Iterates rostered ``players``, builds each one's YTD actuals (from the
    game logs) and preseason projection (from the player's own
    ``.preseason``, attached by hydration to both the user roster and every
    opponent roster), calls :func:`compute_sgp_deviation`, keys the result by
    :func:`pace_dev_key`, and derives hitter/pitcher cutpoints over the
    players with a defined ``sgp_dev``.
    """
    deviations: dict[str, dict[str, Any]] = {}
    devs: dict[str, list[float]] = {"hitter": [], "pitcher": []}

    for player in players:
        norm = normalize_name(player.name)
        if player.player_type == PlayerType.HITTER:
            actuals = hitter_logs.get(norm, {})
            proj_keys = HITTER_PROJ_KEYS
        else:
            actuals = pitcher_logs.get(norm, {})
            proj_keys = PITCHER_PROJ_KEYS
        pre = player.preseason
        if pre is not None:
            projected = {k: getattr(pre, k, 0) for k in proj_keys}
        else:
            projected = {}

        summary = compute_sgp_deviation(actuals, projected, player.player_type, denoms)
        deviations[pace_dev_key(player.name, player.player_type.value)] = summary
        if summary["sgp_dev"] is not None:
            devs[player.player_type.value].append(summary["sgp_dev"])

    return {
        "deviations": deviations,
        "cutpoints": {t: compute_pace_cutpoints(d) for t, d in devs.items()},
    }
