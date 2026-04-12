"""Roto scoring and team stat projection — shared across all modules.

Provides two core functions:
- project_team_stats: sum projected stats for a roster into a
  CategoryStats. Accepts Player dataclass objects OR flat dicts for
  backwards compatibility with draft/script callers that still build
  rosters as plain dicts.
- score_roto: assign roto points (1-N) with fractional tie-breaking
"""

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.standings import CategoryStats
from fantasy_baseball.utils.constants import ALL_CATEGORIES as ALL_CATS  # noqa: F401
from fantasy_baseball.utils.constants import INVERSE_STATS as INVERSE_CATS  # noqa: F401
from fantasy_baseball.utils.constants import safe_float as _safe
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip


def _get(p, key, default=0):
    """Read a field from a Player dataclass or a plain dict."""
    if hasattr(p, key):
        return getattr(p, key)
    if isinstance(p, dict):
        return p.get(key, default)
    return default


def _stat(p, key):
    """Read a stat from a Player's ROS stats or from a flat dict."""
    # Player dataclass: stats live on the .ros attribute
    ros = getattr(p, "ros", None)
    if ros is not None and hasattr(ros, key):
        return _safe(getattr(ros, key, 0))
    # Flat dict (legacy callers, tests, draft scripts)
    if isinstance(p, dict):
        return _safe(p.get(key, 0))
    return 0.0


def project_team_stats(roster) -> CategoryStats:
    """Sum projected stats for a roster into a CategoryStats.

    Accepts Player dataclass objects OR plain dicts with flat stat
    keys. Rate stats (AVG, ERA, WHIP) are computed from component
    totals rather than simple sums, so the result is mathematically
    correct rather than just a naive average.

    The dict-input path exists for backwards compatibility with
    draft-side scripts (``scripts/simulate_draft.py``,
    ``scripts/summary.py``) that build rosters as plain dicts. Those
    scripts are explicitly out of scope for the League data model
    refactor and would need significant rework to use Player objects.
    Step 9 cleanup can revisit.
    """
    r = hr = rbi = sb = h_total = ab_total = 0.0
    w = k = sv = ip_total = er_total = bb_total = ha_total = 0.0

    for p in roster:
        ptype = _get(p, "player_type")
        if ptype == PlayerType.HITTER:
            r += _stat(p, "r")
            hr += _stat(p, "hr")
            rbi += _stat(p, "rbi")
            sb += _stat(p, "sb")
            h_total += _stat(p, "h")
            ab_total += _stat(p, "ab")
        elif ptype == PlayerType.PITCHER:
            w += _stat(p, "w")
            k += _stat(p, "k")
            sv += _stat(p, "sv")
            ip_total += _stat(p, "ip")
            er_total += _stat(p, "er")
            bb_total += _stat(p, "bb")
            ha_total += _stat(p, "h_allowed")

    return CategoryStats(
        r=r, hr=hr, rbi=rbi, sb=sb,
        avg=calculate_avg(h_total, ab_total),
        w=w, k=k, sv=sv,
        era=calculate_era(er_total, ip_total),
        whip=calculate_whip(bb_total, ha_total, ip_total),
    )


def score_roto(
    all_team_stats: dict,
) -> dict[str, dict[str, float]]:
    """Assign roto points with fractional tie-breaking.

    Args:
        all_team_stats: ``{team_name: stats}`` for all teams. Each
            ``stats`` value can be either a plain ``dict[str, float]``
            (legacy callers in draft/trade code) or a
            :class:`CategoryStats` instance (callers that went through
            ``project_team_stats``). Both shapes support ``[cat]``
            indexing, which is all this function needs.

    Returns:
        ``{team_name: {cat_pts: float, ..., "total": float}}`` where
        ``cat_pts`` keys are ``"R_pts"``, ``"HR_pts"``, etc. Points
        range from 1 (worst) to N (best) for N teams.
    """
    teams = list(all_team_stats.keys())
    n = len(teams)
    results: dict[str, dict[str, float]] = {t: {} for t in teams}

    for cat in ALL_CATS:
        rev = cat not in INVERSE_CATS
        ranked = sorted(teams, key=lambda t: all_team_stats[t][cat], reverse=rev)
        i = 0
        while i < n:
            j = i + 1
            while j < n and abs(all_team_stats[ranked[j]][cat] - all_team_stats[ranked[i]][cat]) < 1e-9:
                j += 1
            avg_pts = sum(n - k for k in range(i, j)) / (j - i)
            for k in range(i, j):
                results[ranked[k]][f"{cat}_pts"] = avg_pts
            i = j

    for t in results:
        results[t]["total"] = sum(results[t].get(f"{c}_pts", 0) for c in ALL_CATS)

    return results
