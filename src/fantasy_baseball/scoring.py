"""Roto scoring and team stat projection — shared across all modules.

Provides two core functions:
- project_team_stats: sum projected stats for a roster into roto categories
- score_roto: assign roto points (1-N) with fractional tie-breaking
"""

ALL_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
INVERSE_CATS = {"ERA", "WHIP"}


def project_team_stats(roster: list[dict]) -> dict[str, float]:
    """Sum projected stats for a roster into roto category totals.

    Each player dict must have 'player_type' ("hitter" or "pitcher") and
    the relevant stat keys (lowercase: r, hr, rbi, sb, h, ab for hitters;
    w, k, sv, ip, er, bb, h_allowed for pitchers).

    Rate stats (AVG, ERA, WHIP) are computed from component totals.
    """
    r = hr = rbi = sb = h_total = ab_total = 0.0
    w = k = sv = ip_total = er_total = bb_total = ha_total = 0.0

    for p in roster:
        if p.get("player_type") == "hitter":
            r += p.get("r", 0)
            hr += p.get("hr", 0)
            rbi += p.get("rbi", 0)
            sb += p.get("sb", 0)
            h_total += p.get("h", 0)
            ab_total += p.get("ab", 0)
        elif p.get("player_type") == "pitcher":
            w += p.get("w", 0)
            k += p.get("k", 0)
            sv += p.get("sv", 0)
            ip_total += p.get("ip", 0)
            er_total += p.get("er", 0)
            bb_total += p.get("bb", 0)
            ha_total += p.get("h_allowed", 0)

    return {
        "R": r, "HR": hr, "RBI": rbi, "SB": sb,
        "AVG": h_total / ab_total if ab_total > 0 else 0,
        "W": w, "K": k, "SV": sv,
        "ERA": er_total * 9 / ip_total if ip_total > 0 else 99,
        "WHIP": (bb_total + ha_total) / ip_total if ip_total > 0 else 99,
    }


def score_roto(
    all_team_stats: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Assign roto points with fractional tie-breaking.

    Args:
        all_team_stats: {team_name: {cat: value}} for all teams.

    Returns:
        {team_name: {cat_pts: float, ..., "total": float}} where
        cat_pts keys are "R_pts", "HR_pts", etc.  Points range from
        1 (worst) to N (best) for N teams.
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
