"""Monte Carlo season simulation with injuries and stat variance.

Shared by scripts/simulate_draft.py (post-draft --monte-carlo) and
scripts/summary.py (in-season weekly projections).
"""

import numpy as np

from fantasy_baseball.utils.constants import (
    CLOSER_SV_THRESHOLD,
    HITTING_COUNTING,
    HITTER_CORR_STATS,
    HITTER_CORRELATION,
    INJURY_PROB,
    INJURY_SEVERITY,
    MANAGEMENT_ADJUSTMENT,
    MANAGEMENT_ADJUSTMENT_DEFAULT,
    PITCHER_CORR_STATS,
    PITCHER_CORRELATION,
    PITCHING_COUNTING,
    REPLACEMENT_HITTER,
    REPLACEMENT_RP,
    REPLACEMENT_SP,
    STAT_VARIANCE,
)


def _build_cov_matrix(
    stats: list[str], correlation: list[list[float]],
) -> np.ndarray:
    """Build a covariance matrix from per-stat sigmas and a correlation matrix.

    cov[i][j] = corr[i][j] * sigma_i * sigma_j
    """
    n = len(stats)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            si = STAT_VARIANCE.get(stats[i], 0.0)
            sj = STAT_VARIANCE.get(stats[j], 0.0)
            cov[i, j] = correlation[i][j] * si * sj
    return cov


# Pre-compute covariance matrices at import time (they never change).
HITTER_COV = _build_cov_matrix(HITTER_CORR_STATS, HITTER_CORRELATION)
PITCHER_COV = _build_cov_matrix(PITCHER_CORR_STATS, PITCHER_CORRELATION)

# Map stat name -> index in the correlated draw vector.
HITTER_IDX = {s: i for i, s in enumerate(HITTER_CORR_STATS)}
PITCHER_IDX = {s: i for i, s in enumerate(PITCHER_CORR_STATS)}


def simulate_season(
    team_rosters: dict,
    rng: np.random.Generator,
    h_slots: int = 13,
    p_slots: int = 9,
) -> tuple[dict, dict]:
    """Run one simulated season with injuries and stat variance.

    For each team, applies random injuries (probability-based) and
    performance variance to every player, then selects the best
    h_slots hitters and p_slots pitchers as the active roster.
    Bench players don't contribute stats.

    Args:
        team_rosters: Dict of team_key -> list of player dicts/Series.
            Each player must have ``player_type`` ("hitter"/"pitcher")
            and the stat columns defined in constants (r, hr, rbi, etc.).
        rng: NumPy random generator for reproducibility.
        h_slots: Number of active hitter slots.
        p_slots: Number of active pitcher slots.

    Returns:
        Tuple of (team_stats, injuries):
        - team_stats: {team_key: {R, HR, RBI, SB, AVG, W, K, SV, ERA, WHIP}}
        - injuries: {team_key: [(player_name, frac_missed), ...]}
    """
    team_stats = {}
    injuries = {}

    for team_key, players in team_rosters.items():
        hitters = [p for p in players if p.get("player_type") == "hitter"]
        pitchers = [p for p in players if p.get("player_type") == "pitcher"]
        team_injuries = []

        adj_hitters = _apply_variance(
            hitters, "hitter", rng, team_injuries,
        )
        adj_pitchers = _apply_variance(
            pitchers, "pitcher", rng, team_injuries,
        )

        # Select active roster (bench excluded)
        adj_hitters.sort(
            key=lambda h: h["r"] + h["hr"] + h["rbi"] + h["sb"],
            reverse=True,
        )
        adj_pitchers.sort(
            key=lambda p: (
                p.get("sv", 0) >= CLOSER_SV_THRESHOLD,
                p["w"] + p["k"] + p.get("sv", 0),
            ),
            reverse=True,
        )
        active_h = adj_hitters[:h_slots]
        active_p = adj_pitchers[:p_slots]

        # Aggregate team stats
        total_ab = sum(h["ab"] for h in active_h)
        total_h = sum(h["h"] for h in active_h)
        total_ip = sum(p["ip"] for p in active_p)
        total_er = sum(p["er"] for p in active_p)
        total_bb = sum(p["bb"] for p in active_p)
        total_ha = sum(p["h_allowed"] for p in active_p)

        team_stats[team_key] = {
            "R": sum(h["r"] for h in active_h),
            "HR": sum(h["hr"] for h in active_h),
            "RBI": sum(h["rbi"] for h in active_h),
            "SB": sum(h["sb"] for h in active_h),
            "AVG": total_h / total_ab if total_ab > 0 else 0,
            "W": sum(p["w"] for p in active_p),
            "K": sum(p["k"] for p in active_p),
            "SV": sum(p.get("sv", 0) for p in active_p),
            "ERA": total_er * 9 / total_ip if total_ip > 0 else 99,
            "WHIP": (total_bb + total_ha) / total_ip if total_ip > 0 else 99,
        }
        injuries[team_key] = team_injuries

    return team_stats, injuries


def _apply_variance(
    players: list,
    player_type: str,
    rng: np.random.Generator,
    injuries_out: list,
) -> list[dict]:
    """Apply injury and correlated performance variance to a list of players.

    Uses multivariate normal draws so that correlated stats (e.g. HR and RBI)
    move together realistically, while less-correlated stats (e.g. HR and SB)
    vary more independently. Correlations and per-stat sigmas are calibrated
    from historical projection-vs-actual residuals (2022-2024).

    Mutates injuries_out by appending (name, frac_missed) for injured players.
    """
    is_hitter = player_type == "hitter"
    counting_cols = HITTING_COUNTING if is_hitter else PITCHING_COUNTING
    injury_prob = INJURY_PROB[player_type]
    injury_lo, injury_hi = INJURY_SEVERITY[player_type]
    cov = HITTER_COV if is_hitter else PITCHER_COV
    idx_map = HITTER_IDX if is_hitter else PITCHER_IDX
    n_corr = len(idx_map)
    mean = np.zeros(n_corr)

    n = len(players)
    if n == 0:
        return []

    # Batch all random draws for the entire player list at once
    injury_rolls = rng.random(n)
    injury_severities = rng.uniform(injury_lo, injury_hi, n)
    all_draws = rng.multivariate_normal(mean, cov, size=n)

    adjusted = []
    for i, p in enumerate(players):
        frac_missed = 0.0
        if injury_rolls[i] < injury_prob:
            frac_missed = injury_severities[i]
            injuries_out.append((p.get("name", "?"), frac_missed))

        scale = 1.0 - frac_missed

        if is_hitter:
            repl = REPLACEMENT_HITTER
        else:
            is_closer = p.get("sv", 0) >= CLOSER_SV_THRESHOLD
            repl = REPLACEMENT_RP if is_closer else REPLACEMENT_SP

        draws = all_draws[i]
        row = {}
        for col in counting_cols:
            base = float(p.get(col, 0) or 0)
            repl_contrib = repl.get(col, 0) * frac_missed

            if col in idx_map:
                perf = max(0, 1.0 + draws[idx_map[col]])
                row[col] = base * perf * scale + repl_contrib
            else:
                row[col] = base * scale + repl_contrib

        row["name"] = p.get("name", "?")
        row["player_type"] = player_type
        adjusted.append(row)

    return adjusted


def apply_management_adjustment(
    team_stats: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Scale team stats by in-season management quality before roto scoring.

    For each team, draws a management factor from a normal distribution
    calibrated from historical draft-to-finish performance (2023-2025).
    Good managers (positive adjustment) get a stat boost representing
    waiver pickups, trades, and streaming. Bad managers get a decline.

    Counting stats are scaled by the factor. Rate stats (AVG, ERA, WHIP)
    are adjusted in the favorable direction for positive management.
    The result feeds into score_roto(), which naturally preserves the
    zero-sum roto point constraint.

    Args:
        team_stats: Output of simulate_season() —
            {team: {R, HR, RBI, SB, AVG, W, K, SV, ERA, WHIP}}.
        rng: NumPy random generator.

    Returns:
        New team_stats dict with management-adjusted stat values.
    """
    # Empirically derived: 1 roto point of management quality
    # corresponds to ~0.24% change in counting stats.
    ROTO_TO_STAT = 0.00236

    adjusted = {}
    for team, stats in team_stats.items():
        mean, sd = MANAGEMENT_ADJUSTMENT.get(
            team, MANAGEMENT_ADJUSTMENT_DEFAULT,
        )
        draw = rng.normal(mean, sd)
        factor = 1.0 + draw * ROTO_TO_STAT
        inv_factor = max(0.0, 2.0 - factor)

        adjusted[team] = {
            # Counting stats: scale with management quality
            "R": stats["R"] * factor,
            "HR": stats["HR"] * factor,
            "RBI": stats["RBI"] * factor,
            "SB": stats["SB"] * factor,
            "W": stats["W"] * factor,
            "K": stats["K"] * factor,
            "SV": stats["SV"] * factor,
            # Rate stats: good management improves AVG, lowers ERA/WHIP
            "AVG": stats["AVG"] * factor,
            "ERA": stats["ERA"] * inv_factor,
            "WHIP": stats["WHIP"] * inv_factor,
        }
    return adjusted
