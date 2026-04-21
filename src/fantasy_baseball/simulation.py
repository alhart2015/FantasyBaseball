"""Monte Carlo season simulation with injuries and stat variance.

Shared by scripts/simulate_draft.py (post-draft --monte-carlo),
scripts/summary.py (in-season weekly projections), and
the season dashboard (web/season_data.py).
"""

import numpy as np

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES as ALL_CATS,
)
from fantasy_baseball.utils.constants import (
    CLOSER_SV_THRESHOLD,
    HITTER_CORR_STATS,
    HITTER_CORRELATION,
    HITTING_COUNTING,
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
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip


def _build_cov_matrix(
    stats: list[str],
    correlation: list[list[float]],
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
    team_stats: dict[str, dict[str, float]] = {}
    injuries: dict[str, list[tuple[str, float]]] = {}

    for team_key, players in team_rosters.items():
        hitters = [p for p in players if p.get("player_type") == PlayerType.HITTER]
        pitchers = [p for p in players if p.get("player_type") == PlayerType.PITCHER]
        team_injuries: list[tuple[str, float]] = []

        adj_hitters = _apply_variance(
            hitters,
            PlayerType.HITTER,
            rng,
            team_injuries,
        )
        adj_pitchers = _apply_variance(
            pitchers,
            PlayerType.PITCHER,
            rng,
            team_injuries,
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
            "AVG": calculate_avg(total_h, total_ab),
            "W": sum(p["w"] for p in active_p),
            "K": sum(p["k"] for p in active_p),
            "SV": sum(p.get("sv", 0) for p in active_p),
            "ERA": calculate_era(total_er, total_ip),
            "WHIP": calculate_whip(total_bb, total_ha, total_ip),
        }
        injuries[team_key] = team_injuries

    return team_stats, injuries


# Typical full-season team totals for blending actual + simulated rate stats.
_TYPICAL_TEAM_AB = 5500
_TYPICAL_TEAM_IP = 1450


def simulate_remaining_season(
    actual_standings: dict[str, dict[str, float]],
    team_rosters: dict,
    fraction_remaining: float,
    rng: np.random.Generator,
    h_slots: int = 13,
    p_slots: int = 9,
) -> tuple[dict, dict]:
    """Simulate only the remaining portion of the season and blend with YTD actuals.

    Player ROS projections in `ros_blended_projections` are normalized to
    full-season totals at fetch time (see normalize_rest_of_season_to_full_season in
    data/projections.py). FanGraphs publishes remaining-games-only projections
    for all systems; the pipeline adds accumulated actuals to convert them to
    full-season totals before blending.

    To simulate only the remainder:

    1. Apply variance to full-season player projections (covariance scaled
       by fraction_remaining so uncertainty shrinks as season progresses).
    2. Aggregate to team-level simulated full-season totals.
    3. Subtract actual YTD stats to get simulated remaining production
       (clamped to >= 0 so hot starts are preserved).
    4. Final = actual + remaining.

    For rate stats, component stats (H, AB, IP, ER, etc.) are estimated
    from the actual rates and blended with simulated remaining components.

    Args:
        actual_standings: {team_key: {R, HR, RBI, SB, AVG, W, K, SV, ERA, WHIP}}
            — actual YTD stats for each team.
        team_rosters: {team_key: [player dicts]} with full-season ROS projections.
        fraction_remaining: Float 0.0–1.0, portion of season left to simulate.
        rng: NumPy random generator for reproducibility.
        h_slots: Number of active hitter slots.
        p_slots: Number of active pitcher slots.

    Returns:
        Tuple of (team_stats, injuries):
        - team_stats: {team_key: {R, HR, RBI, SB, AVG, W, K, SV, ERA, WHIP}}
        - injuries: {team_key: [(player_name, frac_missed), ...]}
    """
    team_stats: dict[str, dict[str, float]] = {}
    injuries: dict[str, list[tuple[str, float]]] = {}

    # Season over — just return actuals, no simulation needed
    if fraction_remaining <= 0:
        for team_key in team_rosters:
            team_stats[team_key] = dict(actual_standings.get(team_key, {}))
            injuries[team_key] = []
        return team_stats, injuries

    for team_key, players in team_rosters.items():
        actuals = actual_standings.get(team_key, {})
        hitters = [p for p in players if p.get("player_type") == PlayerType.HITTER]
        pitchers = [p for p in players if p.get("player_type") == PlayerType.PITCHER]
        team_injuries: list[tuple[str, float]] = []

        # Apply variance to full-season projections (covariance scaled down)
        adj_hitters = _apply_variance(
            hitters,
            PlayerType.HITTER,
            rng,
            team_injuries,
            fraction_remaining,
        )
        adj_pitchers = _apply_variance(
            pitchers,
            PlayerType.PITCHER,
            rng,
            team_injuries,
            fraction_remaining,
        )

        # Select active roster (bench excluded) — same logic as simulate_season
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

        # Aggregate simulated full-season team totals
        sim_r = sum(h["r"] for h in active_h)
        sim_hr = sum(h["hr"] for h in active_h)
        sim_rbi = sum(h["rbi"] for h in active_h)
        sim_sb = sum(h["sb"] for h in active_h)
        sim_h = sum(h["h"] for h in active_h)
        sim_ab = sum(h["ab"] for h in active_h)
        sim_w = sum(p["w"] for p in active_p)
        sim_k = sum(p["k"] for p in active_p)
        sim_sv = sum(p.get("sv", 0) for p in active_p)
        sim_ip = sum(p["ip"] for p in active_p)
        sim_er = sum(p["er"] for p in active_p)
        sim_bb = sum(p["bb"] for p in active_p)
        sim_ha = sum(p["h_allowed"] for p in active_p)

        # Subtract actuals to get simulated remaining (clamped to >= 0)
        rem_r = max(0, sim_r - actuals.get("R", 0))
        rem_hr = max(0, sim_hr - actuals.get("HR", 0))
        rem_rbi = max(0, sim_rbi - actuals.get("RBI", 0))
        rem_sb = max(0, sim_sb - actuals.get("SB", 0))
        rem_w = max(0, sim_w - actuals.get("W", 0))
        rem_k = max(0, sim_k - actuals.get("K", 0))
        rem_sv = max(0, sim_sv - actuals.get("SV", 0))

        # Estimate actual component stats from rate stats for blending
        fraction_elapsed = 1.0 - fraction_remaining
        actual_ab = _TYPICAL_TEAM_AB * fraction_elapsed
        actual_ip = _TYPICAL_TEAM_IP * fraction_elapsed
        actual_h = actuals.get("AVG", 0) * actual_ab
        actual_er = actuals.get("ERA", 0) * actual_ip / 9
        actual_h_plus_bb = actuals.get("WHIP", 0) * actual_ip

        # Remaining components: simulated full-season minus actual
        rem_ab = max(0, sim_ab - actual_ab)
        rem_h = max(0, sim_h - actual_h)
        rem_ip = max(0, sim_ip - actual_ip)
        rem_er = max(0, sim_er - actual_er)
        rem_h_plus_bb = max(0, (sim_bb + sim_ha) - (actual_h_plus_bb))

        # Final = actual + remaining
        total_ab = actual_ab + rem_ab
        total_h = actual_h + rem_h
        total_ip = actual_ip + rem_ip
        total_er = actual_er + rem_er
        total_h_plus_bb = actual_h_plus_bb + rem_h_plus_bb

        team_stats[team_key] = {
            "R": actuals.get("R", 0) + rem_r,
            "HR": actuals.get("HR", 0) + rem_hr,
            "RBI": actuals.get("RBI", 0) + rem_rbi,
            "SB": actuals.get("SB", 0) + rem_sb,
            "AVG": calculate_avg(total_h, total_ab),
            "W": actuals.get("W", 0) + rem_w,
            "K": actuals.get("K", 0) + rem_k,
            "SV": actuals.get("SV", 0) + rem_sv,
            "ERA": calculate_era(total_er, total_ip),
            "WHIP": total_h_plus_bb / total_ip if total_ip > 0 else 99,
        }
        injuries[team_key] = team_injuries

    return team_stats, injuries


def _apply_variance(
    players: list,
    player_type: str,
    rng: np.random.Generator,
    injuries_out: list,
    fraction_remaining: float = 1.0,
) -> list[dict]:
    """Apply injury and correlated performance variance to a list of players.

    Uses multivariate normal draws so that correlated stats (e.g. HR and RBI)
    move together realistically, while less-correlated stats (e.g. HR and SB)
    vary more independently. Correlations and per-stat sigmas are calibrated
    from historical projection-vs-actual residuals (2022-2024).

    When fraction_remaining < 1.0, injury probability and covariance are scaled
    down proportionally (less variance and injury risk for a partial season).

    Mutates injuries_out by appending (name, frac_missed) for injured players.
    """
    is_hitter = player_type == PlayerType.HITTER
    counting_cols = HITTING_COUNTING if is_hitter else PITCHING_COUNTING
    injury_prob = INJURY_PROB[player_type] * fraction_remaining
    injury_lo, injury_hi = INJURY_SEVERITY[player_type]
    base_cov = HITTER_COV if is_hitter else PITCHER_COV
    cov = base_cov * fraction_remaining
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
            team,
            MANAGEMENT_ADJUSTMENT_DEFAULT,
        )
        draw = rng.normal(mean, sd)
        factor = 1.0 + draw * ROTO_TO_STAT

        # Back out rate stat components, adjust, and recompute.
        # Management affects quality (hits per AB, runs per IP) with
        # volume held constant, so only numerators are scaled.
        h = stats["AVG"] * _TYPICAL_TEAM_AB
        er = stats["ERA"] * _TYPICAL_TEAM_IP / 9
        bh = stats["WHIP"] * _TYPICAL_TEAM_IP

        adjusted[team] = {
            # Counting stats: scale with management quality
            "R": stats["R"] * factor,
            "HR": stats["HR"] * factor,
            "RBI": stats["RBI"] * factor,
            "SB": stats["SB"] * factor,
            "W": stats["W"] * factor,
            "K": stats["K"] * factor,
            "SV": stats["SV"] * factor,
            # Rate stats: adjust through components
            "AVG": calculate_avg(h * factor, _TYPICAL_TEAM_AB),
            "ERA": calculate_era(er / factor, _TYPICAL_TEAM_IP),
            "WHIP": calculate_whip(0, bh / factor, _TYPICAL_TEAM_IP),
        }
    return adjusted


def run_monte_carlo(
    team_rosters: dict,
    h_slots: int,
    p_slots: int,
    user_team_name: str,
    n_iterations: int = 1000,
    use_management: bool = False,
    seed: int = 42,
    progress_cb=None,
) -> dict:
    """Run a Monte Carlo simulation and return structured results.

    Args:
        team_rosters: {team_name: [player Series/dicts]} for all teams.
        h_slots: Number of active hitter slots.
        p_slots: Number of active pitcher slots.
        user_team_name: Name of user's team (for category risk).
        n_iterations: Number of simulation iterations.
        use_management: If True, apply management adjustment after each sim.
        seed: RNG seed for reproducibility.
        progress_cb: Optional callback(msg: str) called every 200 iterations.

    Returns:
        {"team_results": {team: {median_pts, p10, p90, first_pct, top3_pct}},
         "category_risk": {cat: {median_pts, p10, p90, top3_pct, bot3_pct}}}
    """
    # Convert Player objects to flat dicts for the simulation internals.
    # The internal simulation engine (simulate_season, _apply_variance)
    # works with flat stat dicts; this conversion preserves that while
    # giving callers a typed list[Player] API.
    flat_rosters: dict[str, list[dict]] = {}
    for team_key, players in team_rosters.items():
        flat_rosters[team_key] = [
            p.to_flat_dict() if hasattr(p, "to_flat_dict") else p for p in players
        ]

    rng = np.random.default_rng(seed)
    team_names = list(flat_rosters.keys())

    all_totals: dict[str, list[float]] = {name: [] for name in team_names}
    mc_wins = {name: 0 for name in team_names}
    mc_top3 = {name: 0 for name in team_names}
    user_cat_pts: dict[str, list[float]] = {c.value: [] for c in ALL_CATS}

    for i in range(n_iterations):
        if progress_cb and i % 200 == 0:
            progress_cb(i)
        sim_stats, _ = simulate_season(flat_rosters, rng, h_slots, p_slots)
        if use_management:
            sim_stats = apply_management_adjustment(sim_stats, rng)
        sim_roto = score_roto(sim_stats)
        ranked = sorted(sim_roto.items(), key=lambda x: x[1]["total"], reverse=True)
        for rank, (name, pts) in enumerate(ranked, 1):
            all_totals[name].append(pts["total"])
            if rank == 1:
                mc_wins[name] += 1
            if rank <= 3:
                mc_top3[name] += 1
            if name == user_team_name:
                for c in ALL_CATS:
                    user_cat_pts[c.value].append(pts.get(f"{c.value}_pts", 0))

    n = n_iterations
    team_results = {}
    for name in team_names:
        arr = np.array(all_totals[name])
        team_results[name] = {
            "median_pts": round(float(np.median(arr)), 1),
            "p10": round(float(np.percentile(arr, 10))),
            "p90": round(float(np.percentile(arr, 90))),
            "first_pct": round(mc_wins[name] / n * 100, 1),
            "top3_pct": round(mc_top3[name] / n * 100, 1),
        }

    category_risk = {}
    for c in ALL_CATS:
        arr = np.array(user_cat_pts[c.value])
        category_risk[c.value] = {
            "median_pts": round(float(np.median(arr)), 1),
            "p10": round(float(np.percentile(arr, 10)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "top3_pct": round(float((arr >= 8).sum()) / n * 100, 1),
            "bot3_pct": round(float((arr <= 3).sum()) / n * 100, 1),
        }

    return {"team_results": team_results, "category_risk": category_risk}


def run_ros_monte_carlo(
    team_rosters: dict,
    actual_standings: dict[str, dict[str, float]],
    fraction_remaining: float,
    h_slots: int,
    p_slots: int,
    user_team_name: str,
    n_iterations: int = 1000,
    use_management: bool = False,
    seed: int = 42,
    progress_cb=None,
) -> dict:
    """Run a Monte Carlo simulation over the remaining season.

    Like run_monte_carlo but uses simulate_remaining_season to blend
    actual YTD stats with simulated ROS projections.

    Args:
        team_rosters: {team_name: [player dicts]} with ROS projections.
        actual_standings: {team_name: {R, HR, RBI, SB, AVG, W, K, SV, ERA, WHIP}}
            — actual YTD stats for each team.
        fraction_remaining: Float 0.0–1.0, portion of season left.
        h_slots: Number of active hitter slots.
        p_slots: Number of active pitcher slots.
        user_team_name: Name of user's team (for category risk).
        n_iterations: Number of simulation iterations.
        use_management: If True, apply management adjustment after each sim.
        seed: RNG seed for reproducibility.
        progress_cb: Optional callback(msg: str) called every 200 iterations.

    Returns:
        {"team_results": {team: {median_pts, p10, p90, first_pct, top3_pct}},
         "category_risk": {cat: {median_pts, p10, p90, top3_pct, bot3_pct}}}
    """
    # Convert Player objects to flat dicts for the simulation internals.
    flat_rosters: dict[str, list[dict]] = {}
    for team_key, players in team_rosters.items():
        flat_rosters[team_key] = [
            p.to_flat_dict() if hasattr(p, "to_flat_dict") else p for p in players
        ]

    rng = np.random.default_rng(seed)
    team_names = list(flat_rosters.keys())

    all_totals: dict[str, list[float]] = {name: [] for name in team_names}
    mc_wins = {name: 0 for name in team_names}
    mc_top3 = {name: 0 for name in team_names}
    user_cat_pts: dict[str, list[float]] = {c.value: [] for c in ALL_CATS}

    for i in range(n_iterations):
        if progress_cb and i % 200 == 0:
            progress_cb(i)
        sim_stats, _ = simulate_remaining_season(
            actual_standings,
            flat_rosters,
            fraction_remaining,
            rng,
            h_slots,
            p_slots,
        )
        if use_management:
            sim_stats = apply_management_adjustment(sim_stats, rng)
        sim_roto = score_roto(sim_stats)
        ranked = sorted(sim_roto.items(), key=lambda x: x[1]["total"], reverse=True)
        for rank, (name, pts) in enumerate(ranked, 1):
            all_totals[name].append(pts["total"])
            if rank == 1:
                mc_wins[name] += 1
            if rank <= 3:
                mc_top3[name] += 1
            if name == user_team_name:
                for c in ALL_CATS:
                    user_cat_pts[c.value].append(pts.get(f"{c.value}_pts", 0))

    n = n_iterations
    team_results = {}
    for name in team_names:
        arr = np.array(all_totals[name])
        team_results[name] = {
            "median_pts": round(float(np.median(arr)), 1),
            "p10": round(float(np.percentile(arr, 10))),
            "p90": round(float(np.percentile(arr, 90))),
            "first_pct": round(mc_wins[name] / n * 100, 1),
            "top3_pct": round(mc_top3[name] / n * 100, 1),
        }

    category_risk = {}
    for c in ALL_CATS:
        arr = np.array(user_cat_pts[c.value])
        category_risk[c.value] = {
            "median_pts": round(float(np.median(arr)), 1),
            "p10": round(float(np.percentile(arr, 10)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "top3_pct": round(float((arr >= 8).sum()) / n * 100, 1),
            "bot3_pct": round(float((arr <= 3).sum()) / n * 100, 1),
        }

    return {"team_results": team_results, "category_risk": category_risk}
