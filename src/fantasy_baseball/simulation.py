"""Monte Carlo season simulation with playing-time and stat variance.

Shared by scripts/simulate_draft.py (post-draft --monte-carlo),
scripts/summary.py (in-season weekly projections), and
the season dashboard (web/season_data.py).
"""

from typing import Any, cast

import numpy as np

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.scoring import score_roto_dict
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.constants import (
    AB_PER_PA,
    CLOSER_SV_THRESHOLD,
    HITTER_CORR_STATS,
    HITTER_CORRELATION,
    HITTING_COUNTING,
    MANAGEMENT_ADJUSTMENT,
    MANAGEMENT_ADJUSTMENT_DEFAULT,
    PITCHER_CORR_STATS,
    PITCHER_CORRELATION,
    PITCHING_COUNTING,
    REPLACEMENT_BY_POSITION,
    STAT_VARIANCE,
    role_from_ip,
    safe_float,
)
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES as ALL_CATS,
)
from fantasy_baseball.utils.playing_time import (
    playing_time_params,
    playing_time_shape,
    scale_from_uniform,
)
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

# Minimum simulated playing-time loss worth logging as a notable absence (vs a
# routine few-percent shortfall), since the playing-time scale is continuous.
_NOTABLE_PT_LOSS = 0.15


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


def _flatten_full_season(p: Any) -> dict[str, Any]:
    """Flatten a player to a dict with full-season counting stats at top level.

    Player objects delegate to ``to_flat_dict_full_season``. Dict inputs with
    nested ``full_season_projection`` get it overlaid onto the top level;
    legacy dicts with only flat top-level stats pass through unchanged.
    """
    if hasattr(p, "to_flat_dict_full_season"):
        result: dict[str, Any] = p.to_flat_dict_full_season()
        return result
    fs = p.get("full_season_projection")
    if isinstance(fs, dict):
        return {**p, **fs}
    return cast(dict[str, Any], p)


def simulate_season(
    team_rosters: dict,
    rng: np.random.Generator,
    h_slots: int = 13,
    p_slots: int = 9,
) -> tuple[dict, dict]:
    """Run one simulated season with playing-time and stat variance.

    For each team, applies a two-sided playing-time multiplier and
    correlated performance variance to every player, then selects the best
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


# Fallback full-season team totals for the actual+simulated rate-stat blend,
# used only when a team's real accumulated AB/IP aren't supplied. Real values
# (threaded from Yahoo standings `extras`) are preferred -- these constants are
# a last resort and 1450 IP runs high for a 9-pitcher league.
_TYPICAL_TEAM_AB = 5500
_TYPICAL_TEAM_IP = 1450


def _ytd_playing_time(
    actuals: dict[str, float],
    fraction_elapsed: float,
) -> tuple[float, float]:
    """Return (actual_ab, actual_ip) for the YTD-vs-remaining blend.

    Prefers the team's real accumulated AB/IP when present in ``actuals``
    (the season-dashboard threads them in from Yahoo standings ``extras``;
    AB is derived from real PA at the pipeline boundary). Falls back to the
    league-typical full-season constant scaled by ``fraction_elapsed`` only
    when the real values are absent -- keeping older callers that pass bare
    category dicts working unchanged.
    """
    ab = actuals.get("AB")
    ip = actuals.get("IP")
    actual_ab = float(ab) if ab is not None else _TYPICAL_TEAM_AB * fraction_elapsed
    actual_ip = float(ip) if ip is not None else _TYPICAL_TEAM_IP * fraction_elapsed
    return actual_ab, actual_ip


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
        fraction_remaining: Float 0.0-1.0, portion of season left to simulate.
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

        # Recover YTD component totals (H, ER, H+BB) from the rate stats so they
        # can be added to the simulated remainder before re-deriving rates.
        fraction_elapsed = 1.0 - fraction_remaining
        actual_ab, actual_ip = _ytd_playing_time(actuals, fraction_elapsed)
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


def _projected_volume(p: Any, is_hitter: bool) -> float:
    """Projected full-season playing time for the curve lookup.

    Hitters use projected PA; pitchers projected IP. Falls back to AB/AB_PER_PA
    when a dict caller carries ``ab`` but not ``pa``. ``safe_float`` coerces
    missing/NaN to 0.0 so bad data lands on the lowest (most conservative) curve
    band rather than slipping through as NaN and mapping to the best band.
    """
    if is_hitter:
        pa = safe_float(p.get("pa"))
        if pa > 0:
            return pa
        return safe_float(p.get("ab")) / AB_PER_PA
    return safe_float(p.get("ip"))


def _playing_time_scales(
    players: list,
    player_type: str,
    rng: np.random.Generator,
    fraction_remaining: float,
) -> np.ndarray:
    """Per-player playing-time multiplier (1.0 == exactly as projected).

    Draws ``u ~ Uniform(0, 1)`` per player and maps it through the empirical
    standardized-z ladder for the player's projected volume (``utils.playing_time``),
    then locates/scales by the calibrated ``mean_scale``/``cv_pt``. This reproduces
    the real, volume-dependent SHAPE of realized/projected PA-IP -- left-skewed for
    hitters/SP with a ceiling near full health, role-change upside for relievers --
    instead of a symmetric Normal clipped at a flat (and physically impossible) 2.0.
    Over a partial season only the remaining playing time is at risk, so the haircut
    and spread are damped inside ``scale_from_uniform``.
    """
    is_hitter = player_type == PlayerType.HITTER
    n = len(players)
    us = rng.random(n)
    out = np.empty(n)
    for i, p in enumerate(players):
        vol = _projected_volume(p, is_hitter)
        mean_scale, cv_pt = playing_time_params(player_type, vol)
        ladder = playing_time_shape(player_type, vol)
        out[i] = scale_from_uniform(mean_scale, cv_pt, ladder, float(us[i]), fraction_remaining)
    return out


# Hitter positions are every REPLACEMENT_BY_POSITION key that isn't a pitcher role,
# derived so adding a position to the constant doesn't need a second edit here.
_HITTER_REPL_POS = tuple(k for k in REPLACEMENT_BY_POSITION if k not in ("SP", "RP"))

# Neutral hitter replacement for position-less / UTIL / DH-only hitters: the
# element-wise mean of the position lines. A power bat filling a UTIL slot floors
# here rather than at the speed-maximizing line, so it doesn't get a phantom ~15 SB.
_GENERIC_HITTER_REPL: dict[str, int] = {
    col: round(
        sum(REPLACEMENT_BY_POSITION[pos][col] for pos in _HITTER_REPL_POS) / len(_HITTER_REPL_POS)
    )
    for col in HITTING_COUNTING
}


def _line_sgp(line: dict[str, int]) -> float:
    """SGP of a fixed replacement line (ranks a hitter's eligible positions)."""
    ab = line.get("ab", 0) or 1
    return calculate_player_sgp(
        {
            "player_type": PlayerType.HITTER,
            "r": line["r"],
            "hr": line["hr"],
            "rbi": line["rbi"],
            "sb": line["sb"],
            "avg": line["h"] / ab,
            "ab": ab,
        }
    )


# Per-position replacement SGP, precomputed once: the lines and the SGP denominators
# are all constants, so multi-position routing is a static dict lookup, not a cached
# runtime SGP call.
_HITTER_REPL_SGP: dict[str, float] = {
    pos: _line_sgp(REPLACEMENT_BY_POSITION[pos]) for pos in _HITTER_REPL_POS
}


def _pos_label(pos: object) -> str:
    """Uppercase position label from a Position enum or a string."""
    return (pos.value if hasattr(pos, "value") else str(pos)).upper()


def _replacement_line(p: dict, is_hitter: bool) -> dict:
    """The replacement-level line that backfills an injured player's missed time.

    Position-aware: a streamed catcher gives ~0 SB while a streamed middle
    infielder gives ~15, so the floor depends on where the player plays.

    - Hitters route to the highest-SGP replacement among their eligible Core-8
      positions (flexibility lets you stream the best available fill). A player
      with no Core-8 eligibility (UTIL/DH-only) or no positions falls back to the
      neutral ``_GENERIC_HITTER_REPL`` mean, not the speed-maximizing line.
    - Pitchers route to SP or RP by their ``SP``/``RP`` position eligibility (the
      authoritative signal, present on the flat dict and used by transactions.py),
      falling back to projected IP >= ``STARTER_IP_THRESHOLD`` when neither is
      listed. ``SP`` wins for swingmen eligible at both.
    """
    if not is_hitter:
        pos_set = {_pos_label(pos) for pos in (p.get("positions") or [])}
        if "SP" in pos_set or "RP" in pos_set:
            return REPLACEMENT_BY_POSITION["SP" if "SP" in pos_set else "RP"]
        ip = float(p.get("ip", 0) or 0)
        return REPLACEMENT_BY_POSITION[role_from_ip(ip)]
    elig = [
        lab for pos in (p.get("positions") or []) if (lab := _pos_label(pos)) in _HITTER_REPL_POS
    ]
    if not elig:
        return _GENERIC_HITTER_REPL
    return REPLACEMENT_BY_POSITION[max(elig, key=_HITTER_REPL_SGP.__getitem__)]


def _apply_variance(
    players: list,
    player_type: str,
    rng: np.random.Generator,
    injuries_out: list,
    fraction_remaining: float = 1.0,
) -> list[dict]:
    """Apply playing-time and correlated performance variance to a list of players.

    Two independent sources of variance per player:

    - Playing time: a two-sided multiplier ``scale`` from the calibrated
      playing-time model (``_playing_time_scales``). ``scale`` can exceed 1.0
      (a player beats his projected PA/IP) or fall below it; the missed
      fraction ``max(0, 1 - scale)`` is backfilled with replacement-level
      production and logged on the injuries report when notable.
    - Performance: correlated multivariate-normal draws so related stats
      (e.g. HR and RBI) move together. Per-stat sigmas and correlations are
      calibrated from projection-vs-actual rate residuals (2022-2024); the
      covariance is scaled by ``fraction_remaining`` for partial seasons.

    Mutates ``injuries_out`` by appending ``(name, frac_missed)`` for players
    whose simulated playing-time loss is at least ``_NOTABLE_PT_LOSS``.
    """
    is_hitter = player_type == PlayerType.HITTER
    counting_cols = HITTING_COUNTING if is_hitter else PITCHING_COUNTING
    base_cov = HITTER_COV if is_hitter else PITCHER_COV
    cov = base_cov * fraction_remaining
    idx_map = HITTER_IDX if is_hitter else PITCHER_IDX
    n_corr = len(idx_map)
    mean = np.zeros(n_corr)

    n = len(players)
    if n == 0:
        return []

    # Batch all random draws for the entire player list at once
    scales = _playing_time_scales(players, player_type, rng, fraction_remaining)
    all_draws = rng.multivariate_normal(mean, cov, size=n)

    adjusted = []
    for i, p in enumerate(players):
        scale = float(scales[i])
        frac_missed = max(0.0, 1.0 - scale)
        if frac_missed >= _NOTABLE_PT_LOSS:
            injuries_out.append((p.get("name", "?"), frac_missed))

        # Routing is a cheap dict lookup (per-position SGP is precomputed), so
        # compute it inline rather than memoizing onto the player dict.
        repl = _replacement_line(p, is_hitter)

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

    NOTE (shelved 2026-05-21): not currently surfaced on the season
    dashboard -- the 2023-2025 calibration is too noisy to display. See
    MANAGEMENT_ADJUSTMENT in utils.constants for the SD-discount refinement
    idea (weight each mean by its confidence). Kept intact for reactivation
    once more seasons of data accumulate.

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
        sim_roto = score_roto_dict(sim_stats)
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
        fraction_remaining: Float 0.0-1.0, portion of season left.
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
    # simulate_remaining_season derives ROS from full-season minus YTD; flatten
    # full-season here so that math is well-formed (see _flatten_full_season).
    flat_rosters: dict[str, list[dict]] = {}
    for team_key, players in team_rosters.items():
        flat_rosters[team_key] = [_flatten_full_season(p) for p in players]

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
        sim_roto = score_roto_dict(sim_stats)
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
