"""Monte Carlo season simulation with playing-time and stat variance.

Shared by scripts/simulate_draft.py (post-draft --monte-carlo),
scripts/summary.py (in-season weekly projections), and
the season dashboard (web/season_data.py).
"""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.special import _ufuncs as _scu  # Boost-backed nbinom ppf (see _nbinom_ppf_fast)
from scipy.special import ndtr, pdtr, pdtrik

from fantasy_baseball.distributions import build_distributions
from fantasy_baseball.mc_fill import ActiveSample, BenchSample, allocate_bench_fill
from fantasy_baseball.mc_roster import ActiveBody, EffectiveRoster
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.scoring import score_roto_dict
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.constants import (
    AB_PER_PA,
    CLOSER_SV_THRESHOLD,
    HITTER_CORR_STATS,
    HITTER_CORRELATION,
    HITTING_COUNTING,
    PITCHER_CORR_STATS,
    PITCHER_CORRELATION,
    PITCHING_COUNTING,
    QUANTILE_LEVELS,
    REPLACEMENT_BY_POSITION,
    STAT_DISPERSION,
    ZERO_IP_RATE_SENTINEL,
    role_from_ip,
    safe_float,
)
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES as ALL_CATS,
)
from fantasy_baseball.utils.dispersion import negbin_variance_from_r, resolve_dispersion_r
from fantasy_baseball.utils.playing_time import (
    playing_time_moments,
    playing_time_params,
    playing_time_shape,
    scale_from_uniform,
)
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

# Minimum simulated playing-time loss worth logging as a notable absence (vs a
# routine few-percent shortfall), since the playing-time scale is continuous.
_NOTABLE_PT_LOSS = 0.15


# Map stat name -> index in the correlated draw vector.
HITTER_IDX = {s: i for i, s in enumerate(HITTER_CORR_STATS)}
PITCHER_IDX = {s: i for i, s in enumerate(PITCHER_CORR_STATS)}

# Unit-variance correlated latents for the Gaussian copula (the NegBin sampler's
# Gaussian layer).
HITTER_CORR_MATRIX = np.array(HITTER_CORRELATION)
PITCHER_CORR_MATRIX = np.array(PITCHER_CORRELATION)

_U_EPS = 1e-9

# Phase 4b ROS-direct hitter sampling. When True (default), the hitter
# effective_rosters path samples the rest_of_season line directly with
# pt_mean_fraction=1.0 (full mean over the ROS window) + suppress_repl=True (the
# bench injury-fill owns the backfill). Flip to False to fall back to full-season
# hitter sampling without touching the displacement/fill wiring -- the one-line
# escape hatch gated on the Phase 6 SD backtest (NOT asserted in Phase 4).
_ROS_DIRECT_HITTERS = True

# Phase 5 ROS-direct pitcher sampling. When True (default), the pitcher
# effective_rosters path samples the rest_of_season line directly with
# pt_mean_fraction=0 (eff_mean=1 -> NO playing-time mean haircut, so mean ==
# projection == ERoto, which applies no haircut to pitcher means) +
# suppress_repl=True (no backfill). There is NO bench injury-fill: pitcher
# rich-fill is deferred, so applying the hitter haircut (pt_mean_fraction=1.0)
# without a fill to restore it would deflate pitcher means below ERoto. Flip to
# False to fall back to full-season top-k pitcher sampling -- the one-line escape
# hatch gated on the Phase 6 SD backtest (NOT asserted in Phase 5).
_ROS_DIRECT_PITCHERS = True


@dataclass
class VarianceBatch:
    """Output of ``_apply_variance_batch``.

    ``counts`` carries the SAME per-(iter, player) counting columns the function
    returned before this dataclass existed (back-compat); ``frac_missed`` exposes
    the per-(iter, player) stochastic playing-time shortfall the fill engine
    consumes (previously computed and discarded).
    """

    counts: dict[str, np.ndarray]  # {col: (n_iter, n_players)}
    frac_missed: np.ndarray  # (n_iter, n_players) = max(0, 1 - scales)
    scales: np.ndarray  # (n_iter, n_players) -- unclamped playing-time scale that drove mu


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


def _full_season_pt_volume(player: Any, is_hitter: bool) -> float:
    """Full-season projected playing time for the ROS-direct PT curve lookup.

    The playing-time curve is calibrated and indexed by FULL-SEASON volume. On
    the ROS-direct path the flat dict carries ROS volume (~half a season at
    mid-year), which would misclassify a full-timer as a part-timer (inflated
    ``cv_pt``). Read the player's ``full_season_projection`` instead: ``pa`` for
    hitters (fallback ``ab / AB_PER_PA``), ``ip`` for pitchers.

    If ``full_season_projection`` is unset (preseason) OR its volume is <= 0,
    fall back to the ROS volume (the same ``_projected_volume`` reading on the
    ROS flat dict) so those players keep working. Numeric guards use
    ``is not None`` / ``> 0`` -- never ``x or default`` (a real 0 is bad data, but
    the fallback handles it explicitly rather than via falsy coercion).
    """
    fs = player.full_season_projection
    if fs is not None:
        if is_hitter:
            pa = safe_float(getattr(fs, "pa", 0.0))
            if pa > 0:
                return pa
            ab = safe_float(getattr(fs, "ab", 0.0))
            if ab > 0:
                return ab / AB_PER_PA
        else:
            ip = safe_float(getattr(fs, "ip", 0.0))
            if ip > 0:
                return ip
    # Preseason / missing full-season: mirror _projected_volume on the ROS line.
    return _projected_volume(player.to_flat_dict(), is_hitter)


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


def _nbinom_ppf_fast(u: np.ndarray, r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Vectorized NegBin inverse-CDF, bit-identical to scipy.stats.nbinom.ppf but
    without the generic rv_discrete dispatch wrapper (argsreduce/broadcast/cdf
    round-trip -- the perf bottleneck).

    Modern scipy (>=1.11) overrides nbinom._ppf with a Boost-backed compiled
    kernel, scipy.special._ufuncs._nbinom_ppf, rather than the historical
    ceil(nbdtrik)+correction recipe. The Cephes nbdtr/nbdtrik path is NOT
    bit-identical to that Boost kernel for non-integer r (nbdtr's betainc loses
    ~1-3% accuracy at small k), so it cannot reproduce nbinom.ppf exactly. We
    therefore call the same raw Boost ufunc scipy itself dispatches to, which is
    exact by construction and skips only the wrapper -- verified bit-identical in
    test_fast_nbinom_ppf_bit_matches_scipy."""
    return cast(np.ndarray, _scu._nbinom_ppf(u, r, p))


def _poisson_ppf_fast(u: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Vectorized Poisson inverse-CDF matching scipy.stats.poisson.ppf via raw
    scipy.special pdtrik/pdtr (no generic dispatch wrapper)."""
    k = np.ceil(pdtrik(u, mu))
    k = np.maximum(k, 0.0)
    k1 = k - 1.0
    cdf_k1 = np.where(k1 >= 0.0, pdtr(np.maximum(k1, 0.0), mu), 0.0)
    k = np.where(cdf_k1 >= u, k1, k)
    under = pdtr(k, mu) < u
    k = np.where(under, k + 1.0, k)
    return cast(np.ndarray, k)


def _negbin_copula_counts(
    mu: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    fraction_remaining: float,
) -> np.ndarray:
    """Map correlated standard-normal latents z to NegBin counts via a copula.

    mu: per-element mean (base * scale). r: per-element calibrated dispersion
    (np.inf == Poisson floor). Variance is scaled by fraction_remaining through
    an effective dispersion r_eff; an element whose target variance falls to/below
    its mean (the Poisson floor) is drawn Poisson, so its variance bottoms out at
    mu and does NOT shrink further with fraction_remaining (a count's variance
    cannot go below its mean -- relevant for inf-r stats and very small
    fraction_remaining). u is clamped to [eps, 1-eps] so the ppf never returns inf.
    """
    mu = np.asarray(mu, dtype=float)
    r = np.asarray(r, dtype=float)
    u = np.clip(ndtr(z), _U_EPS, 1.0 - _U_EPS)

    out = np.zeros_like(mu)
    pos = mu > 0
    if not np.any(pos):
        return out

    mu_p = mu[pos]
    r_p = r[pos]
    u_p = u[pos]

    var_full = negbin_variance_from_r(mu_p, r_p)
    var_target = fraction_remaining * var_full

    supra = var_target > mu_p
    res = np.empty_like(mu_p)
    if np.any(supra):
        r_eff = mu_p[supra] ** 2 / (var_target[supra] - mu_p[supra])
        p_eff = r_eff / (r_eff + mu_p[supra])
        res[supra] = _nbinom_ppf_fast(u_p[supra], r_eff, p_eff)
    if np.any(~supra):
        res[~supra] = _poisson_ppf_fast(u_p[~supra], mu_p[~supra])

    out[pos] = res
    return out


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
    - Performance: correlated NegBin counts via a Gaussian copula. Correlated
      unit-variance normal latents (per-type correlation matrix) map through the
      normal CDF to uniforms, then each stat's NegBin (or Poisson-floor) inverse
      CDF; dispersion r (scalar or mu-banded, from STAT_DISPERSION) is
      variance-scaled by fraction_remaining. Replaces the clipped-Gaussian
      multiplier, which biased counts upward and spiked at zero.

    Mutates ``injuries_out`` by appending ``(name, frac_missed)`` for players
    whose simulated playing-time loss is at least ``_NOTABLE_PT_LOSS``.
    """
    is_hitter = player_type == PlayerType.HITTER
    counting_cols = HITTING_COUNTING if is_hitter else PITCHING_COUNTING
    corr_matrix = HITTER_CORR_MATRIX if is_hitter else PITCHER_CORR_MATRIX
    idx_map = HITTER_IDX if is_hitter else PITCHER_IDX
    n_corr = len(idx_map)

    n = len(players)
    if n == 0:
        return []

    scales = _playing_time_scales(players, player_type, rng, fraction_remaining)
    # Correlated unit-variance latents (the Gaussian copula's Gaussian layer).
    all_z = rng.multivariate_normal(np.zeros(n_corr), corr_matrix, size=n)

    # Per-(player, correlated-stat) NegBin mean (projection * playing-time scale)
    # and dispersion r (scalar or mu-banded, resolved from that mean). idx_map's
    # keys are exactly the correlated counting stats, so one pass fills both.
    mu_mat = np.zeros((n, n_corr))
    r_mat = np.full((n, n_corr), np.inf)
    for col, j in idx_map.items():
        mu_mat[:, j] = np.array([safe_float(p.get(col)) for p in players]) * scales
        r_mat[:, j] = resolve_dispersion_r(STAT_DISPERSION[col], mu_mat[:, j])

    # One flattened copula draw over all (player, stat) cells -- collapses the
    # per-stat scipy ppf calls (heavy fixed overhead) into a single nbinom +
    # single poisson call. C-order ravel keeps mu/r/z cells aligned.
    counts = _negbin_copula_counts(
        mu_mat.ravel(), r_mat.ravel(), all_z.ravel(), fraction_remaining
    ).reshape(n, n_corr)

    adjusted = []
    for i, p in enumerate(players):
        scale = float(scales[i])
        frac_missed = max(0.0, 1.0 - scale)
        if frac_missed >= _NOTABLE_PT_LOSS:
            injuries_out.append((p.get("name", "?"), frac_missed))
        repl = _replacement_line(p, is_hitter)
        row: dict[str, Any] = {}
        for col in counting_cols:
            repl_contrib = repl.get(col, 0) * frac_missed
            if col in idx_map:
                row[col] = counts[i, idx_map[col]] + repl_contrib
            else:
                row[col] = safe_float(p.get(col)) * scale + repl_contrib
        row["name"] = p.get("name", "?")
        row["player_type"] = player_type
        adjusted.append(row)

    return adjusted


# Lexicographic pick bonus: larger than any plausible per-pitcher w+k+sv, so a
# closer (sv >= threshold) always outranks a non-closer regardless of secondary
# score -- the scalar tuple key (is_closer, w+k+sv) expressed as one sortable float.
_CLOSER_RANK_BONUS = 1e9


def _gather_sum(stat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Sum the columns picked by ``idx`` (n_iter, k) out of ``stat`` (n_iter, n)."""
    # cast (not np.asarray): stat is already float, so this is a type-only assert
    # to satisfy warn_return_any, with no runtime coercion -- as in _nbinom_ppf_fast.
    return cast(np.ndarray, np.take_along_axis(stat, idx, axis=1).sum(axis=1))


def _topk_indices(key: np.ndarray, k: int) -> np.ndarray:
    """Per-row indices of the ``k`` largest entries of ``key`` (n_iter, n_players).

    Mirrors the scalar active-roster pick (sort by ``key`` descending, take the
    top ``k``); returns shape (n_iter, k), or every column when k >= n_players.
    Tie order is unspecified, but the keys are continuous sums so ties are
    measure-zero and do not change the simulated distribution.
    """
    n_players = key.shape[1]
    if k >= n_players:
        return np.broadcast_to(np.arange(n_players), key.shape)
    return np.argpartition(-key, k - 1, axis=1)[:, :k]


def _apply_variance_batch(
    players: list,
    player_type: str,
    rng: np.random.Generator,
    fraction_remaining: float,
    n_iter: int,
    *,
    pt_mean_fraction: float | None = None,
    suppress_repl: bool = False,
    pt_volumes: np.ndarray | None = None,
) -> VarianceBatch:
    """Vectorized ``_apply_variance`` over ``n_iter`` iterations at once.

    Returns a ``VarianceBatch`` whose ``counts`` is
    ``{counting_col: ndarray(n_iter, n_players)}`` -- the same per-player,
    per-stat values the scalar path yields, but with a leading iteration axis --
    and whose ``frac_missed`` is the per-(iter, player) playing-time shortfall.
    Identical model: per-player playing-time scale (calibrated z-ladder), a
    correlated NegBin/Poisson copula draw, and replacement-level injury backfill.
    Injuries are not reported (the ROS MC discards them); only the stat values
    are needed downstream.

    Keyword-only knobs (both default to legacy behavior, so the default path is
    byte-for-byte identical to before they existed):

    - ``pt_mean_fraction``: the MEAN-horizon term for the playing-time haircut.
      When ``None`` it equals ``fraction_remaining`` (today's behavior). When set
      (e.g. 1.0 for ROS-direct), it drives ONLY the mean haircut; ``eff_sd`` and
      the ``_negbin_copula_counts`` dispersion KEEP ``fraction_remaining`` (the
      variance horizon). ``playing_time_moments`` is closed-form (consumes no
      rng), so splitting its single call into a mean call and an sd call does NOT
      perturb the rng stream.
    - ``suppress_repl``: when True the built-in ``repl_contrib`` is identically
      zero (the new fill engine owns the backfill); default False folds the
      replacement line in exactly as before.
    - ``pt_volumes``: per-player volumes (aligned to ``players``) used for the
      playing-time curve lookup (BOTH ``playing_time_params`` and
      ``playing_time_shape``) instead of ``_projected_volume(p)``. The ROS-direct
      helpers pass FULL-SEASON volumes here because the curve is indexed by
      full-season volume; the flat dict on that path carries ROS volume, which
      would misclassify a full-timer as a part-timer. ``None`` (default) keeps the
      legacy per-player ``_projected_volume`` lookup -> byte-identical.
    """
    is_hitter = player_type == PlayerType.HITTER
    counting_cols = HITTING_COUNTING if is_hitter else PITCHING_COUNTING
    corr_matrix = HITTER_CORR_MATRIX if is_hitter else PITCHER_CORR_MATRIX
    idx_map = HITTER_IDX if is_hitter else PITCHER_IDX
    n_corr = len(idx_map)
    n_players = len(players)
    if n_players == 0:
        return VarianceBatch(
            counts={col: np.zeros((n_iter, 0)) for col in counting_cols},
            frac_missed=np.zeros((n_iter, 0)),
            scales=np.zeros((n_iter, 0)),
        )

    # Mean horizon vs variance horizon. When pt_mean_fraction is None the two
    # coincide (== fraction_remaining), reproducing today's single-call moments.
    fr_mean = pt_mean_fraction if pt_mean_fraction is not None else fraction_remaining

    # Static per-player playing-time moments and z-ladders (iteration-independent),
    # so the only per-iteration playing-time work is the uniform->z interpolation.
    eff_mean = np.empty(n_players)
    eff_sd = np.empty(n_players)
    ladders: list[np.ndarray] = []
    for j, p in enumerate(players):
        vol = float(pt_volumes[j]) if pt_volumes is not None else _projected_volume(p, is_hitter)
        mean_scale, cv_pt = playing_time_params(player_type, vol)
        # playing_time_moments consumes no rng (closed-form); splitting the mean
        # and sd calls keeps the default path's rng stream byte-stable while
        # letting pt_mean_fraction lift ONLY the mean haircut.
        eff_mean[j], _ = playing_time_moments(mean_scale, cv_pt, fr_mean)
        _, eff_sd[j] = playing_time_moments(mean_scale, cv_pt, fraction_remaining)
        ladders.append(np.asarray(playing_time_shape(player_type, vol), dtype=float))

    # Playing-time scale per (iteration, player): the vectorized scale_from_uniform.
    us = rng.random((n_iter, n_players))
    z_pt = np.empty((n_iter, n_players))
    for j in range(n_players):
        z_pt[:, j] = np.interp(us[:, j], QUANTILE_LEVELS, ladders[j])
    scales = np.maximum(0.0, eff_mean[None, :] + z_pt * eff_sd[None, :])

    base = {col: np.array([safe_float(p.get(col)) for p in players]) for col in counting_cols}

    # Per-(iter, player, corr-stat) NegBin mean and dispersion r.
    mu_mat = np.zeros((n_iter, n_players, n_corr))
    r_mat = np.full((n_iter, n_players, n_corr), np.inf)
    for col, j in idx_map.items():
        mu_mat[:, :, j] = base[col][None, :] * scales
        r_mat[:, :, j] = resolve_dispersion_r(STAT_DISPERSION[col], mu_mat[:, :, j])

    # One flattened copula draw over every (iter, player, stat) cell. C-order
    # ravel keeps mu/r/z aligned, same as the scalar path's per-team draw.
    all_z = rng.multivariate_normal(np.zeros(n_corr), corr_matrix, size=(n_iter, n_players))
    counts = _negbin_copula_counts(
        mu_mat.ravel(), r_mat.ravel(), all_z.ravel(), fraction_remaining
    ).reshape(n_iter, n_players, n_corr)

    frac_missed = np.maximum(0.0, 1.0 - scales)
    repl_lines = [_replacement_line(p, is_hitter) for p in players]
    out: dict[str, np.ndarray] = {}
    for col in counting_cols:
        if suppress_repl:
            repl_contrib: float | np.ndarray = 0.0
        else:
            repl_line = np.array([rl.get(col, 0) for rl in repl_lines], dtype=float)
            repl_contrib = repl_line[None, :] * frac_missed
        if col in idx_map:
            out[col] = counts[:, :, idx_map[col]] + repl_contrib
        else:
            out[col] = base[col][None, :] * scales + repl_contrib
    return VarianceBatch(counts=out, frac_missed=frac_missed, scales=scales)


def _simulate_team_hitters_ros_direct(
    effective_roster: EffectiveRoster,
    fraction_remaining: float,
    rng: np.random.Generator,
    n_iter: int,
) -> dict[str, np.ndarray]:
    """Return the team's ROS-ONLY hitter arrays (each shape ``(n_iter,)``):
    ``{R, HR, RBI, SB, ros_h, ros_ab}``.

    ROS-direct: samples the effective HITTER bodies' ``rest_of_season`` lines,
    applies each body's displacement ``factor`` to its sampled counts, runs the
    bench injury-fill, and returns the summed-ROS counting + the ``ros_h``/
    ``ros_ab`` AVG components. It does NOT take or use team_YTD: the CALLER owns
    the YTD blend (``team_total = YTD + ROS``) and the AVG recombine
    (``(YTD_h + ros_h) / (YTD_ab + ros_ab)``). Keeping YTD in one place avoids a
    dead param and keeps this helper a pure ROS sampler.

    Mechanism (HITTERS only -- pitchers never enter here):

    - ``active`` is filtered to HITTER bodies (``EffectiveRoster.active`` is
      ``[*il, *active]``, both player types); their order is the helper's OWN
      column order, never aligned against any flat sublist (the C2 fix).
    - The bench fill pool is ``effective_roster.bench`` (already HITTER-only).
    - Active bodies are sampled via ``_apply_variance_batch`` with
      ``pt_mean_fraction=1.0`` (the projection IS the remaining mean -- apply the
      FULL mean haircut over the ROS window, NOT a re-haircut) and
      ``suppress_repl=True`` (the bench fill replaces the built-in backfill);
      ``fraction_remaining`` keeps the SD + dispersion (the variance horizon).
    - Each body's displacement ``factor`` multiplies its SAMPLED ROS counts. This
      scales BOTH the mean AND the SD of that body's counts by ``factor`` (its
      variance by ``factor^2``), which is INTENDED: the curve lookup inside
      ``_apply_variance_batch`` uses ``pt_volumes`` = each body's FULL-SEASON
      volume (``_full_season_pt_volume``), NOT the ROS volume on the flat dict, so
      the body is sampled with its FULL-VOLUME CV band; multiplying realized
      counts by ``factor`` rescales mean and SD together, holding CV fixed --
      exactly the full-volume band the spec requires (NOT a narrowed, higher-CV
      curve point).
    - Bench per-game lines are the clean DETERMINISTIC base ROS projection
      (``base_ros_total / g_ros_full``), iteration-independent -- built once.
    """
    active_h_bodies = [
        b for b in effective_roster.active if b.player.player_type == PlayerType.HITTER
    ]
    bench_h_bodies = effective_roster.bench  # already HITTER-only

    cats = {"R": "r", "HR": "hr", "RBI": "rbi", "SB": "sb"}
    zeros = np.zeros(n_iter)
    if not active_h_bodies:
        out: dict[str, np.ndarray] = {cat: zeros.copy() for cat in cats}
        out["ros_h"] = zeros.copy()
        out["ros_ab"] = zeros.copy()
        return out

    active_flats = [b.player.to_flat_dict() for b in active_h_bodies]
    pt_volumes = np.array(
        [_full_season_pt_volume(b.player, is_hitter=True) for b in active_h_bodies]
    )
    active_vb = _apply_variance_batch(
        active_flats,
        PlayerType.HITTER,
        rng,
        fraction_remaining,
        n_iter,
        pt_mean_fraction=1.0,
        suppress_repl=True,
        pt_volumes=pt_volumes,
    )

    # Per-active-body realized ROS counts = sampled draw * displacement factor.
    factors = np.array([b.factor for b in active_h_bodies])  # (n_active,)
    realized: dict[str, np.ndarray] = {
        col: active_vb.counts[col] * factors[None, :] for col in HITTING_COUNTING
    }
    frac_missed = active_vb.frac_missed  # (n_iter, n_active)

    # Bench per-game counts: clean BASE ROS projection / g_ros_full (deterministic
    # fill), iteration-independent -- built ONCE.
    bench_samples: list[BenchSample] = []
    for bb in bench_h_bodies:
        base_flat = bb.player.to_flat_dict()
        gf = bb.g_ros_full
        per_game = {
            col: (safe_float(base_flat.get(col)) / gf if gf > 0 else 0.0)
            for col in HITTING_COUNTING
        }
        bench_samples.append(BenchSample(body=bb, per_game_counts=per_game))

    def _repl_for(ab: ActiveBody) -> dict[str, float]:
        return _replacement_line(ab.player.to_flat_dict(), is_hitter=True)

    # Per-iteration fill allocation (the sanctioned small Python loop: <=12 active,
    # <=2 bench). Vectorized sampling is done above; only the allocation loops.
    fill_totals: dict[str, np.ndarray] = {col: np.zeros(n_iter) for col in HITTING_COUNTING}
    for it in range(n_iter):
        actives = [
            ActiveSample(body=body, frac_missed=float(frac_missed[it, idx]))
            for idx, body in enumerate(active_h_bodies)
        ]
        fill = allocate_bench_fill(actives, bench_samples, _repl_for).fill_counts
        for col in HITTING_COUNTING:
            fill_totals[col][it] = fill[col]

    out = {}
    for cat, col in cats.items():
        out[cat] = realized[col].sum(axis=1) + fill_totals[col]
    out["ros_h"] = realized["h"].sum(axis=1) + fill_totals["h"]
    out["ros_ab"] = realized["ab"].sum(axis=1) + fill_totals["ab"]
    return out


def _simulate_team_pitchers_ros_direct(
    effective_roster: EffectiveRoster,
    fraction_remaining: float,
    rng: np.random.Generator,
    n_iter: int,
) -> dict[str, np.ndarray]:
    """Return the team's ROS-ONLY pitcher arrays (each shape ``(n_iter,)``):
    ``{W, K, SV}`` + ``ros_ip``/``ros_er``/``ros_bb``/``ros_ha`` (for the
    ERA/WHIP recombine).

    Mirrors ``_simulate_team_hitters_ros_direct`` but with NO mean haircut and NO
    bench injury-fill (pitcher rich-fill is deferred). Samples the active PITCHER
    bodies' ``rest_of_season`` lines with ``pt_mean_fraction=0`` (so
    ``eff_mean = 1 - (1 - mean_scale) * 0 = 1`` -- NO playing-time mean haircut ->
    mean == projection == ERoto, which applies no haircut to pitcher means; the
    SD term keeps ``cv_pt * sqrt(fraction_remaining)`` so the variance horizon is
    intact) and ``suppress_repl=True`` (no built-in backfill). Each body's
    displacement ``factor`` multiplies its sampled ROS counts, then sums.

    Why no haircut here vs the hitter helper's ``pt_mean_fraction=1.0`` full
    haircut: apply the haircut ONLY when there is a fill to restore it. Hitters
    apply it and ``allocate_bench_fill`` restores it; pitchers have NO fill, so a
    haircut would stay UNRESTORED and deflate the pitcher mean ~15-24% below
    ERoto -- a standings-corrupting bug. With ``pt_mean_fraction=0`` the pitcher
    mean matches ERoto by construction (no haircut, no fill, no premium).

    Healthy bench pitchers are absent from ``EffectiveRoster.active`` (dropped by
    ``build_effective_roster``) so they never contribute. The CALLER owns the YTD
    blend (``team_total = YTD + ROS``, no clamp; ERA/WHIP recombine).
    """
    active_p_bodies = [
        b for b in effective_roster.active if b.player.player_type == PlayerType.PITCHER
    ]
    cats = {"W": "w", "K": "k", "SV": "sv"}
    zeros = np.zeros(n_iter)
    if not active_p_bodies:
        out: dict[str, np.ndarray] = {cat: zeros.copy() for cat in cats}
        for ros_key in ("ros_ip", "ros_er", "ros_bb", "ros_ha"):
            out[ros_key] = zeros.copy()
        return out

    active_flats = [b.player.to_flat_dict() for b in active_p_bodies]
    pt_volumes = np.array(
        [_full_season_pt_volume(b.player, is_hitter=False) for b in active_p_bodies]
    )
    vb = _apply_variance_batch(
        active_flats,
        PlayerType.PITCHER,
        rng,
        fraction_remaining,
        n_iter,
        pt_mean_fraction=0,  # eff_mean=1: NO haircut -> mean == projection == ERoto
        suppress_repl=True,
        pt_volumes=pt_volumes,
    )
    factors = np.array([b.factor for b in active_p_bodies])  # (n_active,)
    realized = {col: vb.counts[col] * factors[None, :] for col in PITCHING_COUNTING}

    out = {cat: realized[col].sum(axis=1) for cat, col in cats.items()}
    out["ros_ip"] = realized["ip"].sum(axis=1)
    out["ros_er"] = realized["er"].sum(axis=1)
    out["ros_bb"] = realized["bb"].sum(axis=1)
    out["ros_ha"] = realized["h_allowed"].sum(axis=1)
    return out


def simulate_remaining_season_batch(
    actual_standings: dict[str, dict[str, float]],
    team_rosters: dict,
    fraction_remaining: float,
    rng: np.random.Generator,
    h_slots: int,
    p_slots: int,
    n_iter: int,
    active_cols: dict | None = None,
    effective_rosters: dict[str, EffectiveRoster] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Vectorized ``simulate_remaining_season`` over ``n_iter`` iterations.

    Returns ``{team: {category: ndarray(n_iter)}}`` for the 10 roto categories --
    the per-iteration team totals the scalar path emits one call at a time,
    stacked along a leading iteration axis. Same blend: simulated full season
    minus YTD (clamped >= 0), re-added to actuals; rate stats via recovered YTD
    components. Drives ``run_ros_monte_carlo`` without the Python per-iteration loop.

    ``active_cols`` optionally pins a fixed active roster instead of the
    per-iteration top-k pick. When ``None`` (default) every team uses top-k.
    When ``active_cols[team] = {"h": ndarray[int], "p": ndarray[int]}`` is
    present, that team's hitter/pitcher contributions are summed over exactly
    those fixed column indices (into the hitter/pitcher sublists in roster
    order) for every iteration. A team absent from ``active_cols`` falls back to
    top-k; an empty index array yields a zero contribution.

    ``effective_rosters`` optionally routes a team's HITTERS through the
    ROS-direct body engine (``_simulate_team_hitters_ros_direct``): a fixed
    active set with IL displacement + bench injury-fill, blended
    ``team_total = YTD + ROS`` (ROS >= 0 makes the YTD floor structural, so NO
    ``max(actual, sim)`` clamp for those hitter cats). When ``None`` or a team is
    absent, that team's hitters fall back to the flat-dict top-k path UNCHANGED
    (the byte-identical anchor). PITCHERS ALWAYS use the existing full-season
    path + its ``max(actual, sim)`` clamp, regardless of ``effective_rosters``.
    """
    cats = [c.value for c in ALL_CATS]
    out: dict[str, dict[str, np.ndarray]] = {}
    if fraction_remaining <= 0:
        for team in team_rosters:
            a = actual_standings.get(team, {})
            out[team] = {c: np.full(n_iter, a.get(c, 0), dtype=float) for c in cats}
        return out

    fraction_elapsed = 1.0 - fraction_remaining
    zeros = np.zeros(n_iter)
    for team, players in team_rosters.items():
        actuals = actual_standings.get(team, {})
        hitters = [p for p in players if p.get("player_type") == PlayerType.HITTER]
        pitchers = [p for p in players if p.get("player_type") == PlayerType.PITCHER]

        # ROS-direct hitter path: when this team has an EffectiveRoster (and the
        # flag is on), route its HITTERS through the body engine (fixed active
        # set + IL displacement + bench injury-fill), bypassing the flat-dict
        # top-k hitter sampling entirely. Pitchers ALWAYS take the path below.
        eff = effective_rosters.get(team) if effective_rosters is not None else None
        use_ros_direct = _ROS_DIRECT_HITTERS and eff is not None
        use_ros_direct_pitchers = _ROS_DIRECT_PITCHERS and eff is not None

        if use_ros_direct:
            assert eff is not None  # use_ros_direct implies eff is not None
            ros = _simulate_team_hitters_ros_direct(eff, fraction_remaining, rng, n_iter)
        else:
            hb = _apply_variance_batch(
                hitters, PlayerType.HITTER, rng, fraction_remaining, n_iter
            ).counts

        team_cols = active_cols.get(team) if active_cols is not None else None

        if not use_ros_direct and hitters:
            if team_cols is not None:
                cols = team_cols["h"]
                h_idx = np.broadcast_to(cols, (n_iter, cols.shape[0]))
            else:
                h_idx = _topk_indices(hb["r"] + hb["hr"] + hb["rbi"] + hb["sb"], h_slots)
            sim_r = _gather_sum(hb["r"], h_idx)
            sim_hr = _gather_sum(hb["hr"], h_idx)
            sim_rbi = _gather_sum(hb["rbi"], h_idx)
            sim_sb = _gather_sum(hb["sb"], h_idx)
            sim_h = _gather_sum(hb["h"], h_idx)
            sim_ab = _gather_sum(hb["ab"], h_idx)
        elif not use_ros_direct:
            sim_r = sim_hr = sim_rbi = sim_sb = sim_h = sim_ab = zeros

        # ROS-direct pitcher path: route this team's PITCHERS through the body
        # engine (fixed active-slot + IL set, displacement factors, NO haircut/NO
        # fill), bypassing the flat-dict top-k pitcher sampling. When off (no
        # effective_roster), sample pb here exactly as before -- this keeps the
        # effective_rosters=None rng stream byte-identical (pb drawn after the
        # hitter hb/ros draw, in the same order).
        if use_ros_direct_pitchers:
            assert eff is not None  # use_ros_direct_pitchers implies eff is not None
            pros = _simulate_team_pitchers_ros_direct(eff, fraction_remaining, rng, n_iter)
        elif pitchers:
            pb = _apply_variance_batch(
                pitchers, PlayerType.PITCHER, rng, fraction_remaining, n_iter
            ).counts
            if team_cols is not None:
                cols = team_cols["p"]
                p_idx = np.broadcast_to(cols, (n_iter, cols.shape[0]))
            else:
                # Closers (sv >= threshold) first, then by w+k+sv -- the scalar tuple key.
                pkey = (pb["sv"] >= CLOSER_SV_THRESHOLD).astype(float) * _CLOSER_RANK_BONUS + (
                    pb["w"] + pb["k"] + pb["sv"]
                )
                p_idx = _topk_indices(pkey, p_slots)
            sim_w = _gather_sum(pb["w"], p_idx)
            sim_k = _gather_sum(pb["k"], p_idx)
            sim_sv = _gather_sum(pb["sv"], p_idx)
            sim_ip = _gather_sum(pb["ip"], p_idx)
            sim_er = _gather_sum(pb["er"], p_idx)
            sim_bb = _gather_sum(pb["bb"], p_idx)
            sim_ha = _gather_sum(pb["h_allowed"], p_idx)
        else:
            sim_w = sim_k = sim_sv = sim_ip = sim_er = sim_bb = sim_ha = zeros

        actual_ab, actual_ip = _ytd_playing_time(actuals, fraction_elapsed)
        actual_h = actuals.get("AVG", 0) * actual_ab
        actual_er = actuals.get("ERA", 0) * actual_ip / 9
        actual_h_plus_bb = actuals.get("WHIP", 0) * actual_ip

        if use_ros_direct_pitchers:
            # ROS-direct pitchers: team_total = YTD + ROS (ROS >= 0 -> structural
            # YTD floor, NO clamp). ERA/WHIP recombine YTD + ROS volume components.
            total_ip = actual_ip + pros["ros_ip"]
            total_er = actual_er + pros["ros_er"]
            total_h_plus_bb = actual_h_plus_bb + pros["ros_bb"] + pros["ros_ha"]
            sim_w_out = actuals.get("W", 0) + pros["W"]
            sim_k_out = actuals.get("K", 0) + pros["K"]
            sim_sv_out = actuals.get("SV", 0) + pros["SV"]
        else:
            # Pitcher rate blend (full-season path, UNCHANGED): actual + clamped
            # sim remainder == max(actual, sim) for the volume components.
            total_ip = actual_ip + np.maximum(0.0, sim_ip - actual_ip)
            total_er = actual_er + np.maximum(0.0, sim_er - actual_er)
            total_h_plus_bb = actual_h_plus_bb + np.maximum(
                0.0, (sim_bb + sim_ha) - actual_h_plus_bb
            )
            sim_w_out = np.maximum(actuals.get("W", 0), sim_w)
            sim_k_out = np.maximum(actuals.get("K", 0), sim_k)
            sim_sv_out = np.maximum(actuals.get("SV", 0), sim_sv)

        if use_ros_direct:
            # ROS-direct: team_total = YTD + ROS (ROS >= 0 -> structural YTD
            # floor, NO clamp). AVG recombines YTD + ROS h/ab components.
            hit_r = actuals.get("R", 0) + ros["R"]
            hit_hr = actuals.get("HR", 0) + ros["HR"]
            hit_rbi = actuals.get("RBI", 0) + ros["RBI"]
            hit_sb = actuals.get("SB", 0) + ros["SB"]
            total_ab = actual_ab + ros["ros_ab"]
            total_h = actual_h + ros["ros_h"]
        else:
            # Top-k fallback: actual + clamped sim remainder == max(actual, sim).
            hit_r = np.maximum(actuals.get("R", 0), sim_r)
            hit_hr = np.maximum(actuals.get("HR", 0), sim_hr)
            hit_rbi = np.maximum(actuals.get("RBI", 0), sim_rbi)
            hit_sb = np.maximum(actuals.get("SB", 0), sim_sb)
            total_ab = actual_ab + np.maximum(0.0, sim_ab - actual_ab)
            total_h = actual_h + np.maximum(0.0, sim_h - actual_h)

        with np.errstate(divide="ignore", invalid="ignore"):
            avg = np.where(total_ab > 0, total_h / total_ab, 0.0)
            era = np.where(total_ip > 0, total_er * 9 / total_ip, ZERO_IP_RATE_SENTINEL)
            whip = np.where(total_ip > 0, total_h_plus_bb / total_ip, ZERO_IP_RATE_SENTINEL)

        # Pitcher counting totals: ROS-direct -> YTD + ROS (no clamp; ROS >= 0
        # makes the YTD floor structural); top-k fallback -> max(actual, sim),
        # never below what's already banked. Both resolved into sim_*_out above.
        out[team] = {
            "R": hit_r,
            "HR": hit_hr,
            "RBI": hit_rbi,
            "SB": hit_sb,
            "AVG": avg,
            "W": sim_w_out,
            "K": sim_k_out,
            "SV": sim_sv_out,
            "ERA": era,
            "WHIP": whip,
        }
    return out


def run_monte_carlo(
    team_rosters: dict,
    h_slots: int,
    p_slots: int,
    user_team_name: str,
    n_iterations: int = 1000,
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
    seed: int = 42,
    progress_cb=None,
    effective_rosters: dict[str, EffectiveRoster] | None = None,
) -> dict:
    """Run a Monte Carlo simulation over the remaining season.

    Like run_monte_carlo but uses simulate_remaining_season_batch to blend
    actual YTD stats with simulated ROS projections.

    ``effective_rosters`` (optional) routes each present team's HITTERS through
    the ROS-direct body engine (fixed active set + IL displacement + bench
    injury-fill); when ``None`` the batch falls entirely to the top-k path
    (byte-identical to pre-Phase-4b). Pitchers are unaffected either way.

    Args:
        team_rosters: {team_name: [player dicts]} with ROS projections.
        actual_standings: {team_name: {R, HR, RBI, SB, AVG, W, K, SV, ERA, WHIP}}
            — actual YTD stats for each team.
        fraction_remaining: Float 0.0-1.0, portion of season left.
        h_slots: Number of active hitter slots.
        p_slots: Number of active pitcher slots.
        user_team_name: Name of user's team (for category risk).
        n_iterations: Number of simulation iterations.
        seed: RNG seed for reproducibility.
        progress_cb: Optional callback(msg: str) called every 200 iterations.

    Returns:
        {"team_results": {team: {median_pts, p10, p90, first_pct, top3_pct}},
         "category_risk": {cat: {median_pts, p10, p90, top3_pct, bot3_pct}},
         "distributions": compact per-team outcome curves (see build_distributions)}
        category_risk is {} when the user team is absent from the rosters.
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
    all_cat_pts: dict[str, dict[str, list[float]]] = {
        name: {c.value: [] for c in ALL_CATS} for name in team_names
    }
    cats = [c.value for c in ALL_CATS]
    # (category value, scoring "{cat}_pts" key) pairs, hoisted out of the
    # per-iteration loop so the f-string is built once, not n_iter * n_teams times.
    cat_pts_keys = [(c.value, f"{c.value}_pts") for c in ALL_CATS]

    # Vectorized: one batched simulation of all iterations replaces the former
    # per-iteration simulate_remaining_season call. Roto scoring stays per
    # iteration (it ranks teams within each draw), reading column i from the batch.
    # The batch below is the heavy step (the scoring loop is cheap), so signal the
    # MC phase before it rather than letting the prior step's message linger.
    if progress_cb:
        progress_cb(0)
    batch = simulate_remaining_season_batch(
        actual_standings,
        flat_rosters,
        fraction_remaining,
        rng,
        h_slots,
        p_slots,
        n_iterations,
        effective_rosters=effective_rosters,
    )

    for i in range(n_iterations):
        if progress_cb and i % 200 == 0 and i != 0:
            progress_cb(i)
        sim_stats = {name: {cat: float(batch[name][cat][i]) for cat in cats} for name in team_names}
        sim_roto = score_roto_dict(sim_stats)
        ranked = sorted(sim_roto.items(), key=lambda x: x[1]["total"], reverse=True)
        for rank, (name, pts) in enumerate(ranked, 1):
            all_totals[name].append(pts["total"])
            if rank == 1:
                mc_wins[name] += 1
            if rank <= 3:
                mc_top3[name] += 1
            team_cat_pts = all_cat_pts[name]
            for val, pts_key in cat_pts_keys:
                team_cat_pts[val].append(pts.get(pts_key, 0))

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
    # category_risk summarizes the user team's per-category point spread. If the
    # user team is absent from the rosters (e.g. a misconfigured team name), there
    # is nothing to summarize -- leave category_risk empty so the dashboard hides
    # the table, rather than computing percentiles on empty arrays (np.percentile
    # raises IndexError on an empty input). The user team's slice is the same
    # per-iteration sequence the old user-only accumulator produced, so the
    # computed values are unchanged when the team is present.
    user_cat_pts = all_cat_pts.get(user_team_name)
    if user_cat_pts is not None:
        for c in ALL_CATS:
            arr = np.array(user_cat_pts[c.value])
            category_risk[c.value] = {
                "median_pts": round(float(np.median(arr)), 1),
                "p10": round(float(np.percentile(arr, 10)), 1),
                "p90": round(float(np.percentile(arr, 90)), 1),
                "top3_pct": round(float((arr >= 8).sum()) / n * 100, 1),
                "bot3_pct": round(float((arr <= 3).sum()) / n * 100, 1),
            }

    distributions = build_distributions(all_totals, batch, all_cat_pts, cats, user_team_name)

    return {
        "team_results": team_results,
        "category_risk": category_risk,
        "distributions": distributions,
    }
