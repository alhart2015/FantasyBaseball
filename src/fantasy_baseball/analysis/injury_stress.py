"""Injury stress-test: how robust is the user's ROS lead to lost playing time?

Rides the existing ROS Monte Carlo (simulation.run_ros_monte_carlo) so every
number reconciles with the season dashboard. See
docs/superpowers/specs/2026-07-22-injury-stress-test-design.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from fantasy_baseball.mc_roster import build_effective_rosters
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.standings import CategoryStats
from fantasy_baseball.simulation import (
    _full_season_pt_volume,
    _replacement_line,
    run_ros_monte_carlo,
)
from fantasy_baseball.utils.constants import (
    AB_PER_PA,
    HITTING_COUNTING,
    PITCHING_COUNTING,
    QUANTILE_LEVELS,
    Category,
)
from fantasy_baseball.utils.playing_time import (
    playing_time_moments,
    playing_time_params,
    playing_time_shape,
)

SIGNIFICANT_TIME_THRESHOLD: float = 0.20
PAIR_TOP_K: int = 8
HEALTH_SAMPLES: int = 20000
DEFAULT_N_ITER: int = 1000
SEED: int = 42


@dataclass(frozen=True)
class HealthProbs:
    p_all_healthy: float
    p_one: float
    p_two_plus: float
    per_player: dict[str, float]
    threshold: float


def health_probabilities(
    active_players: list[Player],
    fraction_remaining: float,
    *,
    threshold: float = SIGNIFICANT_TIME_THRESHOLD,
    n_samples: int = HEALTH_SAMPLES,
    seed: int = SEED,
) -> HealthProbs:
    """P(0 / exactly-1 / 2-or-more active players lose significant time).

    Per player, sample realized playing-time scale with the SAME moments the MC
    uses (mean horizon 1.0 for hitters -> eff_mean == mean_scale; 0.0 for pitchers
    -> eff_mean == 1.0; sd horizon == fraction_remaining), then count a
    "significant" loss when realized scale <= eff_mean * (1 - threshold), i.e. at
    least `threshold` below the player's OWN expected remaining playing time. This
    isolates the injury/availability tail from the systematic mean haircut. Draws
    are independent across players (injuries are ~independent).
    """
    rng = np.random.default_rng(seed)
    n = len(active_players)
    if n == 0:
        return HealthProbs(1.0, 0.0, 0.0, {}, threshold)
    significant = np.zeros((n_samples, n), dtype=bool)
    for j, p in enumerate(active_players):
        is_hitter = p.player_type == PlayerType.HITTER
        vol = _full_season_pt_volume(p, is_hitter=is_hitter)
        mean_scale, cv_pt = playing_time_params(p.player_type, vol)
        fr_mean = 1.0 if is_hitter else 0.0
        eff_mean, _ = playing_time_moments(mean_scale, cv_pt, fr_mean)
        _, eff_sd = playing_time_moments(mean_scale, cv_pt, fraction_remaining)
        ladder = np.asarray(playing_time_shape(p.player_type, vol), dtype=float)
        u = rng.random(n_samples)
        z = np.interp(u, QUANTILE_LEVELS, ladder)
        scale = np.maximum(0.0, eff_mean + z * eff_sd)
        significant[:, j] = scale <= eff_mean * (1.0 - threshold)
    counts = significant.sum(axis=1)
    per_player = {p.name: float(significant[:, j].mean()) for j, p in enumerate(active_players)}
    return HealthProbs(
        p_all_healthy=float((counts == 0).mean()),
        p_one=float((counts == 1).mean()),
        p_two_plus=float((counts >= 2).mean()),
        per_player=per_player,
        threshold=threshold,
    )


@dataclass(frozen=True)
class McInputs:
    team_rosters: dict[str, list[Player]]
    actual_standings: dict[str, dict[str, float]]
    fraction_remaining: float
    h_slots: int
    p_slots: int
    eos_baseline: dict[str, CategoryStats]
    team_sds: dict[str, dict[Category, float]]
    denoms: dict[Category, float]
    user_team_name: str
    projected_margin: float


def _replacement_ros(player: Player) -> HitterStats | PitcherStats:
    """Replacement-level ROS stats object at `player`'s slot, scaled to his ROS
    playing-time volume (AB for hitters, IP for pitchers). Returns a NEW stats
    object; `player.rest_of_season` is not mutated."""
    is_hitter = player.player_type == PlayerType.HITTER
    ros = player.rest_of_season
    repl = _replacement_line(player.to_flat_dict(), is_hitter)
    if isinstance(ros, HitterStats):
        x_ab = float(ros.ab) if ros.ab else 0.0
        factor = (x_ab / repl["ab"]) if repl.get("ab") else 0.0
        s = {c: repl[c] * factor for c in HITTING_COUNTING}
        avg = (s["h"] / s["ab"]) if s["ab"] else 0.0
        return dataclasses.replace(
            ros,
            r=s["r"],
            hr=s["hr"],
            rbi=s["rbi"],
            sb=s["sb"],
            h=s["h"],
            ab=s["ab"],
            pa=(s["ab"] / AB_PER_PA),
            avg=avg,
            sgp=None,
        )
    if isinstance(ros, PitcherStats):
        x_ip = float(ros.ip) if ros.ip else 0.0
        factor = (x_ip / repl["ip"]) if repl.get("ip") else 0.0
        s = {c: repl[c] * factor for c in PITCHING_COUNTING}
        era = (s["er"] * 9.0 / s["ip"]) if s["ip"] else 0.0
        whip = ((s["bb"] + s["h_allowed"]) / s["ip"]) if s["ip"] else 0.0
        return dataclasses.replace(
            ros,
            w=s["w"],
            k=s["k"],
            sv=s["sv"],
            ip=s["ip"],
            er=s["er"],
            bb=s["bb"],
            h_allowed=s["h_allowed"],
            era=era,
            whip=whip,
            sgp=None,
        )
    raise ValueError(f"{player.name!r} has no rest_of_season line to substitute a replacement into")


def substitute_replacement(user_players: list[Player], target_names: list[str]) -> list[Player]:
    """Clone `user_players`, replacing each named player's ROS line with a
    position-matched replacement-level line (see `_replacement_ros`). Non-targets
    are shared unchanged (same object)."""
    targets = set(target_names)
    out: list[Player] = []
    for p in user_players:
        if p.name in targets:
            out.append(dataclasses.replace(p, rest_of_season=_replacement_ros(p)))
        else:
            out.append(p)
    return out


def win_pct(
    inputs: McInputs,
    user_players: list[Player],
    *,
    availability_variance_off: bool = False,
    n_iter: int = DEFAULT_N_ITER,
    seed: int = SEED,
) -> float:
    """User's P(finish 1st) for a given user roster. Rebuilds effective_rosters
    (fixed eos_baseline/team_sds/fraction_remaining) so the substitution takes
    effect in the ROS-direct path, then runs the ROS MC."""
    team_rosters = {**inputs.team_rosters, inputs.user_team_name: user_players}
    eff = build_effective_rosters(
        team_rosters,
        inputs.eos_baseline,
        inputs.team_sds,
        inputs.fraction_remaining,
        denoms=inputs.denoms,
    )
    mc = run_ros_monte_carlo(
        team_rosters=team_rosters,
        actual_standings=inputs.actual_standings,
        fraction_remaining=inputs.fraction_remaining,
        h_slots=inputs.h_slots,
        p_slots=inputs.p_slots,
        user_team_name=inputs.user_team_name,
        n_iterations=n_iter,
        seed=seed,
        effective_rosters=eff,
        availability_variance_off=availability_variance_off,
    )
    return float(mc["team_results"][inputs.user_team_name]["first_pct"])
