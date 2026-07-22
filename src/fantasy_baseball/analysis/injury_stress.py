"""Injury stress-test: how robust is the user's ROS lead to lost playing time?

Rides the existing ROS Monte Carlo (simulation.run_ros_monte_carlo) so every
number reconciles with the season dashboard. See
docs/superpowers/specs/2026-07-22-injury-stress-test-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.simulation import _full_season_pt_volume
from fantasy_baseball.utils.constants import QUANTILE_LEVELS
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
