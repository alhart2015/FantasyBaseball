"""Injury stress-test: how robust is the user's ROS lead to lost playing time?

Rides the existing ROS Monte Carlo (simulation.run_ros_monte_carlo) so every
number reconciles with the season dashboard. See
docs/superpowers/specs/2026-07-22-injury-stress-test-design.md.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fantasy_baseball.mc_roster import build_effective_rosters
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.standings import CategoryStats, Standings, build_eos_baseline
from fantasy_baseball.scoring import _classify_roster, build_team_sds, score_roto_dict
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
    OpportunityStat,
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


def build_actual_standings(standings: Standings) -> dict[str, dict[str, float]]:
    """{team: {R..WHIP, IP, AB}} -- mirrors refresh_pipeline.py:1451-1460 exactly
    (AB from PA * AB_PER_PA), so it matches what the dashboard MC consumed."""
    out: dict[str, dict[str, float]] = {}
    for e in standings.entries:
        row = e.stats.to_dict()
        ip = e.extras.get(OpportunityStat.IP)
        pa = e.extras.get(OpportunityStat.PA)
        if ip is not None:
            row["IP"] = float(ip)
        if pa is not None:
            row["AB"] = float(pa) * AB_PER_PA
        out[e.team_name] = row
    return out


def projected_margin_from_eos(eos_baseline: dict[str, CategoryStats], user_team_name: str) -> float:
    """Signed deterministic projected roto margin: user total minus the best other
    team's total. score_roto_dict accepts the {team: CategoryStats} baseline directly."""
    roto = score_roto_dict(eos_baseline)
    user = roto[user_team_name]["total"]
    others = [v["total"] for t, v in roto.items() if t != user_team_name]
    return float(user - max(others)) if others else float(user)


def load_mc_inputs_from_upstash(config_path: Path | None = None) -> McInputs:
    """Assemble the ROS-MC inputs from STORED (last-refresh vintage) Upstash blobs;
    no Yahoo call. eos_baseline/team_sds are recomputed on the stored
    fraction_remaining so the baseline matches the dashboard's stored MC. (Minor AVG
    drift is possible vs the pipeline's un-persisted ownership-attributed team-AB
    overlay; it cancels in the scenario deltas, which are the deliverable.)"""
    from fantasy_baseball.config import load_config
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key
    from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
    from fantasy_baseball.data.redis_store import get_latest_standings
    from fantasy_baseball.models.positions import BENCH_SLOTS
    from fantasy_baseball.sgp.denominators import get_sgp_denominators

    root = Path(__file__).resolve().parents[3]
    cfg = load_config(config_path or (root / "config" / "league.yaml"))
    kv = build_explicit_upstash_kv()

    def cache(key: CacheKey):  # unwrap the {"_meta","_data"} envelope (cache:* blobs)
        raw = kv.get(redis_key(key))
        if raw is None:
            raise RuntimeError(f"Upstash missing {key}; run a refresh first.")
        o = json.loads(raw) if isinstance(raw, str) else raw
        return o["_data"] if isinstance(o, dict) and "_data" in o else o

    user_blob = cache(CacheKey.ROSTER)
    opp_blob = cache(CacheKey.OPP_ROSTERS)
    proj_blob = cache(CacheKey.PROJECTIONS)

    user_players = [Player.from_dict(p) for p in user_blob]
    opp_players = {t: [Player.from_dict(p) for p in r] for t, r in opp_blob.items()}
    team_rosters = {cfg.team_name: user_players, **opp_players}

    standings = get_latest_standings(kv)  # Standings object (not a cache:* envelope)
    if standings is None:
        raise RuntimeError("Upstash missing standings history; run a refresh first.")
    actual_standings = build_actual_standings(standings)

    fr = float(proj_blob["fraction_remaining"])  # vintage -> matches dashboard MC
    denoms = get_sgp_denominators(cfg.sgp_overrides)

    ytd_by_team = {e.team_name: e.ytd_components() for e in standings.entries}
    eos_baseline = build_eos_baseline(team_rosters, ytd_by_team)
    team_sds = build_team_sds(team_rosters, math.sqrt(fr))

    non_hitter = {str(s) for s in BENCH_SLOTS} | {"P"}
    h_slots = sum(v for k, v in cfg.roster_slots.items() if k not in non_hitter)
    p_slots = cfg.roster_slots.get("P", 9)

    return McInputs(
        team_rosters=team_rosters,
        actual_standings=actual_standings,
        fraction_remaining=fr,
        h_slots=h_slots,
        p_slots=p_slots,
        eos_baseline=eos_baseline,
        team_sds=team_sds,
        denoms=denoms,
        user_team_name=cfg.team_name,
        projected_margin=projected_margin_from_eos(eos_baseline, cfg.team_name),
    )


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


@dataclass(frozen=True)
class PlayerExposure:
    name: str
    player_type: str
    win_pct_cost: float


@dataclass(frozen=True)
class PairExposure:
    name_a: str
    name_b: str
    joint_cost: float
    super_additive: float


@dataclass(frozen=True)
class StressResult:
    baseline_win_pct: float
    availability_off_win_pct: float
    projected_margin: float
    health: HealthProbs
    singles: list[PlayerExposure]
    pairs: list[PairExposure]
    threshold: float
    n_iter: int
    seed: int


def run_stress_test(
    inputs: McInputs,
    *,
    threshold: float = SIGNIFICANT_TIME_THRESHOLD,
    pair_top_k: int = PAIR_TOP_K,
    n_iter: int = DEFAULT_N_ITER,
    seed: int = SEED,
) -> StressResult:
    """Headline win% (baseline + availability-variance-off), per-player health
    probabilities, and ranked single/pair replacement-level counterfactuals for
    the user's active roster. Every `win_pct` call shares `seed` (common random
    numbers) so counterfactual deltas isolate the roster swap, not MC noise."""
    me = inputs.team_rosters[inputs.user_team_name]
    base = win_pct(inputs, me, n_iter=n_iter, seed=seed)
    avail_off = win_pct(inputs, me, availability_variance_off=True, n_iter=n_iter, seed=seed)

    active, _il, _bench = _classify_roster(me)
    health = health_probabilities(active, inputs.fraction_remaining, threshold=threshold, seed=seed)

    singles: list[PlayerExposure] = []
    for p in active:
        wp = win_pct(inputs, substitute_replacement(me, [p.name]), n_iter=n_iter, seed=seed)
        singles.append(PlayerExposure(p.name, p.player_type.value, base - wp))
    singles.sort(key=lambda e: e.win_pct_cost, reverse=True)

    cost_by_name = {e.name: e.win_pct_cost for e in singles}
    top = singles[:pair_top_k]
    pairs: list[PairExposure] = []
    for a, b in itertools.combinations(top, 2):
        wp = win_pct(inputs, substitute_replacement(me, [a.name, b.name]), n_iter=n_iter, seed=seed)
        joint = base - wp
        pairs.append(
            PairExposure(
                a.name, b.name, joint, joint - (cost_by_name[a.name] + cost_by_name[b.name])
            )
        )
    pairs.sort(key=lambda e: e.joint_cost, reverse=True)

    return StressResult(
        baseline_win_pct=base,
        availability_off_win_pct=avail_off,
        projected_margin=inputs.projected_margin,
        health=health,
        singles=singles,
        pairs=pairs,
        threshold=threshold,
        n_iter=n_iter,
        seed=seed,
    )


def _pct(x: float) -> str:
    return f"{x:5.1f}%"


def render_report(result: StressResult) -> str:
    r = result
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("INJURY STRESS-TEST")
    lines.append("=" * 72)

    lines.append("")
    lines.append("1. WHAT INJURY RISK COSTS YOU")
    lines.append("-" * 72)
    margin = r.projected_margin
    side = "ahead of" if margin >= 0 else "behind"
    lines.append(f"  Deterministic projected roto margin : {margin:+.1f} pts ({side} the leader)")
    lines.append(f"  Win% if availability lands as expected: {_pct(r.availability_off_win_pct)}")
    lines.append(f"  Win% (real, with injury risk)         : {_pct(r.baseline_win_pct)}")
    lines.append(
        f"  -> Injury/availability risk costs you : "
        f"{r.availability_off_win_pct - r.baseline_win_pct:+.1f} win pts"
    )

    lines.append("")
    lines.append("2. HOW LIKELY IS EVERYONE STAYS HEALTHY?")
    lines.append("-" * 72)
    thr = round(r.threshold * 100)
    lines.append(f"  (a player 'loses significant time' = >= {thr}% below expected playing time)")
    lines.append(
        f"  P(no active player loses significant time): {_pct(r.health.p_all_healthy * 100)}"
    )
    lines.append(f"  P(exactly one does)                       : {_pct(r.health.p_one * 100)}")
    lines.append(f"  P(two or more)                            : {_pct(r.health.p_two_plus * 100)}")

    lines.append("")
    lines.append("3. WHO ARE YOU MOST EXPOSED TO? (lose one, replaced)")
    lines.append("-" * 72)
    lines.append(f"  {'Player':<24}{'Type':<9}{'win% cost':>10}")
    for e in r.singles:
        lines.append(f"  {e.name[:23]:<24}{e.player_type:<9}{e.win_pct_cost:>9.1f}")

    lines.append("")
    lines.append("4. LOSING TWO (top exposures, ranked by joint win% cost)")
    lines.append("-" * 72)
    lines.append(f"  {'Pair':<40}{'joint':>8}{'vs sum':>9}")
    for p in r.pairs:
        pair = f"{p.name_a[:18]} + {p.name_b[:18]}"
        tag = "  (worse than additive)" if p.super_additive > 0.5 else ""
        lines.append(f"  {pair:<40}{p.joint_cost:>8.1f}{p.super_additive:>+9.1f}{tag}")

    lines.append("")
    lines.append("5. NOTE")
    lines.append("-" * 72)
    lines.append("  Section 2 uses a GENERIC (volume/role) injury model -- every player in a")
    lines.append("  PA/IP band shares the same downside. Per-player injury history is not yet")
    lines.append("  modeled (deferred; see the design's Future work).")
    lines.append(
        f"  MC: n_iter={r.n_iter}, seed={r.seed} (common random numbers across scenarios)."
    )
    return "\n".join(lines)
