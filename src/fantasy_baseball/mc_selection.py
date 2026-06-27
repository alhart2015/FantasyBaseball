"""Active-set selection helpers for the Phase 0 selection-attribution diagnostic.

Produce the fixed column indices consumed by
``simulation.simulate_remaining_season_batch(active_cols=...)``, run the MC under
three selection arms (per-iteration top-k / fixed top-k / active-slot), and
format the comparison. Diagnostic-only: NO games plumbing, NO fill engine.
"""

from __future__ import annotations

import numpy as np

from fantasy_baseball.mc_roster import EffectiveRoster, build_effective_roster
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.scoring import LeagueContext, _classify_roster
from fantasy_baseball.simulation import (
    _CLOSER_RANK_BONUS,
    _flatten_full_season,
    simulate_remaining_season_batch,
)
from fantasy_baseball.utils.constants import ALL_CATEGORIES, CLOSER_SV_THRESHOLD

_CATS = [c.value for c in ALL_CATEGORIES]


def _is_hitter(p) -> bool:
    ptype = p.player_type if isinstance(p, Player) else p.get("player_type")
    return ptype == PlayerType.HITTER


def compute_active_slot_cols(players: list) -> dict[str, np.ndarray]:
    """Columns of the active-slot players (healthy bench AND IL excluded).

    Uses the canonical Player-typed ``_classify_roster``. Identity by object,
    so same-name players never collide.
    """
    active, _il, _bench = _classify_roster([p for p in players if isinstance(p, Player)])
    active_ids = {id(p) for p in active}

    h_cols: list[int] = []
    p_cols: list[int] = []
    hi = pi = 0
    for p in players:
        if _is_hitter(p):
            if id(p) in active_ids:
                h_cols.append(hi)
            hi += 1
        else:
            if id(p) in active_ids:
                p_cols.append(pi)
            pi += 1
    return {"h": np.array(h_cols, dtype=int), "p": np.array(p_cols, dtype=int)}


def compute_fixed_topk_cols(
    flat_players: list[dict], h_slots: int, p_slots: int
) -> dict[str, np.ndarray]:
    """Top-k columns by projected mean stats, fixed once (no per-iteration churn)."""
    h_keys: list[tuple[float, int]] = []
    p_keys: list[tuple[float, int]] = []
    hi = pi = 0
    for p in flat_players:
        if _is_hitter(p):
            key = p.get("r", 0) + p.get("hr", 0) + p.get("rbi", 0) + p.get("sb", 0)
            h_keys.append((key, hi))
            hi += 1
        else:
            sv = p.get("sv", 0)
            bonus = _CLOSER_RANK_BONUS if sv >= CLOSER_SV_THRESHOLD else 0.0
            key = bonus + p.get("w", 0) + p.get("k", 0) + sv
            p_keys.append((key, pi))
            pi += 1

    def _top(keys: list[tuple[float, int]], k: int) -> np.ndarray:
        chosen = [idx for _, idx in sorted(keys, reverse=True)[:k]]
        return np.array(sorted(chosen), dtype=int)

    return {"h": _top(h_keys, h_slots), "p": _top(p_keys, p_slots)}


def _build_effective_rosters(
    team_rosters: dict,
    eos_baseline: dict,
    team_sds: dict,
    fraction_remaining: float,
) -> dict[str, EffectiveRoster]:
    """Build each team's EffectiveRoster from the live Player rosters.

    Mirrors the engine's pipeline setup (refresh_pipeline._run_ros_monte_carlo):
    the SAME LeagueContext (baseline, team_sds, fraction_remaining) the standings
    build used, so the new-engine arm agrees with ERoto by construction.
    """
    out: dict[str, EffectiveRoster] = {}
    for tname, roster in team_rosters.items():
        players = [p for p in roster if isinstance(p, Player)]
        lc = LeagueContext(
            baseline_other_team_stats={t: s for t, s in eos_baseline.items() if t != tname},
            team_sds=team_sds,
            team_name=tname,
            fraction_remaining=fraction_remaining,
        )
        out[tname] = build_effective_roster(players, lc)
    return out


def run_selection_attribution(
    team_rosters: dict,
    actual_standings: dict,
    fraction_remaining: float,
    h_slots: int,
    p_slots: int,
    n_iter: int,
    seed: int,
    eos_baseline: dict | None = None,
    team_sds: dict | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run the MC under up to four selection arms; return per-team category medians.

    Arms:
    - ``topk_per_iter`` (today): per-iteration top-k re-selection.
    - ``topk_fixed``: top-k fixed once on the mean -> isolates re-selection churn.
    - ``active_slot``: active slots only, bench+IL excluded -> isolates bench
      seating on top of churn.
    - ``new_engine``: the ROS-direct body engine -- a fixed active set with IL
      displacement + bench injury-fill, routed via ``effective_rosters`` through
      ``_simulate_team_hitters_ros_direct``. Bypasses ``active_cols`` entirely
      (the body-direct path, mirroring the live engine's dual route).

    The first three arms share the flat-dict ``active_cols`` column mechanism. The
    ``new_engine`` arm requires the standings context (``eos_baseline`` + ``team_sds``)
    to build per-team ``LeagueContext`` + ``EffectiveRoster``. When that context is
    absent (e.g. a slot-less synthetic test), the ``new_engine`` arm is SKIPPED and
    the result contains only the first three arms -- the diagnostic does not crash.
    All arms share one seed.
    """
    flat = {t: [_flatten_full_season(p) for p in players] for t, players in team_rosters.items()}
    active = {t: compute_active_slot_cols(players) for t, players in team_rosters.items()}
    topk = {t: compute_fixed_topk_cols(flat[t], h_slots, p_slots) for t in flat}

    arms: dict[str, dict | None] = {
        "topk_per_iter": None,
        "topk_fixed": topk,
        "active_slot": active,
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for arm, cols in arms.items():
        rng = np.random.default_rng(seed)
        batch = simulate_remaining_season_batch(
            actual_standings,
            flat,
            fraction_remaining,
            rng,
            h_slots,
            p_slots,
            n_iter,
            active_cols=cols,
        )
        out[arm] = {t: {c: float(np.median(batch[t][c])) for c in _CATS} for t in flat}

    # 4th arm: NEW engine (body-direct). Requires the standings context to build
    # EffectiveRosters; absent it, skip the arm (do not crash on slot-less tests).
    if eos_baseline is not None and team_sds is not None:
        effective_rosters = _build_effective_rosters(
            team_rosters, eos_baseline, team_sds, fraction_remaining
        )
        rng = np.random.default_rng(seed)
        batch = simulate_remaining_season_batch(
            actual_standings,
            flat,
            fraction_remaining,
            rng,
            h_slots,
            p_slots,
            n_iter,
            effective_rosters=effective_rosters,
        )
        out["new_engine"] = {t: {c: float(np.median(batch[t][c])) for c in _CATS} for t in flat}
    return out


def format_attribution_table(res: dict, teams: list[str] | None = None) -> str:
    """ASCII table: per team, per category, each arm's median totals.

    Includes the ``new_engine`` column when that arm is present in ``res`` (it is
    skipped when the standings context was unavailable -- see
    ``run_selection_attribution``).
    """
    arms = ["topk_per_iter", "topk_fixed", "active_slot"]
    if "new_engine" in res:
        arms.append("new_engine")
    if teams is None:
        teams = sorted(next(iter(res.values())).keys())
    lines: list[str] = []
    for team in teams:
        lines.append(f"== {team} ==")
        lines.append(f"{'cat':<6}" + "".join(f"{a:>16}" for a in arms))
        for c in _CATS:
            row = "".join(f"{res[a][team][c]:>16.2f}" for a in arms)
            lines.append(f"{c:<6}{row}")
        lines.append("")
    return "\n".join(lines)
