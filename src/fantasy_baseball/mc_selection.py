"""Active-set selection helpers for the Phase 0 selection-attribution diagnostic.

Produce the fixed column indices consumed by
``simulation.simulate_remaining_season_batch(active_cols=...)``, run the MC under
three selection arms (per-iteration top-k / fixed top-k / active-slot), and
format the comparison. Diagnostic-only: NO games plumbing, NO fill engine.
"""

from __future__ import annotations

import numpy as np

from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.scoring import _classify_roster
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


def run_selection_attribution(
    team_rosters: dict,
    actual_standings: dict,
    fraction_remaining: float,
    h_slots: int,
    p_slots: int,
    n_iter: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run the MC under three selection arms; return per-team category medians.

    Arms: ``topk_per_iter`` (today), ``topk_fixed`` (top-k fixed once on the mean
    -> isolates re-selection churn), ``active_slot`` (active slots, bench+IL
    excluded -> isolates bench seating on top of churn). All arms share one seed.
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
    return out


def format_attribution_table(res: dict, teams: list[str] | None = None) -> str:
    """ASCII table: per team, per category, the three arms' median totals."""
    arms = ["topk_per_iter", "topk_fixed", "active_slot"]
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
