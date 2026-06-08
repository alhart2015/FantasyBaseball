"""Throwaway: how does PURE deltaRoto perform in the draft bake-off?

Drops two deltaRoto strategies into the existing simulate_draft harness and
runs them through both arenas -- alongside hand-built baselines -- so the
table is self-contained.

    deltaroto_immediate : pick highest immediate_delta (mirrors live dashboard)
    deltaroto_vopn      : pick highest value_of_picking_now (deltaRoto-native VONA)

PURE deltaRoto: the only constraint is roster-legality (no closer timing, no
AVG floor). Each pick rebuilds 10-team ProjectedStandings from the live sim
rosters and ranks the candidate pool with eroto_recs.rank_candidates -- the
exact engine behind the live dashboard /api/recs.

Methodology notes:
  * Variation in BOTH arenas comes from adp_noise (opponent-side), never
    strategy_noise. The harness's strategy_noise path draws its alternate
    pick from the VAR/VONA recommender, which would inject VAR picks into the
    deltaRoto runs and contaminate the test. adp_noise jitters opponent picks
    across seeds while leaving our own (clean-ADP) deltaRoto pick untouched.
  * deltaRoto ignores scoring_mode; everything runs once under "var".
  * Process-isolated batches (like compare_strategies): a single long-lived
    process accumulates a pandas/native memory-corruption crash over hundreds
    of sims, so each (strategy, arena) batch runs in its own worker process.

Read-only analysis. Reuses the tested run_simulation loop and _score_roto, so
deltaRoto is scored by the same currency as the hand strategies -- a fair fight.
"""

import os

# Pin native math libs to one thread BEFORE numpy/pandas import (inherited by
# spawned workers). Reduces native-crash flakiness in the long opponent-VAR
# pandas loops; process isolation below is the actual stability fix.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from compare_strategies import OPP_STRATEGIES

from fantasy_baseball.draft.eroto_recs import rank_candidates
from fantasy_baseball.draft.recs_integration import (
    _build_replacements,
    build_adp_table,
    build_projected_standings,
    build_team_rosters,
)
from fantasy_baseball.draft.roster_state import RosterState, get_filled_positions
from fantasy_baseball.draft.state import StateKey
from fantasy_baseball.draft.strategy import STRATEGIES, build_player_lookup
from fantasy_baseball.models.player import Player
from fantasy_baseball.scoring import build_team_sds
from fantasy_baseball.utils.positions import can_fill_slot

ITERATIONS = 100  # total per (candidate, position-arm)
CHUNK_ITERS = 5  # iters per worker process; fresh process per chunk bounds the
#                  native pandas memory-corruption crash seen in long sim runs.
SEED_BASE = 5000  # same seed set for every candidate/arm -> paired comparison
POOL_CAP = 200  # top-N candidates by VAR scored per pick (matches dashboard)

# Every value model, evaluated explicitly: (label, strategy_name, scoring_mode).
# VAR/VONA are pick_default under the two scoring modes; the closer strategies
# are the imperative overlays; deltaRoto immediate/vopn are the adapters.
CANDIDATES = [
    ("deltaroto_immediate", "deltaroto_immediate", "var"),
    ("deltaroto_vopn", "deltaroto_vopn", "var"),
    ("VAR", "default", "var"),
    ("VONA", "default", "vona"),
    ("three_closers", "three_closers", "var"),
    ("two_closers", "two_closers", "var"),
    ("nonzero_sv", "nonzero_sv", "var"),
]
DELTAROTO = {
    "deltaroto_immediate": "immediate_delta",
    "deltaroto_vopn": "value_of_picking_now",
}

# Static per-board inputs don't change pick to pick: cache keyed on board id.
_STATIC_CACHE: dict[int, tuple] = {}


def _static_inputs(board, config):
    """Board-derived inputs constant across the draft: per-position replacement
    Players, ADP table, pid->Player map, var-ordered pids, team names.
    """
    key = id(board)
    cached = _STATIC_CACHE.get(key)
    if cached is not None:
        return cached
    board_by_id: dict[str, Player] = {}
    for row in board.to_dict("records"):
        p = Player.from_dict(row)
        if p.yahoo_id:
            board_by_id[p.yahoo_id] = p
    replacements = _build_replacements(board, config.roster_slots, config.num_teams)
    adp_table = build_adp_table(board)
    pool_sorted = board.sort_values("var", ascending=False) if "var" in board.columns else board
    ordered_pids = list(pool_sorted["player_id"])
    team_names = list(config.teams.values())
    cached = (board_by_id, replacements, adp_table, ordered_pids, team_names)
    _STATIC_CACHE[key] = cached
    return cached


def make_deltaroto_pick(sort_attr):
    """Build a strategy fn that picks the candidate maximizing ``sort_attr``
    (``immediate_delta`` or ``value_of_picking_now``) each turn.
    """

    def pick(board, full_board, tracker, _balance, config, _team_filled, **kwargs):
        player_lookup = kwargs.get("player_lookup") or build_player_lookup(board, full_board)
        team_rosters = kwargs.get("team_rosters") or {}
        board_by_id, replacements, adp_table, ordered_pids, team_names = _static_inputs(
            board, config
        )

        # Synthetic draft-state from the live sim rosters (keepers already folded
        # into team_rosters by run_simulation). build_team_rosters does the
        # type-aware replacement padding so early-round standings aren't degenerate.
        picks = [
            {"player_id": pid, "team": config.teams[num], "position": ""}
            for num, pids in team_rosters.items()
            for pid in pids
        ]
        state = {StateKey.KEEPERS.value: [], StateKey.PICKS.value: picks}
        rosters = build_team_rosters(
            state, board_by_id, team_names, config.roster_slots, replacements
        )
        standings = build_projected_standings(rosters)
        team_sds = build_team_sds(rosters, sd_scale=1.0)

        # Candidate pool: undrafted, roster-legal for our team, top-N by var.
        drafted = set(tracker.drafted_ids)
        filled = get_filled_positions(
            tracker.user_roster_ids,
            full_board,
            roster_slots=config.roster_slots,
            player_lookup=player_lookup,
        )
        roster_state = RosterState.from_dicts(filled, config.roster_slots)
        candidates: list[Player] = []
        for pid in ordered_pids:
            if pid in drafted:
                continue
            p = board_by_id.get(pid)
            if p is None:
                continue
            row = player_lookup.get(pid)
            positions = row["positions"] if row is not None else p.positions
            if not roster_state.any_slot_open_for(positions):
                continue
            candidates.append(p)
            if len(candidates) >= POOL_CAP:
                break
        if not candidates:
            return None, None

        rows = rank_candidates(
            candidates=candidates,
            replacements=replacements,
            team_name=config.team_name,
            projected_standings=standings,
            team_sds=team_sds,
            picks_until_next_turn=tracker.picks_until_next_turn,
            adp_table=adp_table,
        )
        if not rows:
            return None, None
        rows.sort(key=lambda r: getattr(r, sort_attr), reverse=True)

        # Position-aware gate + k-th-choice noise, mirroring strategy._choose_rec.
        pool = rows
        if kwargs.get("position_aware"):
            open_starters = roster_state.unfilled_starter_slots()
            if open_starters:
                starter_rows = [
                    r for r in rows if any(can_fill_slot(r.positions, s) for s in open_starters)
                ]
                if starter_rows:
                    pool = starter_rows
        pick_rank = int(kwargs.get("pick_rank", 0) or 0)
        best = pool[min(pick_rank, len(pool) - 1)]
        return best.name, best.player_id

    return pick


def _run_chunk_worker(label, strat, scoring, position_aware, iters, seed_base):
    """Run ``iters`` strategic-field sims for one (candidate, position-arm) in a
    fresh process; return the raw per-iter pts/ranks so chunks aggregate exactly.

    Field noise is on (each pick takes its algorithm's k-th choice -> the field
    varies across seeds). Opponents are the closer mix (OPP_STRATEGIES).
    """
    from simulate_draft import build_board_and_context, run_simulation

    for name, attr in DELTAROTO.items():
        STRATEGIES[name] = make_deltaroto_pick(attr)
    ctx = build_board_and_context()
    config = ctx["config"]

    pts, ranks = [], []
    for i in range(iters):
        r = run_simulation(
            ctx,
            strategy_name=strat,
            scoring_mode=scoring,
            adp_noise=0.0,
            strategy_noise=0.0,
            position_aware=position_aware,
            field_noise=True,
            seed=seed_base + i,
            opponent_strategies_str=OPP_STRATEGIES,
        )
        hart = next(t for t in r["results"] if t["team"] == config.team_name)
        pts.append(hart["tot"])
        ranks.append(r["rank"])
    return {"label": label, "position_aware": position_aware, "pts": pts, "ranks": ranks}


def _summary(label, pts, ranks):
    wins = sum(1 for rk in ranks if rk == 1)
    return {
        "label": label,
        "avg_pts": float(np.mean(pts)),
        "avg_rank": float(np.mean(ranks)),
        "win_pct": wins / len(pts) * 100,
        "floor": min(pts),
        "ceil": max(pts),
        "n": len(pts),
    }


def _print_ranking(rows):
    print(
        f"{'#':>3} {'Strategy':<22} {'Avg':>6} {'AvgRk':>6} {'Win%':>5} {'Floor':>6} {'Ceil':>6} {'n':>4}"
    )
    print("-" * 66)
    for i, r in enumerate(sorted(rows, key=lambda x: -x["avg_pts"]), 1):
        print(
            f"{i:>3} {r['label']:<22} {r['avg_pts']:>6.1f} {r['avg_rank']:>6.2f} "
            f"{r['win_pct']:>5.1f} {r['floor']:>6.0f} {r['ceil']:>6.0f} {r['n']:>4}"
        )


def main():
    n_chunks = ITERATIONS // CHUNK_ITERS
    jobs = []  # (label, strat, scoring, position_aware, iters, seed_base)
    for label, strat, scoring in CANDIDATES:
        for pa in (True, False):
            for c in range(n_chunks):
                jobs.append((label, strat, scoring, pa, CHUNK_ITERS, SEED_BASE + c * CHUNK_ITERS))

    print(
        f"Position-aware experiment: {len(CANDIDATES)} candidates x 2 arms "
        f"x {ITERATIONS} iters = {len(jobs)} chunks of {CHUNK_ITERS}"
    )
    print("Strategic field (closer mix), field_noise on (k-th-choice variance).")
    print(f"Opponents: {OPP_STRATEGIES}\n")

    acc: dict[tuple, dict] = {}
    t0 = time.perf_counter()
    # Fresh process per chunk (max_tasks_per_child=1) bounds the native crash.
    with ProcessPoolExecutor(max_workers=10, max_tasks_per_child=1) as pool:
        futures = {
            pool.submit(_run_chunk_worker, label, strat, scoring, pa, iters, seed): (label, pa)
            for (label, strat, scoring, pa, iters, seed) in jobs
        }
        for done, f in enumerate(as_completed(futures), 1):
            label, pa = futures[f]
            try:
                r = f.result()
                bucket = acc.setdefault((r["label"], r["position_aware"]), {"pts": [], "ranks": []})
                bucket["pts"].extend(r["pts"])
                bucket["ranks"].extend(r["ranks"])
                tag = "aware" if pa else "unaware"
                print(f"  [{done}/{len(jobs)}] {label:<20} {tag:<7} +{len(r['pts'])}", flush=True)
            except Exception as e:  # a crashed chunk is reported, others continue
                tag = "aware" if pa else "unaware"
                print(f"  [{done}/{len(jobs)}] {label:<20} {tag:<7} ERROR: {e}", flush=True)

    print(f"\nAll chunks done in {(time.perf_counter() - t0) / 60:.1f}m")

    aware = [_summary(lbl, b["pts"], b["ranks"]) for (lbl, pa), b in acc.items() if pa and b["pts"]]
    unaware = [
        _summary(lbl, b["pts"], b["ranks"]) for (lbl, pa), b in acc.items() if not pa and b["pts"]
    ]

    print("\n" + "=" * 66)
    print(f"EVERYONE POSITION-AWARE  (strategic field, up to {ITERATIONS} iters)")
    print("=" * 66)
    _print_ranking(aware)

    print("\n" + "=" * 66)
    print(f"EVERYONE POSITION-UNAWARE  (strategic field, up to {ITERATIONS} iters)")
    print("=" * 66)
    _print_ranking(unaware)

    print("\n" + "=" * 66)
    print("POSITION-AWARE EFFECT  (aware avg - unaware avg, per candidate)")
    print("=" * 66)
    ua = {r["label"]: r["avg_pts"] for r in unaware}
    for r in sorted(aware, key=lambda x: -(x["avg_pts"] - ua.get(x["label"], x["avg_pts"]))):
        d = r["avg_pts"] - ua.get(r["label"], r["avg_pts"])
        print(
            f"  {r['label']:<22} aware={r['avg_pts']:>5.1f}  unaware={ua.get(r['label'], 0):>5.1f}  delta={d:+.1f}"
        )


if __name__ == "__main__":
    main()
