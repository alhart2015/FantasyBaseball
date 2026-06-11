"""Strategy comparison -- all strategies x 2 scoring modes.

Parallelized across CPU cores. Each strategy+mode combo runs its
iterations in a worker process.

Part 1: Against strategic opponents (20 iterations with pick noise).
Part 2: Against ADP opponents (20 iterations with ADP jitter).

Usage:
    python scripts/compare_strategies.py
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fantasy_baseball.draft.strategy import STRATEGIES
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD

OPP_STRATEGIES = (
    "1:two_closers,2:two_closers,3:two_closers,4:three_closers,"
    "5:two_closers,6:four_closers,7:two_closers,9:nonzero_sv,10:three_closers"
)

ADP_ITERATIONS = 20
ADP_NOISE = 15.0
STRATEGY_NOISE = 1.0
STRATEGY_ITERATIONS = 20
MAX_WORKERS = 16  # use half of 32 cores — each worker is memory-heavy


def _run_strategy_batch(
    strat, scoring, iterations, opp_str, adp_noise, strategy_noise, seed_base, position_aware
):
    """Run one strategy+mode for N iterations. Called in worker process."""
    # Each worker builds its own context (not picklable)
    from simulate_draft import build_board_and_context, run_simulation

    ctx = build_board_and_context()
    config = ctx["config"]

    pts_list = []
    rank_list = []
    wins = 0
    for i in range(iterations):
        r = run_simulation(
            ctx,
            strategy_name=strat,
            scoring_mode=scoring,
            strategy_noise=strategy_noise,
            adp_noise=adp_noise,
            seed=seed_base + i,
            opponent_strategies_str=opp_str,
            position_aware=position_aware,
        )
        hart = next(t for t in r["results"] if t["team"] == config.team_name)
        pts_list.append(hart["tot"])
        rank_list.append(r["rank"])
        if r["rank"] == 1:
            wins += 1

    return {
        "label": f"{strat}+{scoring}",
        "strat": strat,
        "mode": scoring,
        "avg_pts": float(np.mean(pts_list)),
        "avg_rank": float(np.mean(rank_list)),
        "win_pct": wins / len(pts_list) * 100,
        "floor": min(pts_list),
        "ceil": max(pts_list),
    }


def _flag_inert(results):
    """Mark strategies whose aggregate result is identical to ``default`` for the
    same scoring mode. An overlay that never fires defers to the slot-gated
    default pick, producing a byte-identical draft -- so exact equality across
    all iterations means the overlay added nothing. Mutates ``results`` in place,
    adding an ``inert`` bool. (Honest-analysis guard: never again present a
    no-op overlay as a distinct strategy.)
    """
    default_by_mode = {
        r["mode"]: (r["avg_pts"], r["avg_rank"], r["win_pct"])
        for r in results
        if r.get("strat") == "default"
    }
    for r in results:
        base = default_by_mode.get(r["mode"])
        r["inert"] = (
            r.get("strat") != "default"
            and base is not None
            and (r["avg_pts"], r["avg_rank"], r["win_pct"]) == base
        )


def _print_ranking(results):
    _flag_inert(results)
    n_inert = sum(1 for r in results if r.get("inert"))
    distinct = [r for r in results if not r.get("inert")]
    print(f"{'#':>3} {'Strategy':<30} {'Avg':>5} {'AvgRk':>6} {'Win%':>5} {'Floor':>5} {'Ceil':>5}")
    print("-" * 72)
    for i, r in enumerate(sorted(results, key=lambda x: -x["avg_pts"]), 1):
        tag = "  INERT(==default)" if r.get("inert") else ""
        print(
            f"{i:>3} {r['label']:<30} {r['avg_pts']:>5.1f} {r['avg_rank']:>6.2f} "
            f"{r['win_pct']:>5.1f} {r['floor']:>5.0f} {r['ceil']:>5.0f}{tag}"
        )
    print(
        f"\n  {n_inert}/{len(results)} combos are INERT (overlay never fired -- identical to "
        f"default). {len(distinct)} combos represent genuinely distinct behavior."
    )


def main():
    # Gate: "gated" (position_aware=True, the correct roster discipline -- the
    # dashboard drafts gated) is the default; "ungated" reproduces the old
    # verdict's methodology. See the slot-gate discussion in the draft CLAUDE.md.
    gate_arg = sys.argv[1] if len(sys.argv) > 1 else "gated"
    position_aware = gate_arg != "ungated"

    strategy_names = sorted(STRATEGIES.keys())
    scoring_modes = ["var", "vona", "deltaroto_immediate", "deltaroto_vopn"]
    n_combos = len(strategy_names) * len(scoring_modes)

    print(f"Strategies: {len(strategy_names)}, modes: {len(scoring_modes)}, workers: {MAX_WORKERS}")
    print(f"Gate: {gate_arg} (position_aware={position_aware})")
    print("Each worker builds its own board (~10s startup)\n")

    # ---------------------------------------------------------------
    # PART 1: Strategic opponents with pick noise
    # ---------------------------------------------------------------
    print("=" * 100)
    print(
        f"PART 1: vs STRATEGIC OPPONENTS ({STRATEGY_ITERATIONS} iterations, "
        f"pick_noise={STRATEGY_NOISE}, {MAX_WORKERS} workers)"
    )
    print("=" * 100)
    print(f"  Opponents: {OPP_STRATEGIES}\n")

    t0 = time.perf_counter()
    strat_results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for strat in strategy_names:
            for scoring in scoring_modes:
                f = pool.submit(
                    _run_strategy_batch,
                    strat,
                    scoring,
                    STRATEGY_ITERATIONS,
                    OPP_STRATEGIES,
                    0.0,
                    STRATEGY_NOISE,
                    4000,
                    position_aware,
                )
                futures[f] = f"{strat}+{scoring}"
        for f in as_completed(futures):
            try:
                result = f.result()
                strat_results.append(result)
                done = len(strat_results)
                print(
                    f"{result['label']:<30} {result['avg_pts']:>5.1f} "
                    f"{result['avg_rank']:>6.2f} {result['win_pct']:>5.1f} "
                    f"{result['floor']:>5.0f} {result['ceil']:>5.0f}"
                    f"  [{done}/{n_combos}]",
                    flush=True,
                )
            except Exception as e:
                print(f"{futures[f]:<30} ERROR: {e}", flush=True)
    t1 = time.perf_counter()
    print(f"\nPart 1 completed in {t1 - t0:.1f}s ({(t1 - t0) / 60:.1f}m)\n")
    _print_ranking(strat_results)

    # ---------------------------------------------------------------
    # PART 2: ADP opponents with jitter
    # ---------------------------------------------------------------
    print()
    print("=" * 100)
    print(
        f"PART 2: vs ADP OPPONENTS ({ADP_ITERATIONS} iterations, "
        f"noise={ADP_NOISE}, {MAX_WORKERS} workers)"
    )
    print("=" * 100)
    print()

    t2 = time.perf_counter()
    adp_results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for strat in strategy_names:
            for scoring in scoring_modes:
                f = pool.submit(
                    _run_strategy_batch,
                    strat,
                    scoring,
                    ADP_ITERATIONS,
                    "",
                    ADP_NOISE,
                    0.0,
                    3000,
                    position_aware,
                )
                futures[f] = f"{strat}+{scoring}"
        for f in as_completed(futures):
            try:
                result = f.result()
                adp_results.append(result)
                done = len(adp_results)
                print(
                    f"{result['label']:<30} {result['avg_pts']:>5.1f} "
                    f"{result['avg_rank']:>6.2f} {result['win_pct']:>5.1f} "
                    f"{result['floor']:>5.0f} {result['ceil']:>5.0f}"
                    f"  [{done}/{n_combos}]",
                    flush=True,
                )
            except Exception as e:
                print(f"{futures[f]:<30} ERROR: {e}", flush=True)
    t3 = time.perf_counter()
    print(f"\nPart 2 completed in {t3 - t2:.1f}s ({(t3 - t2) / 60:.1f}m)\n")
    _print_ranking(adp_results)

    print()
    print(f"Total time: {t3 - t0:.1f}s ({(t3 - t0) / 60:.1f}m)")
    print("Done.")


if __name__ == "__main__":
    main()
