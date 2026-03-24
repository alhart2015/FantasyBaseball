"""Batch sweep: run all strategies x scoring modes against strategic opponents.

Supports both deterministic (single run) and jittered (N iterations) modes.
Uses ADP noise to eliminate butterfly-effect artifacts from single-run results.

Usage:
    python scripts/batch_sweep.py                # deterministic (1 iter)
    python scripts/batch_sweep.py --iterations 30 # jittered (30 iters)
"""
import argparse
import csv
import gc
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from simulate_draft import build_board_and_context, run_simulation, save_simulation_output
from fantasy_baseball.draft.strategy import STRATEGIES

STRATEGY_NAMES = list(STRATEGIES.keys())
SCORING_MODES = ["var", "vona"]

# Historical opponent mapping (see analyze_history.py STRATEGY CLASSIFICATION)
# Closer targets based on 2024-2025 avg: Spacemen=4, Jon/HelloP=3,
# Boston/WIP=2, SkeleThor/Springfield=1-2 (took 2 in 2024), TBD=1, Cavalli=new(2)
OPP_STRATEGIES = (
    "1:two_closers,2:two_closers,3:two_closers,4:three_closers,"
    "5:two_closers,6:four_closers,7:two_closers,9:nonzero_sv,10:three_closers"
)

OUT_PATH = PROJECT_ROOT / "data" / "batch_sweep_results.csv"


def main():
    parser = argparse.ArgumentParser(description="Batch strategy sweep")
    parser.add_argument("--iterations", "-n", type=int, default=1,
                        help="Number of iterations per combo (1=deterministic)")
    parser.add_argument("--noise", type=float, default=15.0,
                        help="ADP noise std dev (only used when iterations > 1)")
    args = parser.parse_args()

    iterations = args.iterations
    adp_noise = args.noise if iterations > 1 else 0.0

    t0 = time.time()
    print("Building draft board (once)...")
    ctx = build_board_and_context()
    print(f"Board built in {time.time() - t0:.1f}s")

    total_combos = len(STRATEGY_NAMES) * len(SCORING_MODES)
    total_sims = total_combos * iterations
    mode = f"{iterations} iterations, ADP noise={adp_noise}" if iterations > 1 else "deterministic"
    print(f"Sweeping {len(STRATEGY_NAMES)} strategies x {len(SCORING_MODES)} scoring "
          f"x {iterations} iter = {total_sims} sims ({mode})")
    print(f"Opponents: {OPP_STRATEGIES}")
    print()

    results = []
    sim_start = time.time()
    done_combos = 0
    run_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    for strategy in STRATEGY_NAMES:
        for scoring in SCORING_MODES:
            done_combos += 1
            pts_list = []
            rank_list = []
            wins = 0
            last_result = None

            for i in range(iterations):
                try:
                    r = run_simulation(
                        ctx,
                        strategy_name=strategy,
                        scoring_mode=scoring,
                        adp_noise=adp_noise,
                        seed=2000 + i if iterations > 1 else None,
                        opponent_strategies_str=OPP_STRATEGIES,
                    )
                    pts_list.append(r["pts"])
                    rank_list.append(r["rank"])
                    if r["rank"] == 1:
                        wins += 1
                    last_result = r
                except Exception as e:
                    print(f"    ERROR: {strategy}+{scoring} iter {i}: {e}")
                gc.collect()

            if not pts_list:
                continue

            avg_pts = np.mean(pts_list)
            avg_rank = np.mean(rank_list)
            win_pct = wins / len(pts_list) * 100

            results.append({
                "strategy": strategy,
                "scoring": scoring,
                "avg_pts": round(avg_pts, 1),
                "med_pts": round(np.median(pts_list), 1),
                "min_pts": min(pts_list),
                "max_pts": max(pts_list),
                "avg_rank": round(avg_rank, 2),
                "win_pct": round(win_pct, 1),
                "iterations": len(pts_list),
            })

            # Save full output for last iteration (for roster inspection)
            if iterations == 1 and last_result:
                save_simulation_output(
                    last_result, strategy, scoring, OPP_STRATEGIES, run_ts,
                )

            elapsed = time.time() - sim_start
            done_sims = done_combos * iterations
            rate = done_sims / elapsed if elapsed > 0 else 0
            remaining = total_sims - done_sims
            eta = remaining / rate if rate > 0 else 0

            if iterations > 1:
                print(f"  [{done_combos:>3}/{total_combos}] {strategy:<20} {scoring:<5} -> "
                      f"avg:{avg_pts:>5.1f}pts  rank:{avg_rank:.2f}  "
                      f"win:{win_pct:>4.1f}%  [{min(pts_list):.0f}-{max(pts_list):.0f}]  "
                      f"(~{eta:.0f}s left)")
            else:
                print(f"  [{done_combos:>3}/{total_combos}] {strategy:<20} {scoring:<5} -> "
                      f"{avg_pts:>5.1f}pts rank#{avg_rank:.0f}  "
                      f"(~{eta:.0f}s left)")

    total_time = time.time() - sim_start
    print(f"\nDone in {total_time:.0f}s ({total_sims / total_time:.1f} sims/s)")

    # Write CSV
    fieldnames = ["strategy", "scoring", "avg_pts", "med_pts", "min_pts",
                  "max_pts", "avg_rank", "win_pct", "iterations"]
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"Results saved to {OUT_PATH}")
    if iterations == 1:
        print(f"Full sim outputs saved to data/sim_results/{run_ts}_*.json")

    # Summary
    print(f"\n{'='*90}")
    header = (f"BATCH SWEEP ({mode}, with opponent strategies)"
              if iterations > 1
              else "BATCH SWEEP (deterministic, with opponent strategies)")
    print(header)
    print(f"{'='*90}")

    sorted_results = sorted(results, key=lambda r: r["avg_pts"], reverse=True)

    if iterations > 1:
        print(f"\n{'Strategy':<20} {'Score':<6} {'Avg Pts':>8} {'Avg Rank':>9} "
              f"{'Win%':>6} {'Range':>12}")
        print("-" * 65)
        for r in sorted_results:
            print(f"{r['strategy']:<20} {r['scoring']:<6} {r['avg_pts']:>8.1f} "
                  f"{r['avg_rank']:>9.2f} {r['win_pct']:>5.1f}% "
                  f"[{r['min_pts']:.0f}-{r['max_pts']:.0f}]")
    else:
        print(f"\n{'Strategy':<22} {'Scoring':<8} {'Pts':>5} {'Rank':>5}")
        print("-" * 42)
        for r in sorted_results:
            print(f"{r['strategy']:<22} {r['scoring']:<8} {r['avg_pts']:>5.1f} "
                  f"#{r['avg_rank']:.0f}")

    # Top / bottom 5
    print(f"\nTOP 5:")
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {r['strategy']}+{r['scoring']}: {r['avg_pts']}")
    print(f"\nBOTTOM 5:")
    for i, r in enumerate(sorted_results[-5:], 1):
        print(f"  {i}. {r['strategy']}+{r['scoring']}: {r['avg_pts']}")


if __name__ == "__main__":
    main()
