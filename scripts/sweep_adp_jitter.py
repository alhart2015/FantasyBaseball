"""Sweep strategies against ADP-drafting opponents with random jitter.

Runs N iterations per strategy with different random seeds to get
stable average results. Opponents draft by ADP with Gaussian noise,
always filling active slots first then bench.
"""
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from simulate_draft import build_board_and_context, run_simulation
from fantasy_baseball.draft.strategy import STRATEGIES

STRATEGY_NAMES = [
    "default", "nonzero_sv", "two_closers", "three_closers",
    "no_punt_opp", "avg_anchor", "no_punt_cap3",
]
SCORING_MODES = ["var", "vona"]
ADP_NOISE = 15.0  # Gaussian std dev on ADP (±15 picks of randomness)
ITERATIONS = 50
OUT_PATH = PROJECT_ROOT / "data" / "sweep_adp_jitter_results.csv"


def main():
    t0 = time.time()
    print("Building draft board (once)...")
    ctx = build_board_and_context()
    print(f"Board built in {time.time() - t0:.1f}s")

    total = len(STRATEGY_NAMES) * len(SCORING_MODES)
    print(f"Sweeping {len(STRATEGY_NAMES)} strategies x {len(SCORING_MODES)} scoring "
          f"x {ITERATIONS} iterations = {total * ITERATIONS} sims")
    print(f"ADP noise: {ADP_NOISE} (Gaussian std dev)")
    print(f"Opponents: pure ADP (no strategies)")
    print()

    results = []
    sim_start = time.time()
    done_combos = 0

    for strategy in STRATEGY_NAMES:
        for scoring in SCORING_MODES:
            done_combos += 1
            pts_list = []
            rank_list = []
            wins = 0

            for i in range(ITERATIONS):
                try:
                    r = run_simulation(
                        ctx,
                        strategy_name=strategy,
                        scoring_mode=scoring,
                        adp_noise=ADP_NOISE,
                        seed=1000 + i,
                        opponent_strategies_str="",
                    )
                    pts_list.append(r["pts"])
                    rank_list.append(r["rank"])
                    if r["rank"] == 1:
                        wins += 1
                except Exception as e:
                    print(f"    ERROR: {strategy}+{scoring} iter {i}: {e}")

                gc.collect()

            avg_pts = np.mean(pts_list) if pts_list else 0
            avg_rank = np.mean(rank_list) if rank_list else 0
            med_pts = np.median(pts_list) if pts_list else 0
            min_pts = min(pts_list) if pts_list else 0
            max_pts = max(pts_list) if pts_list else 0
            win_pct = wins / len(pts_list) * 100 if pts_list else 0

            results.append({
                "strategy": strategy,
                "scoring": scoring,
                "avg_pts": round(avg_pts, 1),
                "med_pts": round(med_pts, 1),
                "min_pts": min_pts,
                "max_pts": max_pts,
                "avg_rank": round(avg_rank, 2),
                "win_pct": round(win_pct, 1),
                "iterations": len(pts_list),
            })

            elapsed = time.time() - sim_start
            rate = (done_combos * ITERATIONS) / elapsed if elapsed > 0 else 0
            remaining = (total - done_combos) * ITERATIONS
            eta = remaining / rate if rate > 0 else 0

            print(f"  [{done_combos:>2}/{total}] {strategy:<16} {scoring:<5} -> "
                  f"avg:{avg_pts:>5.1f}pts  rank:{avg_rank:.2f}  "
                  f"win:{win_pct:>4.1f}%  range:[{min_pts:.0f}-{max_pts:.0f}]  "
                  f"(~{eta:.0f}s left)")

    total_time = time.time() - sim_start
    print(f"\nDone in {total_time:.0f}s ({total * ITERATIONS / total_time:.1f} sims/s)")

    # Write CSV
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "strategy", "scoring", "avg_pts", "med_pts", "min_pts",
            "max_pts", "avg_rank", "win_pct", "iterations",
        ])
        w.writeheader()
        w.writerows(results)
    print(f"Results saved to {OUT_PATH}")

    # Summary
    print(f"\n{'='*90}")
    print(f"SWEEP RESULTS ({ITERATIONS} iterations, ADP noise={ADP_NOISE}, no opponent strategies)")
    print(f"{'='*90}")
    print(f"\n{'Strategy':<20} {'Score':<6} {'Avg Pts':>8} {'Avg Rank':>9} "
          f"{'Win%':>6} {'Range':>12}")
    print("-" * 65)

    sorted_results = sorted(results, key=lambda r: r["avg_pts"], reverse=True)
    for r in sorted_results:
        print(f"{r['strategy']:<20} {r['scoring']:<6} {r['avg_pts']:>8.1f} "
              f"{r['avg_rank']:>9.2f} {r['win_pct']:>5.1f}% "
              f"[{r['min_pts']:.0f}-{r['max_pts']:.0f}]")


if __name__ == "__main__":
    main()
