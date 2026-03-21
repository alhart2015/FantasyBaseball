"""Batch sweep: run all strategies x scoring modes x seeds in-process.

Builds the draft board ONCE, then runs all combinations against it.
"""
import csv
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from simulate_draft import build_board_and_context, run_simulation
from fantasy_baseball.draft.strategy import STRATEGIES

STRATEGY_NAMES = list(STRATEGIES.keys())
SCORING_MODES = ["var", "vona"]
SEEDS = list(range(10))
ADP_NOISE = 15

# Historical opponent mapping (see analyze_history.py STRATEGY CLASSIFICATION)
OPP_STRATEGIES = (
    "1:three_closers,2:default,3:nonzero_sv,4:three_closers,"
    "5:nonzero_sv,6:nonzero_sv,7:default,9:nonzero_sv,10:three_closers"
)

OUT_PATH = PROJECT_ROOT / "data" / "batch_sweep_results.csv"


def main():
    t0 = time.time()
    print("Building draft board (once)...")
    ctx = build_board_and_context()
    board_time = time.time() - t0
    print(f"Board built in {board_time:.1f}s")
    print(f"Sweeping {len(STRATEGY_NAMES)} strategies x {len(SCORING_MODES)} scoring x {len(SEEDS)} seeds = {len(STRATEGY_NAMES)*len(SCORING_MODES)*len(SEEDS)} sims")
    print()

    results = []
    total = len(STRATEGY_NAMES) * len(SCORING_MODES) * len(SEEDS)
    done = 0
    sim_start = time.time()

    for strategy in STRATEGY_NAMES:
        for scoring in SCORING_MODES:
            for seed in SEEDS:
                done += 1
                try:
                    r = run_simulation(
                        ctx,
                        strategy_name=strategy,
                        scoring_mode=scoring,
                        adp_noise=ADP_NOISE,
                        seed=seed,
                        opponent_strategies_str=OPP_STRATEGIES,
                    )
                    pts = r["pts"]
                    rank = r["rank"]
                except Exception as e:
                    pts = -1
                    rank = -1

                results.append({
                    "strategy": strategy,
                    "scoring": scoring,
                    "seed": seed,
                    "pts": pts,
                    "rank": rank,
                })

            # Progress after each strategy+scoring combo (every 10 sims)
            elapsed = time.time() - sim_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  [{done:>3}/{total}] {strategy:<20} {scoring:<5} -> "
                  f"med {statistics.median([r['pts'] for r in results[-len(SEEDS):] if r['pts']>0]):>4.0f}pts  "
                  f"({rate:.1f} sims/s, ~{eta:.0f}s left)")

    sim_time = time.time() - sim_start
    print(f"\nAll sims done in {sim_time:.1f}s ({total/sim_time:.1f} sims/s)")

    # Write CSV
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "scoring", "seed", "pts", "rank"])
        w.writeheader()
        w.writerows(results)
    print(f"Results saved to {OUT_PATH}")

    # Summary
    print(f"\n{'='*80}")
    print("BATCH SWEEP RESULTS (with opponent strategies)")
    print(f"{'='*80}")
    print(f"\n{'Strategy':<20} {'Scoring':<8} {'Mean Pts':>9} {'Med Pts':>9} {'Mean Rk':>8} {'Best':>5} {'Worst':>6}")
    print("-" * 75)

    for scoring in SCORING_MODES:
        for strategy in STRATEGY_NAMES:
            subset = [r for r in results if r["strategy"] == strategy and r["scoring"] == scoring and r["pts"] > 0]
            if not subset:
                continue
            pts_list = [r["pts"] for r in subset]
            rk_list = [r["rank"] for r in subset]
            mean_pts = statistics.mean(pts_list)
            med_pts = statistics.median(pts_list)
            mean_rk = statistics.mean(rk_list)
            best = max(pts_list)
            worst = min(pts_list)
            print(f"{strategy:<20} {scoring:<8} {mean_pts:>9.1f} {med_pts:>9.1f} {mean_rk:>8.1f} {best:>5} {worst:>6}")

    # Top / bottom 5
    combos = []
    for scoring in SCORING_MODES:
        for strategy in STRATEGY_NAMES:
            subset = [r for r in results if r["strategy"] == strategy and r["scoring"] == scoring and r["pts"] > 0]
            if subset:
                combos.append((strategy, scoring, statistics.mean([r["pts"] for r in subset])))
    combos.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTOP 5 BY MEAN POINTS:")
    for i, (s, sc, m) in enumerate(combos[:5], 1):
        print(f"  {i}. {s}+{sc}: {m:.1f}")

    print(f"\nBOTTOM 5 BY MEAN POINTS:")
    for i, (s, sc, m) in enumerate(combos[-5:], 1):
        print(f"  {i}. {s}+{sc}: {m:.1f}")


if __name__ == "__main__":
    main()
