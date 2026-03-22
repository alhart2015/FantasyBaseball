"""Batch sweep: run all strategies x scoring modes (deterministic).

Builds the draft board ONCE, then runs all combinations against it.
No ADP noise or random seeds — opponent ADP drafting is deterministic
(take best ADP player who fills an open roster slot).
Saves full simulation output for each combo for later re-analysis.
"""
import csv
import gc
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from simulate_draft import build_board_and_context, run_simulation, save_simulation_output
from fantasy_baseball.draft.strategy import STRATEGIES

STRATEGY_NAMES = list(STRATEGIES.keys())
SCORING_MODES = ["var", "vona"]

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

    total = len(STRATEGY_NAMES) * len(SCORING_MODES)
    print(f"Sweeping {len(STRATEGY_NAMES)} strategies x {len(SCORING_MODES)} scoring = {total} sims (deterministic)")
    print()

    results = []
    done = 0
    sim_start = time.time()
    run_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    for strategy in STRATEGY_NAMES:
        for scoring in SCORING_MODES:
            done += 1
            try:
                r = run_simulation(
                    ctx,
                    strategy_name=strategy,
                    scoring_mode=scoring,
                    opponent_strategies_str=OPP_STRATEGIES,
                )
                pts = r["pts"]
                rank = r["rank"]

                # Save full simulation output
                save_simulation_output(
                    r, strategy, scoring, OPP_STRATEGIES, run_ts,
                )
            except Exception as e:
                pts = -1
                rank = -1
                print(f"    ERROR: {e}")

            results.append({
                "strategy": strategy,
                "scoring": scoring,
                "pts": pts,
                "rank": rank,
            })

            elapsed = time.time() - sim_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  [{done:>3}/{total}] {strategy:<20} {scoring:<5} -> "
                  f"{pts:>4}pts rank#{rank}  "
                  f"({rate:.1f} sims/s, ~{eta:.0f}s left)")

            gc.collect()

    sim_time = time.time() - sim_start
    print(f"\nAll sims done in {sim_time:.1f}s ({total/sim_time:.1f} sims/s)")

    # Write CSV
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "scoring", "pts", "rank"])
        w.writeheader()
        w.writerows(results)
    print(f"Results saved to {OUT_PATH}")
    print(f"Full sim outputs saved to data/sim_results/{run_ts}_*.json")

    # Summary
    print(f"\n{'='*80}")
    print("BATCH SWEEP RESULTS (deterministic, with opponent strategies)")
    print(f"{'='*80}")
    print(f"\n{'Strategy':<22} {'Scoring':<8} {'Pts':>5} {'Rank':>5}")
    print("-" * 42)

    for scoring in SCORING_MODES:
        for strategy in STRATEGY_NAMES:
            subset = [r for r in results
                      if r["strategy"] == strategy
                      and r["scoring"] == scoring
                      and r["pts"] > 0]
            if not subset:
                continue
            r = subset[0]
            print(f"{strategy:<22} {scoring:<8} {r['pts']:>5} {r['rank']:>5}")
        print()

    # Top / bottom 5
    combos = [(r["strategy"], r["scoring"], r["pts"])
              for r in results if r["pts"] > 0]
    combos.sort(key=lambda x: x[2], reverse=True)

    print(f"TOP 5 BY POINTS:")
    for i, (s, sc, p) in enumerate(combos[:5], 1):
        print(f"  {i}. {s}+{sc}: {p}")

    print(f"\nBOTTOM 5 BY POINTS:")
    for i, (s, sc, p) in enumerate(combos[-5:], 1):
        print(f"  {i}. {s}+{sc}: {p}")


if __name__ == "__main__":
    main()
