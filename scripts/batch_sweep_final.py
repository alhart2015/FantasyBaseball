"""Final sweep: all strategies x scoring x 10 seeds with ADP noise.

Uses subprocess isolation to avoid segfaults on long runs.
"""
import csv
import subprocess
import sys
import statistics
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STRATEGY_NAMES = [
    "default", "nonzero_sv", "avg_hedge", "three_closers",
    "no_punt", "no_punt_opp", "no_punt_stagger", "no_punt_cap3",
    "avg_anchor", "closers_avg", "balanced", "anti_fragile",
]
SCORING_MODES = ["var", "vona"]
SEEDS = list(range(10))
ADP_NOISE = 15
OPP_STRATEGIES = (
    "1:three_closers,2:default,3:nonzero_sv,4:three_closers,"
    "5:nonzero_sv,6:nonzero_sv,7:default,9:nonzero_sv,10:three_closers"
)
OUT_PATH = PROJECT_ROOT / "data" / "batch_sweep_results_final.csv"
PYTHON = sys.executable or "C:/Python312/python.exe"

WORKER_SCRIPT = '''
import sys
sys.path.insert(0, r"{src}")
sys.path.insert(0, r"{scripts}")
from simulate_draft import build_board_and_context, run_simulation
ctx = build_board_and_context()
strategy = "{strategy}"
scoring = "{scoring}"
opp = "{opp}"
for seed in range({n_seeds}):
    try:
        r = run_simulation(ctx, strategy_name=strategy, scoring_mode=scoring,
                           adp_noise={noise}, seed=seed, opponent_strategies_str=opp)
        print(f"RESULT:{{strategy}}:{{scoring}}:{{seed}}:{{r['pts']}}:{{r['rank']}}")
    except Exception as e:
        print(f"ERROR:{{strategy}}:{{scoring}}:{{seed}}:{{e}}", file=sys.stderr)
    del r
'''


def main():
    src = str(PROJECT_ROOT / "src").replace("\\", "\\\\")
    scripts = str(PROJECT_ROOT / "scripts").replace("\\", "\\\\")

    total = len(STRATEGY_NAMES) * len(SCORING_MODES)
    print(f"Final sweep: {len(STRATEGY_NAMES)} strategies x {len(SCORING_MODES)} scoring "
          f"x {len(SEEDS)} seeds = {total * len(SEEDS)} sims")
    print(f"Using subprocess isolation (one process per strategy+scoring combo)")
    print()

    results = []
    done = 0
    t0 = time.time()

    for strategy in STRATEGY_NAMES:
        for scoring in SCORING_MODES:
            done += 1
            script = WORKER_SCRIPT.format(
                src=src, scripts=scripts, strategy=strategy,
                scoring=scoring, opp=OPP_STRATEGIES,
                noise=ADP_NOISE, n_seeds=len(SEEDS),
            )
            proc = subprocess.run(
                [PYTHON, "-c", script],
                capture_output=True, text=True, timeout=300,
            )

            seed_results = []
            for line in proc.stdout.strip().split("\n"):
                if line.startswith("RESULT:"):
                    parts = line.split(":")
                    seed_results.append({
                        "strategy": parts[1],
                        "scoring": parts[2],
                        "seed": int(parts[3]),
                        "pts": int(parts[4]),
                        "rank": int(parts[5]),
                    })

            results.extend(seed_results)

            pts_list = [r["pts"] for r in seed_results if r["pts"] > 0]
            rk_list = [r["rank"] for r in seed_results if r["rank"] > 0]
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0

            if pts_list:
                wins = sum(1 for r in rk_list if r == 1)
                print(f"  [{done:>3}/{total}] {strategy:<22} {scoring:<5} "
                      f"mean={statistics.mean(pts_list):>5.1f} "
                      f"med={statistics.median(pts_list):>4.0f} "
                      f"min={min(pts_list):>3} max={max(pts_list):>3} "
                      f"win={wins}/{len(rk_list)} "
                      f"(~{eta:.0f}s left)")
            else:
                print(f"  [{done:>3}/{total}] {strategy:<22} {scoring:<5} FAILED")
                if proc.stderr:
                    print(f"    stderr: {proc.stderr[:200]}")

    sim_time = time.time() - t0
    print(f"\nAll sims done in {sim_time:.1f}s")

    # Write CSV
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "scoring", "seed", "pts", "rank"])
        w.writeheader()
        w.writerows(results)
    print(f"Results saved to {OUT_PATH}")

    # Summary
    print(f"\n{'='*100}")
    print("FINAL SWEEP RESULTS (10 seeds, ADP noise=15, opponent strategies)")
    print(f"{'='*100}")
    print(f"\n{'Strategy+Scoring':<28} {'Mean':>6} {'Med':>4} {'Std':>5} "
          f"{'Min':>4} {'Max':>4} {'Win%':>5} {'Top3%':>6} {'Bot3%':>6}")
    print("-" * 90)

    combos = []
    for scoring in SCORING_MODES:
        for strategy in STRATEGY_NAMES:
            subset = [r for r in results
                      if r["strategy"] == strategy
                      and r["scoring"] == scoring
                      and r["pts"] > 0]
            if not subset:
                continue
            pts_list = [r["pts"] for r in subset]
            rk_list = [r["rank"] for r in subset]
            mean_pts = statistics.mean(pts_list)
            med_pts = statistics.median(pts_list)
            std_pts = statistics.stdev(pts_list) if len(pts_list) > 1 else 0
            win_rate = sum(1 for r in rk_list if r == 1) / len(rk_list) * 100
            top3 = sum(1 for r in rk_list if r <= 3) / len(rk_list) * 100
            bot3 = sum(1 for r in rk_list if r >= 8) / len(rk_list) * 100
            combos.append((mean_pts, strategy, scoring, med_pts, std_pts,
                           min(pts_list), max(pts_list), win_rate, top3, bot3))

    combos.sort(reverse=True)
    for mean_pts, strategy, scoring, med, std, worst, best, wr, t3, b3 in combos:
        label = f"{strategy}+{scoring}"
        print(f"{label:<28} {mean_pts:>6.1f} {med:>4.0f} {std:>5.1f} "
              f"{worst:>4} {best:>4} {wr:>5.0f} {t3:>6.0f} {b3:>6.0f}")

    print(f"\nTOP 5:")
    for i, (m, s, sc, med, std, worst, best, wr, t3, b3) in enumerate(combos[:5], 1):
        print(f"  {i}. {s}+{sc}: mean={m:.1f} med={med:.0f} "
              f"win={wr:.0f}% top3={t3:.0f}% range=[{worst}-{best}]")

    print(f"\nBOTTOM 5:")
    for i, (m, s, sc, med, std, worst, best, wr, t3, b3) in enumerate(combos[-5:], 1):
        print(f"  {i}. {s}+{sc}: mean={m:.1f} med={med:.0f} "
              f"win={wr:.0f}% top3={t3:.0f}% range=[{worst}-{best}]")


if __name__ == "__main__":
    main()
