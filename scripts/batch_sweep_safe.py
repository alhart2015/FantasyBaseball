"""Batch sweep using subprocess per strategy+scoring combo to avoid segfaults."""
import subprocess, sys, csv, statistics, re, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = str(PROJECT_ROOT / "scripts" / "batch_sweep_worker.py")

STRATEGIES = [
    "default", "nonzero_sv", "avg_hedge", "three_closers",
    "no_punt", "no_punt_opp", "avg_anchor", "closers_avg",
    "balanced", "anti_fragile",
]
SCORING_MODES = ["var", "vona"]
SEEDS = list(range(10))

results = []
total = len(STRATEGIES) * len(SCORING_MODES)
done = 0
t0 = time.time()

for strategy in STRATEGIES:
    for scoring in SCORING_MODES:
        done += 1
        cmd = [sys.executable, SCRIPT, strategy, scoring] + [str(s) for s in SEEDS]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        for line in out.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                parts = line.split(":")
                results.append({
                    "strategy": parts[1], "scoring": parts[2],
                    "seed": int(parts[3]), "pts": int(parts[4]), "rank": int(parts[5]),
                })
        subset = [r for r in results[-len(SEEDS):] if r["pts"] > 0]
        med = statistics.median([r["pts"] for r in subset]) if subset else -1
        elapsed = time.time() - t0
        eta = (total - done) / (done / elapsed) if done > 0 else 0
        print(f"  [{done:>2}/{total}] {strategy:<20} {scoring:<5} med={med:.0f}pts  (~{eta:.0f}s left)")

# Save
out_path = PROJECT_ROOT / "data" / "batch_sweep_results.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["strategy", "scoring", "seed", "pts", "rank"])
    w.writeheader()
    w.writerows(results)

# Summary
print(f"\n{'Strategy':<20} {'Scoring':<8} {'Mean':>6} {'Med':>5}")
print("-" * 45)
combos = []
for scoring in SCORING_MODES:
    for strategy in STRATEGIES:
        subset = [r for r in results if r["strategy"] == strategy and r["scoring"] == scoring and r["pts"] > 0]
        if subset:
            m = statistics.mean([r["pts"] for r in subset])
            md = statistics.median([r["pts"] for r in subset])
            combos.append((strategy, scoring, m))
            print(f"{strategy:<20} {scoring:<8} {m:>6.1f} {md:>5.0f}")
combos.sort(key=lambda x: x[2], reverse=True)
print(f"\nTOP 5: {[(s,sc,f'{m:.1f}') for s,sc,m in combos[:5]]}")
print(f"BOTTOM 5: {[(s,sc,f'{m:.1f}') for s,sc,m in combos[-5:]]}")
