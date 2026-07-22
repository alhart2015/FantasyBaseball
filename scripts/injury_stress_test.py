"""Injury stress-test CLI: how robust is my ROS lead to injuries?

Reads live season state from remote Upstash and prints a ranked report:
headline attribution, P(everyone healthy), single + pair lose-a-player
counterfactuals. See docs/superpowers/specs/2026-07-22-injury-stress-test-design.md.

Usage: python scripts/injury_stress_test.py [--threshold 0.20] [--pair-top-k 8]
                                            [--n-iter 1000] [--out report.md]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fantasy_baseball.analysis.injury_stress import (  # noqa: E402
    PAIR_TOP_K,
    SIGNIFICANT_TIME_THRESHOLD,
    load_mc_inputs_from_upstash,
    render_report,
    run_stress_test,
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # names may be non-ASCII
    ap = argparse.ArgumentParser(description="Injury stress-test for the ROS lead.")
    ap.add_argument("--threshold", type=float, default=SIGNIFICANT_TIME_THRESHOLD)
    ap.add_argument("--pair-top-k", type=int, default=PAIR_TOP_K)
    ap.add_argument("--n-iter", type=int, default=1000)
    ap.add_argument("--out", type=str, default=None, help="write the report to this path")
    args = ap.parse_args()

    print("Loading live season state from Upstash ...", file=sys.stderr)
    inputs = load_mc_inputs_from_upstash()
    print(f"Running stress test ({args.n_iter} iters) ...", file=sys.stderr)
    result = run_stress_test(
        inputs, threshold=args.threshold, pair_top_k=args.pair_top_k, n_iter=args.n_iter
    )
    report = render_report(result)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
