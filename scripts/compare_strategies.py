"""Definitive strategy comparison — all 14 strategies × 2 scoring modes.

Runs every strategy in a single process with a single shared context.
No file I/O for results — everything printed directly.

Part 1: Against strategic opponents (deterministic, single run each).
Part 2: Against ADP opponents (50 iterations with jitter for stable averages).

Usage:
    python scripts/compare_strategies.py
"""
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from simulate_draft import build_board_and_context, run_simulation
from fantasy_baseball.draft.strategy import STRATEGIES
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD

OPP_STRATEGIES = (
    "1:two_closers,2:two_closers,3:two_closers,4:three_closers,"
    "5:two_closers,6:four_closers,7:two_closers,9:nonzero_sv,10:three_closers"
)

CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
ADP_ITERATIONS = 50
ADP_NOISE = 15.0


def count_closers(user_roster_ids, board, full_board):
    count = 0
    for pid in user_roster_ids:
        rows = board[board["player_id"] == pid]
        if rows.empty:
            rows = full_board[full_board["player_id"] == pid]
        if not rows.empty and rows.iloc[0].get("sv", 0) >= CLOSER_SV_THRESHOLD:
            count += 1
    return count


def main():
    print("Building draft board...")
    ctx = build_board_and_context()
    config = ctx["config"]
    board = ctx["board"]
    full_board = ctx["full_board"]
    pick_order = ctx.get("pick_order")

    print(f"  Board: {len(board)} players")
    print(f"  Keepers: {len(config.keepers)}")
    print(f"  Systems: {config.projection_systems}")
    print(f"  Custom draft order: {'yes' if pick_order else 'no'}")
    if pick_order:
        hart_picks = sum(1 for tn in pick_order if tn == config.draft_position)
        print(f"  Hart post-keeper picks: {hart_picks}")

    strategy_names = sorted(STRATEGIES.keys())
    scoring_modes = ["vona", "var"]

    # ---------------------------------------------------------------
    # PART 1: Strategic opponents (deterministic)
    # ---------------------------------------------------------------
    print()
    print("=" * 100)
    print("PART 1: vs STRATEGIC OPPONENTS (deterministic, single run)")
    print("=" * 100)
    print(f"  Opponents: {OPP_STRATEGIES}")
    print()

    header = f"{'Strategy':<30} {'Pts':>5} {'Rank':>4} {'CL':>3}"
    for c in CATS:
        header += f" {c:>5}"
    print(header)
    print("-" * len(header))

    t0 = time.perf_counter()
    strat_results = []
    for strat in strategy_names:
        for scoring in scoring_modes:
            label = f"{strat}+{scoring}"
            try:
                r = run_simulation(
                    ctx,
                    strategy_name=strat,
                    scoring_mode=scoring,
                    opponent_strategies_str=OPP_STRATEGIES,
                )
                hart = next(t for t in r["results"] if t["team"] == config.team_name)
                n_cl = count_closers(r["user_roster_ids"], board, full_board)
                row = f"{label:<30} {hart['tot']:>5.1f} {r['rank']:>4} {n_cl:>3}"
                for c in CATS:
                    row += f" {hart[f'{c}_p']:>5.1f}"
                print(row, flush=True)
                strat_results.append((label, hart["tot"], r["rank"], n_cl))
            except Exception as e:
                print(f"{label:<30} ERROR: {e}", flush=True)
    t1 = time.perf_counter()
    print(f"\nPart 1 completed in {t1 - t0:.1f}s ({len(strat_results)}/28 strategies)")

    print()
    print("RANKING (strategic opponents):")
    print(f"{'#':>3} {'Strategy':<30} {'Pts':>5} {'Rank':>4} {'CL':>3}")
    print("-" * 50)
    for i, (label, pts, rank, n_cl) in enumerate(
        sorted(strat_results, key=lambda x: -x[1]), 1
    ):
        print(f"{i:>3} {label:<30} {pts:>5.1f} {rank:>4} {n_cl:>3}")

    # ---------------------------------------------------------------
    # PART 2: ADP opponents with jitter (50 iterations)
    # ---------------------------------------------------------------
    print()
    print("=" * 100)
    print(f"PART 2: vs ADP OPPONENTS ({ADP_ITERATIONS} iterations, noise={ADP_NOISE})")
    print("=" * 100)
    print()

    header2 = f"{'Strategy':<30} {'Avg':>5} {'AvgRk':>6} {'Win%':>5} {'Min':>4} {'Max':>4}"
    print(header2)
    print("-" * len(header2))

    t2 = time.perf_counter()
    adp_results = []
    for strat in strategy_names:
        for scoring in scoring_modes:
            label = f"{strat}+{scoring}"
            pts_list = []
            rank_list = []
            wins = 0
            try:
                for i in range(ADP_ITERATIONS):
                    r = run_simulation(
                        ctx,
                        strategy_name=strat,
                        scoring_mode=scoring,
                        adp_noise=ADP_NOISE,
                        seed=3000 + i,
                        opponent_strategies_str="",
                    )
                    pts_list.append(r["pts"])
                    rank_list.append(r["rank"])
                    if r["rank"] == 1:
                        wins += 1
                avg_pts = np.mean(pts_list)
                avg_rank = np.mean(rank_list)
                win_pct = wins / len(pts_list) * 100
                done = len(adp_results) + 1
                print(
                    f"{label:<30} {avg_pts:>5.1f} {avg_rank:>6.2f} "
                    f"{win_pct:>5.1f} {min(pts_list):>4.0f} {max(pts_list):>4.0f}"
                    f"  [{done}/28]",
                    flush=True,
                )
                adp_results.append(
                    (label, avg_pts, avg_rank, win_pct, min(pts_list), max(pts_list))
                )
            except Exception as e:
                print(f"{label:<30} ERROR: {e}", flush=True)
    t3 = time.perf_counter()
    print(f"\nPart 2 completed in {t3 - t2:.1f}s")

    print()
    print("RANKING (ADP opponents):")
    print(
        f"{'#':>3} {'Strategy':<30} {'Avg':>5} {'AvgRk':>6} "
        f"{'Win%':>5} {'Floor':>5} {'Ceil':>5}"
    )
    print("-" * 65)
    for i, (label, avg, rank, win, mn, mx) in enumerate(
        sorted(adp_results, key=lambda x: -x[1]), 1
    ):
        print(
            f"{i:>3} {label:<30} {avg:>5.1f} {rank:>6.2f} "
            f"{win:>5.1f} {mn:>5.0f} {mx:>5.0f}"
        )

    print()
    print(f"Total time: {t3 - t0:.1f}s ({(t3 - t0) / 60:.1f}m)")
    print("Done.")


if __name__ == "__main__":
    main()
