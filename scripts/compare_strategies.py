"""Definitive strategy comparison.

Runs both strategies in a single process with a single shared context.
No file I/O for results — everything printed directly.

Against strategic opponents: single deterministic run (jitter has no
effect since all opponents use strategy functions, not ADP fallback).

Against ADP opponents: 50 iterations with jitter for stable averages.
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from simulate_draft import build_board_and_context, run_simulation
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD

STRATEGIES_TO_COMPARE = [
    ("two_closers", "vona"),
    ("no_punt_opp", "vona"),
    ("two_closers", "var"),
    ("default", "vona"),
]

OPP_STRATEGIES = (
    "1:two_closers,2:two_closers,3:two_closers,4:three_closers,"
    "5:two_closers,6:four_closers,7:two_closers,9:nonzero_sv,10:three_closers"
)

CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
ADP_ITERATIONS = 50
ADP_NOISE = 15.0


def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def count_closers(user_roster_ids, board, full_board):
    count = 0
    for pid in user_roster_ids:
        rows = board[board["player_id"] == pid]
        if rows.empty:
            rows = full_board[full_board["player_id"] == pid]
        if not rows.empty and rows.iloc[0].get("sv", 0) >= CLOSER_SV_THRESHOLD:
            count += 1
    return count


def print_roto_breakdown(results, team_name):
    hart = next(t for t in results if t["team"] == team_name)
    for c in CATS:
        val = hart[c]
        pts = hart[f"{c}_p"]
        if c == "AVG":
            print(f"    {c:<5}: .{int(val*1000):03d}  ({pts:>4.1f}/10)")
        elif c in ("ERA", "WHIP"):
            print(f"    {c:<5}: {val:.2f}  ({pts:>4.1f}/10)")
        else:
            print(f"    {c:<5}: {val:>5.0f}  ({pts:>4.1f}/10)")
    print(f"    TOTAL: {hart['tot']:.1f}/100")
    return hart


def print_draft_picks(r, ctx, label):
    config = ctx["config"]
    board = ctx["board"]
    full_board = ctx["full_board"]
    pick_order = ctx.get("pick_order")
    num_keepers = len(config.keepers)

    draft_names = r["tracker"].drafted_players[num_keepers:]
    draft_ids = r["tracker"].drafted_ids[num_keepers:]
    user_ids = set(r["user_roster_ids"])

    print(f"  {label} draft picks:")
    for pick_idx, (name, pid) in enumerate(zip(draft_names, draft_ids)):
        if pid not in user_ids:
            continue

        overall_pick = pick_idx + 1 + num_keepers
        rnd = (overall_pick - 1) // config.num_teams + 1

        rows = board[board["player_id"] == pid]
        if rows.empty:
            rows = full_board[full_board["player_id"] == pid]
        if not rows.empty:
            p = rows.iloc[0]
            sv = p.get("sv", 0)
            if p.get("player_type") == "pitcher" and sv >= CLOSER_SV_THRESHOLD:
                tag = "CLOSER"
            elif p.get("player_type") == "pitcher":
                tag = "SP"
            else:
                tag = "hitter"
            print(f"    R{rnd:>2} pick {overall_pick:>3}: {name:<28} [{tag}]")
        else:
            print(f"    R{rnd:>2} pick {overall_pick:>3}: {name:<28} [?]")


def main():
    print("Building draft board...")
    ctx = build_board_and_context()
    config = ctx["config"]
    board = ctx["board"]
    full_board = ctx["full_board"]
    pick_order = ctx.get("pick_order")

    # Verify context
    print(f"  Board: {len(board)} players")
    print(f"  Keepers: {len(config.keepers)}")
    print(f"  Custom draft order: {'yes' if pick_order else 'no'}")
    if pick_order:
        hart_picks = sum(1 for tn in pick_order if tn == config.draft_position)
        print(f"  Hart post-keeper picks: {hart_picks}")

    # ---------------------------------------------------------------
    # PART 1: Strategic opponents (deterministic)
    # ---------------------------------------------------------------
    print_header("PART 1: vs STRATEGIC OPPONENTS (deterministic, single run)")
    print(f"  Opponents: {OPP_STRATEGIES}")

    strat_results = {}
    for strat, scoring in STRATEGIES_TO_COMPARE:
        label = f"{strat}+{scoring}"
        r = run_simulation(
            ctx,
            strategy_name=strat,
            scoring_mode=scoring,
            opponent_strategies_str=OPP_STRATEGIES,
        )
        n_closers = count_closers(r["user_roster_ids"], board, full_board)
        strat_results[label] = r

        print(f"\n  --- {label} ---")
        print(f"  Roster: {len(r['user_roster'])} players, {n_closers} closers")
        hart = print_roto_breakdown(r["results"], config.team_name)
        print(f"  Result: {hart['tot']:.1f} pts, rank #{r['rank']}")

    # Side-by-side category comparison for top 2
    labels = list(strat_results.keys())[:2]
    print(f"\n  {'Category':<8}", end="")
    for label in labels:
        print(f"  {label:>20}", end="")
    print(f"  {'Edge':>12}")
    print("  " + "-" * 65)
    for c in CATS:
        print(f"  {c:<8}", end="")
        pts_vals = []
        for label in labels:
            r = strat_results[label]
            hart = next(t for t in r["results"] if t["team"] == config.team_name)
            pts = hart[f"{c}_p"]
            pts_vals.append(pts)
            print(f"  {pts:>20.1f}", end="")
        diff = pts_vals[0] - pts_vals[1]
        edge = labels[0].split("+")[0] if diff > 0 else labels[1].split("+")[0] if diff < 0 else "tie"
        print(f"  {edge:>12}")
    # Total
    print(f"  {'TOTAL':<8}", end="")
    for label in labels:
        r = strat_results[label]
        hart = next(t for t in r["results"] if t["team"] == config.team_name)
        print(f"  {hart['tot']:>20.1f}", end="")
    print()

    # Draft picks and full rosters for top 2
    print()
    for label in labels:
        print_draft_picks(strat_results[label], ctx, label)

        # Full roster with stats
        r = strat_results[label]
        print(f"\n  {label} full roster:")
        for pid in r["user_roster_ids"]:
            rows = board[board["player_id"] == pid]
            if rows.empty:
                rows = full_board[full_board["player_id"] == pid]
            if rows.empty:
                continue
            p = rows.iloc[0]
            sv = p.get("sv", 0)
            if p.get("player_type") == "pitcher" and sv >= CLOSER_SV_THRESHOLD:
                tag = "CLOSER"
            elif p.get("player_type") == "pitcher":
                tag = "SP"
            else:
                tag = "hitter"
            if p.get("player_type") == "hitter":
                stats = (f"R:{p.get('r',0):>3.0f} HR:{p.get('hr',0):>2.0f} "
                         f"RBI:{p.get('rbi',0):>3.0f} SB:{p.get('sb',0):>2.0f} "
                         f"AVG:{p.get('avg',0):.3f}")
            else:
                stats = (f"W:{p.get('w',0):>2.0f} K:{p.get('k',0):>3.0f} "
                         f"SV:{sv:>2.0f} ERA:{p.get('era',0):.2f} "
                         f"IP:{p.get('ip',0):>3.0f}")
            print(f"    {p.get('name','?'):<28} [{tag:<6}] {stats}")
        print()

    # ---------------------------------------------------------------
    # PART 2: ADP opponents with jitter (50 iterations)
    # ---------------------------------------------------------------
    print_header(f"PART 2: vs ADP OPPONENTS ({ADP_ITERATIONS} iterations, noise={ADP_NOISE})")

    for strat, scoring in STRATEGIES_TO_COMPARE:
        label = f"{strat}+{scoring}"
        pts_list = []
        rank_list = []
        wins = 0

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

        print(f"  {label:<25} avg:{avg_pts:>5.1f}  rank:{avg_rank:.2f}  "
              f"win:{win_pct:>4.1f}%  [{min(pts_list):.0f}-{max(pts_list):.0f}]")

    # ---------------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------------
    print_header("VERDICT")
    for label in labels:
        r = strat_results[label]
        hart = next(t for t in r["results"] if t["team"] == config.team_name)
        n_cl = count_closers(r["user_roster_ids"], board, full_board)
        print(f"  {label}: {hart['tot']:.1f} pts, rank #{r['rank']}, {n_cl} closers")


if __name__ == "__main__":
    main()
