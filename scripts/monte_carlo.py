"""Monte Carlo simulation of draft outcomes.

Usage:
    python scripts/monte_carlo.py [--iterations N]

Reads the draft result from data/sim_state.json (or draft_state.json), reconstructs all team
rosters, then runs N simulations (default 1000) with random injuries and
stat variance to estimate win probability and risk profile.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.draft.board import build_draft_board, apply_keepers
from fantasy_baseball.scoring import score_roto, ALL_CATS, INVERSE_CATS
from fantasy_baseball.simulation import simulate_season
from fantasy_baseball.utils.name_utils import normalize_name

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
SIM_STATE_PATH = PROJECT_ROOT / "data" / "sim_state.json"
LIVE_STATE_PATH = PROJECT_ROOT / "data" / "draft_state.json"



def reconstruct_rosters(config, board, state):
    """Reconstruct per-team rosters from the draft state."""
    drafted_names = state["drafted_players"]
    drafted_ids = state["drafted_ids"]
    num_teams = config.num_teams

    team_players = {i: [] for i in range(1, num_teams + 1)}

    # Keepers
    for keeper in config.keepers:
        for num, name in config.teams.items():
            if name == keeper["team"]:
                norm = normalize_name(keeper["name"])
                matches = board[board["name_normalized"] == norm]
                if not matches.empty:
                    best = matches.loc[matches["var"].idxmax()]
                    team_players[num].append(best)
                break

    # Draft picks
    num_keepers = len(config.keepers)
    draft_entries = list(zip(drafted_names[num_keepers:], drafted_ids[num_keepers:]))
    for pick_num, (name, pid) in enumerate(draft_entries, 1):
        rnd = (pick_num - 1) // num_teams + 1
        pos = (pick_num - 1) % num_teams + 1
        team = pos if rnd % 2 == 1 else num_teams - pos + 1
        rows = board[board["player_id"] == pid]
        if rows.empty:
            # Try the full board (keepers removed from board but still on full)
            continue
        team_players[team].append(rows.iloc[0])

    return team_players




def main():
    parser = argparse.ArgumentParser(description="Monte Carlo draft simulation")
    parser.add_argument("--iterations", "-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)

    # Compute active roster slot counts from config
    h_slots = sum(
        v for k, v in config.roster_slots.items() if k not in ("P", "BN", "IL")
    )
    p_slots = config.roster_slots.get("P", 9)

    user_team_num = None
    for num, name in config.teams.items():
        if name == config.team_name:
            user_team_num = num
            break

    print(f"Monte Carlo Simulation | {config.team_name}")
    print(f"Iterations: {args.iterations}")
    print()

    # Build board and load state
    print("Building draft board...")
    board = build_draft_board(
        projections_dir=PROJECTIONS_DIR,
        positions_path=POSITIONS_PATH,
        systems=config.projection_systems,
        weights=config.projection_weights or None,
        sgp_overrides=config.sgp_overrides or None,
        roster_slots=config.roster_slots or None,
        num_teams=config.num_teams,
    )
    board_after_keepers = apply_keepers(board, config.keepers)

    state_path = SIM_STATE_PATH if SIM_STATE_PATH.exists() else LIVE_STATE_PATH
    with open(state_path) as f:
        state = json.load(f)
    print(f"Loaded state from {state_path.name}")

    print("Reconstructing rosters...")
    team_players = reconstruct_rosters(config, board_after_keepers, state)

    for tn, players in team_players.items():
        tname = config.teams.get(tn, f"Team {tn}")
        h = sum(1 for p in players if p["player_type"] == "hitter")
        p = sum(1 for p in players if p["player_type"] == "pitcher")
        marker = " <<<" if tn == user_team_num else ""
        print(f"  {tname:<32} {h}H/{p}P{marker}")

    print(f"\nRunning {args.iterations} simulations...")
    rng = np.random.default_rng(args.seed)

    # Storage
    all_finishes = {tn: [] for tn in team_players}
    all_cat_finishes = {tn: {cat: [] for cat in ALL_CATS} for tn in team_players}
    all_totals = {tn: [] for tn in team_players}

    # Track best/worst for user team
    user_seasons = []  # list of (total_pts, stats, injuries, roto_pts)

    for i in range(args.iterations):
        team_stats, injuries = simulate_season(team_players, rng, h_slots, p_slots)
        roto = score_roto(team_stats)

        for tn in team_players:
            total = roto[tn]["total"]
            all_totals[tn].append(total)
            # Compute finish position
            rank = 1 + sum(1 for other_tn in team_players
                           if roto[other_tn]["total"] > total)
            all_finishes[tn].append(rank)
            for cat in ALL_CATS:
                all_cat_finishes[tn][cat].append(roto[tn][f"{cat}_pts"])

        if user_team_num:
            user_seasons.append({
                "total": roto[user_team_num]["total"],
                "stats": dict(team_stats[user_team_num]),
                "injuries": injuries.get(user_team_num, []),
                "cat_pts": {cat: roto[user_team_num][f"{cat}_pts"] for cat in ALL_CATS},
            })

    # === Output ===
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Win rates for all teams
    print(f"\n{'Team':<32} {'Med':>4} {'P10':>4} {'P90':>4}  "
          f"{'1st':>5} {'Top3':>5} {'Bot3':>5}")
    print("-" * 75)

    team_order = sorted(team_players.keys(),
                        key=lambda tn: np.median(all_totals[tn]), reverse=True)
    for tn in team_order:
        tname = config.teams.get(tn, f"Team {tn}")
        totals = np.array(all_totals[tn])
        finishes = np.array(all_finishes[tn])
        med = np.median(totals)
        p10 = np.percentile(totals, 10)
        p90 = np.percentile(totals, 90)
        win_pct = np.mean(finishes == 1) * 100
        top3_pct = np.mean(finishes <= 3) * 100
        bot3_pct = np.mean(finishes >= config.num_teams - 2) * 100
        marker = " <<<" if tn == user_team_num else ""
        print(f"{tname:<32} {med:>4.0f} {p10:>4.0f} {p90:>4.0f}  "
              f"{win_pct:>4.1f}% {top3_pct:>4.1f}% {bot3_pct:>4.1f}%{marker}")

    # Category risk profile for user team
    if user_team_num:
        print(f"\n{'=' * 70}")
        print(f"CATEGORY RISK PROFILE — {config.team_name}")
        print(f"{'=' * 70}")
        print(f"{'Cat':>5} {'Med':>4} {'P10':>4} {'P90':>4}  "
              f"{'Top3':>5} {'Bot3':>5}")
        print("-" * 40)
        for cat in ALL_CATS:
            pts = np.array(all_cat_finishes[user_team_num][cat])
            med = np.median(pts)
            p10 = np.percentile(pts, 10)
            p90 = np.percentile(pts, 90)
            top3 = np.mean(pts >= config.num_teams - 2) * 100
            bot3 = np.mean(pts <= 3) * 100
            print(f"{cat:>5} {med:>4.0f} {p10:>4.0f} {p90:>4.0f}  "
                  f"{top3:>4.1f}% {bot3:>4.1f}%")

        # Best and worst seasons
        sorted_seasons = sorted(user_seasons, key=lambda s: s["total"])
        worst3 = sorted_seasons[:3]
        best3 = sorted_seasons[-3:][::-1]

        def _print_season(label, season):
            s = season["stats"]
            cp = season["cat_pts"]
            inj = season["injuries"]
            print(f"\n  {label}: {season['total']} roto pts")
            print(f"    R:{s['R']:>6.0f}({cp['R']:>2}) HR:{s['HR']:>4.0f}({cp['HR']:>2}) "
                  f"RBI:{s['RBI']:>5.0f}({cp['RBI']:>2}) SB:{s['SB']:>4.0f}({cp['SB']:>2}) "
                  f"AVG:{s['AVG']:>.3f}({cp['AVG']:>2})")
            print(f"    W:{s['W']:>6.0f}({cp['W']:>2}) K:{s['K']:>5.0f}({cp['K']:>2}) "
                  f"SV:{s['SV']:>4.0f}({cp['SV']:>2}) "
                  f"ERA:{s['ERA']:>5.2f}({cp['ERA']:>2}) WHIP:{s['WHIP']:>.3f}({cp['WHIP']:>2})")
            if inj:
                names = [f"{n} (-{f*100:.0f}%)" for n, f in inj]
                print(f"    Injuries: {', '.join(names)}")
            else:
                print(f"    Injuries: none")

        print(f"\n{'=' * 70}")
        print(f"BEST 3 SEASONS")
        print(f"{'=' * 70}")
        for i, s in enumerate(best3, 1):
            _print_season(f"#{i}", s)

        print(f"\n{'=' * 70}")
        print(f"WORST 3 SEASONS")
        print(f"{'=' * 70}")
        for i, s in enumerate(worst3, 1):
            _print_season(f"#{i}", s)

    print()


if __name__ == "__main__":
    main()
