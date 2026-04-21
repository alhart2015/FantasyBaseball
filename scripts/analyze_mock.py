"""Analyze the most recent mock draft results."""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.data.db import get_connection
from fantasy_baseball.draft.board import build_draft_board
from fantasy_baseball.scoring import score_roto_dict
from fantasy_baseball.simulation import simulate_season
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES as ALL_CATS,
)
from fantasy_baseball.utils.constants import (
    INVERSE_STATS as INVERSE,
)
from fantasy_baseball.utils.name_utils import normalize_name

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
SIM_STATE_PATH = PROJECT_ROOT / "data" / "sim_state.json"
LIVE_STATE_PATH = PROJECT_ROOT / "data" / "draft_state.json"


def main():
    config = load_config(CONFIG_PATH)
    state_path = SIM_STATE_PATH if SIM_STATE_PATH.exists() else LIVE_STATE_PATH
    with open(state_path) as f:
        state = json.load(f)

    conn = get_connection()
    board = build_draft_board(
        conn=conn,
        sgp_overrides=config.sgp_overrides or None,
        roster_slots=config.roster_slots or None,
        num_teams=10,
    )
    conn.close()

    num_teams = 10
    num_keepers = state.get("num_keepers", 0)
    drafted = state["drafted_players"]
    drafted_ids = state["drafted_ids"]

    team_players = {i: [] for i in range(1, num_teams + 1)}

    # Keepers are at the front of the drafted list — assign by config
    if num_keepers > 0:
        for keeper in config.keepers[:num_keepers]:
            for num, name in config.teams.items():
                if name == keeper["team"]:
                    norm = normalize_name(keeper["name"])
                    matches = board[board["name_normalized"] == norm]
                    if not matches.empty:
                        team_players[num].append(matches.loc[matches["var"].idxmax()])
                    break

    # Draft picks (after keepers) follow snake order
    for pick_num, (name, pid) in enumerate(
        zip(drafted[num_keepers:], drafted_ids[num_keepers:]), 1
    ):
        rnd = (pick_num - 1) // num_teams + 1
        pos = (pick_num - 1) % num_teams + 1
        team = pos if rnd % 2 == 1 else num_teams - pos + 1
        rows = board[board["player_id"] == pid]
        if not rows.empty:
            team_players[team].append(rows.iloc[0])

    user_team = config.draft_position
    print(f"Your team: T{user_team}")

    h_slots = sum(v for k, v in config.roster_slots.items() if k not in ("P", "BN", "IL"))
    p_slots = config.roster_slots.get("P", 9)

    rng = np.random.default_rng(42)
    N = 1000
    all_totals = {t: [] for t in team_players}
    all_finishes = {t: [] for t in team_players}
    all_cat_pts = {t: {c: [] for c in ALL_CATS} for t in team_players}
    all_cat_vals = {t: {c: [] for c in ALL_CATS} for t in team_players}

    for _ in range(N):
        stats, _ = simulate_season(team_players, rng, h_slots, p_slots)
        results = score_roto_dict(stats)
        ranked = sorted(results.items(), key=lambda x: x[1]["total"], reverse=True)
        for rank, (t, pts) in enumerate(ranked, 1):
            all_totals[t].append(pts["total"])
            all_finishes[t].append(rank)
            for cat in ALL_CATS:
                all_cat_pts[t][cat].append(pts.get(f"{cat}_pts", 0))
                all_cat_vals[t][cat].append(stats[t][cat])

    order = sorted(team_players.keys(), key=lambda t: np.median(all_totals[t]), reverse=True)

    # Projected stats
    print("\n" + "=" * 132)
    print("PROJECTED ROTO STANDINGS (median of 1000 MC simulations)")
    print("=" * 132)
    print(f"{'Rk':<3} {'Team':>6} {'Pts':>4}  "
          f"{'R':>5} {'HR':>4} {'RBI':>5} {'SB':>4} {'AVG':>6}  "
          f"{'W':>4} {'K':>5} {'SV':>4} {'ERA':>5} {'WHIP':>6}  "
          f"{'H':>2}/{'P':>2}")
    print("-" * 132)
    for i, tn in enumerate(order, 1):
        med = np.median(all_totals[tn])
        vals = {c: np.median(all_cat_vals[tn][c]) for c in ALL_CATS}
        nh = sum(1 for p in team_players[tn] if p["player_type"] == "hitter")
        np_ = sum(1 for p in team_players[tn] if p["player_type"] == "pitcher")
        marker = " <<<" if tn == user_team else ""
        print(f"{i:<3} T{tn:>4} {med:>4.0f}  "
              f"{vals['R']:>5.0f} {vals['HR']:>4.0f} {vals['RBI']:>5.0f} "
              f"{vals['SB']:>4.0f} {vals['AVG']:>6.3f}  "
              f"{vals['W']:>4.0f} {vals['K']:>5.0f} {vals['SV']:>4.0f} "
              f"{vals['ERA']:>5.2f} {vals['WHIP']:>6.3f}  "
              f"{nh:>2}/{np_:>2}{marker}")

    # Category points
    print(f"\n{'=' * 100}")
    print("MEDIAN ROTO POINTS BY CATEGORY")
    print("=" * 100)
    print(f"{'Rk':<3} {'Team':>6}  ", end="")
    for c in ALL_CATS:
        print(f"{c:>5}", end="")
    print(f"{'TOT':>6}")
    print("-" * 100)
    for i, tn in enumerate(order, 1):
        marker = " <<<" if tn == user_team else ""
        print(f"{i:<3} T{tn:>4}  ", end="")
        for c in ALL_CATS:
            print(f"{np.median(all_cat_pts[tn][c]):>5.0f}", end="")
        print(f"{np.median(all_totals[tn]):>6.0f}{marker}")

    # Win rates
    print(f"\n{'=' * 75}")
    print("WIN PROBABILITY")
    print("=" * 75)
    print(f"{'Rk':<3} {'Team':>6} {'Med':>5} {'P10':>5} {'P90':>5}  "
          f"{'Win%':>6} {'Top3':>6} {'Bot3':>6}")
    print("-" * 75)
    for i, tn in enumerate(order, 1):
        tots = np.array(all_totals[tn])
        fins = np.array(all_finishes[tn])
        marker = " <<<" if tn == user_team else ""
        print(f"{i:<3} T{tn:>4} {np.median(tots):>5.0f} {np.percentile(tots, 10):>5.0f} "
              f"{np.percentile(tots, 90):>5.0f}  "
              f"{np.mean(fins == 1) * 100:>5.1f}% "
              f"{np.mean(fins <= 3) * 100:>5.1f}% "
              f"{np.mean(fins >= 8) * 100:>5.1f}%{marker}")

    # Category risk
    print(f"\n{'=' * 60}")
    print(f"YOUR CATEGORY RISK PROFILE (Team {user_team})")
    print("=" * 60)
    print(f"{'Cat':>5} {'Med':>4} {'P10':>4} {'P90':>4}  {'Top3':>5} {'Bot3':>5}")
    print("-" * 40)
    for cat in ALL_CATS:
        pts = np.array(all_cat_pts[user_team][cat])
        print(f"{cat:>5} {np.median(pts):>4.0f} {np.percentile(pts, 10):>4.0f} "
              f"{np.percentile(pts, 90):>4.0f}  "
              f"{np.mean(pts >= 8) * 100:>4.1f}% "
              f"{np.mean(pts <= 3) * 100:>4.1f}%")

    # Strengths / weaknesses
    print(f"\n{'=' * 60}")
    print("YOUR STRENGTHS AND WEAKNESSES")
    print("=" * 60)
    cats_sorted = sorted(ALL_CATS, key=lambda c: np.median(all_cat_pts[user_team][c]), reverse=True)
    strengths = [(c, np.median(all_cat_pts[user_team][c]), np.median(all_cat_vals[user_team][c]))
                 for c in cats_sorted if np.median(all_cat_pts[user_team][c]) >= 7]
    mid = [(c, np.median(all_cat_pts[user_team][c]), np.median(all_cat_vals[user_team][c]))
           for c in cats_sorted if 4 <= np.median(all_cat_pts[user_team][c]) <= 6]
    weak = [(c, np.median(all_cat_pts[user_team][c]), np.median(all_cat_vals[user_team][c]))
            for c in cats_sorted if np.median(all_cat_pts[user_team][c]) <= 3]

    if strengths:
        print("\nStrengths (7+ pts):")
        for c, p, v in strengths:
            fmt = ".3f" if c in ("AVG", "ERA", "WHIP") else ".0f"
            print(f"  {c:>5}: {v:{fmt}} ({p:.0f} pts)")
    if mid:
        print("\nMiddle (4-6 pts):")
        for c, p, v in mid:
            fmt = ".3f" if c in ("AVG", "ERA", "WHIP") else ".0f"
            print(f"  {c:>5}: {v:{fmt}} ({p:.0f} pts)")
    if weak:
        print("\nWeak (1-3 pts):")
        for c, p, v in weak:
            fmt = ".3f" if c in ("AVG", "ERA", "WHIP") else ".0f"
            print(f"  {c:>5}: {v:{fmt}} ({p:.0f} pts)")


if __name__ == "__main__":
    main()
