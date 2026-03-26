"""Analyze the most recent mock draft results."""
import sys
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.draft.board import build_draft_board
from fantasy_baseball.utils.name_utils import normalize_name

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
SIM_STATE_PATH = PROJECT_ROOT / "data" / "sim_state.json"
LIVE_STATE_PATH = PROJECT_ROOT / "data" / "draft_state.json"

from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES as ALL_CATS,
    CLOSER_SV_THRESHOLD,
    HITTING_COUNTING,
    INJURY_PROB,
    INJURY_SEVERITY,
    INVERSE_STATS as INVERSE,
    PITCHING_COUNTING,
    REPLACEMENT_HITTER,
    REPLACEMENT_RP,
    REPLACEMENT_SP,
    STAT_VARIANCE,
)


def sim_season(team_players, rng, h_slots, p_slots):
    stats = {}
    for tn, players in team_players.items():
        hitters = [p for p in players if p["player_type"] == "hitter"]
        pitchers = [p for p in players if p["player_type"] == "pitcher"]
        ah, ap = [], []
        for h in hitters:
            frac = rng.uniform(*INJURY_SEVERITY["hitter"]) if rng.random() < INJURY_PROB["hitter"] else 0
            scale = 1 - frac
            row = {}
            for col in HITTING_COUNTING:
                base = float(h.get(col, 0) or 0)
                repl_val = REPLACEMENT_HITTER.get(col, 0) * frac
                sigma = STAT_VARIANCE.get(col, 0.0)
                if sigma > 0:
                    perf = max(0, 1 + rng.normal(0, sigma))
                    row[col] = base * perf * scale + repl_val
                else:
                    row[col] = base * scale + repl_val
            ah.append(row)
        for p in pitchers:
            frac = rng.uniform(*INJURY_SEVERITY["pitcher"]) if rng.random() < INJURY_PROB["pitcher"] else 0
            repl = REPLACEMENT_RP if float(p.get("sv", 0) or 0) >= 15 else REPLACEMENT_SP
            scale = 1 - frac
            row = {}
            for col in PITCHING_COUNTING:
                base = float(p.get(col, 0) or 0)
                repl_val = repl.get(col, 0) * frac
                sigma = STAT_VARIANCE.get(col, 0.0)
                if sigma > 0:
                    perf = max(0, 1 + rng.normal(0, sigma))
                    row[col] = base * perf * scale + repl_val
                else:
                    row[col] = base * perf * scale + repl_val
            ap.append(row)
        ah.sort(key=lambda x: x["r"] + x["hr"] + x["rbi"] + x["sb"], reverse=True)
        ap.sort(key=lambda x: (x.get("sv", 0) >= 15, x["w"] + x["k"] + x["sv"]), reverse=True)
        ah, ap = ah[:h_slots], ap[:p_slots]
        r = sum(x["r"] for x in ah); hr = sum(x["hr"] for x in ah)
        rbi = sum(x["rbi"] for x in ah); sb = sum(x["sb"] for x in ah)
        th = sum(x["h"] for x in ah); tab = sum(x["ab"] for x in ah)
        avg = th / tab if tab > 0 else 0
        w = sum(x["w"] for x in ap); k = sum(x["k"] for x in ap)
        sv = sum(x["sv"] for x in ap)
        tip = sum(x["ip"] for x in ap); ter = sum(x["er"] for x in ap)
        tbb = sum(x["bb"] for x in ap); tha = sum(x["h_allowed"] for x in ap)
        era = ter * 9 / tip if tip > 0 else 99
        whip = (tbb + tha) / tip if tip > 0 else 99
        stats[tn] = {"R": r, "HR": hr, "RBI": rbi, "SB": sb, "AVG": avg,
                     "W": w, "K": k, "SV": sv, "ERA": era, "WHIP": whip}
    return stats


def main():
    config = load_config(CONFIG_PATH)
    state_path = SIM_STATE_PATH if SIM_STATE_PATH.exists() else LIVE_STATE_PATH
    with open(state_path) as f:
        state = json.load(f)

    board = build_draft_board(
        projections_dir=PROJECT_ROOT / "data" / "projections" / str(config.season_year),
        positions_path=PROJECT_ROOT / "data" / "player_positions.json",
        systems=config.projection_systems,
        weights=config.projection_weights or None,
        sgp_overrides=config.sgp_overrides or None,
        roster_slots=config.roster_slots or None,
        num_teams=10,
    )

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
        stats = sim_season(team_players, rng, h_slots, p_slots)
        results = {}
        for cat in ALL_CATS:
            rev = cat not in INVERSE
            ranked = sorted(stats.keys(), key=lambda t: stats[t][cat], reverse=rev)
            for i, t in enumerate(ranked):
                results.setdefault(t, {})[f"{cat}_pts"] = num_teams - i
        for t in results:
            results[t]["total"] = sum(results[t][f"{c}_pts"] for c in ALL_CATS)
        for t in team_players:
            total = results[t]["total"]
            all_totals[t].append(total)
            rank = 1 + sum(1 for o in team_players if results[o]["total"] > total)
            all_finishes[t].append(rank)
            for cat in ALL_CATS:
                all_cat_pts[t][cat].append(results[t][f"{cat}_pts"])
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
