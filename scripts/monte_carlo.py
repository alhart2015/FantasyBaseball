"""Monte Carlo simulation of draft outcomes.

Usage:
    python scripts/monte_carlo.py [--iterations N]

Reads the draft result from data/draft_state.json, reconstructs all team
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
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD
from fantasy_baseball.utils.name_utils import normalize_name

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
STATE_PATH = PROJECT_ROOT / "data" / "draft_state.json"

# Active roster slot counts (set from config in main)
ACTIVE_HITTER_SLOTS = 13
ACTIVE_PITCHER_SLOTS = 9

# Injury model parameters
INJURY_PROB = {"pitcher": 0.45, "hitter": 0.18}
# Fraction of season missed if injured: (min, max) uniform
INJURY_SEVERITY = {"pitcher": (0.20, 0.60), "hitter": (0.15, 0.40)}

# Stat variance: per-player standard deviation as fraction of projected value
STAT_VARIANCE = 0.12  # 12% std dev around projection

HITTING_COUNTING = ["r", "hr", "rbi", "sb", "h", "ab"]
PITCHING_COUNTING = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]

# Replacement-level full-season stats for waiver pickups.
# Derived from typical replacement-level player in a 10-team league.
# Hitter: ~500 AB, .250 AVG, modest counting stats
REPLACEMENT_HITTER = {
    "r": 55, "hr": 12, "rbi": 50, "sb": 5, "h": 125, "ab": 500,
}
# Pitcher (SP): ~140 IP, 4.50 ERA, 1.35 WHIP, ~7 W, ~120 K
REPLACEMENT_SP = {
    "w": 7, "k": 120, "sv": 0, "ip": 140,
    "er": 70, "bb": 50, "h_allowed": 139,
}
# Pitcher (RP/closer): ~60 IP, 4.50 ERA, 1.35 WHIP, ~2 W, ~55 K, ~5 SV
REPLACEMENT_RP = {
    "w": 2, "k": 55, "sv": 5, "ip": 60,
    "er": 30, "bb": 21, "h_allowed": 60,
}
ALL_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
INVERSE = {"ERA", "WHIP"}


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


def simulate_season(team_players, rng, h_slots=None, p_slots=None):
    """Run one simulated season with injuries and stat variance.

    Only counts stats from active-roster players (top h_slots hitters,
    top p_slots pitchers by value). Bench players are excluded.

    Returns dict of team_num -> {cat: value} for all roto categories,
    plus an injuries dict of team_num -> list of (name, frac_missed).
    """
    if h_slots is None:
        h_slots = ACTIVE_HITTER_SLOTS
    if p_slots is None:
        p_slots = ACTIVE_PITCHER_SLOTS
    team_stats = {}
    injuries = {}

    for team_num, players in team_players.items():
        hitters = [p for p in players if p["player_type"] == "hitter"]
        pitchers = [p for p in players if p["player_type"] == "pitcher"]
        team_injuries = []

        # Apply injuries and variance to hitters
        adj_hitters = []
        for h in hitters:
            frac_missed = 0.0
            if rng.random() < INJURY_PROB["hitter"]:
                lo, hi = INJURY_SEVERITY["hitter"]
                frac_missed = rng.uniform(lo, hi)
                team_injuries.append((h["name"], frac_missed))

            row = {}
            scale = 1.0 - frac_missed
            for col in HITTING_COUNTING:
                base = h.get(col, 0)
                # Player's contribution (variance + injury scaling)
                varied = max(0, base * (1.0 + rng.normal(0, STAT_VARIANCE)))
                player_contrib = varied * scale
                # Replacement player fills in for missed time
                repl_contrib = REPLACEMENT_HITTER.get(col, 0) * frac_missed
                row[col] = player_contrib + repl_contrib
            row["player_type"] = "hitter"
            row["name"] = h["name"]
            adj_hitters.append(row)

        # Apply injuries and variance to pitchers
        adj_pitchers = []
        for p in pitchers:
            frac_missed = 0.0
            if rng.random() < INJURY_PROB["pitcher"]:
                lo, hi = INJURY_SEVERITY["pitcher"]
                frac_missed = rng.uniform(lo, hi)
                team_injuries.append((p["name"], frac_missed))

            # Choose replacement profile based on whether this is a closer
            is_closer = p.get("sv", 0) >= CLOSER_SV_THRESHOLD
            repl_profile = REPLACEMENT_RP if is_closer else REPLACEMENT_SP

            row = {}
            scale = 1.0 - frac_missed
            for col in PITCHING_COUNTING:
                base = p.get(col, 0)
                varied = max(0, base * (1.0 + rng.normal(0, STAT_VARIANCE)))
                player_contrib = varied * scale
                repl_contrib = repl_profile.get(col, 0) * frac_missed
                row[col] = player_contrib + repl_contrib
            row["player_type"] = "pitcher"
            row["name"] = p["name"]
            adj_pitchers.append(row)

        # Select active roster only (bench players don't contribute stats)
        adj_hitters.sort(
            key=lambda h: h["r"] + h["hr"] + h["rbi"] + h["sb"],
            reverse=True,
        )
        adj_pitchers.sort(
            key=lambda p: (p.get("sv", 0) >= CLOSER_SV_THRESHOLD, p["w"] + p["k"] + p["sv"]),
            reverse=True,
        )
        active_h = adj_hitters[:h_slots]
        active_p = adj_pitchers[:p_slots]

        # Aggregate team stats from active players only
        r = sum(h["r"] for h in active_h)
        hr = sum(h["hr"] for h in active_h)
        rbi = sum(h["rbi"] for h in active_h)
        sb = sum(h["sb"] for h in active_h)
        total_h = sum(h["h"] for h in active_h)
        total_ab = sum(h["ab"] for h in active_h)
        avg = total_h / total_ab if total_ab > 0 else 0

        w = sum(p["w"] for p in active_p)
        k = sum(p["k"] for p in active_p)
        sv = sum(p["sv"] for p in active_p)
        total_ip = sum(p["ip"] for p in active_p)
        total_er = sum(p["er"] for p in active_p)
        total_bb = sum(p["bb"] for p in active_p)
        total_ha = sum(p["h_allowed"] for p in active_p)
        era = total_er * 9 / total_ip if total_ip > 0 else 99.0
        whip = (total_bb + total_ha) / total_ip if total_ip > 0 else 99.0

        team_stats[team_num] = {
            "R": r, "HR": hr, "RBI": rbi, "SB": sb, "AVG": avg,
            "W": w, "K": k, "SV": sv, "ERA": era, "WHIP": whip,
        }
        injuries[team_num] = team_injuries

    return team_stats, injuries


def score_roto(team_stats, num_teams):
    """Compute roto points for all teams. Returns dict of team_num -> {cat_pts, total}."""
    results = {tn: {} for tn in team_stats}
    for cat in ALL_CATS:
        rev = cat not in INVERSE
        ranked = sorted(team_stats.keys(), key=lambda tn: team_stats[tn][cat], reverse=rev)
        # Fractional tie-breaking: tied teams share the average of their points
        i = 0
        while i < len(ranked):
            j = i + 1
            while j < len(ranked) and team_stats[ranked[j]][cat] == team_stats[ranked[i]][cat]:
                j += 1
            avg_pts = sum(num_teams - k for k in range(i, j)) / (j - i)
            for k in range(i, j):
                results[ranked[k]][f"{cat}_pts"] = avg_pts
            i = j

    for tn in results:
        results[tn]["total"] = sum(results[tn][f"{c}_pts"] for c in ALL_CATS)

    return results


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo draft simulation")
    parser.add_argument("--iterations", "-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)

    # Compute active roster slot counts from config
    global ACTIVE_HITTER_SLOTS, ACTIVE_PITCHER_SLOTS
    ACTIVE_HITTER_SLOTS = sum(
        v for k, v in config.roster_slots.items() if k not in ("P", "BN", "IL")
    )
    ACTIVE_PITCHER_SLOTS = config.roster_slots.get("P", 9)

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

    with open(STATE_PATH) as f:
        state = json.load(f)

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
        team_stats, injuries = simulate_season(team_players, rng)
        roto = score_roto(team_stats, config.num_teams)

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
