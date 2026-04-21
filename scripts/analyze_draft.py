"""Analyze a completed draft: static projection + Monte Carlo.

Usage:
    python scripts/analyze_draft.py                          # latest mock
    python scripts/analyze_draft.py data/drafts/mock_*.json  # specific file
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from simulate_draft import _active_slot_counts, _select_active_players, build_board_and_context

from fantasy_baseball.scoring import score_roto_dict
from fantasy_baseball.simulation import simulate_season
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES as ALL_CATS,
)
from fantasy_baseball.utils.constants import (
    HITTING_COUNTING,
    PITCHING_COUNTING,
)
from fantasy_baseball.utils.constants import (
    INVERSE_STATS as INVERSE,
)

ITERS = 1000


def main():
    if len(sys.argv) < 2:
        drafts_dir = PROJECT_ROOT / "data" / "drafts"
        mocks = sorted(drafts_dir.glob("*.json"))
        if not mocks:
            print("No drafts found. Provide a path as argument.")
            sys.exit(1)
        draft_path = mocks[-1]
    else:
        draft_path = Path(sys.argv[1])

    print(f"Analyzing: {draft_path.name}")

    with open(draft_path) as f:
        draft_data = json.load(f)

    meta = draft_data["metadata"]

    # Determine user's team name from metadata, falling back to config
    HART = meta.get("user_team")
    if not HART:
        # Legacy files without user_team: derive from draft_position
        pos = meta.get("draft_position", 8)
        ctx_tmp = build_board_and_context()
        HART = ctx_tmp["config"].teams.get(pos, f"Team {pos}")
        print(f"  (no user_team in metadata, inferred '{HART}' from position {pos})")

    print(f"  {'Mock' if meta['mock'] else 'Real'} draft | Position {meta['draft_position']} | "
          f"{meta['num_teams']} teams | {meta['strategy']} + {meta['scoring_mode']}")
    print(f"  User team: {HART}")
    print(f"  Picks: {meta['picks_completed']}/{meta['total_picks']} | "
          f"Complete: {meta['complete']}")

    # Build board
    print("\nBuilding draft board...")
    ctx = build_board_and_context()
    config = ctx["config"]
    full_board = ctx["full_board"]
    board = ctx["board"]
    h_slots, p_slots = _active_slot_counts(config.roster_slots)

    # Reconstruct all team rosters
    team_players = {}
    for entry in draft_data["draft_log"]:
        tn = entry["team"]
        if tn not in team_players:
            team_players[tn] = []
        pid = entry["player_id"]
        rows = board[board["player_id"] == pid]
        if rows.empty:
            rows = full_board[full_board["player_id"] == pid]
        if not rows.empty:
            team_players[tn].append(rows.iloc[0])

    team_names = sorted(team_players.keys())
    num_teams = len(team_names)

    # ================================================================
    # STATIC PROJECTION
    # ================================================================
    print()
    print("=" * 130)
    print("STATIC POST-DRAFT PROJECTED STANDINGS")
    print("=" * 130)

    results = []
    for tn in team_names:
        players = team_players[tn]
        all_h = [p for p in players if p["player_type"] == "hitter"]
        all_p = [p for p in players if p["player_type"] == "pitcher"]
        hitters, pitchers = _select_active_players(all_h, all_p, config.roster_slots)

        r = sum(h.get("r", 0) for h in hitters)
        hr = sum(h.get("hr", 0) for h in hitters)
        rbi = sum(h.get("rbi", 0) for h in hitters)
        sb = sum(h.get("sb", 0) for h in hitters)
        th = sum(h.get("h", 0) for h in hitters)
        tab = sum(h.get("ab", 0) for h in hitters)
        avg = th / tab if tab > 0 else 0
        w = sum(p.get("w", 0) for p in pitchers)
        k = sum(p.get("k", 0) for p in pitchers)
        sv = sum(p.get("sv", 0) for p in pitchers)
        tip = sum(p.get("ip", 0) for p in pitchers)
        ter = sum(p.get("er", 0) for p in pitchers)
        tbb = sum(p.get("bb", 0) for p in pitchers)
        tha = sum(p.get("h_allowed", 0) for p in pitchers)
        era = ter * 9 / tip if tip > 0 else 0
        whip = (tbb + tha) / tip if tip > 0 else 0
        results.append({
            "team": tn, "R": r, "HR": hr, "RBI": rbi, "SB": sb, "AVG": avg,
            "W": w, "K": k, "SV": sv, "ERA": era, "WHIP": whip,
            "nh": len(hitters), "np": len(pitchers),
        })

    for cat in ALL_CATS:
        rev = cat not in INVERSE
        st = sorted(results, key=lambda x: x[cat], reverse=rev)
        for i, t in enumerate(st):
            t[f"{cat}_p"] = num_teams - i

    for t in results:
        t["tot"] = sum(t[f"{c}_p"] for c in ALL_CATS)

    results.sort(key=lambda x: x["tot"], reverse=True)

    print(f"\n{'Rk':<3} {'Team':<32} {'Pts':>4}  "
          f"{'R':>5} {'HR':>4} {'RBI':>5} {'SB':>4} {'AVG':>6}  "
          f"{'W':>4} {'K':>5} {'SV':>4} {'ERA':>5} {'WHIP':>6}")
    print("-" * 120)
    for i, t in enumerate(results, 1):
        m = " <<<" if t["team"] == HART else ""
        print(f"{i:<3} {t['team']:<32} {t['tot']:>4}  "
              f"{t['R']:>5.0f} {t['HR']:>4.0f} {t['RBI']:>5.0f} "
              f"{t['SB']:>4.0f} {t['AVG']:>6.3f}  "
              f"{t['W']:>4.0f} {t['K']:>5.0f} {t['SV']:>4.0f} "
              f"{t['ERA']:>5.2f} {t['WHIP']:>6.3f}{m}")

    # Category points
    print(f"\n{'Team':<32} ", end="")
    for c in ALL_CATS:
        print(f"{c:>5}", end="")
    print(f"{'TOT':>6}")
    print("-" * 90)
    for t in results:
        m = " <<<" if t["team"] == HART else ""
        print(f"{t['team']:<32} ", end="")
        for c in ALL_CATS:
            print(f"{t[f'{c}_p']:>5}", end="")
        print(f"{t['tot']:>6}{m}")

    hart_static = next(t for t in results if t["team"] == HART)
    rank_static = next(i + 1 for i, t in enumerate(results) if t["team"] == HART)
    suf = {1: "st", 2: "nd", 3: "rd"}.get(rank_static, "th")
    print(f"\nHart of the Order: {rank_static}{suf} place ({hart_static['tot']} pts)")
    top = [(c, hart_static[f"{c}_p"]) for c in ALL_CATS if hart_static[f"{c}_p"] >= 8]
    bot = [(c, hart_static[f"{c}_p"]) for c in ALL_CATS if hart_static[f"{c}_p"] <= 3]
    if top:
        print(f"  Strengths: {', '.join(f'{c}({p})' for c, p in sorted(top, key=lambda x: -x[1]))}")
    if bot:
        print(f"  Weaknesses: {', '.join(f'{c}({p})' for c, p in sorted(bot, key=lambda x: x[1]))}")

    # ================================================================
    # MONTE CARLO
    # ================================================================
    print()
    print("=" * 130)
    print(f"MONTE CARLO SIMULATION ({ITERS} iterations)")
    print("=" * 130)

    rosters = {}
    for tn in team_names:
        roster = []
        for p in team_players[tn]:
            entry = {"player_type": p["player_type"]}
            for s in HITTING_COUNTING + PITCHING_COUNTING:
                v = p.get(s, 0)
                entry[s] = float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0.0
            roster.append(entry)
        rosters[tn] = roster

    rng = np.random.default_rng(42)
    all_pts = {tn: [] for tn in team_names}
    all_ranks = {tn: [] for tn in team_names}

    t0 = time.time()
    for _it in range(ITERS):
        team_stats, _ = simulate_season(rosters, rng, h_slots, p_slots)
        roto = score_roto_dict(team_stats)

        sorted_teams = sorted(roto.items(), key=lambda x: x[1]["total"], reverse=True)
        for rank_i, (tn, pts) in enumerate(sorted_teams, 1):
            all_pts[tn].append(pts["total"])
            all_ranks[tn].append(rank_i)

    elapsed = time.time() - t0
    print(f"Completed {ITERS} iterations in {elapsed:.1f}s\n")

    print(f"{'Team':<32} {'Mean':>5} {'Std':>5} {'Med':>4} "
          f"{'Win%':>5} {'Top3':>5} {'Bot3':>5} "
          f"{'P10':>4} {'P90':>4} {'AvgFn':>5}")
    print("-" * 100)

    mc_results = []
    for tn in team_names:
        pts = np.array(all_pts[tn])
        ranks = np.array(all_ranks[tn])
        mc_results.append({
            "team": tn,
            "mean": float(np.mean(pts)),
            "std": float(np.std(pts)),
            "median": float(np.median(pts)),
            "win_pct": float(np.mean(ranks == 1) * 100),
            "top3": float(np.mean(ranks <= 3) * 100),
            "bot3": float(np.mean(ranks >= 8) * 100),
            "p10": float(np.percentile(pts, 10)),
            "p90": float(np.percentile(pts, 90)),
            "avg_finish": float(np.mean(ranks)),
        })
    mc_results.sort(key=lambda x: x["mean"], reverse=True)

    for r in mc_results:
        m = " <<<" if r["team"] == HART else ""
        print(f"{r['team']:<32} {r['mean']:>5.1f} {r['std']:>5.1f} {r['median']:>4.0f} "
              f"{r['win_pct']:>5.1f} {r['top3']:>5.1f} {r['bot3']:>5.1f} "
              f"{r['p10']:>4.0f} {r['p90']:>4.0f} {r['avg_finish']:>5.2f}{m}")

    hart_mc = next(r for r in mc_results if r["team"] == HART)
    hart_mc_rank = next(i + 1 for i, r in enumerate(mc_results) if r["team"] == HART)
    print("\nHart of the Order MC summary:")
    print(f"  Mean: {hart_mc['mean']:.1f} pts | Median: {hart_mc['median']:.0f} | Std: {hart_mc['std']:.1f}")
    print(f"  Win rate: {hart_mc['win_pct']:.1f}% | Top 3: {hart_mc['top3']:.1f}% | Bottom 3: {hart_mc['bot3']:.1f}%")
    print(f"  Floor (P10): {hart_mc['p10']:.0f} | Ceiling (P90): {hart_mc['p90']:.0f}")
    print(f"  Average finish: {hart_mc['avg_finish']:.2f} (ranked {hart_mc_rank} of {len(mc_results)} by MC mean)")


if __name__ == "__main__":
    main()
