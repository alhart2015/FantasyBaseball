"""Backtest 2025 draft using actual 2025 preseason projections (Steamer+ZiPS)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fantasy_baseball.data.fangraphs import parse_hitting_csv, parse_pitching_csv
from fantasy_baseball.data.projections import _blend_hitters, _blend_pitchers
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.name_utils import normalize_name
from simulate_draft import _select_active_players, _active_slot_counts
from backtest_2025 import DRAFT_2025, ACTUAL, ALL_CATS, INVERSE

ROSTER_SLOTS = {
    "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1,
    "OF": 4, "UTIL": 2, "P": 9, "BN": 2, "IL": 2,
}


def main():
    # Load 2025 preseason projections
    h1 = parse_hitting_csv(PROJECT_ROOT / "data/projections/fangraphs-leaderboard-projections-steamer-hitters-2025.csv")
    h2 = parse_hitting_csv(PROJECT_ROOT / "data/projections/fangraphs-leaderboard-projections-zips-hitters-2025.csv")
    p1 = parse_pitching_csv(PROJECT_ROOT / "data/projections/fangraphs-leaderboard-projections-steamer-pitchers-2025.csv")
    p2 = parse_pitching_csv(PROJECT_ROOT / "data/projections/fangraphs-leaderboard-projections-zips-pitchers-2025.csv")

    h1["_weight"] = 0.5
    h2["_weight"] = 0.5
    p1["_weight"] = 0.5
    p2["_weight"] = 0.5

    hitters = _blend_hitters([h1, h2])
    pitchers = _blend_pitchers([p1, p2])
    pool = pd.concat([hitters, pitchers], ignore_index=True)

    denoms = get_sgp_denominators()
    pool["total_sgp"] = pool.apply(lambda row: calculate_player_sgp(row, denoms=denoms), axis=1)
    pool["name_norm"] = pool["name"].apply(normalize_name)

    print(f"2025 preseason projections: {len(hitters)} hitters, {len(pitchers)} pitchers")

    # Build lookup
    lookup = {}
    for idx, row in pool.iterrows():
        key = row["name_norm"]
        if key not in lookup or row["total_sgp"] > pool.loc[lookup[key], "total_sgp"]:
            lookup[key] = idx

    # Match draft picks
    team_rosters = {}
    matched = 0
    missed = []
    for rnd, player, team in DRAFT_2025:
        if team not in team_rosters:
            team_rosters[team] = []
        key = normalize_name(player)
        if key in lookup:
            team_rosters[team].append(pool.loc[lookup[key]])
            matched += 1
        else:
            key_clean = key.replace(" jr.", "").replace(" sr.", "").replace(" ii", "").strip()
            found = False
            for bkey, bidx in lookup.items():
                if key_clean == bkey or (len(key_clean) > 5 and key_clean in bkey):
                    team_rosters[team].append(pool.loc[bidx])
                    matched += 1
                    found = True
                    break
            if not found:
                missed.append((rnd, player, team))

    print(f"Matched: {matched}/230, Missed: {len(missed)}")
    for rnd, player, team in missed:
        print(f"  R{rnd}: {player} ({team})")

    # Static projection
    h_slots, p_slots = _active_slot_counts(ROSTER_SLOTS)
    results = []
    for tn, players in team_rosters.items():
        all_h = [p for p in players if p["player_type"] == "hitter"]
        all_p = [p for p in players if p["player_type"] == "pitcher"]
        active_h, active_p = _select_active_players(all_h, all_p, ROSTER_SLOTS)

        r = sum(h.get("r", 0) for h in active_h)
        hr = sum(h.get("hr", 0) for h in active_h)
        rbi = sum(h.get("rbi", 0) for h in active_h)
        sb = sum(h.get("sb", 0) for h in active_h)
        th = sum(h.get("h", 0) for h in active_h)
        tab = sum(h.get("ab", 0) for h in active_h)
        avg = th / tab if tab > 0 else 0
        w = sum(p.get("w", 0) for p in active_p)
        k = sum(p.get("k", 0) for p in active_p)
        sv = sum(p.get("sv", 0) for p in active_p)
        tip = sum(p.get("ip", 0) for p in active_p)
        ter = sum(p.get("er", 0) for p in active_p)
        tbb = sum(p.get("bb", 0) for p in active_p)
        tha = sum(p.get("h_allowed", 0) for p in active_p)
        era = ter * 9 / tip if tip > 0 else 0
        whip = (tbb + tha) / tip if tip > 0 else 0
        results.append({
            "team": tn, "R": r, "HR": hr, "RBI": rbi, "SB": sb, "AVG": avg,
            "W": w, "K": k, "SV": sv, "ERA": era, "WHIP": whip,
        })

    for cat in ALL_CATS:
        rev = cat not in INVERSE
        st = sorted(results, key=lambda x: x[cat], reverse=rev)
        for i, t in enumerate(st):
            t[f"{cat}_p"] = 10 - i
    for t in results:
        t["tot"] = sum(t[f"{c}_p"] for c in ALL_CATS)
    results.sort(key=lambda x: x["tot"], reverse=True)

    # Compute actual roto points
    act_pts_map = {}
    for cat in ALL_CATS:
        rev = cat not in INVERSE
        ranked = sorted(ACTUAL.items(), key=lambda x: x[1][cat], reverse=rev)
        for i, (tn, _) in enumerate(ranked):
            act_pts_map.setdefault(tn, {})[cat] = 10 - i

    print()
    print("=" * 130)
    print("2025 DRAFT EVALUATION (Steamer+ZiPS 2025 preseason, 50/50 blend)")
    print("=" * 130)
    print(f"\n{'Rk':>2} {'Team':<32} {'Proj':>4} {'Actual':>6} {'ActRk':>5} {'Delta':>6}")
    print("-" * 60)
    for i, t in enumerate(results, 1):
        a = ACTUAL.get(t["team"], {})
        act_pts = a.get("total", "?")
        act_rk = a.get("rank", "?")
        delta = f"{act_pts - t['tot']:+.1f}" if isinstance(act_pts, (int, float)) else "?"
        m = " <<<" if t["team"] == "Hart of the Order" else ""
        print(f"{i:>2} {t['team']:<32} {t['tot']:>4} {act_pts:>6} {act_rk:>5} {delta:>6}{m}")

    # Hart detailed comparison
    hart = next(t for t in results if t["team"] == "Hart of the Order")
    hart_rank = next(i + 1 for i, t in enumerate(results) if t["team"] == "Hart of the Order")
    a = ACTUAL["Hart of the Order"]

    print()
    print("=" * 90)
    print("HART OF THE ORDER: 2025 Preseason Projection vs Actual")
    print("=" * 90)
    print(f"\n  Projected: {hart_rank}{'st' if hart_rank == 1 else 'nd' if hart_rank == 2 else 'rd' if hart_rank == 3 else 'th'} place ({hart['tot']} pts)")
    print(f"  Actual:    {a['rank']}{'st' if a['rank'] == 1 else 'nd' if a['rank'] == 2 else 'rd' if a['rank'] == 3 else 'th'} place ({a['total']} pts)")
    print(f"  Delta:     {a['rank'] - hart_rank:+d} places, {a['total'] - hart['tot']:+.0f} pts")

    print(f"\n{'Cat':>5} {'Projected':>10} {'Actual':>10} {'Diff%':>8} {'Proj Pts':>8} {'Act Pts':>8}  Assessment")
    print("-" * 85)

    hart_act_pts = act_pts_map["Hart of the Order"]
    for cat in ALL_CATS:
        p_val = hart[cat]
        a_val = a[cat]
        proj_pts = hart[f"{cat}_p"]
        act_p = hart_act_pts[cat]

        if cat in ("AVG", "ERA", "WHIP"):
            diff_pct = -(a_val - p_val) / p_val * 100 if cat in INVERSE else (a_val - p_val) / p_val * 100
            fmt_p, fmt_a = f"{p_val:.3f}", f"{a_val:.3f}"
        else:
            diff_pct = (a_val - p_val) / p_val * 100 if p_val != 0 else 0
            fmt_p, fmt_a = f"{p_val:.0f}", f"{a_val:.0f}"

        pts_delta = act_p - proj_pts
        assessment = ""
        if abs(diff_pct) < 5:
            assessment = "on target"
        elif diff_pct > 20:
            assessment = "MASSIVE overperformance"
        elif diff_pct > 10:
            assessment = "strong overperformance"
        elif diff_pct > 5:
            assessment = "mild overperformance"
        elif diff_pct < -10:
            assessment = "significant underperformance"
        elif diff_pct < -5:
            assessment = "mild underperformance"

        if pts_delta != 0:
            assessment += f" ({pts_delta:+d} roto pts)"

        print(f"{cat:>5} {fmt_p:>10} {fmt_a:>10} {diff_pct:>+7.1f}% {proj_pts:>8} {act_p:>8}  {assessment}")

    # Summary
    proj_total = hart["tot"]
    act_total = a["total"]
    pts_from_mgmt = act_total - proj_total

    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")
    print(f"\n  Draft grade: ", end="")
    if hart_rank <= 2:
        print("A")
    elif hart_rank <= 4:
        print("B+")
    elif hart_rank <= 6:
        print("B-")
    elif hart_rank <= 8:
        print("C")
    else:
        print("D")
    print(f"    Projected {hart_rank}th of 10 using 2025 preseason projections")

    print(f"\n  Management grade: ", end="")
    if pts_from_mgmt >= 15:
        print("A+")
    elif pts_from_mgmt >= 10:
        print("A")
    elif pts_from_mgmt >= 5:
        print("B+")
    elif pts_from_mgmt >= 0:
        print("B")
    else:
        print("C")
    print(f"    Gained {pts_from_mgmt:+.0f} pts vs projection ({hart_rank}th -> {a['rank']}rd)")

    # Biggest gains/losses
    gains = []
    losses = []
    for cat in ALL_CATS:
        delta = hart_act_pts[cat] - hart[f"{cat}_p"]
        if delta > 0:
            gains.append((cat, delta))
        elif delta < 0:
            losses.append((cat, delta))
    gains.sort(key=lambda x: -x[1])
    losses.sort(key=lambda x: x[1])

    if gains:
        print(f"\n  Biggest category gains: {', '.join(f'{c} {d:+d}pts' for c, d in gains)}")
    if losses:
        print(f"  Biggest category losses: {', '.join(f'{c} {d:+d}pts' for c, d in losses)}")


if __name__ == "__main__":
    main()
