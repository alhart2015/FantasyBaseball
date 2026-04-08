"""Weekly summary — rosters, projections, monte carlo, lineup, waivers, trades.

Pulls current rosters from Yahoo, runs static + monte carlo analysis,
then generates lineup, waiver, and trade recommendations.

Usage:
    python scripts/summary.py [--iterations N] [--seed S]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.auth.yahoo_auth import get_yahoo_session, get_league
from fantasy_baseball.config import load_config
from fantasy_baseball.data.db import get_connection, get_blended_projections
from fantasy_baseball.lineup.yahoo_roster import fetch_injuries, fetch_roster, fetch_standings
from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
from fantasy_baseball.lineup.waivers import scan_waivers, detect_open_slots, fetch_and_match_free_agents
from fantasy_baseball.sgp.rankings import compute_combined_sgp_rankings
from fantasy_baseball.data.projections import match_roster_to_projections
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_hitter, is_pitcher
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD, IL_STATUSES
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.scoring import project_team_stats, score_roto, ALL_CATS, INVERSE_CATS
from fantasy_baseball.simulation import simulate_season, apply_management_adjustment, run_monte_carlo

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"

# ── Helpers ───────────────────────────────────────────────────────────

# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weekly fantasy baseball summary")
    parser.add_argument("--iterations", "-n", type=int, default=1000,
                        help="Monte Carlo iterations (default: 1000)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    team_name = config.team_name
    print(f"Weekly Summary | {team_name}")
    print()

    # ── Connect to Yahoo ──────────────────────────────────────────────
    print("Connecting to Yahoo...")
    session = get_yahoo_session()
    league = get_league(session, config.league_id, config.game_code)
    teams = league.teams()

    user_team_key = None
    for key, td in teams.items():
        if normalize_name(td["name"]) == normalize_name(team_name):
            user_team_key = key
            break

    # ── Fetch all rosters ─────────────────────────────────────────────
    print("Fetching rosters...")
    all_rosters_raw = {}
    for key, td in teams.items():
        all_rosters_raw[td["name"]] = fetch_roster(league, key)

    print("Fetching standings...")
    try:
        standings = fetch_standings(league)
        if not standings or all(sum(s.get("stats", {}).values()) == 0 for s in standings if "stats" in s):
            print("  Pre-season — no standings data yet. Using equal leverage weights.")
            standings = None
    except Exception:
        print("  Could not parse standings. Using equal leverage weights.")
        standings = None

    # ── Load projections ──────────────────────────────────────────────
    print("Loading projections...")
    conn = get_connection()
    hitters_proj, pitchers_proj = get_blended_projections(conn)
    conn.close()

    # Precompute normalized names for fast projection matching
    if not hitters_proj.empty:
        hitters_proj["_name_norm"] = hitters_proj["name"].apply(normalize_name)
    if not pitchers_proj.empty:
        pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)

    # ── Match rosters to projections ──────────────────────────────────
    all_rosters = {}
    for name, raw in all_rosters_raw.items():
        all_rosters[name] = match_roster_to_projections(raw, hitters_proj, pitchers_proj)

    user_roster = all_rosters[team_name]
    print(f"  {team_name}: {len(user_roster)} players matched")

    # ── 1. CURRENT STANDINGS ─────────────────────────────────────────
    if standings:
        print()
        print("=" * 90)
        print("CURRENT STANDINGS")
        print("=" * 90)

        standings_sorted = sorted(standings, key=lambda t: t.get("rank", 99))
        print(f"\n{'Team':<32} {'Rank':>4}   " + "  ".join(f"{c:>5}" for c in ALL_CATS))
        print("-" * 90)
        for t in standings_sorted:
            s = t.get("stats", {})
            marker = " <<<" if t["name"] == team_name else ""
            line = f"{t['name']:<32} {t.get('rank', '?'):>4}  "
            for c in ALL_CATS:
                v = s.get(c, 0)
                if c == "AVG":
                    line += f" {v:.3f}"
                elif c in ("ERA", "WHIP"):
                    line += f" {v:>5.2f}"
                else:
                    line += f" {v:>5.0f}"
            print(line + marker)

    # ── 2. STATIC PROJECTED STANDINGS ─────────────────────────────────
    print()
    print("=" * 90)
    print("PROJECTED ROTO STANDINGS (static)")
    print("=" * 90)

    all_stats = {}
    for name, roster in all_rosters.items():
        all_stats[name] = project_team_stats(roster)

    roto = score_roto(all_stats)
    sorted_teams = sorted(roto.items(), key=lambda x: x[1]["total"], reverse=True)

    print(f"\n{'Team':<32} {'Pts':>4}   " + "  ".join(f"{c:>5}" for c in ALL_CATS))
    print("-" * 90)
    for name, pts in sorted_teams:
        s = all_stats[name]
        marker = " <<<" if name == team_name else ""
        line = f"{name:<32} {pts['total']:>4.0f}  "
        for c in ALL_CATS:
            if c == "AVG":
                line += f" {s[c]:.3f}"
            elif c in ("ERA", "WHIP"):
                line += f" {s[c]:>5.2f}"
            else:
                line += f" {s[c]:>5.0f}"
        print(line + marker)

    print(f"\n{'':>32} {'':>4}   " + "  ".join(f"{c:>5}" for c in ALL_CATS))
    print(f"{'Roto points':<32} {'':>4}  ", end="")
    hart_roto = roto[team_name]
    for c in ALL_CATS:
        print(f" {hart_roto.get(f'{c}_pts', 0):>5.0f}", end="")
    print()

    # ── 3. MONTE CARLO ────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"MONTE CARLO ({args.iterations} iterations)")
    print("=" * 90)

    # Compute active slots from config
    h_slots = sum(v for k, v in config.roster_slots.items() if k not in ("P", "BN", "IL"))
    p_slots = config.roster_slots.get("P", 9)

    # Run both modes: without and with management adjustments
    for mc_label, use_mgmt in [("Roster strength only", False),
                                ("With in-season management", True)]:
        mc = run_monte_carlo(
            all_rosters, h_slots, p_slots, team_name,
            n_iterations=args.iterations, use_management=use_mgmt, seed=args.seed,
        )

        print(f"\n  {mc_label}")
        print(f"  {'Team':<32} {'Med':>4} {'P10':>4} {'P90':>4}  {'1st':>5} {'Top3':>5}")
        print("  " + "-" * 66)
        mc_sorted = sorted(mc["team_results"].items(),
                           key=lambda x: x[1]["median_pts"], reverse=True)
        for name, res in mc_sorted:
            marker = " <<<" if name == team_name else ""
            print(f"  {name:<32} {res['median_pts']:>4.0f} {res['p10']:>4.0f} {res['p90']:>4.0f}"
                  f"  {res['first_pct']:>5.1f}% {res['top3_pct']:>5.1f}%{marker}")

    # Category risk uses the management-adjusted run (last iteration's mc)
    print(f"\n{'Category risk — ' + team_name}")
    print(f"  {'Cat':>4} {'Med':>4} {'P10':>4} {'P90':>4}  {'Top3':>5} {'Bot3':>5}")
    print("  " + "-" * 40)
    for c in ALL_CATS:
        r = mc["category_risk"][c]
        print(f"  {c:>4} {r['median_pts']:>4.0f} {r['p10']:>4.0f} {r['p90']:>4.0f}"
              f"  {r['top3_pct']:>5.1f}% {r['bot3_pct']:>5.1f}%")

    # ── 4. LINEUP RECOMMENDATIONS ─────────────────────────────────────
    print()
    print("=" * 90)
    print("LINEUP RECOMMENDATIONS")
    print("=" * 90)

    # Build projected standings for leverage
    projected_standings = [
        {"name": name, "stats": all_stats[name]} for name in all_stats
    ]

    if standings:
        leverage = calculate_leverage(
            standings, team_name, projected_standings=projected_standings,
        )
    else:
        # Pre-season: equal weights across all categories
        leverage = {c: 1.0 / len(ALL_CATS) for c in ALL_CATS}

    il_players = [p for p in user_roster if p.status in IL_STATUSES]
    active_roster = [p for p in user_roster if p.status not in IL_STATUSES]
    if il_players:
        print(f"\nExcluding {len(il_players)} IL player(s):")
        for p in il_players:
            print(f"  {p.name} ({p.status})")

    user_hitters = []
    user_pitchers = []
    denoms = get_sgp_denominators()
    for p in active_roster:
        p_series = pd.Series(p)
        p_series["total_sgp"] = calculate_player_sgp(p_series, denoms=denoms)
        wsgp = calculate_weighted_sgp(p_series, leverage)
        p["wsgp"] = wsgp
        if p["player_type"] == "hitter":
            user_hitters.append(p)
        else:
            user_pitchers.append(p)

    # Hitter lineup
    hitter_lineup = optimize_hitter_lineup(
        [pd.Series(h) for h in user_hitters], leverage,
        roster_slots=config.roster_slots,
    )

    print("\nOptimal hitter lineup:")
    for slot, name in hitter_lineup.items():
        wsgp = next((h["wsgp"] for h in user_hitters if h["name"] == name), 0)
        print(f"  {slot:<5} {name:<28} wSGP: {wsgp:>5.2f}")

    # Pitcher lineup
    starter_pitchers, bench_pitchers = optimize_pitcher_lineup(
        [pd.Series(p) for p in user_pitchers], leverage,
        slots=config.roster_slots.get("P", 9),
    )

    print("\nOptimal pitcher lineup:")
    for p in starter_pitchers:
        print(f"  P    {p['name']:<28} wSGP: {p['wsgp']:>5.2f}")
    if bench_pitchers:
        for p in bench_pitchers:
            print(f"  BN   {p['name']:<28} wSGP: {p['wsgp']:>5.2f}")

    # Check for start/sit changes vs current Yahoo positions
    yahoo_roster = all_rosters_raw[team_name]
    current_bench = {p["name"].replace(" (Batter)", "").replace(" (Pitcher)", "")
                     for p in yahoo_roster if p["selected_position"] == "BN"}
    current_il = {p["name"].replace(" (Batter)", "").replace(" (Pitcher)", "")
                  for p in yahoo_roster if p["selected_position"] == "IL"}

    opt_active_names = set(hitter_lineup.values()) | {p["name"] for p in starter_pitchers}
    opt_bench_names = {h["name"] for h in user_hitters if h["name"] not in hitter_lineup.values()}
    opt_bench_names |= {p["name"] for p in bench_pitchers}

    should_bench = (current_bench | current_il) - opt_bench_names - current_il
    should_start = current_bench & opt_active_names

    if should_start or should_bench:
        print("\nSTART/SIT CHANGES:")
        for name in sorted(should_start):
            print(f"  START: {name} (currently benched)")
        new_bench = opt_bench_names - current_bench - current_il
        for name in sorted(new_bench):
            if name not in current_il:
                print(f"  BENCH: {name} (currently starting)")
    else:
        print("\nNo start/sit changes needed — current lineup is optimal.")

    # ── 5. INJURY MANAGEMENT ──────────────────────────────────────────
    print()
    print("=" * 90)
    print("INJURY MANAGEMENT")
    print("=" * 90)

    injuries = fetch_injuries(league, user_team_key)

    if injuries:
        print(f"\n  {'Player':<25} {'Status':<20} {'Injury':<15} {'Slot'}")
        print("  " + "-" * 70)
        on_il = []
        not_on_il = []
        for inj in injuries:
            name = inj["name"]
            status = inj.get("status_full") or inj.get("status", "?")
            note = inj.get("injury_note", "")
            slot = inj.get("selected_position", "?")
            print(f"  {name:<25} {status:<20} {note:<15} {slot}")
            if slot in ("IL", "IL+"):
                on_il.append(inj)
            else:
                not_on_il.append(inj)

        if not_on_il:
            print("\n  NOT IN IL SLOT:")
            for inj in not_on_il:
                name = inj["name"]
                status = inj.get("status", "")
                if "IL" in status:
                    print(f"    {name} — IL-eligible but in {inj['selected_position']} slot. "
                          f"Move to IL to free a roster spot.")
                else:
                    print(f"    {name} — {status} (day-to-day)")
    else:
        print("\n  No injured players on roster.")

    # ── 6. WAIVER WIRE ────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("WAIVER WIRE")
    print("=" * 90)

    yahoo_roster = all_rosters_raw[team_name]
    open_hitter_slots, open_pitcher_slots, open_bench_slots = detect_open_slots(
        yahoo_roster, config.roster_slots,
    )
    total_open = open_hitter_slots + open_pitcher_slots + open_bench_slots
    if total_open:
        parts = []
        if open_hitter_slots:
            parts.append(f"{open_hitter_slots} hitter")
        if open_pitcher_slots:
            parts.append(f"{open_pitcher_slots} pitcher")
        if open_bench_slots:
            parts.append(f"{open_bench_slots} bench")
        print(f"\n  Empty slots: {', '.join(parts)}")

    print("  Scanning free agents...")
    fa_players, fa_fetched = fetch_and_match_free_agents(
        league, hitters_proj, pitchers_proj,
    )

    if fa_fetched == 0:
        print("  No available players returned by Yahoo")
    else:
        print(f"  Found {fa_fetched} free agents, {len(fa_players)} matched projections")

    user_roster_series = [pd.Series(p) for p in user_roster]
    waiver_recs = scan_waivers(
        user_roster_series, fa_players, leverage, max_results=5,
        open_hitter_slots=open_hitter_slots,
        open_pitcher_slots=open_pitcher_slots,
        open_bench_slots=open_bench_slots,
        roster_slots=config.roster_slots,
    )

    if waiver_recs:
        print(f"\n  Top {len(waiver_recs)} waiver moves:")
        for i, rec in enumerate(waiver_recs, 1):
            if rec["drop"].startswith("(empty"):
                print(f"    {i}. ADD {rec['add']:<22} {rec['drop']}  "
                      f"value: +{rec['sgp_gain']:.2f} wSGP")
            else:
                print(f"    {i}. ADD {rec['add']:<22} DROP {rec['drop']:<22} "
                      f"gain: +{rec['sgp_gain']:.2f} wSGP")
                if rec.get("categories"):
                    gains = [f"+{c}" for c, v in rec["categories"].items() if v > 0.01]
                    losses = [f"-{c}" for c, v in rec["categories"].items() if v < -0.01]
                    if gains or losses:
                        print(f"       gains {', '.join(gains)}  |  costs {', '.join(losses)}")
    else:
        print("\n  No positive waiver moves found — your roster is solid.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
