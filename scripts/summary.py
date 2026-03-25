"""Weekly summary — rosters, projections, monte carlo, lineup, waivers, trades.

Pulls current rosters from Yahoo, runs static + monte carlo analysis,
then generates lineup, waiver, and trade recommendations.

Usage:
    python scripts/summary.py [--iterations N] [--seed S]
"""
import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.auth.yahoo_auth import get_yahoo_session, get_league
from fantasy_baseball.config import load_config
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.lineup.yahoo_roster import (
    fetch_roster, fetch_standings, fetch_free_agents,
)
from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
from fantasy_baseball.lineup.waivers import scan_waivers
from fantasy_baseball.trades.evaluate import find_trades, compute_roto_points_by_cat
from fantasy_baseball.trades.pitch import generate_pitch
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_hitter, is_pitcher
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.scoring import project_team_stats, score_roto, ALL_CATS, INVERSE_CATS

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
INJURIES_PATH = PROJECT_ROOT / "data" / "injuries.yaml"

# Monte carlo parameters
INJURY_PROB = {"pitcher": 0.45, "hitter": 0.18}
INJURY_SEVERITY = {"pitcher": (0.20, 0.60), "hitter": (0.15, 0.40)}
STAT_VARIANCE = {"hitter": 0.10, "pitcher": 0.18}
HITTING_COUNTING = ["r", "hr", "rbi", "sb", "h", "ab"]
PITCHING_COUNTING = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]
REPLACEMENT_HITTER = {"r": 55, "hr": 12, "rbi": 50, "sb": 5, "h": 125, "ab": 500}
REPLACEMENT_SP = {"w": 7, "k": 120, "sv": 0, "ip": 140, "er": 70, "bb": 50, "h_allowed": 139}
REPLACEMENT_RP = {"w": 2, "k": 55, "sv": 5, "ip": 60, "er": 30, "bb": 21, "h_allowed": 60}


# ── Helpers ───────────────────────────────────────────────────────────

def load_injuries():
    """Load injury tracker from data/injuries.yaml. Returns list of dicts."""
    if not INJURIES_PATH.exists():
        return []
    with open(INJURIES_PATH) as f:
        data = yaml.safe_load(f)
    entries = data.get("injuries") if data else []
    return entries or []


def match_roster_to_projections(roster, hitters_proj, pitchers_proj):
    """Match roster players to projections by name. Returns enriched dicts."""
    matched = []
    for player in roster:
        name = player["name"].replace(" (Batter)", "").replace(" (Pitcher)", "")
        name_norm = normalize_name(name)
        positions = player["positions"]

        proj = None
        ptype = None
        if is_hitter(positions) and not hitters_proj.empty:
            matches = hitters_proj[hitters_proj["name"].apply(normalize_name) == name_norm]
            if not matches.empty:
                proj = matches.iloc[0]
                ptype = "hitter"
        if proj is None and is_pitcher(positions) and not pitchers_proj.empty:
            matches = pitchers_proj[pitchers_proj["name"].apply(normalize_name) == name_norm]
            if not matches.empty:
                proj = matches.iloc[0]
                ptype = "pitcher"
        if proj is None:
            for df, pt in [(hitters_proj, "hitter"), (pitchers_proj, "pitcher")]:
                if df.empty:
                    continue
                matches = df[df["name"].apply(normalize_name) == name_norm]
                if not matches.empty:
                    proj = matches.iloc[0]
                    ptype = pt
                    break

        if proj is not None:
            entry = {
                "name": name, "positions": positions,
                "player_type": ptype, "selected_position": player.get("selected_position", ""),
            }
            if ptype == "hitter":
                for col in ["r", "hr", "rbi", "sb", "avg", "h", "ab", "pa"]:
                    entry[col] = float(proj.get(col, 0) or 0)
            else:
                for col in ["w", "k", "sv", "era", "whip", "ip", "er", "bb", "h_allowed"]:
                    entry[col] = float(proj.get(col, 0) or 0)
            matched.append(entry)
    return matched


def simulate_season(team_rosters, rng, h_slots=13, p_slots=9):
    """One monte carlo season with injuries + variance. Returns team_stats dict."""
    team_stats = {}
    for tname, roster in team_rosters.items():
        hitters = [p for p in roster if p["player_type"] == "hitter"]
        pitchers = [p for p in roster if p["player_type"] == "pitcher"]

        adj_hitters = []
        for h in hitters:
            frac_missed = 0.0
            if rng.random() < INJURY_PROB["hitter"]:
                frac_missed = rng.uniform(*INJURY_SEVERITY["hitter"])
            scale = 1.0 - frac_missed
            perf = max(0, 1.0 + rng.normal(0, STAT_VARIANCE["hitter"]))
            row = {}
            for col in HITTING_COUNTING:
                base = h.get(col, 0)
                row[col] = base * perf * scale + REPLACEMENT_HITTER.get(col, 0) * frac_missed
            adj_hitters.append(row)

        adj_pitchers = []
        for p in pitchers:
            frac_missed = 0.0
            if rng.random() < INJURY_PROB["pitcher"]:
                frac_missed = rng.uniform(*INJURY_SEVERITY["pitcher"])
            scale = 1.0 - frac_missed
            perf = max(0, 1.0 + rng.normal(0, STAT_VARIANCE["pitcher"]))
            is_closer = p.get("sv", 0) >= CLOSER_SV_THRESHOLD
            repl = REPLACEMENT_RP if is_closer else REPLACEMENT_SP
            row = {}
            for col in PITCHING_COUNTING:
                base = p.get(col, 0)
                row[col] = base * perf * scale + repl.get(col, 0) * frac_missed
            row["sv_base"] = p.get("sv", 0)
            adj_pitchers.append(row)

        adj_hitters.sort(key=lambda h: h["r"] + h["hr"] + h["rbi"] + h["sb"], reverse=True)
        adj_pitchers.sort(
            key=lambda p: (p.get("sv_base", 0) >= CLOSER_SV_THRESHOLD, p["w"] + p["k"] + p.get("sv", 0)),
            reverse=True,
        )
        ah = adj_hitters[:h_slots]
        ap = adj_pitchers[:p_slots]

        total_ab = sum(h["ab"] for h in ah)
        total_h = sum(h["h"] for h in ah)
        total_ip = sum(p["ip"] for p in ap)
        total_er = sum(p["er"] for p in ap)
        total_bb = sum(p["bb"] for p in ap)
        total_ha = sum(p["h_allowed"] for p in ap)

        team_stats[tname] = {
            "R": sum(h["r"] for h in ah),
            "HR": sum(h["hr"] for h in ah),
            "RBI": sum(h["rbi"] for h in ah),
            "SB": sum(h["sb"] for h in ah),
            "AVG": total_h / total_ab if total_ab > 0 else 0,
            "W": sum(p["w"] for p in ap),
            "K": sum(p["k"] for p in ap),
            "SV": sum(p["sv"] for p in ap),
            "ERA": total_er * 9 / total_ip if total_ip > 0 else 99,
            "WHIP": (total_bb + total_ha) / total_ip if total_ip > 0 else 99,
        }
    return team_stats


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
    weights = config.projection_weights if config.projection_weights else None
    hitters_proj, pitchers_proj = blend_projections(
        PROJECTIONS_DIR, config.projection_systems, weights,
    )

    # ── Match rosters to projections ──────────────────────────────────
    all_rosters = {}
    for name, raw in all_rosters_raw.items():
        all_rosters[name] = match_roster_to_projections(raw, hitters_proj, pitchers_proj)

    user_roster = all_rosters[team_name]
    print(f"  {team_name}: {len(user_roster)} players matched")

    # ── 1. STATIC PROJECTED STANDINGS ─────────────────────────────────
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

    # ── 2. MONTE CARLO ────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"MONTE CARLO ({args.iterations} iterations)")
    print("=" * 90)

    # Compute active slots from config
    h_slots = sum(v for k, v in config.roster_slots.items() if k not in ("P", "BN", "IL"))
    p_slots = config.roster_slots.get("P", 9)

    rng = np.random.default_rng(args.seed)
    mc_totals = {name: [] for name in all_rosters}
    mc_wins = {name: 0 for name in all_rosters}
    mc_top3 = {name: 0 for name in all_rosters}
    mc_cat_pts = {name: {c: [] for c in ALL_CATS} for name in all_rosters}

    for _ in range(args.iterations):
        sim_stats = simulate_season(all_rosters, rng, h_slots, p_slots)
        sim_roto = score_roto(sim_stats)
        ranked = sorted(sim_roto.items(), key=lambda x: x[1]["total"], reverse=True)
        for rank, (name, pts) in enumerate(ranked, 1):
            mc_totals[name].append(pts["total"])
            if rank == 1:
                mc_wins[name] += 1
            if rank <= 3:
                mc_top3[name] += 1
            for c in ALL_CATS:
                mc_cat_pts[name][c].append(pts.get(f"{c}_pts", 0))

    n = args.iterations
    print(f"\n{'Team':<32} {'Med':>4} {'P10':>4} {'P90':>4}  {'1st':>5} {'Top3':>5}")
    print("-" * 70)
    mc_sorted = sorted(mc_totals.items(), key=lambda x: np.median(x[1]), reverse=True)
    for name, pts_list in mc_sorted:
        marker = " <<<" if name == team_name else ""
        med = np.median(pts_list)
        p10 = np.percentile(pts_list, 10)
        p90 = np.percentile(pts_list, 90)
        win_pct = mc_wins[name] / n * 100
        top3_pct = mc_top3[name] / n * 100
        print(f"{name:<32} {med:>4.0f} {p10:>4.0f} {p90:>4.0f}  {win_pct:>5.1f}% {top3_pct:>5.1f}%{marker}")

    print(f"\n{'Category risk — ' + team_name}")
    print(f"  {'Cat':>4} {'Med':>4} {'P10':>4} {'P90':>4}  {'Top3':>5} {'Bot3':>5}")
    print("  " + "-" * 40)
    for c in ALL_CATS:
        pts = mc_cat_pts[team_name][c]
        med = np.median(pts)
        p10 = np.percentile(pts, 10)
        p90 = np.percentile(pts, 90)
        top3 = sum(1 for p in pts if p >= 8) / n * 100
        bot3 = sum(1 for p in pts if p <= 3) / n * 100
        print(f"  {c:>4} {med:>4.0f} {p10:>4.0f} {p90:>4.0f}  {top3:>5.1f}% {bot3:>5.1f}%")

    # ── 3. LINEUP RECOMMENDATIONS ─────────────────────────────────────
    print()
    print("=" * 90)
    print("LINEUP RECOMMENDATIONS")
    print("=" * 90)

    if standings:
        leverage = calculate_leverage(standings, team_name)
    else:
        # Pre-season: equal weights across all categories
        leverage = {c: 1.0 / len(ALL_CATS) for c in ALL_CATS}

    user_hitters = []
    user_pitchers = []
    denoms = get_sgp_denominators()
    for p in user_roster:
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

    # ── 4. INJURY MANAGEMENT ──────────────────────────────────────────
    print()
    print("=" * 90)
    print("INJURY MANAGEMENT")
    print("=" * 90)

    injuries = load_injuries()
    today = date.today()

    if injuries:
        print(f"\n  {'Player':<25} {'IL Date':<12} {'Return':<12} {'Days Left':>9}  {'Replacement'}")
        print("  " + "-" * 80)
        returning_soon = []
        needs_replacement = []
        for inj in injuries:
            name = inj["name"]
            il_date = inj.get("il_date", "?")
            ret = inj.get("expected_return")
            replacement = inj.get("replacement")
            notes = inj.get("notes", "")

            if ret:
                if isinstance(ret, str):
                    ret_date = datetime.strptime(ret, "%Y-%m-%d").date()
                else:
                    ret_date = ret
                days_left = (ret_date - today).days
                days_str = f"{days_left:>6}d"
                if days_left <= 7:
                    returning_soon.append((name, days_left, replacement))
            else:
                days_str = "      ?"

            repl_str = replacement or "(none)"
            print(f"  {name:<25} {str(il_date):<12} {str(ret or '?'):<12} {days_str}  {repl_str}")
            if notes:
                print(f"  {'':>25} {notes}")
            if not replacement:
                needs_replacement.append(inj)

        if returning_soon:
            print("\n  RETURNING SOON:")
            for name, days, repl in returning_soon:
                if days <= 0:
                    print(f"    {name} — eligible to return NOW")
                    if repl:
                        print(f"      -> Consider dropping {repl} to activate")
                else:
                    print(f"    {name} — returning in ~{days} days")
                    if repl:
                        print(f"      -> Plan to drop {repl} to activate")

        if needs_replacement:
            print("\n  NEEDS REPLACEMENT:")
            for inj in needs_replacement:
                name = inj["name"]
                # Find the injured player's type to scope waiver search
                player_entry = next((p for p in user_roster if p["name"] == name), None)
                if player_entry:
                    ptype = player_entry["player_type"]
                    print(f"    {name} ({ptype}) — no replacement picked up yet")
                else:
                    print(f"    {name} — no replacement picked up yet")
    else:
        print("\nNo injuries tracked. Edit data/injuries.yaml to add IL players.")

    # ── 5. WAIVER WIRE ────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("WAIVER WIRE")
    print("=" * 90)

    free_agents = []
    for pos in ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]:
        try:
            fa = fetch_free_agents(league, pos, count=25)
            free_agents.extend(fa)
        except Exception:
            pass

    # Deduplicate by name
    seen = set()
    unique_fa = []
    for fa in free_agents:
        if fa["name"] not in seen:
            seen.add(fa["name"])
            unique_fa.append(fa)

    fa_matched = match_roster_to_projections(unique_fa, hitters_proj, pitchers_proj)

    waiver_recs = scan_waivers(user_roster, fa_matched, leverage, max_results=5)

    if waiver_recs:
        print(f"\nTop {len(waiver_recs)} waiver moves:")
        for i, rec in enumerate(waiver_recs, 1):
            print(f"\n  {i}. ADD: {rec['add']:<25} DROP: {rec['drop']}")
            print(f"     wSGP gain: {rec['sgp_gain']:+.2f}")
            if rec.get("categories"):
                parts = [f"{c}: {v:+.2f}" for c, v in rec["categories"].items() if abs(v) > 0.01]
                if parts:
                    print(f"     Categories: {', '.join(parts)}")
    else:
        print("\nNo positive waiver moves found — your roster is solid.")

    # ── 6. TRADE RECOMMENDATIONS ──────────────────────────────────────
    print()
    print("=" * 90)
    print("TRADE RECOMMENDATIONS")
    print("=" * 90)

    opp_rosters = {n: r for n, r in all_rosters.items() if n != team_name}

    if standings:
        leverage_by_team = {}
        for team in standings:
            leverage_by_team[team["name"]] = calculate_leverage(standings, team["name"])

        current_ranks = compute_roto_points_by_cat(standings)

        trades = find_trades(
            hart_name=team_name,
            hart_roster=user_roster,
            opp_rosters=opp_rosters,
            standings=standings,
            leverage_by_team=leverage_by_team,
            roster_slots=config.roster_slots,
            max_results=5,
        )
    else:
        trades = []
        current_ranks = {}
        print("\n  Pre-season — trade analysis requires standings data.")

    if trades:
        for i, trade in enumerate(trades, 1):
            opp = trade["opponent"]
            send_pos = "/".join(trade["send_positions"][:2])
            recv_pos = "/".join(trade["receive_positions"][:2])

            print(f"\n  {i}. SEND: {trade['send']:<22} ({send_pos})  ->  {opp}")
            print(f"     GET:  {trade['receive']:<22} ({recv_pos})  <-  {opp}")

            hart_parts = [f"{d:+d} {c}" for c, d in trade["hart_cat_deltas"].items() if d != 0]
            opp_parts = [f"{d:+d} {c}" for c, d in trade["opp_cat_deltas"].items() if d != 0]
            print(f"     You gain: {trade['hart_delta']:+d} roto pts ({', '.join(hart_parts) if hart_parts else 'no change'})")
            print(f"     They gain: {trade['opp_delta']:+d} roto pts ({', '.join(opp_parts) if opp_parts else 'no change'})")

            opp_ranks = current_ranks.get(opp, {})
            pitch = generate_pitch(opp, trade["opp_cat_deltas"], opp_ranks)
            print(f"     Pitch: \"{pitch}\"")
    else:
        print("\nNo mutually beneficial trades found.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
