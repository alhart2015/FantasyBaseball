"""Monte Carlo robustness analysis of saved draft simulation results.

For each of the 24 saved sim results (12 strategies x 2 scoring modes),
runs 1000 Monte Carlo iterations with random injuries and stat variance,
then reports Hart's team finish distribution and roto point statistics.

Usage:
    python data/run_monte_carlo.py
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (mirrored from src/fantasy_baseball/draft/projections.py)
# ---------------------------------------------------------------------------
INJURY_PROB = {"pitcher": 0.45, "hitter": 0.18}
INJURY_SEVERITY = {"pitcher": (0.20, 0.60), "hitter": (0.15, 0.40)}
STAT_VARIANCE = 0.12

HITTING_COUNTING = ["r", "hr", "rbi", "sb", "h", "ab"]
PITCHING_COUNTING = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]
ALL_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
INVERSE = {"ERA", "WHIP"}

REPLACEMENT_HITTER = {"r": 55, "hr": 12, "rbi": 50, "sb": 5, "h": 125, "ab": 500}
REPLACEMENT_SP = {"w": 7, "k": 120, "sv": 0, "ip": 140, "er": 70, "bb": 50, "h_allowed": 139}
REPLACEMENT_RP = {"w": 2, "k": 55, "sv": 5, "ip": 60, "er": 30, "bb": 21, "h_allowed": 60}

# Roster slot configuration
ROSTER_SLOTS = {
    "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1,
    "OF": 4, "UTIL": 2, "P": 9, "BN": 2, "IL": 2,
}
H_SLOTS = sum(v for k, v in ROSTER_SLOTS.items() if k not in ("P", "BN", "IL"))  # 12
P_SLOTS = ROSTER_SLOTS["P"]  # 9
NUM_TEAMS = 10
ITERATIONS = 1000
HART_TEAM = "Hart of the Order"


def safe_float(val, default=0.0):
    """Convert a value to float, treating None/NaN as default."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def load_sim_json(filepath):
    """Load a simulation result JSON, handling NaN values."""
    with open(filepath, "r") as f:
        content = f.read()
    # Replace NaN with null for valid JSON parsing
    content = content.replace("NaN", "null")
    return json.loads(content)


def parse_rosters(sim_data):
    """Extract team rosters as {team_name: [player_dicts]}.

    Cleans up NaN values in stats to 0.0.
    """
    rosters = {}
    for team_name, players in sim_data["rosters"].items():
        cleaned = []
        for p in players:
            entry = {
                "name": p.get("name", ""),
                "player_type": p.get("player_type", ""),
                "positions": p.get("positions", []),
            }
            for stat in HITTING_COUNTING + PITCHING_COUNTING:
                entry[stat] = safe_float(p.get(stat))
            cleaned.append(entry)
        rosters[team_name] = cleaned
    return rosters


def simulate_season_from_rosters(rosters, rng, injury_draws, variance_draws):
    """Run one simulated season with pre-drawn random values.

    rosters: {team_name: [player_dicts]}
    injury_draws: dict of team_name -> list of (uniform_draw, severity_draw) per player
    variance_draws: dict of team_name -> list of stat_multiplier arrays per player

    Returns team_stats: {team_name: {cat: value}}
    """
    team_stats = {}

    for team_name, players in rosters.items():
        hitters = [p for p in players if p["player_type"] == "hitter"]
        pitchers = [p for p in players if p["player_type"] == "pitcher"]

        i_draws = injury_draws[team_name]
        v_draws = variance_draws[team_name]

        # Process hitters
        adj_hitters = []
        h_idx = 0
        for h in hitters:
            inj_roll, inj_sev = i_draws[h_idx]
            frac_missed = 0.0
            if inj_roll < INJURY_PROB["hitter"]:
                lo, hi = INJURY_SEVERITY["hitter"]
                frac_missed = lo + inj_sev * (hi - lo)

            var_mults = v_draws[h_idx]  # array of len(HITTING_COUNTING)
            row = {}
            scale = 1.0 - frac_missed
            for ci, col in enumerate(HITTING_COUNTING):
                base = h.get(col, 0.0)
                varied = max(0, base * (1.0 + var_mults[ci]))
                row[col] = varied * scale + REPLACEMENT_HITTER.get(col, 0) * frac_missed
            adj_hitters.append(row)
            h_idx += 1

        # Process pitchers
        adj_pitchers = []
        p_idx = h_idx  # Continue from where hitters left off
        for p in pitchers:
            inj_roll, inj_sev = i_draws[p_idx]
            frac_missed = 0.0
            if inj_roll < INJURY_PROB["pitcher"]:
                lo, hi = INJURY_SEVERITY["pitcher"]
                frac_missed = lo + inj_sev * (hi - lo)

            is_closer = p.get("sv", 0.0) >= 15
            repl = REPLACEMENT_RP if is_closer else REPLACEMENT_SP

            var_mults = v_draws[p_idx]  # array of len(PITCHING_COUNTING)
            row = {}
            scale = 1.0 - frac_missed
            for ci, col in enumerate(PITCHING_COUNTING):
                base = p.get(col, 0.0)
                varied = max(0, base * (1.0 + var_mults[ci]))
                row[col] = varied * scale + repl.get(col, 0) * frac_missed
            row["_sv_base"] = p.get("sv", 0.0)
            adj_pitchers.append(row)
            p_idx += 1

        # Select active roster only
        adj_hitters.sort(
            key=lambda h: h["r"] + h["hr"] + h["rbi"] + h["sb"],
            reverse=True,
        )
        adj_hitters = adj_hitters[:H_SLOTS]

        adj_pitchers.sort(
            key=lambda p: (p.get("sv", 0) >= 15, p["w"] + p["k"] + p["sv"]),
            reverse=True,
        )
        adj_pitchers = adj_pitchers[:P_SLOTS]

        # Aggregate
        r = sum(h["r"] for h in adj_hitters)
        hr = sum(h["hr"] for h in adj_hitters)
        rbi = sum(h["rbi"] for h in adj_hitters)
        sb = sum(h["sb"] for h in adj_hitters)
        total_h = sum(h["h"] for h in adj_hitters)
        total_ab = sum(h["ab"] for h in adj_hitters)
        avg = total_h / total_ab if total_ab > 0 else 0

        w = sum(p["w"] for p in adj_pitchers)
        k = sum(p["k"] for p in adj_pitchers)
        sv = sum(p["sv"] for p in adj_pitchers)
        total_ip = sum(p["ip"] for p in adj_pitchers)
        total_er = sum(p["er"] for p in adj_pitchers)
        total_bb = sum(p["bb"] for p in adj_pitchers)
        total_ha = sum(p["h_allowed"] for p in adj_pitchers)
        era = total_er * 9 / total_ip if total_ip > 0 else 99.0
        whip = (total_bb + total_ha) / total_ip if total_ip > 0 else 99.0

        team_stats[team_name] = {
            "R": r, "HR": hr, "RBI": rbi, "SB": sb, "AVG": avg,
            "W": w, "K": k, "SV": sv, "ERA": era, "WHIP": whip,
        }

    return team_stats


def score_roto(team_stats):
    """Compute roto points. Returns {team_name: total_points}."""
    results = {tn: {} for tn in team_stats}
    for cat in ALL_CATS:
        rev = cat not in INVERSE
        ranked = sorted(team_stats.keys(), key=lambda tn: team_stats[tn][cat], reverse=rev)
        for i, tn in enumerate(ranked):
            results[tn][f"{cat}_pts"] = NUM_TEAMS - i

    totals = {}
    for tn in results:
        totals[tn] = sum(results[tn][f"{c}_pts"] for c in ALL_CATS)

    return totals


def pre_draw_randoms(rng, rosters, n_iterations):
    """Pre-draw all random values for all iterations.

    Returns:
        injury_draws: list of n_iterations dicts, each {team_name: [(roll, sev), ...]}
        variance_draws: list of n_iterations dicts, each {team_name: [array_of_mults, ...]}

    Using shared draws across strategies for fairer comparison.
    """
    # Find max players per team position across all rosters
    # We need consistent indexing, so draw for a fixed max roster size
    max_players = max(len(players) for players in rosters.values())

    # For consistency, we'll use a standard team order
    team_names = sorted(rosters.keys())

    all_injury_rolls = {}
    all_injury_sevs = {}
    all_variance = {}

    for tn in team_names:
        n_players = len(rosters[tn])
        # Shape: (n_iterations, n_players)
        all_injury_rolls[tn] = rng.random((n_iterations, n_players))
        all_injury_sevs[tn] = rng.random((n_iterations, n_players))
        # For variance, we need per-stat draws
        # Max stats per player: max(len(HITTING_COUNTING), len(PITCHING_COUNTING)) = 7
        max_stats = max(len(HITTING_COUNTING), len(PITCHING_COUNTING))
        all_variance[tn] = rng.normal(0, STAT_VARIANCE, (n_iterations, n_players, max_stats))

    return team_names, all_injury_rolls, all_injury_sevs, all_variance


def run_monte_carlo_for_file(filepath, team_names_ref, all_injury_rolls, all_injury_sevs,
                              all_variance, n_iterations):
    """Run Monte Carlo for a single sim result file.

    Uses pre-drawn random values for fair cross-strategy comparison.
    """
    sim_data = load_sim_json(filepath)
    rosters = parse_rosters(sim_data)
    metadata = sim_data["metadata"]

    strategy = metadata["strategy"]
    scoring = metadata["scoring_mode"]

    hart_totals = []
    hart_finishes = []

    for it in range(n_iterations):
        # Build per-iteration injury/variance draws from pre-drawn arrays
        injury_draws = {}
        variance_draws = {}
        for tn in team_names_ref:
            if tn not in rosters:
                continue
            n_players = len(rosters[tn])
            draws = []
            for pi in range(n_players):
                draws.append((all_injury_rolls[tn][it, pi], all_injury_sevs[tn][it, pi]))
            injury_draws[tn] = draws

            var_list = []
            for pi in range(n_players):
                var_list.append(all_variance[tn][it, pi])
            variance_draws[tn] = var_list

        team_stats = simulate_season_from_rosters(rosters, None, injury_draws, variance_draws)
        totals = score_roto(team_stats)

        hart_pts = totals.get(HART_TEAM, 0)
        hart_totals.append(hart_pts)
        # Finish position: 1 + number of teams with more points
        finish = 1 + sum(1 for tn, pts in totals.items() if pts > hart_pts)
        hart_finishes.append(finish)

    totals_arr = np.array(hart_totals)
    finishes_arr = np.array(hart_finishes)

    return {
        "strategy": strategy,
        "scoring": scoring,
        "mean_pts": round(float(np.mean(totals_arr)), 2),
        "std_pts": round(float(np.std(totals_arr)), 2),
        "median_pts": float(np.median(totals_arr)),
        "win_rate": round(float(np.mean(finishes_arr == 1) * 100), 1),
        "top3_rate": round(float(np.mean(finishes_arr <= 3) * 100), 1),
        "bot3_rate": round(float(np.mean(finishes_arr >= 8) * 100), 1),
        "floor_p10": float(np.percentile(totals_arr, 10)),
        "ceiling_p90": float(np.percentile(totals_arr, 90)),
        "mean_finish": round(float(np.mean(finishes_arr)), 2),
        "static_pts": metadata["pts"],
        "static_rank": metadata["rank"],
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    sim_dir = project_root / "data" / "sim_results"

    # Find all 24 files from the latest sweep
    prefix = "2026-03-22_082450_"
    sim_files = sorted(sim_dir.glob(f"{prefix}*.json"))
    print(f"Found {len(sim_files)} simulation result files")

    if not sim_files:
        print("ERROR: No simulation files found!")
        sys.exit(1)

    # We need a reference roster structure (team names and sizes) that's
    # consistent across strategies. All strategies have the same 10 teams
    # but possibly different player counts. We'll pre-draw for the max
    # roster size across all files to ensure the random draws are reusable.

    # First pass: figure out team names and max roster sizes
    print("Scanning roster sizes...")
    team_max_players = {}
    all_team_names = None

    for fpath in sim_files:
        sim_data = load_sim_json(fpath)
        rosters = parse_rosters(sim_data)
        if all_team_names is None:
            all_team_names = sorted(rosters.keys())
        for tn, players in rosters.items():
            if tn not in team_max_players:
                team_max_players[tn] = len(players)
            else:
                team_max_players[tn] = max(team_max_players[tn], len(players))

    print(f"Teams: {len(all_team_names)}")
    for tn in all_team_names:
        print(f"  {tn}: up to {team_max_players[tn]} players")

    # Pre-draw random values for ALL iterations
    # We draw for the max roster size per team so draws are reusable
    print(f"\nPre-drawing random values for {ITERATIONS} iterations...")
    rng = np.random.default_rng(42)
    max_stats = max(len(HITTING_COUNTING), len(PITCHING_COUNTING))

    all_injury_rolls = {}
    all_injury_sevs = {}
    all_variance = {}

    for tn in all_team_names:
        n = team_max_players[tn]
        all_injury_rolls[tn] = rng.random((ITERATIONS, n))
        all_injury_sevs[tn] = rng.random((ITERATIONS, n))
        all_variance[tn] = rng.normal(0, STAT_VARIANCE, (ITERATIONS, n, max_stats))

    print("Random draws complete.")

    # Run Monte Carlo for each file
    results = []
    for i, fpath in enumerate(sim_files):
        t0 = time.time()
        label = fpath.stem.replace(prefix, "")
        print(f"\n[{i+1}/{len(sim_files)}] {label}...", end=" ", flush=True)

        result = run_monte_carlo_for_file(
            fpath, all_team_names,
            all_injury_rolls, all_injury_sevs, all_variance,
            ITERATIONS,
        )
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) | mean={result['mean_pts']:.1f} "
              f"win={result['win_rate']:.1f}% top3={result['top3_rate']:.1f}%")
        results.append(result)

    # Sort by mean points descending
    results.sort(key=lambda r: r["mean_pts"], reverse=True)

    # Print summary table
    print("\n" + "=" * 130)
    print("MONTE CARLO RESULTS (1000 iterations, sorted by mean points)")
    print("=" * 130)
    print(f"{'Strategy':<25} {'Score':>5} {'Mean':>6} {'Std':>5} {'Med':>5} "
          f"{'Win%':>5} {'Top3%':>6} {'Bot3%':>6} {'Floor':>6} {'Ceil':>6} "
          f"{'AvgFin':>6} {'Static':>6}")
    print("-" * 130)

    for r in results:
        label = f"{r['strategy']}_{r['scoring']}"
        print(f"{label:<25} {r['static_pts']:>5} {r['mean_pts']:>6.1f} {r['std_pts']:>5.1f} "
              f"{r['median_pts']:>5.0f} {r['win_rate']:>5.1f} {r['top3_rate']:>6.1f} "
              f"{r['bot3_rate']:>6.1f} {r['floor_p10']:>6.0f} {r['ceiling_p90']:>6.0f} "
              f"{r['mean_finish']:>6.2f} {r['static_rank']:>6}")

    # Save to CSV
    csv_path = project_root / "data" / "monte_carlo_results.csv"
    cols = ["strategy", "scoring", "static_pts", "static_rank", "mean_pts", "std_pts",
            "median_pts", "win_rate", "top3_rate", "bot3_rate", "floor_p10",
            "ceiling_p90", "mean_finish"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            vals = [str(r[c]) for c in cols]
            f.write(",".join(vals) + "\n")

    print(f"\nResults saved to {csv_path}")

    # Print top 5 analysis
    print("\n" + "=" * 80)
    print("TOP 5 STRATEGIES BY MEAN ROTO POINTS")
    print("=" * 80)
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. {r['strategy']} ({r['scoring']})")
        print(f"   Mean: {r['mean_pts']:.1f} pts (std: {r['std_pts']:.1f})")
        print(f"   Median: {r['median_pts']:.0f} | Floor: {r['floor_p10']:.0f} | Ceiling: {r['ceiling_p90']:.0f}")
        print(f"   Win: {r['win_rate']:.1f}% | Top 3: {r['top3_rate']:.1f}% | Bottom 3: {r['bot3_rate']:.1f}%")
        print(f"   Avg finish: {r['mean_finish']:.2f} | Static: {r['static_pts']} pts (rank {r['static_rank']})")

    # Print bottom 5
    print("\n" + "=" * 80)
    print("BOTTOM 5 STRATEGIES BY MEAN ROTO POINTS")
    print("=" * 80)
    for i, r in enumerate(results[-5:], len(results) - 4):
        print(f"\n{i}. {r['strategy']} ({r['scoring']})")
        print(f"   Mean: {r['mean_pts']:.1f} pts (std: {r['std_pts']:.1f})")
        print(f"   Median: {r['median_pts']:.0f} | Floor: {r['floor_p10']:.0f} | Ceiling: {r['ceiling_p90']:.0f}")
        print(f"   Win: {r['win_rate']:.1f}% | Top 3: {r['top3_rate']:.1f}% | Bottom 3: {r['bot3_rate']:.1f}%")
        print(f"   Avg finish: {r['mean_finish']:.2f} | Static: {r['static_pts']} pts (rank {r['static_rank']})")

    # VAR vs VONA comparison
    print("\n" + "=" * 80)
    print("VAR vs VONA COMPARISON (same strategy)")
    print("=" * 80)
    by_strategy = {}
    for r in results:
        s = r["strategy"]
        if s not in by_strategy:
            by_strategy[s] = {}
        by_strategy[s][r["scoring"]] = r

    print(f"{'Strategy':<20} {'VAR mean':>8} {'VONA mean':>9} {'Diff':>6} "
          f"{'VAR win%':>8} {'VONA win%':>9}")
    print("-" * 70)
    for s in sorted(by_strategy.keys()):
        var_r = by_strategy[s].get("var", {})
        vona_r = by_strategy[s].get("vona", {})
        var_mean = var_r.get("mean_pts", 0)
        vona_mean = vona_r.get("mean_pts", 0)
        diff = vona_mean - var_mean
        print(f"{s:<20} {var_mean:>8.1f} {vona_mean:>9.1f} {diff:>+6.1f} "
              f"{var_r.get('win_rate', 0):>8.1f} {vona_r.get('win_rate', 0):>9.1f}")


if __name__ == "__main__":
    main()
