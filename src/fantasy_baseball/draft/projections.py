"""Monte Carlo season projection engine.

Provides reusable simulation functions for projecting roto standings
from partial or complete rosters. Used by both the live draft predictor
and the standalone monte_carlo.py script.
"""
import numpy as np
import pandas as pd
from fantasy_baseball.utils.name_utils import normalize_name

# Injury model parameters
INJURY_PROB = {"pitcher": 0.45, "hitter": 0.18}
INJURY_SEVERITY = {"pitcher": (0.20, 0.60), "hitter": (0.15, 0.40)}
STAT_VARIANCE = 0.12

HITTING_COUNTING = ["r", "hr", "rbi", "sb", "h", "ab"]
PITCHING_COUNTING = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]
ALL_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
INVERSE = {"ERA", "WHIP"}

# Replacement-level full-season stats
REPLACEMENT_HITTER = {"r": 55, "hr": 12, "rbi": 50, "sb": 5, "h": 125, "ab": 500}
REPLACEMENT_SP = {"w": 7, "k": 120, "sv": 0, "ip": 140, "er": 70, "bb": 50, "h_allowed": 139}
REPLACEMENT_RP = {"w": 2, "k": 55, "sv": 5, "ip": 60, "er": 30, "bb": 21, "h_allowed": 60}


def pad_roster_to_full(
    players: list,
    roster_slots: dict[str, int],
    board: pd.DataFrame,
) -> list:
    """Pad a partial roster with replacement-level players to fill all slots.

    Counts current hitters and pitchers, then adds replacement-level
    entries for any unfilled active slots.
    """
    n_hitters = sum(1 for p in players if p.get("player_type") == "hitter")
    n_pitchers = sum(1 for p in players if p.get("player_type") == "pitcher")

    # Target hitter slots: C + 1B + 2B + 3B + SS + IF + OF + UTIL
    hitter_slots = sum(
        v for k, v in roster_slots.items()
        if k not in ("P", "BN", "IL")
    )
    pitcher_slots = roster_slots.get("P", 9)

    padded = list(players)

    # Add replacement hitters
    for i in range(max(0, hitter_slots - n_hitters)):
        repl = dict(REPLACEMENT_HITTER)
        repl["player_type"] = "hitter"
        repl["name"] = f"Repl Hitter {i+1}"
        repl["positions"] = ["OF"]
        padded.append(repl)

    # Add replacement pitchers (SPs)
    for i in range(max(0, pitcher_slots - n_pitchers)):
        repl = dict(REPLACEMENT_SP)
        repl["player_type"] = "pitcher"
        repl["name"] = f"Repl SP {i+1}"
        repl["positions"] = ["SP"]
        padded.append(repl)

    return padded


def simulate_season(team_players, rng):
    """Run one simulated season with injuries and stat variance."""
    team_stats = {}

    for team_num, players in team_players.items():
        hitters = [p for p in players if p.get("player_type") == "hitter"]
        pitchers = [p for p in players if p.get("player_type") == "pitcher"]

        # Hitters
        adj_hitters = []
        for h in hitters:
            frac_missed = 0.0
            if rng.random() < INJURY_PROB["hitter"]:
                lo, hi = INJURY_SEVERITY["hitter"]
                frac_missed = rng.uniform(lo, hi)

            row = {}
            scale = 1.0 - frac_missed
            for col in HITTING_COUNTING:
                base = h.get(col, 0)
                varied = max(0, base * (1.0 + rng.normal(0, STAT_VARIANCE)))
                row[col] = varied * scale + REPLACEMENT_HITTER.get(col, 0) * frac_missed
            adj_hitters.append(row)

        # Pitchers
        adj_pitchers = []
        for p in pitchers:
            frac_missed = 0.0
            if rng.random() < INJURY_PROB["pitcher"]:
                lo, hi = INJURY_SEVERITY["pitcher"]
                frac_missed = rng.uniform(lo, hi)

            is_closer = p.get("sv", 0) >= 15
            repl = REPLACEMENT_RP if is_closer else REPLACEMENT_SP

            row = {}
            scale = 1.0 - frac_missed
            for col in PITCHING_COUNTING:
                base = p.get(col, 0)
                varied = max(0, base * (1.0 + rng.normal(0, STAT_VARIANCE)))
                row[col] = varied * scale + repl.get(col, 0) * frac_missed
            adj_pitchers.append(row)

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

        team_stats[team_num] = {
            "R": r, "HR": hr, "RBI": rbi, "SB": sb, "AVG": avg,
            "W": w, "K": k, "SV": sv, "ERA": era, "WHIP": whip,
        }

    return team_stats


def score_roto(team_stats, num_teams):
    """Compute roto points. Returns dict of team_num -> {cat_pts, total}."""
    results = {tn: {} for tn in team_stats}
    for cat in ALL_CATS:
        rev = cat not in INVERSE
        ranked = sorted(team_stats.keys(), key=lambda tn: team_stats[tn][cat], reverse=rev)
        for i, tn in enumerate(ranked):
            results[tn][f"{cat}_pts"] = num_teams - i

    for tn in results:
        results[tn]["total"] = sum(results[tn][f"{c}_pts"] for c in ALL_CATS)

    return results


def run_projections(
    team_players: dict[int, list],
    roster_slots: dict[str, int],
    board: pd.DataFrame,
    num_teams: int,
    iterations: int = 1000,
    seed: int | None = None,
) -> dict:
    """Run Monte Carlo projection on partial rosters.

    Pads each team to full roster with replacement-level players,
    then simulates `iterations` seasons.

    Returns a dict suitable for JSON serialization with standings,
    win rates, and category profiles for each team.
    """
    rng = np.random.default_rng(seed)

    # Pad all rosters
    padded = {}
    for tn, players in team_players.items():
        padded[tn] = pad_roster_to_full(players, roster_slots, board)

    # Run simulations
    all_totals = {tn: [] for tn in padded}
    all_finishes = {tn: [] for tn in padded}

    for _ in range(iterations):
        stats = simulate_season(padded, rng)
        roto = score_roto(stats, num_teams)
        for tn in padded:
            total = roto[tn]["total"]
            all_totals[tn].append(total)
            rank = 1 + sum(1 for otn in padded if roto[otn]["total"] > total)
            all_finishes[tn].append(rank)

    # Build results
    standings = []
    for tn in sorted(padded.keys()):
        totals = np.array(all_totals[tn])
        finishes = np.array(all_finishes[tn])
        standings.append({
            "team_num": tn,
            "median": int(np.median(totals)),
            "p10": int(np.percentile(totals, 10)),
            "p90": int(np.percentile(totals, 90)),
            "win_pct": round(float(np.mean(finishes == 1) * 100), 1),
            "top3_pct": round(float(np.mean(finishes <= 3) * 100), 1),
            "bot3_pct": round(float(np.mean(finishes >= num_teams - 2) * 100), 1),
        })

    standings.sort(key=lambda s: s["median"], reverse=True)
    return {"standings": standings}


def reconstruct_rosters_from_draft(config, board, tracker, num_teams_override=None):
    """Build per-team player lists from in-progress draft tracker."""
    num_teams = num_teams_override or config.num_teams
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

    # Draft picks (skip keepers at front of drafted list)
    num_keepers = len(config.keepers)
    drafted_names = tracker.drafted_players[num_keepers:]
    drafted_ids = tracker.drafted_ids[num_keepers:]

    for pick_num, (name, pid) in enumerate(zip(drafted_names, drafted_ids), 1):
        rnd = (pick_num - 1) // num_teams + 1
        pos = (pick_num - 1) % num_teams + 1
        team = pos if rnd % 2 == 1 else num_teams - pos + 1
        rows = board[board["player_id"] == pid]
        if not rows.empty:
            team_players[team].append(rows.iloc[0])

    return team_players
