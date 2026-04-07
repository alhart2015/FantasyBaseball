"""Backtest trade recommender using 2025 season data.

At each trade deadline checkpoint, generate trade recommendations using
mid-season standings computed from game logs, then compare projected
roto impact vs actual impact using real end-of-season data.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest_2025 import DRAFT_2025, ACTUAL, ALL_CATS, INVERSE
from fantasy_baseball.sgp.rankings import compute_sgp_rankings
from fantasy_baseball.trades.evaluate import (
    find_trades, compute_roto_points,
    compute_trade_impact, _player_ros_stats,
)
from fantasy_baseball.trades.pitch import generate_pitch
from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.utils.name_utils import normalize_name

GAME_LOG_CACHE = PROJECT_ROOT / "data" / "stats" / "game_logs_2025.json"
PROJ_DIR = PROJECT_ROOT / "data" / "projections"

CHECKPOINTS = ["2025-06-15", "2025-07-15", "2025-08-15"]
ROSTER_SLOTS = {
    "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1,
    "OF": 4, "UTIL": 2, "P": 9, "BN": 2, "IL": 2,
}
HART_TEAM = "Hart of the Order"


def load_game_logs() -> dict[int, dict]:
    """Load cached game logs keyed by MLBAMID."""
    with open(GAME_LOG_CACHE) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_preseason_projections() -> dict[int, dict]:
    """Load blended Steamer+ZiPS 2025 projections keyed by MLBAMID."""
    projections = {}

    for system in ["steamer", "zips"]:
        for ptype, filename in [("hitter", f"{system}-hitters-2025.csv"),
                                ("pitcher", f"{system}-pitchers-2025.csv")]:
            path = PROJ_DIR / filename
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df.dropna(subset=["MLBAMID"])
            for _, row in df.iterrows():
                mid = int(row["MLBAMID"])
                if mid not in projections:
                    projections[mid] = {"_systems": [], "type": ptype, "name": row["Name"]}
                entry = projections[mid]
                if entry["type"] != ptype:
                    continue  # Don't mix hitter/pitcher
                if ptype == "hitter":
                    entry["_systems"].append({
                        "pa": float(row.get("PA", 0) or 0),
                        "ab": float(row.get("AB", 0) or 0),
                        "h": float(row.get("H", 0) or 0),
                        "hr": float(row.get("HR", 0) or 0),
                        "r": float(row.get("R", 0) or 0),
                        "rbi": float(row.get("RBI", 0) or 0),
                        "sb": float(row.get("SB", 0) or 0),
                    })
                else:
                    entry["_systems"].append({
                        "ip": float(row.get("IP", 0) or 0),
                        "w": float(row.get("W", 0) or 0),
                        "k": float(row.get("SO", 0) or 0),
                        "sv": float(row.get("SV", 0) or 0),
                        "er": float(row.get("ER", 0) or 0),
                        "bb": float(row.get("BB", 0) or 0),
                        "h_allowed": float(row.get("H", 0) or 0),
                        "gs": float(row.get("GS", 0) or 0),
                        "g": float(row.get("G", 0) or 0),
                    })

    # Average across systems
    for mid, entry in projections.items():
        systems = entry.pop("_systems", [])
        if not systems:
            continue
        n = len(systems)
        if entry["type"] == "hitter":
            for col in ["pa", "ab", "h", "hr", "r", "rbi", "sb"]:
                entry[col] = sum(s.get(col, 0) for s in systems) / n
            if entry["ab"] > 0:
                entry["avg"] = entry["h"] / entry["ab"]
            else:
                entry["avg"] = 0
        else:
            for col in ["ip", "w", "k", "sv", "er", "bb", "h_allowed", "gs", "g"]:
                entry[col] = sum(s.get(col, 0) for s in systems) / n
            if entry["ip"] > 0:
                entry["era"] = entry["er"] * 9 / entry["ip"]
                entry["whip"] = (entry["bb"] + entry["h_allowed"]) / entry["ip"]
            else:
                entry["era"] = 0
                entry["whip"] = 0

    return projections


def build_team_rosters(projections, game_logs):
    """Build team rosters from DRAFT_2025, enriched with projections and game log IDs."""
    # Build name -> MLBAMID lookup from projections
    name_to_mid = {}
    for mid, entry in projections.items():
        name_to_mid[normalize_name(entry["name"])] = mid

    team_rosters = {}  # team_name -> [{name, mlbam_id, type, positions, ...stats}]
    for _, player_name, team in DRAFT_2025:
        if team not in team_rosters:
            team_rosters[team] = []

        name_norm = normalize_name(player_name)
        mid = name_to_mid.get(name_norm)
        if mid is None:
            clean = name_norm.replace(" jr.", "").replace(" jr", "").replace(" ii", "").strip()
            for nk, nid in name_to_mid.items():
                nclean = nk.replace(" jr.", "").replace(" jr", "").replace(" ii", "").strip()
                if clean == nclean and len(clean) > 4:
                    mid = nid
                    break

        if mid is None:
            continue

        proj = projections[mid]
        ptype = proj["type"]

        # Build positions list (approximate from type)
        if ptype == "hitter":
            positions = ["OF", "UTIL"]  # simplified
        else:
            positions = ["SP", "P"] if proj.get("gs", 0) > 5 else ["RP", "P"]

        entry = {
            "name": player_name,
            "mlbam_id": mid,
            "player_type": ptype,
            "positions": positions,
        }
        if ptype == "hitter":
            for col in ["r", "hr", "rbi", "sb", "avg", "h", "ab", "pa"]:
                entry[col] = proj.get(col, 0)
        else:
            for col in ["w", "k", "sv", "era", "whip", "ip", "er", "bb", "h_allowed"]:
                entry[col] = proj.get(col, 0)

        team_rosters[team].append(entry)

    return team_rosters


def compute_team_stats_at_date(team_roster, game_logs, end_date):
    """Compute cumulative team stats through end_date from game logs."""
    total_r, total_hr, total_rbi, total_sb = 0, 0, 0, 0
    total_h, total_ab = 0, 0
    total_w, total_k, total_sv = 0, 0, 0
    total_ip, total_er, total_bb, total_ha = 0, 0, 0, 0

    for player in team_roster:
        mid = player["mlbam_id"]
        log_entry = game_logs.get(mid, {})
        games = log_entry.get("games", [])
        before = [g for g in games if g["date"] < end_date]

        if player["player_type"] == "hitter":
            for g in before:
                total_r += g.get("r", 0)
                total_hr += g.get("hr", 0)
                total_rbi += g.get("rbi", 0)
                total_sb += g.get("sb", 0)
                total_h += g.get("h", 0)
                total_ab += g.get("ab", 0)
        else:
            for g in before:
                total_w += g.get("w", 0)
                total_k += g.get("k", 0)
                total_sv += g.get("sv", 0)
                total_ip += g.get("ip", 0)
                total_er += g.get("er", 0)
                total_bb += g.get("bb", 0)
                total_ha += g.get("h_allowed", 0)

    avg = total_h / total_ab if total_ab > 0 else 0
    era = total_er * 9 / total_ip if total_ip > 0 else 0
    whip = (total_bb + total_ha) / total_ip if total_ip > 0 else 0

    return {
        "R": total_r, "HR": total_hr, "RBI": total_rbi, "SB": total_sb,
        "AVG": avg, "W": total_w, "K": total_k, "SV": total_sv,
        "ERA": era, "WHIP": whip,
    }


def compute_player_ros_stats(player, game_logs, start_date):
    """Compute a player's actual ROS stats from start_date through end of season."""
    mid = player["mlbam_id"]
    log_entry = game_logs.get(mid, {})
    games = log_entry.get("games", [])
    after = [g for g in games if g["date"] >= start_date]

    if player["player_type"] == "hitter":
        ab = sum(g.get("ab", 0) for g in after)
        h = sum(g.get("h", 0) for g in after)
        return {
            "R": sum(g.get("r", 0) for g in after),
            "HR": sum(g.get("hr", 0) for g in after),
            "RBI": sum(g.get("rbi", 0) for g in after),
            "SB": sum(g.get("sb", 0) for g in after),
            "AVG": h / ab if ab > 0 else 0,
            "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0,
            "ab": ab, "ip": 0,
        }
    else:
        ip = sum(g.get("ip", 0) for g in after)
        er = sum(g.get("er", 0) for g in after)
        bb = sum(g.get("bb", 0) for g in after)
        ha = sum(g.get("h_allowed", 0) for g in after)
        gs = sum(g.get("gs", 0) for g in after)
        g_count = sum(g.get("g", 1) for g in after)
        return {
            "R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
            "W": sum(g.get("w", 0) for g in after),
            "K": sum(g.get("k", 0) for g in after),
            "SV": sum(g.get("sv", 0) for g in after),
            "ERA": er * 9 / ip if ip > 0 else 0,
            "WHIP": (bb + ha) / ip if ip > 0 else 0,
            "ab": 0, "ip": ip,
        }


def compute_player_projected_ros(player, checkpoint):
    """Scale preseason projection to ROS from checkpoint.

    Rough scaling: fraction of season remaining * full-season projection.
    Returns dict with lowercase keys matching trade finder expectations.
    """
    season_start = "2025-03-27"
    season_end = "2025-09-28"
    total_days = (datetime.strptime(season_end, "%Y-%m-%d") - datetime.strptime(season_start, "%Y-%m-%d")).days
    remaining_days = (datetime.strptime(season_end, "%Y-%m-%d") - datetime.strptime(checkpoint, "%Y-%m-%d")).days
    frac = max(0, remaining_days / total_days)

    if player["player_type"] == "hitter":
        ab = player.get("ab", 0) * frac
        return {
            "r": player.get("r", 0) * frac,
            "hr": player.get("hr", 0) * frac,
            "rbi": player.get("rbi", 0) * frac,
            "sb": player.get("sb", 0) * frac,
            "avg": player.get("avg", 0),
            "ab": ab,
            "h": player.get("avg", 0) * ab,
            "pa": ab * 1.1,
        }
    else:
        ip = player.get("ip", 0) * frac
        return {
            "w": player.get("w", 0) * frac,
            "k": player.get("k", 0) * frac,
            "sv": player.get("sv", 0) * frac,
            "era": player.get("era", 0),
            "whip": player.get("whip", 0),
            "ip": ip,
            "er": player.get("era", 0) * ip / 9 if ip > 0 else 0,
            "bb": player.get("whip", 0) * ip * 0.4 if ip > 0 else 0,
            "h_allowed": player.get("whip", 0) * ip * 0.6 if ip > 0 else 0,
        }


def find_player_in_roster(name, rosters):
    """Find a player dict by name across all rosters."""
    name_norm = normalize_name(name)
    for team, players in rosters.items():
        for p in players:
            if normalize_name(p["name"]) == name_norm:
                return p
    return None


def main():
    print("=" * 80)
    print("TRADE RECOMMENDER BACKTEST — 2025 Season")
    print("=" * 80)

    print("\nLoading data...")
    game_logs = load_game_logs()
    projections = load_preseason_projections()
    team_rosters = build_team_rosters(projections, game_logs)

    team_names = sorted(team_rosters.keys())
    print(f"Teams: {len(team_names)}")
    for tn in team_names:
        print(f"  {tn}: {len(team_rosters[tn])} players")

    all_trade_results = []

    for checkpoint in CHECKPOINTS:
        print(f"\n{'=' * 80}")
        print(f"CHECKPOINT: {checkpoint}")
        print(f"{'=' * 80}")

        # Build mid-season standings from game logs
        standings = []
        for team_name in team_names:
            stats = compute_team_stats_at_date(team_rosters[team_name], game_logs, checkpoint)
            standings.append({"name": team_name, "stats": stats})

        # Rank for display
        points = compute_roto_points(standings)
        standings_sorted = sorted(standings, key=lambda t: points[t["name"]], reverse=True)

        print(f"\nStandings at {checkpoint}:")
        for i, team in enumerate(standings_sorted, 1):
            pts = points[team["name"]]
            marker = " <<<" if team["name"] == HART_TEAM else ""
            print(f"  {i:>2}. {team['name']:<32} {pts} pts{marker}")

        # Compute leverage for all teams
        leverage_by_team = {}
        for team in standings:
            leverage_by_team[team["name"]] = calculate_leverage(standings, team["name"])

        # Build roster dicts for trade finder using availability-adjusted
        # projections: if a player has minimal games through the checkpoint,
        # scale down their ROS projection (simulates recency blend for injured
        # players). This prevents proposing trades for injured players.
        def build_adjusted_roster(roster):
            adjusted = []
            for p in roster:
                entry = dict(p)
                ros = compute_player_projected_ros(p, checkpoint)

                # Check actual games played through checkpoint
                mid = p["mlbam_id"]
                log_entry = game_logs.get(mid, {})
                games_before = [g for g in log_entry.get("games", []) if g["date"] < checkpoint]

                if p["player_type"] == "hitter":
                    actual_pa = sum(g.get("pa", 0) for g in games_before)
                    # If less than 50 PA by this point, they're likely injured
                    # Scale ROS projection by availability fraction
                    season_start = "2025-03-27"
                    days_elapsed = (datetime.strptime(checkpoint, "%Y-%m-%d") - datetime.strptime(season_start, "%Y-%m-%d")).days
                    expected_pa = p.get("pa", 500) * days_elapsed / 185  # rough pro-rate
                    avail_frac = min(1.0, actual_pa / expected_pa) if expected_pa > 0 else 0
                    for col in ["r", "hr", "rbi", "sb", "h", "ab", "pa"]:
                        ros[col] = ros.get(col, 0) * avail_frac
                    if avail_frac < 0.3:
                        ros["avg"] = ros.get("avg", 0) * avail_frac
                else:
                    actual_ip = sum(g.get("ip", 0) for g in games_before)
                    season_start = "2025-03-27"
                    days_elapsed = (datetime.strptime(checkpoint, "%Y-%m-%d") - datetime.strptime(season_start, "%Y-%m-%d")).days
                    expected_ip = p.get("ip", 150) * days_elapsed / 185
                    avail_frac = min(1.0, actual_ip / expected_ip) if expected_ip > 0 else 0
                    for col in ["w", "k", "sv", "ip"]:
                        ros[col] = ros.get(col, 0) * avail_frac
                    if avail_frac < 0.3:
                        ros["era"] = ros.get("era", 0) + (1 - avail_frac) * 5  # degrade toward bad
                        ros["whip"] = ros.get("whip", 0) + (1 - avail_frac) * 0.5
                    ros["er"] = ros.get("era", 0) * ros.get("ip", 0) / 9 if ros.get("ip", 0) > 0 else 0
                    ros["bb"] = ros.get("whip", 0) * ros.get("ip", 0) * 0.4 if ros.get("ip", 0) > 0 else 0
                    ros["h_allowed"] = ros.get("whip", 0) * ros.get("ip", 0) * 0.6 if ros.get("ip", 0) > 0 else 0

                entry.update(ros)
                adjusted.append(entry)
            return adjusted

        hart_roster_for_trades = build_adjusted_roster(team_rosters[HART_TEAM])

        opp_rosters_for_trades = {}
        for team_name in team_names:
            if team_name == HART_TEAM:
                continue
            opp_rosters_for_trades[team_name] = build_adjusted_roster(team_rosters[team_name])

        # Build rankings from projections for perception-based filtering
        hitter_rows = []
        pitcher_rows = []
        for mid, proj in projections.items():
            row = {"name": proj["name"], "fg_id": str(mid)}
            if proj["type"] == "hitter":
                row["player_type"] = "hitter"
                for col in ["r", "hr", "rbi", "sb", "avg", "ab", "pa", "h"]:
                    row[col] = proj.get(col, 0)
                hitter_rows.append(row)
            else:
                row["player_type"] = "pitcher"
                for col in ["w", "k", "sv", "era", "whip", "ip"]:
                    row[col] = proj.get(col, 0)
                pitcher_rows.append(row)
        rankings = compute_sgp_rankings(
            pd.DataFrame(hitter_rows) if hitter_rows else pd.DataFrame(),
            pd.DataFrame(pitcher_rows) if pitcher_rows else pd.DataFrame(),
        )

        # Find trades
        trades = find_trades(
            hart_name=HART_TEAM,
            hart_roster=hart_roster_for_trades,
            opp_rosters=opp_rosters_for_trades,
            standings=standings,
            leverage_by_team=leverage_by_team,
            roster_slots=ROSTER_SLOTS,
            rankings=rankings,
            max_results=5,
        )

        if not trades:
            print("\n  No mutually beneficial trades found at this checkpoint.")
            continue

        print(f"\n  TOP {len(trades)} TRADE PROPOSALS:")

        for i, trade in enumerate(trades, 1):
            opp = trade["opponent"]
            send_name = trade["send"]
            recv_name = trade["receive"]

            # Projected impact (from the recommender)
            proj_hart_delta = trade["hart_delta"]
            proj_opp_delta = trade["opp_delta"]
            proj_hart_cats = trade["hart_cat_deltas"]
            proj_opp_cats = trade["opp_cat_deltas"]

            # Generate pitch
            pitch = generate_pitch(
                send_rank=trade.get("send_rank", 0),
                receive_rank=trade.get("receive_rank", 0),
                send_positions=trade.get("send_positions", []),
                receive_positions=trade.get("receive_positions", []),
            )

            # Compute ACTUAL ROS impact
            send_player = find_player_in_roster(send_name, team_rosters)
            recv_player = find_player_in_roster(recv_name, team_rosters)

            actual_hart_delta = "?"
            actual_opp_delta = "?"
            actual_hart_cats = {}
            actual_opp_cats = {}

            if send_player and recv_player:
                # Actual ROS stats from game logs
                actual_send_ros = compute_player_ros_stats(send_player, game_logs, checkpoint)
                actual_recv_ros = compute_player_ros_stats(recv_player, game_logs, checkpoint)

                actual_impact = compute_trade_impact(
                    standings=standings,
                    hart_name=HART_TEAM,
                    opp_name=opp,
                    hart_loses_ros=actual_send_ros,
                    hart_gains_ros=actual_recv_ros,
                    opp_loses_ros=actual_recv_ros,
                    opp_gains_ros=actual_send_ros,
                )
                actual_hart_delta = actual_impact["hart_delta"]
                actual_opp_delta = actual_impact["opp_delta"]
                actual_hart_cats = actual_impact["hart_cat_deltas"]
                actual_opp_cats = actual_impact["opp_cat_deltas"]

            # Display
            print(f"\n  {i}. SEND: {send_name:<22} ->  {opp}")
            print(f"     GET:  {recv_name:<22} <-  {opp}")

            # Projected
            proj_parts = [f"{d:+d} {c}" for c, d in proj_hart_cats.items() if d != 0]
            print(f"\n     PROJECTED Hart: {proj_hart_delta:+d} roto pts ({', '.join(proj_parts) if proj_parts else 'no change'})")
            proj_opp_parts = [f"{d:+d} {c}" for c, d in proj_opp_cats.items() if d != 0]
            print(f"     PROJECTED them: {proj_opp_delta:+d} roto pts ({', '.join(proj_opp_parts) if proj_opp_parts else 'no change'})")

            # Actual
            if isinstance(actual_hart_delta, int):
                act_parts = [f"{d:+d} {c}" for c, d in actual_hart_cats.items() if d != 0]
                print(f"     ACTUAL Hart:    {actual_hart_delta:+d} roto pts ({', '.join(act_parts) if act_parts else 'no change'})")
                act_opp_parts = [f"{d:+d} {c}" for c, d in actual_opp_cats.items() if d != 0]
                print(f"     ACTUAL them:    {actual_opp_delta:+d} roto pts ({', '.join(act_opp_parts) if act_opp_parts else 'no change'})")

                accuracy = "GOOD" if (proj_hart_delta > 0 and actual_hart_delta > 0) else \
                           "BAD" if (proj_hart_delta > 0 and actual_hart_delta < 0) else \
                           "NEUTRAL"
                print(f"     Verdict: {accuracy} call — projected {proj_hart_delta:+d}, actual {actual_hart_delta:+d}")
            else:
                print(f"     ACTUAL: could not compute (player not found in game logs)")

            print(f"\n     Pitch: \"{pitch}\"")

            all_trade_results.append({
                "checkpoint": checkpoint,
                "rank": i,
                "send": send_name,
                "receive": recv_name,
                "opponent": opp,
                "proj_hart_delta": proj_hart_delta,
                "actual_hart_delta": actual_hart_delta if isinstance(actual_hart_delta, int) else None,
                "proj_opp_delta": proj_opp_delta,
                "actual_opp_delta": actual_opp_delta if isinstance(actual_opp_delta, int) else None,
            })

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    good = sum(1 for t in all_trade_results
               if t["actual_hart_delta"] is not None and t["proj_hart_delta"] > 0 and t["actual_hart_delta"] > 0)
    bad = sum(1 for t in all_trade_results
              if t["actual_hart_delta"] is not None and t["proj_hart_delta"] > 0 and t["actual_hart_delta"] < 0)
    neutral = sum(1 for t in all_trade_results
                  if t["actual_hart_delta"] is not None and t["actual_hart_delta"] == 0)
    total = good + bad + neutral

    print(f"\n  Total trade proposals: {len(all_trade_results)}")
    print(f"  Evaluable: {total}")
    print(f"  Good calls (projected +, actual +): {good}")
    print(f"  Bad calls (projected +, actual -): {bad}")
    print(f"  Neutral (actual 0): {neutral}")
    if total > 0:
        print(f"  Accuracy: {good / total:.0%}")

    # Best trade at each checkpoint
    print(f"\n  Best actual trade at each checkpoint:")
    for cp in CHECKPOINTS:
        cp_trades = [t for t in all_trade_results if t["checkpoint"] == cp and t["actual_hart_delta"] is not None]
        if cp_trades:
            best = max(cp_trades, key=lambda t: t["actual_hart_delta"])
            print(f"    {cp}: send {best['send']}, get {best['receive']} from {best['opponent']} "
                  f"(projected {best['proj_hart_delta']:+d}, actual {best['actual_hart_delta']:+d})")


if __name__ == "__main__":
    main()
