"""Monte Carlo season projection engine.

Provides reusable simulation functions for projecting roto standings
from partial or complete rosters. Used by both the live draft predictor
and the simulate_draft.py --monte-carlo mode.
"""

from typing import Any

import numpy as np
import pandas as pd

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.scoring import score_roto_dict
from fantasy_baseball.simulation import (
    HITTER_COV,
    HITTER_IDX,
    PITCHER_COV,
    PITCHER_IDX,
)
from fantasy_baseball.utils.constants import (
    CLOSER_SV_THRESHOLD,
    HITTING_COUNTING,
    INJURY_PROB,
    INJURY_SEVERITY,
    PITCHING_COUNTING,
    REPLACEMENT_HITTER,
    REPLACEMENT_RP,
    REPLACEMENT_SP,
)
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip


def pad_roster_to_full(
    players: list,
    roster_slots: dict[str, int],
) -> list:
    """Pad a partial roster with replacement-level players to fill all slots.

    Counts current hitters and pitchers, then adds replacement-level
    entries for any unfilled active slots.
    """
    n_hitters = sum(1 for p in players if p.get("player_type") == PlayerType.HITTER)
    n_pitchers = sum(1 for p in players if p.get("player_type") == PlayerType.PITCHER)

    # Target hitter slots: C + 1B + 2B + 3B + SS + IF + OF + UTIL
    hitter_slots = sum(v for k, v in roster_slots.items() if k not in ("P", "BN", "IL"))
    pitcher_slots = roster_slots.get("P", 9)

    padded = list(players)

    # Add replacement hitters
    for i in range(max(0, hitter_slots - n_hitters)):
        repl: dict[str, Any] = dict(REPLACEMENT_HITTER)
        repl["player_type"] = PlayerType.HITTER
        repl["name"] = f"Repl Hitter {i + 1}"
        repl["positions"] = ["OF"]
        padded.append(repl)

    # Add replacement pitchers (SPs)
    for i in range(max(0, pitcher_slots - n_pitchers)):
        repl = dict(REPLACEMENT_SP)
        repl["player_type"] = PlayerType.PITCHER
        repl["name"] = f"Repl SP {i + 1}"
        repl["positions"] = ["SP"]
        padded.append(repl)

    return padded


def simulate_season(team_players, rng, h_slots=None, p_slots=None):
    """Run one simulated season with injuries and stat variance.

    Only counts stats from active-roster players (top h_slots hitters,
    top p_slots pitchers by value). Bench players are excluded.
    """
    team_stats = {}

    for team_num, players in team_players.items():
        hitters = [p for p in players if p.get("player_type") == PlayerType.HITTER]
        pitchers = [p for p in players if p.get("player_type") == PlayerType.PITCHER]

        # Hitters — perf affects quality stats (H, R, HR, RBI, SB) but
        # not volume stats (AB, PA) so rate stats like AVG actually vary.
        h_mean = np.zeros(len(HITTER_IDX))
        adj_hitters = []
        for h in hitters:
            frac_missed = 0.0
            if rng.random() < INJURY_PROB["hitter"]:
                lo, hi = INJURY_SEVERITY["hitter"]
                frac_missed = rng.uniform(lo, hi)

            row = {}
            scale = 1.0 - frac_missed
            draws = rng.multivariate_normal(h_mean, HITTER_COV)
            for col in HITTING_COUNTING:
                base = h.get(col, 0)
                repl_val = REPLACEMENT_HITTER.get(col, 0) * frac_missed
                if col in HITTER_IDX:
                    perf = max(0, 1.0 + draws[HITTER_IDX[col]])
                    row[col] = base * perf * scale + repl_val
                else:
                    row[col] = base * scale + repl_val
            adj_hitters.append(row)

        p_mean = np.zeros(len(PITCHER_IDX))
        adj_pitchers = []
        for p in pitchers:
            frac_missed = 0.0
            if rng.random() < INJURY_PROB["pitcher"]:
                lo, hi = INJURY_SEVERITY["pitcher"]
                frac_missed = rng.uniform(lo, hi)

            is_closer = p.get("sv", 0) >= CLOSER_SV_THRESHOLD
            repl = REPLACEMENT_RP if is_closer else REPLACEMENT_SP

            row = {}
            scale = 1.0 - frac_missed
            draws = rng.multivariate_normal(p_mean, PITCHER_COV)
            for col in PITCHING_COUNTING:
                base = p.get(col, 0)
                repl_val = repl.get(col, 0) * frac_missed
                if col in PITCHER_IDX:
                    perf = max(0, 1.0 + draws[PITCHER_IDX[col]])
                    row[col] = base * perf * scale + repl_val
                else:
                    row[col] = base * scale + repl_val
            adj_pitchers.append(row)

        # Select active roster only (bench players don't contribute stats)
        if h_slots is not None:
            adj_hitters.sort(
                key=lambda h: h["r"] + h["hr"] + h["rbi"] + h["sb"],
                reverse=True,
            )
            adj_hitters = adj_hitters[:h_slots]
        if p_slots is not None:
            adj_pitchers.sort(
                key=lambda p: (p.get("sv", 0) >= CLOSER_SV_THRESHOLD, p["w"] + p["k"] + p["sv"]),
                reverse=True,
            )
            adj_pitchers = adj_pitchers[:p_slots]

        # Aggregate
        r = sum(h["r"] for h in adj_hitters)
        hr = sum(h["hr"] for h in adj_hitters)
        rbi = sum(h["rbi"] for h in adj_hitters)
        sb = sum(h["sb"] for h in adj_hitters)
        total_h = sum(h["h"] for h in adj_hitters)
        total_ab = sum(h["ab"] for h in adj_hitters)
        avg = calculate_avg(total_h, total_ab)

        w = sum(p["w"] for p in adj_pitchers)
        k = sum(p["k"] for p in adj_pitchers)
        sv = sum(p["sv"] for p in adj_pitchers)
        total_ip = sum(p["ip"] for p in adj_pitchers)
        total_er = sum(p["er"] for p in adj_pitchers)
        total_bb = sum(p["bb"] for p in adj_pitchers)
        total_ha = sum(p["h_allowed"] for p in adj_pitchers)
        era = calculate_era(total_er, total_ip)
        whip = calculate_whip(total_bb, total_ha, total_ip)

        team_stats[team_num] = {
            "R": r,
            "HR": hr,
            "RBI": rbi,
            "SB": sb,
            "AVG": avg,
            "W": w,
            "K": k,
            "SV": sv,
            "ERA": era,
            "WHIP": whip,
        }

    return team_stats


def run_projections(
    team_players: dict[int, list],
    roster_slots: dict[str, int],
    board: pd.DataFrame,  # noqa: ARG001  (unused; kept for API)
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

    # Compute active roster slot counts
    h_slots = sum(v for k, v in roster_slots.items() if k not in ("P", "BN", "IL"))
    p_slots = roster_slots.get("P", 9)

    # Pad all rosters
    padded = {}
    for tn, players in team_players.items():
        padded[tn] = pad_roster_to_full(players, roster_slots)

    # Run simulations
    all_totals: dict[int, list[float]] = {tn: [] for tn in padded}
    all_finishes: dict[int, list[int]] = {tn: [] for tn in padded}

    for _ in range(iterations):
        stats = simulate_season(padded, rng, h_slots=h_slots, p_slots=p_slots)
        # score_roto_dict wants str keys (they flow into team_name: str on
        # ProjectedStandingsEntry); convert here and back since draft teams
        # are numbered internally.
        stats_by_name = {str(tn): s for tn, s in stats.items()}
        roto = score_roto_dict(stats_by_name)
        for tn in padded:
            total = roto[str(tn)]["total"]
            all_totals[tn].append(total)
            rank = 1 + sum(1 for otn in padded if roto[str(otn)]["total"] > total)
            all_finishes[tn].append(rank)

    # Build results
    standings = []
    for tn in sorted(padded.keys()):
        totals = np.array(all_totals[tn])
        finishes = np.array(all_finishes[tn])
        standings.append(
            {
                "team_num": tn,
                "median": int(np.median(totals)),
                "p10": int(np.percentile(totals, 10)),
                "p90": int(np.percentile(totals, 90)),
                "win_pct": round(float(np.mean(finishes == 1) * 100), 1),
                "top3_pct": round(float(np.mean(finishes <= 3) * 100), 1),
                "bot3_pct": round(float(np.mean(finishes >= num_teams - 2) * 100), 1),
            }
        )

    standings.sort(key=lambda s: s["median"], reverse=True)
    return {"standings": standings}


def reconstruct_rosters_from_draft(config, board, tracker, num_teams_override=None, keepers=None):
    """Build per-team player lists from in-progress draft tracker.

    *keepers* overrides ``config.keepers`` — pass an empty list for mock
    drafts that don't use keepers.
    """
    num_teams = num_teams_override or config.num_teams
    team_players: dict[int, list[Any]] = {i: [] for i in range(1, num_teams + 1)}

    if keepers is None:
        keepers = config.keepers

    # Keepers
    for keeper in keepers:
        for num, name in config.teams.items():
            if name == keeper["team"]:
                norm = normalize_name(keeper["name"])
                matches = board[board["name_normalized"] == norm]
                if not matches.empty:
                    best = matches.loc[matches["var"].idxmax()]
                    team_players[num].append(best)
                break

    # Draft picks (skip keepers at front of drafted list)
    num_keepers = len(keepers)
    drafted_names = tracker.drafted_players[num_keepers:]
    drafted_ids = tracker.drafted_ids[num_keepers:]

    for pick_num, (_name, pid) in enumerate(zip(drafted_names, drafted_ids, strict=False), 1):
        rnd = (pick_num - 1) // num_teams + 1
        pos = (pick_num - 1) % num_teams + 1
        team = pos if rnd % 2 == 1 else num_teams - pos + 1
        rows = board[board["player_id"] == pid]
        if not rows.empty:
            team_players[team].append(rows.iloc[0])

    return team_players
