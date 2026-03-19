"""Simulate a full draft to evaluate strategy.

Usage:
    python scripts/simulate_draft.py [--strategy default|nonzero_sv|avg_hedge]

- Your team: picks according to the selected strategy (default: 'default').
- Other teams: take the highest-ADP available player they can legally roster.
- Roster limits are enforced for all teams.

Outputs projected roto standings at the end.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.draft.board import build_draft_board, apply_keepers
from fantasy_baseball.draft.tracker import DraftTracker
from fantasy_baseball.draft.balance import CategoryBalance, calculate_draft_leverage
from fantasy_baseball.draft.recommender import (
    get_recommendations,
    get_filled_positions,
)
from fantasy_baseball.draft.strategy import STRATEGIES
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import can_fill_slot

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"


class TeamTrackerProxy:
    """Lightweight proxy that makes strategy functions work for any team.

    Strategy functions access tracker.user_roster_ids, tracker.user_roster,
    tracker.drafted_ids, tracker.current_pick, tracker.current_round, etc.
    This proxy redirects "user" fields to the specified team's roster.
    """

    def __init__(self, real_tracker, team_roster, team_roster_ids):
        self._real = real_tracker
        self.user_roster = team_roster
        self.user_roster_ids = team_roster_ids
        self.drafted_players = real_tracker.drafted_players
        self.drafted_ids = real_tracker.drafted_ids
        self.num_teams = real_tracker.num_teams
        self.rounds = real_tracker.rounds

    @property
    def current_pick(self):
        return self._real.current_pick

    @property
    def current_round(self):
        return self._real.current_round

    @property
    def total_picks(self):
        return self._real.total_picks


def _active_slot_counts(roster_slots):
    """Return (active_hitter_slots, active_pitcher_slots) from config."""
    hitter_slots = sum(
        v for k, v in roster_slots.items() if k not in ("P", "BN", "IL")
    )
    pitcher_slots = roster_slots.get("P", 9)
    return hitter_slots, pitcher_slots


def _select_active_players(hitters, pitchers, roster_slots):
    """Return only the active-roster hitters and pitchers.

    Ranks hitters by (R + HR + RBI + SB) and pitchers by (W + K + SV),
    then takes the top N to fill active slots. Bench players are excluded.
    """
    h_slots, p_slots = _active_slot_counts(roster_slots)

    ranked_h = sorted(
        hitters,
        key=lambda h: h.get("r", 0) + h.get("hr", 0) + h.get("rbi", 0) + h.get("sb", 0),
        reverse=True,
    )
    ranked_p = sorted(
        pitchers,
        key=lambda p: p.get("w", 0) + p.get("k", 0) + p.get("sv", 0),
        reverse=True,
    )
    return ranked_h[:h_slots], ranked_p[:p_slots]


def _can_fill_active_slot(player_positions, filled, roster_slots):
    """Check if a player can fill an active (non-bench/IL) slot."""
    for pos, total in roster_slots.items():
        if pos in ("BN", "IL"):
            continue
        if filled.get(pos, 0) < total and can_fill_slot(player_positions, pos):
            return True
    return False


def _can_roster(player_positions, filled, roster_slots):
    """Check if a player can fit in any open slot (including bench/IL)."""
    for pos, total in roster_slots.items():
        if filled.get(pos, 0) < total and can_fill_slot(player_positions, pos):
            return True
    return False


def _assign_slot(player_positions, filled, roster_slots):
    """Assign a player to the best available slot, updating filled in place."""
    # Try specific slots first, then flex, then bench, then IL
    active = [p for p in roster_slots if p not in ("BN", "IL", "IF", "UTIL")]
    flex = ["IF", "UTIL"]
    overflow = ["BN", "IL"]
    for slot in active + flex + overflow:
        if slot not in roster_slots:
            continue
        if filled.get(slot, 0) < roster_slots[slot] and can_fill_slot(player_positions, slot):
            filled[slot] = filled.get(slot, 0) + 1
            return slot
    return None


def main():
    parser = argparse.ArgumentParser(description="Simulate a fantasy baseball draft")
    parser.add_argument(
        "--strategy", "-s", choices=list(STRATEGIES.keys()),
        default="default",
        help="Draft strategy for your team (default: %(default)s)",
    )
    parser.add_argument(
        "--closer-deadlines", type=str, default=None,
        help="Comma-separated closer deadline rounds for three_closers strategy (e.g. 4,8,12)",
    )
    parser.add_argument(
        "--adp-noise", type=float, default=0.0,
        help="Std dev of noise added to opponent ADP (e.g. 20 = +/- ~20 ADP spots)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for ADP noise",
    )
    parser.add_argument(
        "--opponent-strategies", type=str, default=None,
        help="Assign strategies to opponents: '3:default,5:three_closers' (team_num:strategy)",
    )
    args = parser.parse_args()

    # Apply custom closer deadlines if provided
    if args.closer_deadlines:
        import fantasy_baseball.draft.strategy as strat_mod
        deadlines = [int(r.strip()) for r in args.closer_deadlines.split(",")]
        strat_mod.THREE_CLOSERS_DEADLINES = deadlines

    strategy_fn = STRATEGIES[args.strategy]

    config = load_config(CONFIG_PATH)
    print(f"Simulating draft | {config.team_name} at position {config.draft_position}")
    print(f"Strategy: {args.strategy}")
    print(f"League: {config.num_teams} teams, {sum(config.roster_slots.values())} roster slots")
    print()

    # Build board
    print("Building draft board...")
    full_board = build_draft_board(
        projections_dir=PROJECTIONS_DIR,
        positions_path=POSITIONS_PATH,
        systems=config.projection_systems,
        weights=config.projection_weights or None,
        sgp_overrides=config.sgp_overrides or None,
        roster_slots=config.roster_slots or None,
        num_teams=config.num_teams,
    )
    board = apply_keepers(full_board, config.keepers)

    # Build ADP ranking for other teams (lower ADP = picked earlier)
    # Use the board which has ADP from blending
    adp_board = board.copy()
    if "adp" not in adp_board.columns:
        print("WARNING: No ADP data found. Other teams will use VAR ranking.")
        adp_board["adp"] = range(len(adp_board))

    # Add noise to simulate unpredictable opponents
    if args.adp_noise > 0:
        import numpy as np
        rng = np.random.default_rng(args.seed)
        noise = rng.normal(0, args.adp_noise, size=len(adp_board))
        adp_board = adp_board.copy()
        adp_board["adp"] = adp_board["adp"] + noise

    adp_board = adp_board.sort_values("adp", ascending=True)

    # Initialize tracker
    user_keepers = [k for k in config.keepers if k.get("team") == config.team_name]
    rounds = sum(config.roster_slots.values()) - len(user_keepers)
    tracker = DraftTracker(
        num_teams=config.num_teams,
        user_position=config.draft_position,
        rounds=rounds,
    )
    balance = CategoryBalance()

    # Per-team filled positions tracking
    team_filled = {i: {} for i in range(1, config.num_teams + 1)}

    # Register keepers
    for keeper in config.keepers:
        for num, name in config.teams.items():
            if name == keeper["team"]:
                norm = normalize_name(keeper["name"])
                matches = full_board[full_board["name_normalized"] == norm]
                if not matches.empty:
                    best = matches.loc[matches["var"].idxmax()]
                    is_user = keeper.get("team") == config.team_name
                    if is_user:
                        balance.add_player(best)
                    tracker.draft_player(best["name"], is_user=is_user,
                                         player_id=best["player_id"])
                    _assign_slot(best["positions"], team_filled[num],
                                 config.roster_slots)
                break

    # Parse opponent strategies
    opp_strategies = {}  # team_num -> strategy_fn
    opp_balances = {}    # team_num -> CategoryBalance
    opp_rosters = {}     # team_num -> [player_ids]
    opp_roster_names = {}  # team_num -> [names]
    if args.opponent_strategies:
        for pair in args.opponent_strategies.split(","):
            tn_str, strat_name = pair.strip().split(":")
            tn = int(tn_str)
            if strat_name not in STRATEGIES:
                print(f"WARNING: Unknown strategy '{strat_name}' for team {tn}, using ADP")
                continue
            opp_strategies[tn] = STRATEGIES[strat_name]
            opp_balances[tn] = CategoryBalance()
            opp_rosters[tn] = []
            opp_roster_names[tn] = []

    # Add keeper projections to opponent balance trackers
    for keeper in config.keepers:
        for num, name in config.teams.items():
            if name == keeper["team"] and num in opp_balances:
                norm = normalize_name(keeper["name"])
                matches = full_board[full_board["name_normalized"] == norm]
                if not matches.empty:
                    best = matches.loc[matches["var"].idxmax()]
                    opp_balances[num].add_player(best)
                    opp_rosters[num].append(best["player_id"])
                    opp_roster_names[num].append(best["name"])
            break

    print(f"Draft pool: {len(board)} players")
    print(f"Keepers registered: {len(config.keepers)}")
    if opp_strategies:
        for tn, fn in opp_strategies.items():
            tname = config.teams.get(tn, f"Team {tn}")
            sname = [k for k, v in STRATEGIES.items() if v is fn][0]
            print(f"  {tname}: {sname}")
    print()

    # Run draft
    pick_log = []
    while tracker.current_pick <= tracker.total_picks:
        team_num = tracker.picking_team
        team_label = config.teams.get(team_num, f"Team {team_num}")
        is_user = tracker.is_user_pick

        if is_user:
            # Use the selected strategy
            pick_name, pid = strategy_fn(
                board, full_board, tracker, balance, config, team_filled,
                total_rounds=rounds,
            )
            if pick_name is None:
                # Fallback: pick best available by ADP
                available_ids = set(tracker.drafted_ids)
                for _, row in adp_board.iterrows():
                    if row["player_id"] not in available_ids:
                        pick_name = row["name"]
                        pid = row["player_id"]
                        break

            pick_pos = ""
            if pid:
                rows = board[board["player_id"] == pid]
                if not rows.empty:
                    pick_pos = rows.iloc[0].get("best_position", "")

            if pick_name:
                tracker.draft_player(pick_name, is_user=True, player_id=pid)
                row = board[board["player_id"] == pid]
                if not row.empty:
                    balance.add_player(row.iloc[0])
                    _assign_slot(row.iloc[0]["positions"],
                                 team_filled[team_num], config.roster_slots)
            else:
                pick_name = "(no pick)"
                pick_pos = ""
        elif team_num in opp_strategies:
            # This opponent uses a named strategy
            opp_fn = opp_strategies[team_num]
            proxy = TeamTrackerProxy(
                tracker, opp_roster_names[team_num], opp_rosters[team_num],
            )
            pick_name, pid = opp_fn(
                board, full_board, proxy, opp_balances[team_num],
                config, team_filled, total_rounds=rounds,
            )
            if pick_name is None:
                # Fallback to ADP
                available_ids = set(tracker.drafted_ids)
                for _, row in adp_board.iterrows():
                    if row["player_id"] not in available_ids:
                        pick_name = row["name"]
                        pid = row["player_id"]
                        break

            pick_pos = ""
            if pid:
                rows = board[board["player_id"] == pid]
                if not rows.empty:
                    pick_pos = rows.iloc[0].get("best_position", "")

            if pick_name:
                tracker.draft_player(pick_name, is_user=False, player_id=pid)
                row = board[board["player_id"] == pid]
                if not row.empty:
                    opp_balances[team_num].add_player(row.iloc[0])
                    opp_rosters[team_num].append(pid)
                    opp_roster_names[team_num].append(pick_name)
                    _assign_slot(row.iloc[0]["positions"],
                                 team_filled[team_num], config.roster_slots)
            else:
                pick_name = "(no pick)"
                pick_pos = ""
        else:
            # Other teams: pick best available by ADP, preferring players
            # who fill an active roster slot over bench/IL.
            available_ids = set(tracker.drafted_ids)
            pick_name = None
            pick_pos = ""
            pid = ""

            # First pass: find the best ADP player who fills an ACTIVE slot
            for _, row in adp_board.iterrows():
                if row["player_id"] in available_ids:
                    continue
                positions = row["positions"]
                if _can_fill_active_slot(positions, team_filled[team_num],
                                         config.roster_slots):
                    pick_name = row["name"]
                    pid = row["player_id"]
                    pick_pos = row.get("best_position", "")
                    tracker.draft_player(pick_name, is_user=False,
                                         player_id=pid)
                    _assign_slot(positions, team_filled[team_num],
                                 config.roster_slots)
                    break

            # Second pass: if all active slots are full, fill bench
            if pick_name is None:
                for _, row in adp_board.iterrows():
                    if row["player_id"] in available_ids:
                        continue
                    positions = row["positions"]
                    if _can_roster(positions, team_filled[team_num],
                                   config.roster_slots):
                        pick_name = row["name"]
                        pid = row["player_id"]
                        pick_pos = row.get("best_position", "")
                        tracker.draft_player(pick_name, is_user=False,
                                             player_id=pid)
                        _assign_slot(positions, team_filled[team_num],
                                     config.roster_slots)
                        break

            if pick_name is None:
                pick_name = "(no pick)"
                tracker.advance()
                continue

        rnd = tracker.current_round
        pick = tracker.current_pick
        marker = " <<<" if is_user else ""
        pick_log.append({
            "round": rnd, "pick": pick, "team": team_label,
            "player": pick_name, "pos": pick_pos, "is_user": is_user,
        })

        if is_user:
            print(f"  R{rnd:>2} #{pick:>3} {team_label:<30} {pick_name:<25} {pick_pos}{marker}")
        elif rnd <= 5:
            # Show early rounds for context
            print(f"  R{rnd:>2} #{pick:>3} {team_label:<30} {pick_name:<25} {pick_pos}")

        tracker.advance()

    # Save state so monte_carlo.py can read it
    state_path = PROJECT_ROOT / "data" / "draft_state.json"
    state_data = {
        "current_pick": tracker.current_pick,
        "current_round": tracker.current_round,
        "drafted_players": list(tracker.drafted_players),
        "drafted_ids": list(tracker.drafted_ids),
        "user_roster": list(tracker.user_roster),
        "user_roster_ids": list(tracker.user_roster_ids),
        "roster_slots": dict(config.roster_slots),
    }
    import json as _json
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        _json.dump(state_data, f, indent=2)
    print(f"\nState saved to {state_path}")

    # === Results ===
    print()
    print("=" * 80)
    print("DRAFT COMPLETE")
    print("=" * 80)

    # Show user roster
    print(f"\n{config.team_name} ROSTER:")
    for name in tracker.user_roster:
        rows = full_board[full_board["name"] == name]
        if not rows.empty:
            r = rows.iloc[0]
            print(f"  {r['name']:<25} {'/'.join(r['positions'][:3]):<12} "
                  f"{'hitter' if r['player_type']=='hitter' else 'pitcher'}")

    # Reconstruct all team rosters and project standings
    team_players = {i: [] for i in range(1, config.num_teams + 1)}

    # Keepers
    for keeper in config.keepers:
        for num, name in config.teams.items():
            if name == keeper["team"]:
                norm = normalize_name(keeper["name"])
                matches = full_board[full_board["name_normalized"] == norm]
                if not matches.empty:
                    team_players[num].append(matches.loc[matches["var"].idxmax()])
                break

    # Draft picks
    num_keepers = len(config.keepers)
    draft_entries = list(zip(
        tracker.drafted_players[num_keepers:],
        tracker.drafted_ids[num_keepers:],
    ))
    for pick_num, (name, pid) in enumerate(draft_entries, 1):
        rnd = (pick_num - 1) // config.num_teams + 1
        pos = (pick_num - 1) % config.num_teams + 1
        team = pos if rnd % 2 == 1 else config.num_teams - pos + 1
        rows = board[board["player_id"] == pid]
        if not rows.empty:
            team_players[team].append(rows.iloc[0])

    # Project stats (active roster only — bench players don't count)
    results = []
    for tn in range(1, config.num_teams + 1):
        tname = config.teams.get(tn, f"Team {tn}")
        all_hitters = [p for p in team_players[tn] if p["player_type"] == "hitter"]
        all_pitchers = [p for p in team_players[tn] if p["player_type"] == "pitcher"]
        hitters, pitchers = _select_active_players(
            all_hitters, all_pitchers, config.roster_slots,
        )
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
            "team": tname, "R": r, "HR": hr, "RBI": rbi, "SB": sb, "AVG": avg,
            "W": w, "K": k, "SV": sv, "ERA": era, "WHIP": whip,
            "nh": len(hitters), "np": len(pitchers),
        })

    # Roto points
    all_cats = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
    inverse = {"ERA", "WHIP"}
    for cat in all_cats:
        rev = cat not in inverse
        st = sorted(results, key=lambda x: x[cat], reverse=rev)
        for i, t in enumerate(st):
            t[f"{cat}_p"] = config.num_teams - i

    for t in results:
        t["tot"] = sum(t[f"{c}_p"] for c in all_cats)

    results.sort(key=lambda x: x["tot"], reverse=True)

    # Print standings
    print(f"\nPROJECTED ROTO STANDINGS")
    print("=" * 132)
    print(f"{'Rk':<3} {'Team':<32} {'Pts':>4}  "
          f"{'R':>5} {'HR':>4} {'RBI':>5} {'SB':>4} {'AVG':>6}  "
          f"{'W':>4} {'K':>5} {'SV':>4} {'ERA':>5} {'WHIP':>6}  "
          f"{'H':>2}/{'P':>2}")
    print("-" * 132)
    for i, t in enumerate(results, 1):
        m = " <<<" if t["team"] == config.team_name else ""
        print(f"{i:<3} {t['team']:<32} {t['tot']:>4}  "
              f"{t['R']:>5.0f} {t['HR']:>4.0f} {t['RBI']:>5.0f} "
              f"{t['SB']:>4.0f} {t['AVG']:>6.3f}  "
              f"{t['W']:>4.0f} {t['K']:>5.0f} {t['SV']:>4.0f} "
              f"{t['ERA']:>5.2f} {t['WHIP']:>6.3f}  "
              f"{t['nh']:>2}/{t['np']:>2}{m}")

    # Category breakdown
    print(f"\nROTO POINTS BY CATEGORY (10=best, 1=worst)")
    print("=" * 97)
    print(f"{'Team':<32} ", end="")
    for c in all_cats:
        print(f"{c:>5}", end="")
    print(f"{'TOT':>6}")
    print("-" * 97)
    for t in results:
        m = " <<<" if t["team"] == config.team_name else ""
        print(f"{t['team']:<32} ", end="")
        for c in all_cats:
            print(f"{t[f'{c}_p']:>5}", end="")
        print(f"{t['tot']:>6}{m}")

    # User team summary
    hart = next(t for t in results if t["team"] == config.team_name)
    rank = next(i + 1 for i, t in enumerate(results) if t["team"] == config.team_name)
    suf = {1: "st", 2: "nd", 3: "rd"}.get(rank, "th")
    print(f"\n{'=' * 60}")
    print(f"{config.team_name} - Projected {rank}{suf} place ({hart['tot']} pts)")
    print(f"{'=' * 60}")
    print(f"Roster: {hart['nh']}H / {hart['np']}P")

    def fmt(c, v):
        return f"{v:.3f}" if c in ("AVG", "ERA", "WHIP") else f"{v:.0f}"

    top = [(c, hart[f"{c}_p"], hart[c]) for c in all_cats if hart[f"{c}_p"] >= 8]
    mid = [(c, hart[f"{c}_p"], hart[c]) for c in all_cats if 4 <= hart[f"{c}_p"] <= 7]
    bot = [(c, hart[f"{c}_p"], hart[c]) for c in all_cats if hart[f"{c}_p"] <= 3]

    if top:
        print("\nStrengths (8-10 pts):")
        for c, p, v in sorted(top, key=lambda x: -x[1]):
            print(f"  {c:>4}: {fmt(c, v):>7} ({p} pts)")
    if mid:
        print("\nMiddle of pack (4-7 pts):")
        for c, p, v in sorted(mid, key=lambda x: -x[1]):
            print(f"  {c:>4}: {fmt(c, v):>7} ({p} pts)")
    if bot:
        print("\nWeak categories (1-3 pts):")
        for c, p, v in sorted(bot, key=lambda x: x[1]):
            print(f"  {c:>4}: {fmt(c, v):>7} ({p} pts)")


if __name__ == "__main__":
    main()
