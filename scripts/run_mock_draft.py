"""Mock Draft — practice the no_punt strategy against ADP opponents.

Usage:
    python scripts/run_mock_draft.py --position 8 --teams 10
    python scripts/run_mock_draft.py --position 3 --teams 12

Other teams auto-draft by ADP. No keepers. You get the full no_punt
recommendation engine with closer deadline alerts and AVG floor warnings.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.draft.board import build_draft_board
from fantasy_baseball.draft.tracker import DraftTracker
from fantasy_baseball.draft.balance import CategoryBalance, calculate_draft_leverage
from fantasy_baseball.draft.search import find_player
from fantasy_baseball.draft.recommender import (
    get_recommendations,
    get_filled_positions,
)
from fantasy_baseball.draft.strategy import (
    CLOSER_SV_THRESHOLD, NO_PUNT_SV_DEADLINE, NO_PUNT_AVG_FLOOR,
    OPP_CLOSER_ADP_BUFFER,
)
from fantasy_baseball.utils.positions import can_fill_slot
from fantasy_baseball.utils.name_utils import normalize_name

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"


def _can_fill_active_slot(player_positions, filled, roster_slots):
    for pos, total in roster_slots.items():
        if pos in ("BN", "IL"):
            continue
        if filled.get(pos, 0) < total and can_fill_slot(player_positions, pos):
            return True
    return False


def _can_roster(player_positions, filled, roster_slots):
    for pos, total in roster_slots.items():
        if filled.get(pos, 0) < total and can_fill_slot(player_positions, pos):
            return True
    return False


def _assign_slot(player_positions, filled, roster_slots):
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


def _get_draft_leverage(balance, tracker):
    totals = balance.get_totals()
    picks_made = len(tracker.user_roster)
    total_picks = tracker.rounds
    return calculate_draft_leverage(totals, picks_made, total_picks)


def _handle_user_pick(board, tracker, balance, roster_slots, num_teams):
    """Show no_punt recommendations and get user input."""
    filled = get_filled_positions(
        tracker.user_roster_ids, board, roster_slots=roster_slots,
    )

    # Picks until next turn
    peek_pick = tracker.current_pick + 1
    picks_gap = 1
    while peek_pick <= tracker.total_picks:
        peek_round = (peek_pick - 1) // tracker.num_teams + 1
        peek_pos = (peek_pick - 1) % tracker.num_teams + 1
        if peek_round % 2 == 1:
            peek_team = peek_pos
        else:
            peek_team = tracker.num_teams - peek_pos + 1
        if peek_team == tracker.user_position:
            break
        peek_pick += 1
        picks_gap += 1

    leverage = _get_draft_leverage(balance, tracker)
    recs = get_recommendations(
        board, drafted=tracker.drafted_ids,
        user_roster=tracker.user_roster,
        n=5, filled_positions=filled,
        picks_until_next=picks_gap,
        roster_slots=roster_slots,
        num_teams=num_teams,
        draft_leverage=leverage,
    )

    # --- No-punt strategy logic ---
    strategy_alerts = []

    has_closer = False
    for pid in tracker.user_roster_ids:
        rows = board[board["player_id"] == pid]
        if not rows.empty and rows.iloc[0].get("sv", 0) >= CLOSER_SV_THRESHOLD:
            has_closer = True
            break

    closer_rec = None
    if not has_closer:
        available = board[~board["player_id"].isin(tracker.drafted_ids)]
        closers = available[
            available.apply(lambda r: r.get("sv", 0) >= CLOSER_SV_THRESHOLD, axis=1)
        ]
        if not closers.empty:
            best_closer = closers.sort_values("var", ascending=False).iloc[0]
            closer_adp = best_closer.get("adp", 999)
            current_pick = tracker.current_pick
            num_keepers = len(tracker.drafted_players) - (current_pick - 1)
            effective_pick = current_pick + num_keepers

            if effective_pick >= closer_adp - OPP_CLOSER_ADP_BUFFER:
                closer_rec = {
                    "name": best_closer["name"],
                    "best_position": "RP",
                    "var": best_closer.get("var", 0),
                    "score": None,
                    "need_flag": True,
                    "note": f"FALLING closer — ADP {closer_adp:.0f}, effective pick {effective_pick}",
                    "player_type": "pitcher",
                }
                strategy_alerts.append(
                    f"CLOSER OPPORTUNITY: {best_closer['name']} "
                    f"(ADP {closer_adp:.0f}) is falling"
                )
            elif tracker.current_round >= NO_PUNT_SV_DEADLINE:
                closer_rec = {
                    "name": best_closer["name"],
                    "best_position": "RP",
                    "var": best_closer.get("var", 0),
                    "score": None,
                    "need_flag": True,
                    "note": f"DEADLINE — no closer by round {NO_PUNT_SV_DEADLINE}",
                    "player_type": "pitcher",
                }
                strategy_alerts.append(
                    f"CLOSER DEADLINE: Draft {best_closer['name']} now — "
                    f"round {NO_PUNT_SV_DEADLINE} backstop reached"
                )

    # AVG floor
    current_h = sum(h.get("h", 0) for h in balance._hitters)
    current_ab = sum(h.get("ab", 0) for h in balance._hitters)
    avg_warnings = []
    if current_ab > 0:
        for rec in recs:
            if rec.get("player_type") != "hitter":
                continue
            rows = board[board["name"] == rec["name"]]
            if rows.empty:
                continue
            player = rows.iloc[0]
            new_h = current_h + player.get("h", 0)
            new_ab = current_ab + player.get("ab", 0)
            projected_avg = new_h / new_ab if new_ab > 0 else 0
            if projected_avg < NO_PUNT_AVG_FLOOR:
                avg_warnings.append(rec["name"])

    # Reorder: push low-AVG hitters below safe recs
    if avg_warnings:
        safe_recs = [r for r in recs if r["name"] not in avg_warnings]
        risky_recs = [r for r in recs if r["name"] in avg_warnings]
        recs = safe_recs + risky_recs

    # Closer on top if triggered
    if closer_rec:
        recs = [r for r in recs if r["name"] != closer_rec["name"]]
        recs = [closer_rec] + recs[:4]

    # Display
    print(f"\nPicks until next turn: {picks_gap}")

    if strategy_alerts:
        print()
        for alert in strategy_alerts:
            print(f"  >>> {alert}")

    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(recs, 1):
        flag = " [NEED]" if rec.get("need_flag") else ""
        note = f" ({rec['note']})" if rec.get("note") else ""
        score_str = f" score: {rec['score']:.1f}" if rec.get("score") is not None else ""
        avg_warn = " [LOW AVG]" if rec["name"] in avg_warnings else ""
        print(f"  {i}. {rec['name']} ({rec.get('best_position', '?')}) "
              f"VAR: {rec['var']:.1f}{score_str}{flag}{avg_warn}{note}")

    # Balance
    totals = balance.get_totals()
    print(f"\nROSTER BALANCE:")
    print(f"  R:{totals['R']:.0f} HR:{totals['HR']:.0f} RBI:{totals['RBI']:.0f} "
          f"SB:{totals['SB']:.0f} AVG:{totals['AVG']:.3f}")
    era_str = f"{totals['ERA']:.2f}" if totals["ERA"] is not None else "N/A"
    whip_str = f"{totals['WHIP']:.3f}" if totals["WHIP"] is not None else "N/A"
    print(f"  W:{totals['W']:.0f} K:{totals['K']:.0f} SV:{totals['SV']:.0f} "
          f"ERA:{era_str} WHIP:{whip_str}")

    # Get input
    while True:
        raw = input("\nYour pick (name, number, or 'skip'): ").strip()
        if not raw:
            continue
        if raw.lower() == "skip":
            return None, None

        # Number selection
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(recs):
                name = recs[idx]["name"]
                rows = board[board["name"] == name]
                pid = rows.iloc[0]["player_id"] if not rows.empty else name
                return name, pid

        # Name search
        result = find_player(raw, board, tracker.drafted_ids)
        if result:
            return result["name"], result["player_id"]
        print(f"  Player not found: '{raw}'. Try again.")


def _auto_pick(adp_board, tracker, team_filled, roster_slots):
    """ADP-based auto-pick for opponents."""
    available_ids = set(tracker.drafted_ids)

    # Active slots first
    for _, row in adp_board.iterrows():
        if row["player_id"] in available_ids:
            continue
        if _can_fill_active_slot(row["positions"], team_filled, roster_slots):
            return row["name"], row["player_id"], row["positions"]

    # Bench
    for _, row in adp_board.iterrows():
        if row["player_id"] in available_ids:
            continue
        if _can_roster(row["positions"], team_filled, roster_slots):
            return row["name"], row["player_id"], row["positions"]

    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Mock draft with no_punt strategy")
    parser.add_argument("--position", "-p", type=int, required=True,
                        help="Your draft position (1-indexed)")
    parser.add_argument("--teams", "-t", type=int, default=10,
                        help="Number of teams (default: 10)")
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    roster_slots = config.roster_slots
    rounds = sum(roster_slots.values())

    print(f"MOCK DRAFT | Position {args.position} of {args.teams}")
    print(f"Strategy: no_punt (R{NO_PUNT_SV_DEADLINE} closer deadline, AVG floor {NO_PUNT_AVG_FLOOR})")
    print(f"Rounds: {rounds} | Opponents: ADP auto-draft")
    print()

    # Build board (no keepers)
    print("Building draft board...")
    board = build_draft_board(
        projections_dir=PROJECTIONS_DIR,
        positions_path=POSITIONS_PATH,
        systems=config.projection_systems,
        weights=config.projection_weights or None,
        sgp_overrides=config.sgp_overrides or None,
        roster_slots=roster_slots or None,
        num_teams=args.teams,
    )
    adp_board = board.sort_values("adp", ascending=True)
    print(f"Draft pool: {len(board)} players")
    print()

    tracker = DraftTracker(
        num_teams=args.teams,
        user_position=args.position,
        rounds=rounds,
    )
    balance = CategoryBalance()
    team_filled = {i: {} for i in range(1, args.teams + 1)}

    try:
        while tracker.current_pick <= tracker.total_picks:
            team_num = tracker.picking_team
            is_user = tracker.is_user_pick

            print("=" * 60)
            print(f"ROUND {tracker.current_round} | Pick {tracker.current_pick} "
                  f"| Team {team_num}", end="")
            if is_user:
                print(" *** YOUR PICK ***")
            else:
                print()
            print("=" * 60)

            if is_user:
                name, pid = _handle_user_pick(
                    board, tracker, balance, roster_slots, args.teams,
                )
                if name:
                    tracker.draft_player(name, is_user=True, player_id=pid)
                    rows = board[board["player_id"] == pid]
                    if not rows.empty:
                        balance.add_player(rows.iloc[0])
                        _assign_slot(rows.iloc[0]["positions"],
                                     team_filled[team_num], roster_slots)
                    print(f"\n  -> Drafted: {name}")
                else:
                    print("  -> Skipped")
            else:
                name, pid, positions = _auto_pick(
                    adp_board, tracker, team_filled[team_num], roster_slots,
                )
                if name:
                    tracker.draft_player(name, is_user=False, player_id=pid)
                    _assign_slot(positions, team_filled[team_num], roster_slots)
                    pos_str = "/".join(positions[:2]) if positions else ""
                    print(f"  Auto-pick: {name} ({pos_str})")
                else:
                    print("  (no pick available)")

            tracker.advance()
            print()

        print("\n" + "=" * 60)
        print("MOCK DRAFT COMPLETE")
        print("=" * 60)
        print(f"\nYour roster ({len(tracker.user_roster)} players):")
        for name in tracker.user_roster:
            rows = board[board["name"] == name]
            if not rows.empty:
                r = rows.iloc[0]
                pos = "/".join(r["positions"][:3])
                ptype = "hitter" if r["player_type"] == "hitter" else "pitcher"
                print(f"  {r['name']:<25} {pos:<12} {ptype}")

        # Show balance
        totals = balance.get_totals()
        print(f"\nProjected stats:")
        print(f"  R:{totals['R']:.0f} HR:{totals['HR']:.0f} RBI:{totals['RBI']:.0f} "
              f"SB:{totals['SB']:.0f} AVG:{totals['AVG']:.3f}")
        era_str = f"{totals['ERA']:.2f}" if totals["ERA"] is not None else "N/A"
        whip_str = f"{totals['WHIP']:.3f}" if totals["WHIP"] is not None else "N/A"
        print(f"  W:{totals['W']:.0f} K:{totals['K']:.0f} SV:{totals['SV']:.0f} "
              f"ERA:{era_str} WHIP:{whip_str}")

    except (KeyboardInterrupt, EOFError):
        print("\n\nMock draft ended early.")
        print(f"Your roster so far ({len(tracker.user_roster)} players):")
        for name in tracker.user_roster:
            print(f"  {name}")


if __name__ == "__main__":
    main()
