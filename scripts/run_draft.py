"""Interactive Draft Assistant for Fantasy Baseball.

Usage:
    python scripts/run_draft.py

Pre-requisites:
    1. FanGraphs projection CSVs in data/projections/
    2. Run: python scripts/fetch_positions.py
    3. config/league.yaml with keepers and settings
"""

import json
import sys
import threading
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.data.db import get_connection
from fantasy_baseball.draft.balance import CategoryBalance
from fantasy_baseball.draft.board import apply_keepers, build_draft_board
from fantasy_baseball.draft.recommender import (
    Recommendation,
    calculate_vona_scores,
    get_filled_positions,
    get_recommendations,
    get_roster_by_position,
)
from fantasy_baseball.draft.search import find_player, split_team_and_player
from fantasy_baseball.draft.state import serialize_board, serialize_state, write_board, write_state
from fantasy_baseball.draft.strategy import (
    NO_PUNT_AVG_FLOOR,
    OPP_CLOSER_ADP_BUFFER,
    STRATEGIES,
    _count_closers,
    _force_closer,
    _sv_in_danger,
)
from fantasy_baseball.draft.tracker import DraftTracker
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD, Category
from fantasy_baseball.web.app import create_app

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
DRAFT_ORDER_PATH = PROJECT_ROOT / "config" / "draft_order.json"
STATE_PATH = PROJECT_ROOT / "data" / "draft_state.json"
BOARD_PATH = PROJECT_ROOT / "data" / "draft_state_board.json"
DRAFTS_DIR = PROJECT_ROOT / "data" / "drafts"

FLASK_PORT = 5000


def _start_flask_server(state_path: Path) -> None:
    """Start the Flask dashboard server in a background daemon thread."""
    from waitress import serve

    app = create_app(state_path=state_path)
    server_thread = threading.Thread(
        target=lambda: serve(
            app,
            host="127.0.0.1",
            port=FLASK_PORT,
            _quiet=True,
        ),
        daemon=True,
        name="flask-dashboard",
    )
    server_thread.start()
    print(f"Dashboard running at http://127.0.0.1:{FLASK_PORT}")


def _load_draft_order(path, num_teams):
    """Load custom draft order with trade detection.

    Returns a list of dicts, one per pick (0-indexed), with keys:
        team: team name for this pick
        round: 1-indexed round
        slot: 1-indexed slot within round
        traded: True if this pick was traded (team differs from standard snake)
        original_team: the team that would have picked in standard snake (if traded)
    """
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    rounds = data["rounds"]
    trades = {(t["round"], t["slot"]): t for t in data.get("trades", [])}
    picks = []
    for rnd_idx, round_teams in enumerate(rounds):
        rnd = rnd_idx + 1
        for slot_idx, team_name in enumerate(round_teams):
            slot = slot_idx + 1
            trade_info = trades.get((rnd, slot))
            picks.append(
                {
                    "team": team_name,
                    "round": rnd,
                    "slot": slot,
                    "traded": trade_info is not None,
                    "original_team": trade_info["from"] if trade_info else None,
                }
            )
    return picks


def _save_draft_log(
    tracker, balance, config, full_board, mock=False, draft_position=None, run_timestamp=None
):
    """Save a timestamped draft log to data/drafts/ for later analysis."""
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "mock_" if mock else "draft_"
    filename = f"{prefix}{run_timestamp}.json"
    out_path = DRAFTS_DIR / filename

    num_teams = config.num_teams
    num_keepers = len(config.keepers) if not mock else 0

    # Build draft log from tracker
    draft_log = []
    draft_entries = list(
        zip(
            tracker.drafted_players[num_keepers:],
            tracker.drafted_ids[num_keepers:],
            strict=False,
        )
    )
    for pick_num, (name, pid) in enumerate(draft_entries, 1):
        rnd = (pick_num - 1) // num_teams + 1
        pos = (pick_num - 1) % num_teams + 1
        team_num = pos if rnd % 2 == 1 else num_teams - pos + 1
        team_name = config.teams.get(team_num, f"Team {team_num}")
        draft_log.append(
            {
                "pick": pick_num,
                "round": rnd,
                "team_num": team_num,
                "team": team_name,
                "player": name,
                "player_id": pid,
            }
        )

    # User roster details
    user_roster = []
    for pid in tracker.user_roster_ids:
        rows = full_board[full_board["player_id"] == pid]
        if not rows.empty:
            p = rows.iloc[0]
            entry = {
                "name": str(p.get("name", "")),
                "player_id": str(p.get("player_id", "")),
                "player_type": str(p.get("player_type", "")),
                "positions": [str(x) for x in p.get("positions", [])],
                "var": round(float(p.get("var", 0)), 2),
            }
            for stat in [
                "r",
                "hr",
                "rbi",
                "sb",
                "h",
                "ab",
                "avg",
                "w",
                "k",
                "sv",
                "ip",
                "er",
                "bb",
                "h_allowed",
            ]:
                val = p.get(stat, 0)
                if val is not None and val != 0:
                    entry[stat] = round(float(val), 4)
            user_roster.append(entry)

    output = {
        "metadata": {
            "timestamp": run_timestamp,
            "mock": mock,
            "draft_position": draft_position or config.draft_position,
            "num_teams": num_teams,
            "user_team": config.teams.get(
                draft_position or config.draft_position, f"Team {draft_position}"
            ),
            "strategy": config.strategy,
            "scoring_mode": config.scoring_mode,
            "picks_completed": len(draft_entries),
            "total_picks": tracker.total_picks,
            "complete": tracker.current_pick > tracker.total_picks,
        },
        "user_roster": user_roster,
        "draft_log": draft_log,
        "balance": {cat.value: v for cat, v in balance.get_totals().items()},
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return str(out_path)


def _write_dashboard_state(
    tracker,
    balance,
    board,
    recs,
    filled,
    roster_slots=None,
    roster_by_pos=None,
    teams=None,
    num_keepers=0,
    vona_scores=None,
):
    """Serialize and atomically write dashboard state to disk."""
    state = serialize_state(
        tracker=tracker,
        balance=balance,
        board=board,
        recommendations=recs,
        filled_positions=filled,
        roster_slots=roster_slots,
        roster_by_position=roster_by_pos,
        teams=teams,
        num_keepers=num_keepers,
        vona_scores=vona_scores,
    )
    write_state(state, STATE_PATH)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Fantasy Baseball Draft Assistant")
    parser.add_argument(
        "--mock", action="store_true", help="Mock draft mode: no keepers, override position/teams"
    )
    parser.add_argument(
        "--position",
        "-p",
        type=int,
        default=None,
        help="Draft position (mock mode, default: from config)",
    )
    parser.add_argument(
        "--teams",
        "-t",
        type=int,
        default=None,
        help="Number of teams (mock mode, default: from config)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # Load config
    config = load_config(CONFIG_PATH)

    # Timestamp for this draft session (used for saving)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Mock mode overrides
    mock = args.mock
    num_teams = args.teams or config.num_teams
    draft_position = args.position or config.draft_position
    keepers = [] if mock else config.keepers

    strategy = config.strategy
    scoring_mode = config.scoring_mode

    if mock:
        print(f"MOCK DRAFT | Position {draft_position} of {num_teams}")
        print(f"Strategy: {strategy} + {scoring_mode} | No keepers")
    else:
        print(f"League {config.league_id} | Draft position: {draft_position}")
        print(f"Team: {config.team_name}")
        print(f"Strategy: {strategy} + {scoring_mode}")
        print(f"Keepers: {len(keepers)} players across {num_teams} teams")
    print()

    # Build draft board (keep full board for keeper lookups)
    print("Building draft board...")
    conn = get_connection()
    full_board = build_draft_board(
        conn=conn,
        sgp_overrides=config.sgp_overrides or None,
        roster_slots=config.roster_slots or None,
        num_teams=num_teams,
    )
    conn.close()
    board = apply_keepers(full_board, keepers)
    print(f"Draft pool: {len(board)} players (after removing {len(keepers)} keepers)")
    print()

    # Initialize tracker and balance
    user_keepers = [k for k in keepers if k.get("team") == config.team_name]
    draftable_slots = sum(v for k, v in config.roster_slots.items() if k != "IL")
    rounds = draftable_slots - len(user_keepers)
    tracker = DraftTracker(
        num_teams=num_teams,
        user_position=draft_position,
        rounds=rounds,
    )
    balance = CategoryBalance()

    # Add keeper projections to balance and mark all keepers as drafted
    from fantasy_baseball.utils.name_utils import normalize_name

    for keeper in keepers:
        is_user = keeper.get("team") == config.team_name
        norm = normalize_name(keeper["name"])
        matches = full_board[full_board["name_normalized"] == norm]
        if not matches.empty:
            # Pick the highest-VAR match (the real keeper, not a namesake)
            best = matches.loc[matches["var"].idxmax()]
            if is_user:
                balance.add_player(best)
            tracker.draft_player(best["name"], is_user=is_user, player_id=best["player_id"])
        else:
            tracker.draft_player(keeper["name"], is_user=is_user)

    # Load custom draft order (with traded picks)
    draft_order = _load_draft_order(DRAFT_ORDER_PATH, num_teams)
    if draft_order:
        num_keepers = len(keepers)
        # Only use picks after keepers
        draft_picks = draft_order[num_keepers:]
        user_pick_count = sum(1 for p in draft_picks if p["team"] == config.team_name)
        trade_count = sum(1 for p in draft_picks if p["traded"])
        print(
            f"Draft order loaded: {len(draft_picks)} picks, "
            f"{user_pick_count} yours, {trade_count} traded picks"
        )
    else:
        draft_picks = None
        print("No custom draft order found — using standard snake")
    print()

    # Write the full board once (clients fetch via /api/board on page load)
    board_data = serialize_board(board)
    write_board(board_data, BOARD_PATH)
    print(f"Board snapshot written ({len(board_data)} players)")

    # Start Flask dashboard server
    _start_flask_server(STATE_PATH)

    # Write initial state (use full_board for roster lookups so keepers are found)
    filled = get_filled_positions(
        tracker.user_roster_ids, full_board, roster_slots=config.roster_slots
    )
    by_pos = get_roster_by_position(
        tracker.user_roster_ids, full_board, roster_slots=config.roster_slots
    )
    recs = get_recommendations(
        board,
        tracker.drafted_ids,
        tracker.user_roster,
        n=5,
        filled_positions=filled,
        picks_until_next=tracker.picks_until_next_turn,
        roster_slots=config.roster_slots,
        num_teams=num_teams,
        scoring_mode=scoring_mode,
    )
    available = board[~board["player_id"].isin(tracker.drafted_ids)]
    vona = calculate_vona_scores(available, tracker.picks_until_next_turn)
    _write_dashboard_state(
        tracker,
        balance,
        board,
        recs,
        filled,
        roster_slots=config.roster_slots,
        roster_by_pos=by_pos,
        teams=config.teams,
        num_keepers=len(keepers),
        vona_scores=vona,
    )

    # Show pre-draft rankings
    print("=" * 70)
    print("TOP 25 AVAILABLE PLAYERS")
    print("=" * 70)
    _show_top_players(board, tracker.drafted_ids, 25)
    print()

    # Build team name list for input parsing
    team_names = list(config.teams.values()) if config.teams else []

    # Main draft loop
    num_keeper_picks = len(keepers)
    keeper_rounds = num_keeper_picks // num_teams
    pick_index = 0  # index into draft_picks (post-keeper picks)
    try:
        while tracker.current_pick <= tracker.total_picks:
            # Overall pick number including keeper picks
            overall_pick = tracker.current_pick + num_keeper_picks

            # Determine who's picking: custom order or standard snake
            if draft_picks and pick_index < len(draft_picks):
                pick_info = draft_picks[pick_index]
                team_label = pick_info["team"]
                is_user = team_label == config.team_name
                is_traded = pick_info["traded"]
                pick_round = pick_info["round"]
            else:
                team_num = tracker.picking_team
                team_label = config.teams.get(team_num, f"Team {team_num}")
                is_user = tracker.is_user_pick
                is_traded = False
                pick_round = tracker.current_round + keeper_rounds

            print("=" * 70)
            if is_traded:
                original = pick_info.get("original_team", "???")
                print("  !!!!! TRADED PICK !!!!!")
                print(f"  Originally {original}'s pick -> now {team_label}'s")
                print("  !!!!! TRADED PICK !!!!!")
            print(f"ROUND {pick_round} | Pick {overall_pick} | {team_label}", end="")
            if is_user:
                print(" *** YOUR PICK ***")
            else:
                print()
            print("=" * 70)

            if is_user:
                _handle_user_pick(
                    board,
                    full_board,
                    tracker,
                    balance,
                    roster_slots=config.roster_slots,
                    num_teams=num_teams,
                    team_names=team_names,
                    user_team_name=config.team_name,
                    config=config,
                    draft_picks=draft_picks,
                    pick_index=pick_index,
                )
            else:
                # Peek at input — if "mine", treat as user pick (traded pick)
                raw = input("\nPick (player, 'team player', or 'mine'): ").strip()
                if raw.lower() == "mine":
                    print("  >>> YOUR PICK (traded to you)")
                    _handle_user_pick(
                        board,
                        full_board,
                        tracker,
                        balance,
                        roster_slots=config.roster_slots,
                        num_teams=num_teams,
                        config=config,
                        draft_picks=draft_picks,
                        pick_index=pick_index,
                    )
                else:
                    # Process the input that was already typed
                    _handle_other_pick(
                        board,
                        full_board,
                        tracker,
                        balance,
                        team_names,
                        config.team_name,
                        prefilled_input=raw,
                    )

            # Advance to next pick so dashboard shows upcoming pick, not the one just made
            tracker.advance()
            pick_index += 1

            # Write updated state for the dashboard after every pick
            filled = get_filled_positions(
                tracker.user_roster_ids, full_board, roster_slots=config.roster_slots
            )
            by_pos = get_roster_by_position(
                tracker.user_roster_ids, full_board, roster_slots=config.roster_slots
            )
            recs = get_recommendations(
                board,
                tracker.drafted_ids,
                tracker.user_roster,
                n=5,
                filled_positions=filled,
                picks_until_next=tracker.picks_until_next_turn,
                roster_slots=config.roster_slots,
                num_teams=num_teams,
                scoring_mode=scoring_mode,
            )

            available = board[~board["player_id"].isin(tracker.drafted_ids)]
            vona = calculate_vona_scores(available, tracker.picks_until_next_turn)
            _write_dashboard_state(
                tracker,
                balance,
                board,
                recs,
                filled,
                roster_slots=config.roster_slots,
                roster_by_pos=by_pos,
                teams=config.teams,
                num_keepers=len(keepers),
                vona_scores=vona,
            )

            # Show updated top 10
            print()
            _show_top_players(board, tracker.drafted_ids, 10)
            print()

        print("\nDraft complete!")
    except (KeyboardInterrupt, EOFError):
        print("\n\nDraft paused.")

    # Save timestamped draft log
    log_path = _save_draft_log(
        tracker,
        balance,
        config,
        full_board,
        mock=mock,
        draft_position=draft_position,
        run_timestamp=run_timestamp,
    )
    print(f"\nDraft log saved to {log_path}")

    print("\nYour roster:")
    for name in tracker.user_roster:
        print(f"  {name}")


def _handle_user_pick(
    board,
    full_board,
    tracker,
    balance,
    roster_slots=None,
    num_teams=None,
    team_names=None,
    user_team_name=None,
    config=None,
    draft_picks=None,
    pick_index=0,
):
    """Handle the user's draft pick with recommendations."""
    scoring_mode = config.scoring_mode if config else "var"
    strategy = config.strategy if config else "two_closers"
    team_name = config.team_name if config else "Hart of the Order"
    filled = get_filled_positions(tracker.user_roster_ids, full_board, roster_slots=roster_slots)
    # Calculate gap to NEXT user turn after this one.
    if draft_picks:
        # Use custom draft order to find next user pick
        picks_gap = 0
        for future in draft_picks[pick_index + 1 :]:
            picks_gap += 1
            if future["team"] == team_name:
                break
        if picks_gap == 0:
            picks_gap = 1  # fallback
    else:
        # Standard snake fallback
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

    recs = get_recommendations(
        board,
        drafted=tracker.drafted_ids,
        user_roster=tracker.user_roster,
        n=5,
        filled_positions=filled,
        picks_until_next=picks_gap,
        roster_slots=roster_slots,
        num_teams=num_teams,
        scoring_mode=scoring_mode,
    )

    # --- Strategy alerts ---
    strategy_alerts = []
    closer_rec = None

    # Count closers on roster
    closer_count = _count_closers(tracker, board, full_board)

    # n-closers strategies: force closers at staggered deadlines
    _closer_deadlines = {
        "two_closers": ([8, 14], 2),
        "three_closers": ([5, 9, 13], 3),
        "four_closers": ([5, 8, 12, 16], 4),
        "no_punt": ([9], 1),
        "no_punt_stagger": ([13, 17, 20], 3),
        "no_punt_cap3": ([13, 17, 20], 3),
        "avg_hedge": ([9], 1),
        "closers_avg": ([5, 9, 13], 3),
    }
    if strategy in _closer_deadlines:
        deadlines, target = _closer_deadlines[strategy]
        if closer_count < target:
            deadline_idx = closer_count
            if deadline_idx < len(deadlines):
                deadline = deadlines[deadline_idx]
                if tracker.current_round >= deadline:
                    strategy_alerts.append(
                        f"CLOSER DEADLINE: Round {deadline} — need closer "
                        f"#{closer_count + 1} of {target}"
                    )
                    available = board[~board["player_id"].isin(tracker.drafted_ids)]
                    closers_avail = available[available["sv"].fillna(0) >= CLOSER_SV_THRESHOLD]
                    if not closers_avail.empty:
                        best_closer = closers_avail.sort_values("var", ascending=False).iloc[0]
                        closer_rec = Recommendation(
                            name=best_closer["name"],
                            var=float(best_closer.get("var", 0) or 0),
                            score=None,
                            best_position="RP",
                            positions=list(best_closer["positions"])
                            if isinstance(best_closer["positions"], list)
                            else [best_closer["positions"]],
                            player_type=PlayerType.PITCHER,
                            need_flag=True,
                            note=f"CLOSER DEADLINE — draft closer #{closer_count + 1}",
                        )

    # no_punt_opp: dynamic SV monitoring + opportunistic closer grabs
    elif strategy == "no_punt_opp":
        # Build team_rosters from tracker for SV danger check
        # (In the interactive draft we don't have full team_rosters,
        # so we check closer count as a simpler proxy)
        if closer_count == 0 and tracker.current_round >= 8:
            strategy_alerts.append("SV DANGER: You have zero closers — consider drafting one now")
            available = board[~board["player_id"].isin(tracker.drafted_ids)]
            closers_avail = available[
                available.apply(lambda r: r.get("sv", 0) >= CLOSER_SV_THRESHOLD, axis=1)
            ]
            if not closers_avail.empty:
                best_closer = closers_avail.sort_values("var", ascending=False).iloc[0]
                closer_rec = Recommendation(
                    name=best_closer["name"],
                    var=float(best_closer.get("var", 0) or 0),
                    score=None,
                    best_position="RP",
                    positions=list(best_closer["positions"])
                    if isinstance(best_closer["positions"], list)
                    else [best_closer["positions"]],
                    player_type=PlayerType.PITCHER,
                    need_flag=True,
                    note="SV DANGER — draft a closer",
                )

        # Opportunistic closer grab: closer falling past ADP
        if closer_rec is None and closer_count < 3:
            available = board[~board["player_id"].isin(tracker.drafted_ids)]
            closers_avail = available[
                available.apply(lambda r: r.get("sv", 0) >= CLOSER_SV_THRESHOLD, axis=1)
            ]
            if not closers_avail.empty:
                effective_pick = tracker.current_pick
                falling = closers_avail[
                    effective_pick >= closers_avail["adp"] - OPP_CLOSER_ADP_BUFFER
                ]
                if not falling.empty:
                    best_falling = falling.sort_values("var", ascending=False).iloc[0]
                    strategy_alerts.append(
                        f"OPPORTUNISTIC: {best_falling['name']} (closer) has fallen past ADP"
                    )

    # Check AVG floor and flag low-AVG hitters
    current_h = sum(h.get("h", 0) for h in balance._hitters)
    current_ab = sum(h.get("ab", 0) for h in balance._hitters)
    avg_warnings = []
    if current_ab > 0:
        for rec in recs:
            if rec.player_type != PlayerType.HITTER:
                continue
            rows = board[board["name"] == rec.name]
            if rows.empty:
                continue
            player = rows.iloc[0]
            new_h = current_h + player.get("h", 0)
            new_ab = current_ab + player.get("ab", 0)
            projected_avg = new_h / new_ab if new_ab > 0 else 0
            if projected_avg < NO_PUNT_AVG_FLOOR:
                avg_warnings.append(rec.name)

    # Reorder: push low-AVG hitters below non-flagged recs
    if avg_warnings:
        safe_recs = [r for r in recs if r.name not in avg_warnings]
        risky_recs = [r for r in recs if r.name in avg_warnings]
        recs = safe_recs + risky_recs

    # Build final recommendation list: closer on top if triggered
    if closer_rec:
        closer_in_recs = any(r.name == closer_rec.name for r in recs)
        if closer_in_recs:
            recs = [r for r in recs if r.name != closer_rec.name]
        recs = [closer_rec, *recs[:4]]

    # Show recommendations
    print(f"\nPicks until next turn: {picks_gap}")

    if strategy_alerts:
        print()
        for alert in strategy_alerts:
            print(f"  >>> {alert}")

    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(recs, 1):
        flag = " [NEED]" if rec.need_flag else ""
        note = f" ({rec.note})" if rec.note else ""
        score_str = f" score: {rec.score:.1f}" if rec.score is not None else ""
        avg_warn = " [LOW AVG]" if rec.name in avg_warnings else ""
        print(
            f"  {i}. {rec.name} ({rec.best_position}) "
            f"VAR: {rec.var:.1f}{score_str}{flag}{avg_warn}{note}"
        )

    # Show category balance
    totals = balance.get_totals()
    warnings = balance.get_warnings()
    print("\nROSTER BALANCE:")
    print(
        f"  R:{totals[Category.R]:.0f} HR:{totals[Category.HR]:.0f} "
        f"RBI:{totals[Category.RBI]:.0f} SB:{totals[Category.SB]:.0f} "
        f"AVG:{totals[Category.AVG]:.3f}"
    )
    era_str = f"{totals[Category.ERA]:.2f}" if totals[Category.ERA] is not None else "N/A"
    whip_str = f"{totals[Category.WHIP]:.3f}" if totals[Category.WHIP] is not None else "N/A"
    print(
        f"  W:{totals[Category.W]:.0f} K:{totals[Category.K]:.0f} "
        f"SV:{totals[Category.SV]:.0f} ERA:{era_str} WHIP:{whip_str}"
    )
    if warnings:
        print(f"  Warnings: {', '.join(warnings)}")

    # Get user input — pass team_names so "spacemen gausman" is recognized as traded away
    name, pid, matched_team = _get_player_input(
        board,
        tracker,
        current_recs=recs,
        team_names=team_names,
    )
    if name:
        # If a different team was specified, this pick was traded away
        traded_away = (
            matched_team is not None
            and user_team_name is not None
            and matched_team != user_team_name
        )
        if traded_away:
            tracker.draft_player(name, is_user=False, player_id=pid)
            print(f"  -> Drafted: {name} ({matched_team} via trade)")
        else:
            tracker.draft_player(name, is_user=True, player_id=pid)
            rows = board[board["player_id"] == pid] if pid else board[board["name"] == name]
            if not rows.empty:
                balance.add_player(rows.iloc[0])
            print(f"  -> Drafted: {name}")


def _handle_other_pick(
    board, full_board, tracker, balance, team_names=None, user_team_name=None, prefilled_input=None
):
    """Handle another team's pick (or a traded pick for the user's team).

    If prefilled_input is provided, it's used as the first input line
    instead of prompting (for when the main loop already read input to
    check for 'mine').
    """
    name, pid, matched_team = _get_player_input(
        board, tracker, team_names=team_names, prefilled_input=prefilled_input
    )
    if name:
        is_user = (
            matched_team is not None
            and user_team_name is not None
            and matched_team == user_team_name
        )
        tracker.draft_player(name, is_user=is_user, player_id=pid)
        if is_user:
            rows = board[board["player_id"] == pid] if pid else board[board["name"] == name]
            if rows.empty and pid:
                rows = full_board[full_board["player_id"] == pid]
            if not rows.empty:
                balance.add_player(rows.iloc[0])
            print(f"  -> Drafted: {name} (YOUR PICK via trade)")
        else:
            print(f"  -> Drafted: {name}")


def _get_player_input(board, tracker, team_names=None, current_recs=None, prefilled_input=None):
    """Get and fuzzy-match a player name from user input.

    Returns (name, player_id, team) or (None, None, None).

    *team* is the matched team-name prefix (e.g. "Hart of the Order") if
    the user typed one, otherwise None.

    If *current_recs* is provided, number selection uses those recs directly
    (matching what was displayed) instead of regenerating them.

    If *team_names* is provided, the input may optionally start with a team
    name prefix (e.g. "hello peanuts logan webb").  The team prefix is
    stripped before the player fuzzy search.

    If *prefilled_input* is provided, it's used as the first input line
    instead of prompting.
    """
    available = board[~board["player_id"].isin(tracker.drafted_ids)]
    available_names = available["name"].tolist()

    def _lookup_id(name):
        """Find the player_id for a matched name."""
        rows = available[available["name"] == name]
        if not rows.empty:
            return rows.iloc[0]["player_id"]
        return name + "::unknown"

    first_loop = True
    while True:
        if first_loop and prefilled_input is not None:
            raw = prefilled_input
            first_loop = False
        else:
            raw = input("\nEnter player name (or 'skip' to skip): ").strip()
        if raw.lower() == "skip":
            return None, None, None
        if raw.lower() == "quit":
            sys.exit(0)

        # Try number selection (for recommendations)
        if raw.isdigit():
            idx = int(raw) - 1
            recs = current_recs
            if recs is None:
                filled = get_filled_positions(tracker.user_roster_ids, board)
                recs = get_recommendations(
                    board, tracker.drafted_ids, tracker.user_roster, n=5, filled_positions=filled
                )
            if 0 <= idx < len(recs):
                return recs[idx].name, _lookup_id(recs[idx].name), None

        # Try to split off a team-name prefix
        player_query = raw
        matched_team = None

        # Quoted team name: "crews control" chris sale
        if raw.startswith('"') and '"' in raw[1:]:
            closing = raw.index('"', 1)
            quoted_team = raw[1:closing].strip()
            remainder = raw[closing + 1 :].strip()
            if remainder:
                # Match quoted text against team names (case-insensitive)
                for tn in team_names or []:
                    if tn.lower() == quoted_team.lower():
                        matched_team = tn
                        player_query = remainder
                        print(f"  (team: {tn})")
                        break
                if matched_team is None:
                    # Quoted text didn't match a known team — treat as new/unknown team
                    matched_team = quoted_team
                    player_query = remainder
                    print(f"  (team: {quoted_team})")

        # Unquoted: try fuzzy team prefix matching
        if matched_team is None and team_names:
            team, remainder = split_team_and_player(raw, team_names)
            if team:
                print(f"  (team: {team})")
                player_query = remainder
                matched_team = team

        # Fuzzy search
        match = find_player(player_query, available_names)
        if match:
            confirm = input(f"  -> {match}? (y/n): ").strip().lower()
            if confirm in ("y", "yes", ""):
                return match, _lookup_id(match), matched_team
            # Show alternatives
            alts = find_player(player_query, available_names, return_top_n=5)
            if alts:
                print("  Alternatives:")
                for i, alt in enumerate(alts, 1):
                    print(f"    {i}. {alt}")
                choice = input("  Pick # (or type again): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(alts):
                    picked = alts[int(choice) - 1]
                    return picked, _lookup_id(picked), matched_team
        else:
            print("  No match found. Try again.")


def _show_top_players(board, drafted_ids, n):
    """Display top N available players."""
    available = board[~board["player_id"].isin(drafted_ids)]
    for i, (_, p) in enumerate(available.head(n).iterrows(), 1):
        pos_str = (
            "/".join(p["positions"][:3]) if isinstance(p["positions"], list) else p["best_position"]
        )
        print(f"  {i:>3}. {p['name']:<25} {pos_str:<12} VAR: {p['var']:>6.1f}")


if __name__ == "__main__":
    main()
