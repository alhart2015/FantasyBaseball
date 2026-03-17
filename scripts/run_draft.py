"""Interactive Draft Assistant for Fantasy Baseball.

Usage:
    python scripts/run_draft.py

Pre-requisites:
    1. FanGraphs projection CSVs in data/projections/
    2. Run: python scripts/fetch_positions.py
    3. config/league.yaml with keepers and settings
"""
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.draft.board import build_draft_board, apply_keepers
from fantasy_baseball.draft.tracker import DraftTracker
from fantasy_baseball.draft.balance import CategoryBalance
from fantasy_baseball.draft.search import find_player
from fantasy_baseball.draft.recommender import (
    get_recommendations,
    get_filled_positions,
)
from fantasy_baseball.draft.state import serialize_state, write_state
from fantasy_baseball.web.app import create_app

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
STATE_PATH = PROJECT_ROOT / "data" / "draft_state.json"

FLASK_PORT = 5000


def _start_flask_server(state_path: Path) -> None:
    """Start the Flask dashboard server in a background daemon thread."""
    app = create_app(state_path=state_path)
    server_thread = threading.Thread(
        target=lambda: app.run(
            host="127.0.0.1",
            port=FLASK_PORT,
            use_reloader=False,
            debug=False,
        ),
        daemon=True,
        name="flask-dashboard",
    )
    server_thread.start()
    print(f"Dashboard running at http://127.0.0.1:{FLASK_PORT}")


def _write_dashboard_state(tracker, balance, board, recs, filled):
    """Serialize and atomically write dashboard state to disk."""
    state = serialize_state(
        tracker=tracker,
        balance=balance,
        board=board,
        recommendations=recs,
        filled_positions=filled,
    )
    write_state(state, STATE_PATH)


def main():
    # Load config
    config = load_config(CONFIG_PATH)
    print(f"League {config.league_id} | Draft position: {config.draft_position}")
    print(f"Team: {config.team_name}")
    print(f"Keepers: {len(config.keepers)} players across {config.num_teams} teams")
    print()

    # Build draft board (keep full board for keeper lookups)
    print("Building draft board...")
    full_board = build_draft_board(
        projections_dir=PROJECTIONS_DIR,
        positions_path=POSITIONS_PATH,
        systems=config.projection_systems,
        weights=config.projection_weights or None,
        sgp_overrides=config.sgp_overrides or None,
    )
    board = apply_keepers(full_board, config.keepers)
    print(f"Draft pool: {len(board)} players (after removing {len(config.keepers)} keepers)")
    print()

    # Initialize tracker and balance
    user_keepers = [k for k in config.keepers if k.get("team") == config.team_name]
    rounds = sum(config.roster_slots.values()) - len(user_keepers)
    tracker = DraftTracker(
        num_teams=config.num_teams,
        user_position=config.draft_position,
        rounds=rounds,
    )
    balance = CategoryBalance()

    # Add keeper projections to balance and mark all keepers as drafted
    for keeper in config.keepers:
        is_user = keeper.get("team") == config.team_name
        if is_user:
            rows = full_board[full_board["name"] == keeper["name"]]
            if not rows.empty:
                balance.add_player(rows.iloc[0])
        tracker.draft_player(keeper["name"], is_user=is_user)

    # Start Flask dashboard server
    _start_flask_server(STATE_PATH)

    # Write initial state
    filled = get_filled_positions(tracker.user_roster, board)
    recs = get_recommendations(board, tracker.drafted_players, tracker.user_roster,
                               n=5, filled_positions=filled)
    _write_dashboard_state(tracker, balance, board, recs, filled)

    # Show pre-draft rankings
    print("=" * 70)
    print("TOP 25 AVAILABLE PLAYERS")
    print("=" * 70)
    _show_top_players(board, tracker.drafted_players, 25)
    print()

    # Main draft loop
    while tracker.current_pick <= tracker.total_picks:
        print("=" * 70)
        print(f"ROUND {tracker.current_round} | Pick {tracker.current_pick} "
              f"| Team {tracker.picking_team}", end="")
        if tracker.is_user_pick:
            print(" *** YOUR PICK ***")
        else:
            print()
        print("=" * 70)

        if tracker.is_user_pick:
            _handle_user_pick(board, tracker, balance)
        else:
            _handle_other_pick(board, tracker)

        # Write updated state for the dashboard after every pick
        filled = get_filled_positions(tracker.user_roster, board)
        recs = get_recommendations(board, tracker.drafted_players, tracker.user_roster,
                                   n=5, filled_positions=filled)
        _write_dashboard_state(tracker, balance, board, recs, filled)

        # Show updated top 10
        print()
        _show_top_players(board, tracker.drafted_players, 10)
        print()

        tracker.advance()

    print("\nDraft complete!")
    print("\nYour roster:")
    for name in tracker.user_roster:
        print(f"  {name}")


def _handle_user_pick(board, tracker, balance):
    """Handle the user's draft pick with recommendations."""
    filled = get_filled_positions(tracker.user_roster, board)
    # Calculate gap to NEXT user turn after this one
    save = tracker.current_pick
    tracker.current_pick += 1
    picks_gap = tracker.picks_until_user_turn + 1
    tracker.current_pick = save

    recs = get_recommendations(
        board,
        drafted=tracker.drafted_players,
        user_roster=tracker.user_roster,
        n=5,
        filled_positions=filled,
        picks_until_next=picks_gap,
    )

    # Show recommendations
    print(f"\nPicks until next turn: {picks_gap}")
    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(recs, 1):
        flag = " [NEED]" if rec["need_flag"] else ""
        note = f" ({rec['note']})" if rec["note"] else ""
        print(f"  {i}. {rec['name']} ({rec['best_position']}) "
              f"VAR: {rec['var']:.1f}{flag}{note}")

    # Show category balance
    totals = balance.get_totals()
    warnings = balance.get_warnings()
    print(f"\nROSTER BALANCE:")
    print(f"  R:{totals['R']:.0f} HR:{totals['HR']:.0f} RBI:{totals['RBI']:.0f} "
          f"SB:{totals['SB']:.0f} AVG:{totals['AVG']:.3f}")
    print(f"  W:{totals['W']:.0f} K:{totals['K']:.0f} SV:{totals['SV']:.0f} "
          f"ERA:{totals['ERA']:.2f} WHIP:{totals['WHIP']:.3f}")
    if warnings:
        print(f"  Warnings: {', '.join(warnings)}")

    # Get user input
    name = _get_player_input(board, tracker)
    if name:
        tracker.draft_player(name, is_user=True)
        rows = board[board["name"] == name]
        if not rows.empty:
            balance.add_player(rows.iloc[0])
        print(f"  -> Drafted: {name}")


def _handle_other_pick(board, tracker):
    """Handle another team's pick."""
    name = _get_player_input(board, tracker)
    if name:
        tracker.draft_player(name, is_user=False)
        print(f"  -> Drafted: {name}")


def _get_player_input(board, tracker):
    """Get and fuzzy-match a player name from user input."""
    available_names = board[~board["name"].isin(tracker.drafted_players)]["name"].tolist()
    while True:
        raw = input("\nEnter player name (or 'skip' to skip): ").strip()
        if raw.lower() == "skip":
            return None
        if raw.lower() == "quit":
            sys.exit(0)

        # Try number selection (for recommendations)
        if raw.isdigit():
            idx = int(raw) - 1
            filled = get_filled_positions(tracker.user_roster, board)
            recs = get_recommendations(board, tracker.drafted_players, tracker.user_roster,
                                       n=5, filled_positions=filled)
            if 0 <= idx < len(recs):
                return recs[idx]["name"]

        # Fuzzy search
        match = find_player(raw, available_names)
        if match:
            confirm = input(f"  -> {match}? (y/n): ").strip().lower()
            if confirm in ("y", "yes", ""):
                return match
            # Show alternatives
            alts = find_player(raw, available_names, return_top_n=5)
            if alts:
                print("  Alternatives:")
                for i, alt in enumerate(alts, 1):
                    print(f"    {i}. {alt}")
                choice = input("  Pick # (or type again): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(alts):
                    return alts[int(choice) - 1]
        else:
            print("  No match found. Try again.")


def _show_top_players(board, drafted, n):
    """Display top N available players."""
    available = board[~board["name"].isin(drafted)]
    for i, (_, p) in enumerate(available.head(n).iterrows(), 1):
        pos_str = "/".join(p["positions"][:3]) if isinstance(p["positions"], list) else p["best_position"]
        print(f"  {i:>3}. {p['name']:<25} {pos_str:<12} VAR: {p['var']:>6.1f}")


if __name__ == "__main__":
    main()
