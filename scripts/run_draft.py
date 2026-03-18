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
from fantasy_baseball.draft.search import find_player, split_team_and_player
from fantasy_baseball.draft.recommender import (
    get_recommendations,
    get_filled_positions,
    get_roster_by_position,
)
from fantasy_baseball.draft.state import serialize_state, serialize_board, write_state, write_board
from fantasy_baseball.web.app import create_app

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
STATE_PATH = PROJECT_ROOT / "data" / "draft_state.json"
BOARD_PATH = PROJECT_ROOT / "data" / "draft_state_board.json"

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


def _write_dashboard_state(tracker, balance, board, recs, filled,
                           roster_slots=None, roster_by_pos=None, teams=None):
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
        roster_slots=config.roster_slots or None,
        num_teams=config.num_teams,
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
    from fantasy_baseball.utils.name_utils import normalize_name
    for keeper in config.keepers:
        is_user = keeper.get("team") == config.team_name
        norm = normalize_name(keeper["name"])
        matches = full_board[full_board["name_normalized"] == norm]
        if not matches.empty:
            # Pick the highest-VAR match (the real keeper, not a namesake)
            best = matches.loc[matches["var"].idxmax()]
            if is_user:
                balance.add_player(best)
            tracker.draft_player(best["name"], is_user=is_user,
                                 player_id=best["player_id"])
        else:
            tracker.draft_player(keeper["name"], is_user=is_user)

    # Write the full board once (clients fetch via /api/board on page load)
    board_data = serialize_board(board)
    write_board(board_data, BOARD_PATH)
    print(f"Board snapshot written ({len(board_data)} players)")

    # Start Flask dashboard server
    _start_flask_server(STATE_PATH)

    # Write initial state (use full_board for roster lookups so keepers are found)
    filled = get_filled_positions(tracker.user_roster_ids, full_board,
                                  roster_slots=config.roster_slots)
    by_pos = get_roster_by_position(tracker.user_roster_ids, full_board,
                                    roster_slots=config.roster_slots)
    recs = get_recommendations(board, tracker.drafted_ids, tracker.user_roster,
                               n=5, filled_positions=filled,
                               roster_slots=config.roster_slots,
                               num_teams=config.num_teams)
    _write_dashboard_state(tracker, balance, board, recs, filled,
                           roster_slots=config.roster_slots,
                           roster_by_pos=by_pos,
                           teams=config.teams)

    # Show pre-draft rankings
    print("=" * 70)
    print("TOP 25 AVAILABLE PLAYERS")
    print("=" * 70)
    _show_top_players(board, tracker.drafted_ids, 25)
    print()

    # Build team name list for input parsing
    team_names = list(config.teams.values()) if config.teams else []

    # Main draft loop
    try:
        while tracker.current_pick <= tracker.total_picks:
            team_num = tracker.picking_team
            team_label = config.teams.get(team_num, f"Team {team_num}")
            print("=" * 70)
            print(f"ROUND {tracker.current_round} | Pick {tracker.current_pick} "
                  f"| {team_label}", end="")
            if tracker.is_user_pick:
                print(" *** YOUR PICK ***")
            else:
                print()
            print("=" * 70)

            if tracker.is_user_pick:
                _handle_user_pick(board, full_board, tracker, balance,
                                  roster_slots=config.roster_slots,
                                  num_teams=config.num_teams)
            else:
                _handle_other_pick(board, full_board, tracker, balance,
                                   team_names, config.team_name)

            # Advance to next pick so dashboard shows upcoming pick, not the one just made
            tracker.advance()

            # Write updated state for the dashboard after every pick
            filled = get_filled_positions(tracker.user_roster_ids, full_board,
                                          roster_slots=config.roster_slots)
            by_pos = get_roster_by_position(tracker.user_roster_ids, full_board,
                                            roster_slots=config.roster_slots)
            recs = get_recommendations(board, tracker.drafted_ids, tracker.user_roster,
                                       n=5, filled_positions=filled,
                                       roster_slots=config.roster_slots,
                                       num_teams=config.num_teams)
            _write_dashboard_state(tracker, balance, board, recs, filled,
                                   roster_slots=config.roster_slots,
                                   roster_by_pos=by_pos,
                                   teams=config.teams)

            # Show updated top 10
            print()
            _show_top_players(board, tracker.drafted_ids, 10)
            print()

        print("\nDraft complete!")
    except (KeyboardInterrupt, EOFError):
        print("\n\nDraft paused. State saved. Re-run to resume.")
    print("\nYour roster:")
    for name in tracker.user_roster:
        print(f"  {name}")


def _handle_user_pick(board, full_board, tracker, balance, roster_slots=None,
                      num_teams=None):
    """Handle the user's draft pick with recommendations."""
    filled = get_filled_positions(tracker.user_roster_ids, full_board,
                                  roster_slots=roster_slots)
    # Calculate gap to NEXT user turn after this one.
    # Use a local variable instead of mutating tracker.current_pick so that
    # an exception cannot leave the tracker in a corrupted state.
    peek_pick = tracker.current_pick + 1
    picks_gap = 1  # count the peek step itself
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
    era_str = f"{totals['ERA']:.2f}" if totals["ERA"] is not None else "N/A"
    whip_str = f"{totals['WHIP']:.3f}" if totals["WHIP"] is not None else "N/A"
    print(f"  W:{totals['W']:.0f} K:{totals['K']:.0f} SV:{totals['SV']:.0f} "
          f"ERA:{era_str} WHIP:{whip_str}")
    if warnings:
        print(f"  Warnings: {', '.join(warnings)}")

    # Get user input
    name, pid, _team = _get_player_input(board, tracker, current_recs=recs)
    if name:
        tracker.draft_player(name, is_user=True, player_id=pid)
        rows = board[board["player_id"] == pid] if pid else board[board["name"] == name]
        if not rows.empty:
            balance.add_player(rows.iloc[0])
        print(f"  -> Drafted: {name}")


def _handle_other_pick(board, full_board, tracker, balance,
                       team_names=None, user_team_name=None):
    """Handle another team's pick (or a traded pick for the user's team)."""
    name, pid, matched_team = _get_player_input(board, tracker,
                                                team_names=team_names)
    if name:
        is_user = (matched_team is not None
                   and user_team_name is not None
                   and matched_team == user_team_name)
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


def _get_player_input(board, tracker, team_names=None, current_recs=None):
    """Get and fuzzy-match a player name from user input.

    Returns (name, player_id, team) or (None, None, None).

    *team* is the matched team-name prefix (e.g. "Hart of the Order") if
    the user typed one, otherwise None.

    If *current_recs* is provided, number selection uses those recs directly
    (matching what was displayed) instead of regenerating them.

    If *team_names* is provided, the input may optionally start with a team
    name prefix (e.g. "hello peanuts logan webb").  The team prefix is
    stripped before the player fuzzy search.
    """
    available = board[~board["player_id"].isin(tracker.drafted_ids)]
    available_names = available["name"].tolist()

    def _lookup_id(name):
        """Find the player_id for a matched name."""
        rows = available[available["name"] == name]
        if not rows.empty:
            return rows.iloc[0]["player_id"]
        return name + "::unknown"

    while True:
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
                recs = get_recommendations(board, tracker.drafted_ids,
                                           tracker.user_roster, n=5,
                                           filled_positions=filled)
            if 0 <= idx < len(recs):
                return recs[idx]["name"], _lookup_id(recs[idx]["name"]), None

        # Try to split off a team-name prefix
        player_query = raw
        matched_team = None
        if team_names:
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
        pos_str = "/".join(p["positions"][:3]) if isinstance(p["positions"], list) else p["best_position"]
        print(f"  {i:>3}. {p['name']:<25} {pos_str:<12} VAR: {p['var']:>6.1f}")


if __name__ == "__main__":
    main()
