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
from fantasy_baseball.draft.balance import CategoryBalance, calculate_draft_leverage
from fantasy_baseball.draft.search import find_player, split_team_and_player
from fantasy_baseball.draft.recommender import (
    get_recommendations,
    get_filled_positions,
    get_roster_by_position,
)
from fantasy_baseball.draft.state import serialize_state, serialize_board, write_state, write_board
from fantasy_baseball.draft.projections import run_projections, reconstruct_rosters_from_draft
from fantasy_baseball.draft.strategy import (
    CLOSER_SV_THRESHOLD, NO_PUNT_SV_DEADLINE, NO_PUNT_AVG_FLOOR,
    OPP_CLOSER_ADP_BUFFER,
)
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


def _get_draft_leverage(balance, tracker):
    """Compute category leverage weights from the user's current roster balance."""
    totals = balance.get_totals()
    picks_made = len(tracker.user_roster)
    total_picks = tracker.rounds
    return calculate_draft_leverage(totals, picks_made, total_picks)


def _write_dashboard_state(tracker, balance, board, recs, filled,
                           roster_slots=None, roster_by_pos=None, teams=None,
                           projection_data=None):
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
    if projection_data is not None:
        state["projections"] = projection_data
    write_state(state, STATE_PATH)


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Fantasy Baseball Draft Assistant")
    parser.add_argument("--mock", action="store_true",
                        help="Mock draft mode: no keepers, override position/teams")
    parser.add_argument("--position", "-p", type=int, default=None,
                        help="Draft position (mock mode, default: from config)")
    parser.add_argument("--teams", "-t", type=int, default=None,
                        help="Number of teams (mock mode, default: from config)")
    return parser.parse_args()


def main():
    args = _parse_args()

    # Load config
    config = load_config(CONFIG_PATH)

    # Mock mode overrides
    mock = args.mock
    num_teams = args.teams or config.num_teams
    draft_position = args.position or config.draft_position
    keepers = [] if mock else config.keepers

    if mock:
        print(f"MOCK DRAFT | Position {draft_position} of {num_teams}")
        print(f"Strategy: no_punt | No keepers")
    else:
        print(f"League {config.league_id} | Draft position: {draft_position}")
        print(f"Team: {config.team_name}")
        print(f"Keepers: {len(keepers)} players across {num_teams} teams")
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
        num_teams=num_teams,
    )
    board = apply_keepers(full_board, keepers)
    print(f"Draft pool: {len(board)} players (after removing {len(keepers)} keepers)")
    print()

    # Initialize tracker and balance
    user_keepers = [k for k in keepers if k.get("team") == config.team_name]
    rounds = sum(config.roster_slots.values()) - len(user_keepers)
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
    leverage = _get_draft_leverage(balance, tracker)
    recs = get_recommendations(board, tracker.drafted_ids, tracker.user_roster,
                               n=5, filled_positions=filled,
                               roster_slots=config.roster_slots,
                               num_teams=num_teams,
                               draft_leverage=leverage)
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
    last_projected_round = 0
    projection_data = None

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
                                  num_teams=num_teams,
                                  team_names=team_names,
                                  user_team_name=config.team_name)
            else:
                # Peek at input — if "mine", treat as user pick (traded pick)
                raw = input("\nPick (player, 'team player', or 'mine'): ").strip()
                if raw.lower() == "mine":
                    print(f"  >>> YOUR PICK (traded to you)")
                    _handle_user_pick(board, full_board, tracker, balance,
                                      roster_slots=config.roster_slots,
                                      num_teams=num_teams)
                else:
                    # Process the input that was already typed
                    _handle_other_pick(board, full_board, tracker, balance,
                                       team_names, config.team_name,
                                       prefilled_input=raw)

            # Advance to next pick so dashboard shows upcoming pick, not the one just made
            tracker.advance()

            # Write updated state for the dashboard after every pick
            filled = get_filled_positions(tracker.user_roster_ids, full_board,
                                          roster_slots=config.roster_slots)
            by_pos = get_roster_by_position(tracker.user_roster_ids, full_board,
                                            roster_slots=config.roster_slots)
            leverage = _get_draft_leverage(balance, tracker)
            recs = get_recommendations(board, tracker.drafted_ids, tracker.user_roster,
                                       n=5, filled_positions=filled,
                                       roster_slots=config.roster_slots,
                                       num_teams=num_teams,
                                       draft_leverage=leverage)

            # Run projections at the end of each completed round.
            # After advance(), current_pick points to the NEXT pick.
            # A round just completed if the previous pick was the last in its round,
            # i.e. (current_pick - 1) is exactly divisible by num_teams.
            just_finished_round = (tracker.current_pick - 1) // num_teams
            if just_finished_round > last_projected_round and just_finished_round >= 1:
                last_projected_round = just_finished_round
                print(f"\n  Running projected standings (round {just_finished_round} complete)...")
                team_rosters = reconstruct_rosters_from_draft(
                    config, full_board, tracker,
                    num_teams_override=num_teams)
                projection_data = run_projections(
                    team_rosters, config.roster_slots, full_board,
                    num_teams, iterations=1000,
                )
                # Annotate with team names and user flag for dashboard
                user_num = draft_position
                for s in projection_data["standings"]:
                    s["team_name"] = config.teams.get(
                        s["team_num"], f"Team {s['team_num']}")
                    s["is_user"] = s["team_num"] == user_num

                # Print compact standings
                print(f"  {'Team':<28} {'Med':>4} {'Win%':>5} {'Top3':>5}")
                print(f"  {'-'*46}")
                for s in projection_data["standings"]:
                    marker = " <<<" if s["is_user"] else ""
                    print(f"  {s['team_name']:<28} {s['median']:>4} "
                          f"{s['win_pct']:>4.1f}% {s['top3_pct']:>4.1f}%{marker}")

            _write_dashboard_state(tracker, balance, board, recs, filled,
                                   roster_slots=config.roster_slots,
                                   roster_by_pos=by_pos,
                                   teams=config.teams,
                                   projection_data=projection_data)

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
                      num_teams=None, team_names=None, user_team_name=None):
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

    leverage = _get_draft_leverage(balance, tracker)
    recs = get_recommendations(
        board,
        drafted=tracker.drafted_ids,
        user_roster=tracker.user_roster,
        n=5,
        filled_positions=filled,
        picks_until_next=picks_gap,
        roster_slots=roster_slots,
        num_teams=num_teams,
        draft_leverage=leverage,
    )

    # --- No-punt strategy logic ---
    strategy_alerts = []

    # Check if we have a closer
    has_closer = False
    for pid in tracker.user_roster_ids:
        rows = board[board["player_id"] == pid]
        if rows.empty:
            rows = full_board[full_board["player_id"] == pid]
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
            # Effective pick in ADP terms = draft pick + keepers already off the board
            num_keepers = len(tracker.drafted_players) - (current_pick - 1)
            effective_pick = current_pick + num_keepers

            if effective_pick >= closer_adp - OPP_CLOSER_ADP_BUFFER:
                # Closer is falling past their ADP — opportunity
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
                    f"(ADP {closer_adp:.0f}) is falling — grab before someone else does"
                )
            elif tracker.current_round >= NO_PUNT_SV_DEADLINE:
                # Deadline: must draft a closer now
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

    # Check AVG floor and flag low-AVG hitters
    current_h = sum(h.get("h", 0) for h in balance._hitters)
    current_ab = sum(h.get("ab", 0) for h in balance._hitters)
    avg_warnings = []
    if current_ab > 0:
        for rec in recs:
            if rec["player_type"] != "hitter":
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

    # Reorder: push low-AVG hitters below non-flagged recs
    if avg_warnings:
        safe_recs = [r for r in recs if r["name"] not in avg_warnings]
        risky_recs = [r for r in recs if r["name"] in avg_warnings]
        recs = safe_recs + risky_recs

    # Build final recommendation list: closer on top if triggered
    if closer_rec:
        # Check if the closer is already in recs
        closer_in_recs = any(r["name"] == closer_rec["name"] for r in recs)
        if closer_in_recs:
            recs = [r for r in recs if r["name"] != closer_rec["name"]]
        recs = [closer_rec] + recs[:4]

    # Show recommendations
    print(f"\nPicks until next turn: {picks_gap}")

    if strategy_alerts:
        print()
        for alert in strategy_alerts:
            print(f"  >>> {alert}")

    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(recs, 1):
        flag = " [NEED]" if rec["need_flag"] else ""
        note = f" ({rec['note']})" if rec["note"] else ""
        score_str = f" score: {rec['score']:.1f}" if rec.get("score") is not None else ""
        avg_warn = " [LOW AVG]" if rec["name"] in avg_warnings else ""
        print(f"  {i}. {rec['name']} ({rec['best_position']}) "
              f"VAR: {rec['var']:.1f}{score_str}{flag}{avg_warn}{note}")

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

    # Get user input — pass team_names so "spacemen gausman" is recognized as traded away
    name, pid, matched_team = _get_player_input(
        board, tracker, current_recs=recs, team_names=team_names,
    )
    if name:
        # If a different team was specified, this pick was traded away
        traded_away = (matched_team is not None
                       and user_team_name is not None
                       and matched_team != user_team_name)
        if traded_away:
            tracker.draft_player(name, is_user=False, player_id=pid)
            print(f"  -> Drafted: {name} ({matched_team} via trade)")
        else:
            tracker.draft_player(name, is_user=True, player_id=pid)
            rows = board[board["player_id"] == pid] if pid else board[board["name"] == name]
            if not rows.empty:
                balance.add_player(rows.iloc[0])
            print(f"  -> Drafted: {name}")


def _handle_other_pick(board, full_board, tracker, balance,
                       team_names=None, user_team_name=None,
                       prefilled_input=None):
    """Handle another team's pick (or a traded pick for the user's team).

    If prefilled_input is provided, it's used as the first input line
    instead of prompting (for when the main loop already read input to
    check for 'mine').
    """
    name, pid, matched_team = _get_player_input(board, tracker,
                                                team_names=team_names,
                                                prefilled_input=prefilled_input)
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


def _get_player_input(board, tracker, team_names=None, current_recs=None,
                      prefilled_input=None):
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
