"""Draft strategies for the simulation engine.

Each strategy is a function that receives the current draft state and
returns the name + player_id of the player to pick.

Strategies:
    default     — Pure leverage-weighted recommendation (current behavior).
    nonzero_sv  — Same as default, but forces a closer (SV >= 20) by a
                  configurable round if none has been drafted yet.
    avg_hedge   — Same as default, but applies a penalty to hitters whose
                  AVG would drag the team below a configurable floor.
"""
import pandas as pd
from fantasy_baseball.draft.balance import CategoryBalance, calculate_draft_leverage
from fantasy_baseball.draft.recommender import get_recommendations, get_filled_positions
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.utils.positions import can_fill_slot

# Minimum projected SV to count as a closer
CLOSER_SV_THRESHOLD = 20
# Draft a closer by this round if you have none
CLOSER_DEADLINE_ROUND = 10
# Don't let team AVG fall below this
AVG_FLOOR = 0.255


def pick_default(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """Default strategy: take the #1 leverage-weighted recommendation."""
    filled = get_filled_positions(
        tracker.user_roster_ids, full_board,
        roster_slots=config.roster_slots,
    )
    leverage = calculate_draft_leverage(
        balance.get_totals(),
        picks_made=len(tracker.user_roster),
        total_picks=kwargs.get("total_rounds", 22),
    )
    recs = get_recommendations(
        board, drafted=tracker.drafted_ids,
        user_roster=tracker.user_roster,
        n=5, filled_positions=filled,
        roster_slots=config.roster_slots,
        num_teams=config.num_teams,
        draft_leverage=leverage,
    )
    if recs:
        return recs[0]["name"], _lookup_pid(board, recs[0]["name"])
    return None, None


def pick_nonzero_sv(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """Force a closer by CLOSER_DEADLINE_ROUND if none has been drafted."""
    # Check if we already have a closer
    has_closer = False
    for pid in tracker.user_roster_ids:
        rows = board[board["player_id"] == pid]
        if rows.empty:
            rows = full_board[full_board["player_id"] == pid]
        if not rows.empty and rows.iloc[0].get("sv", 0) >= CLOSER_SV_THRESHOLD:
            has_closer = True
            break

    current_round = tracker.current_round

    # If no closer and we're at or past the deadline, force one
    if not has_closer and current_round >= CLOSER_DEADLINE_ROUND:
        available = board[~board["player_id"].isin(tracker.drafted_ids)]
        closers = available[
            available.apply(lambda r: r.get("sv", 0) >= CLOSER_SV_THRESHOLD, axis=1)
        ]
        if not closers.empty:
            # Pick the closer with the best ADP (most drafted consensus)
            closers = closers.sort_values("adp", ascending=True)
            best = closers.iloc[0]
            # Verify they can be rostered
            filled = get_filled_positions(
                tracker.user_roster_ids, full_board,
                roster_slots=config.roster_slots,
            )
            if _can_roster_player(best, filled, config.roster_slots):
                return best["name"], best["player_id"]

    # Otherwise, fall back to default
    return pick_default(board, full_board, tracker, balance, config,
                        team_filled, **kwargs)


def pick_avg_hedge(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """Penalize hitters that would drag team AVG below the floor."""
    filled = get_filled_positions(
        tracker.user_roster_ids, full_board,
        roster_slots=config.roster_slots,
    )
    leverage = calculate_draft_leverage(
        balance.get_totals(),
        picks_made=len(tracker.user_roster),
        total_picks=kwargs.get("total_rounds", 22),
    )
    recs = get_recommendations(
        board, drafted=tracker.drafted_ids,
        user_roster=tracker.user_roster,
        n=10, filled_positions=filled,
        roster_slots=config.roster_slots,
        num_teams=config.num_teams,
        draft_leverage=leverage,
    )
    if not recs:
        return None, None

    totals = balance.get_totals()
    current_h = sum(h.get("h", 0) for h in balance._hitters)
    current_ab = sum(h.get("ab", 0) for h in balance._hitters)

    for rec in recs:
        if rec["player_type"] != "hitter":
            # Pitchers don't affect AVG — always acceptable
            return rec["name"], _lookup_pid(board, rec["name"])

        # Simulate what team AVG would be if we add this hitter
        rows = board[board["name"] == rec["name"]]
        if rows.empty:
            continue
        player = rows.iloc[0]
        new_h = current_h + player.get("h", 0)
        new_ab = current_ab + player.get("ab", 0)
        projected_avg = new_h / new_ab if new_ab > 0 else 0

        if projected_avg >= AVG_FLOOR or current_ab == 0:
            return rec["name"], _lookup_pid(board, rec["name"])
        # else: skip this low-AVG hitter and try the next rec

    # If all hitters would tank AVG, take the best one anyway
    return recs[0]["name"], _lookup_pid(board, recs[0]["name"])


def _lookup_pid(board, name):
    rows = board[board["name"] == name]
    if not rows.empty:
        return rows.iloc[0]["player_id"]
    return name + "::unknown"


def _can_roster_player(player, filled, roster_slots):
    positions = player["positions"]
    for pos, total in roster_slots.items():
        if pos == "IL":
            continue
        if filled.get(pos, 0) < total and can_fill_slot(positions, pos):
            return True
    return False


STRATEGIES = {
    "default": pick_default,
    "nonzero_sv": pick_nonzero_sv,
    "avg_hedge": pick_avg_hedge,
}
