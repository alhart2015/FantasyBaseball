"""Draft strategies for the simulation engine.

Each strategy is a function that receives the current draft state and
returns the name + player_id of the player to pick.

Strategies:
    default          — Pure leverage-weighted recommendation (current behavior).
    nonzero_sv       — Forces a closer (SV >= 20) by a configurable round.
    avg_hedge        — Penalizes hitters whose AVG would drag team below floor.
    three_closers    — Drafts exactly 3 closers at configurable deadline rounds.
    no_punt          — Ensures no category finishes last (SV + AVG floors).
    avg_anchor       — Targets a high-AVG hitter (.285+) in first 3 hitter picks.
    closers_avg      — Combines three_closers + avg_anchor.
    balanced         — Alternates hitter/pitcher picks to diversify risk.
    anti_fragile     — Discounts high-IP pitchers, prefers durable mid-tier arms.
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
# three_closers strategy: how many closers and spacing
THREE_CLOSERS_TARGET = 3
# Draft first closer by round 5, second by round 9, third by round 13
THREE_CLOSERS_DEADLINES = [5, 9, 13]
# no_punt: force a closer by this round if SV == 0, and AVG floor
NO_PUNT_SV_DEADLINE = 9
NO_PUNT_AVG_FLOOR = 0.250
# opportunistic: grab a closer if they've fallen past their ADP
# (effective_pick >= ADP = they "should" already be gone, someone else will grab them)
OPP_CLOSER_ADP_BUFFER = 0  # trigger only when actually past ADP
# avg_anchor: minimum AVG to qualify as an anchor, and deadline
AVG_ANCHOR_MIN = 0.285
AVG_ANCHOR_DEADLINE_HITTER = 3  # must draft anchor within first 3 hitter picks
# balanced: max allowed imbalance between hitters and pitchers
BALANCED_MAX_SKEW = 2
# anti_fragile: IP threshold above which pitchers get discounted
ANTI_FRAGILE_IP_THRESHOLD = 170
ANTI_FRAGILE_DISCOUNT = 0.25  # 25% VAR penalty per 30 IP above threshold


def pick_default(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """Default strategy: take the #1 leverage-weighted recommendation."""
    recs = _get_recs(board, full_board, tracker, balance, config, n=5, **kwargs)
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


def pick_no_punt_opp(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """No-punt with opportunistic closer acquisition.

    Like no_punt, but also grabs a closer early if one is available
    past their ADP (i.e., they're a bargain that could get sniped).
    Still enforces the round 8 deadline as a backstop.
    """
    closer_count = _count_closers(tracker, board, full_board)
    current_round = tracker.current_round
    current_pick = tracker.current_pick

    # Opportunistic: if we don't have a closer yet, check if a good one
    # is available past their ADP (they're falling and could get sniped)
    if closer_count == 0:
        available = board[~board["player_id"].isin(tracker.drafted_ids)]
        closers = available[
            available.apply(lambda r: r.get("sv", 0) >= CLOSER_SV_THRESHOLD, axis=1)
        ]
        if not closers.empty:
            # Effective pick in ADP terms = draft pick + keepers already off the board
            num_keepers = len(tracker.drafted_players) - (current_pick - 1)
            effective_pick = current_pick + num_keepers
            # Find closers whose ADP says they should be gone by now
            falling = closers[effective_pick >= closers["adp"] - OPP_CLOSER_ADP_BUFFER]
            if not falling.empty:
                # Take the best one by VAR
                falling = falling.sort_values("var", ascending=False)
                filled = get_filled_positions(
                    tracker.user_roster_ids, full_board,
                    roster_slots=config.roster_slots,
                )
                for _, best in falling.iterrows():
                    if _can_roster_player(best, filled, config.roster_slots):
                        return best["name"], best["player_id"]

    # Deadline backstop: force a closer by round 8
    if closer_count == 0 and current_round >= NO_PUNT_SV_DEADLINE:
        result = _force_closer(board, tracker, full_board, config)
        if result:
            return result

    # Otherwise: default with AVG floor (same as no_punt)
    recs = _get_recs(board, full_board, tracker, balance, config, n=10, **kwargs)
    if not recs:
        return None, None

    current_h = sum(h.get("h", 0) for h in balance._hitters)
    current_ab = sum(h.get("ab", 0) for h in balance._hitters)

    for rec in recs:
        if rec["player_type"] != "hitter":
            return rec["name"], _lookup_pid(board, rec["name"])

        rows = board[board["name"] == rec["name"]]
        if rows.empty:
            continue
        player = rows.iloc[0]
        new_h = current_h + player.get("h", 0)
        new_ab = current_ab + player.get("ab", 0)
        projected_avg = new_h / new_ab if new_ab > 0 else 0

        if projected_avg >= NO_PUNT_AVG_FLOOR or current_ab == 0:
            return rec["name"], _lookup_pid(board, rec["name"])

    return recs[0]["name"], _lookup_pid(board, recs[0]["name"])


def pick_three_closers(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """Draft exactly 3 closers, spaced across the draft.

    Uses deadlines to ensure closers are drafted by rounds 6, 10, and 14.
    Between deadlines, falls back to default leverage-weighted picks.
    When a closer is needed, picks the best available closer by VAR
    (not ADP), so we get the highest-value closer remaining.
    """
    # Count how many closers we already have
    closer_count = 0
    for pid in tracker.user_roster_ids:
        rows = board[board["player_id"] == pid]
        if rows.empty:
            rows = full_board[full_board["player_id"] == pid]
        if not rows.empty and rows.iloc[0].get("sv", 0) >= CLOSER_SV_THRESHOLD:
            closer_count += 1

    current_round = tracker.current_round

    # Check if we need to force a closer this pick
    need_closer = False
    if closer_count < THREE_CLOSERS_TARGET:
        # Which deadline applies?
        deadline_idx = closer_count  # 0th closer -> deadline[0], etc.
        if deadline_idx < len(THREE_CLOSERS_DEADLINES):
            deadline = THREE_CLOSERS_DEADLINES[deadline_idx]
            if current_round >= deadline:
                need_closer = True

    if need_closer:
        available = board[~board["player_id"].isin(tracker.drafted_ids)]
        closers = available[
            available.apply(lambda r: r.get("sv", 0) >= CLOSER_SV_THRESHOLD, axis=1)
        ]
        if not closers.empty:
            # Pick by best VAR (value above replacement)
            closers = closers.sort_values("var", ascending=False)
            filled = get_filled_positions(
                tracker.user_roster_ids, full_board,
                roster_slots=config.roster_slots,
            )
            for _, best in closers.iterrows():
                if _can_roster_player(best, filled, config.roster_slots):
                    return best["name"], best["player_id"]

    # Otherwise, use the default recommendation engine.
    # But also let the default engine pick a closer naturally if it wants —
    # we only force when deadlines hit.
    return pick_default(board, full_board, tracker, balance, config,
                        team_filled, **kwargs)


def _count_closers(tracker, board, full_board):
    """Count how many closers are on the user's roster."""
    count = 0
    for pid in tracker.user_roster_ids:
        rows = board[board["player_id"] == pid]
        if rows.empty:
            rows = full_board[full_board["player_id"] == pid]
        if not rows.empty and rows.iloc[0].get("sv", 0) >= CLOSER_SV_THRESHOLD:
            count += 1
    return count


def _count_hitters(tracker, board, full_board):
    """Count hitters on the user's roster."""
    count = 0
    for pid in tracker.user_roster_ids:
        rows = board[board["player_id"] == pid]
        if rows.empty:
            rows = full_board[full_board["player_id"] == pid]
        if not rows.empty and rows.iloc[0].get("player_type") == "hitter":
            count += 1
    return count


def _count_pitchers(tracker, board, full_board):
    """Count pitchers on the user's roster."""
    return len(tracker.user_roster) - _count_hitters(tracker, board, full_board)


def _force_closer(board, tracker, full_board, config):
    """Pick the best available closer by VAR. Returns (name, pid) or None."""
    available = board[~board["player_id"].isin(tracker.drafted_ids)]
    closers = available[
        available.apply(lambda r: r.get("sv", 0) >= CLOSER_SV_THRESHOLD, axis=1)
    ]
    if closers.empty:
        return None
    closers = closers.sort_values("var", ascending=False)
    filled = get_filled_positions(
        tracker.user_roster_ids, full_board,
        roster_slots=config.roster_slots,
    )
    for _, best in closers.iterrows():
        if _can_roster_player(best, filled, config.roster_slots):
            return best["name"], best["player_id"]
    return None


def _get_recs(board, full_board, tracker, balance, config, n=10, **kwargs):
    """Get leverage-weighted recommendations (shared helper)."""
    filled = get_filled_positions(
        tracker.user_roster_ids, full_board,
        roster_slots=config.roster_slots,
    )
    leverage = calculate_draft_leverage(
        balance.get_totals(),
        picks_made=len(tracker.user_roster),
        total_picks=kwargs.get("total_rounds", 22),
    )
    return get_recommendations(
        board, drafted=tracker.drafted_ids,
        user_roster=tracker.user_roster,
        n=n, filled_positions=filled,
        roster_slots=config.roster_slots,
        num_teams=config.num_teams,
        draft_leverage=leverage,
        scoring_mode=kwargs.get("scoring_mode", "var"),
    )


def pick_no_punt(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """Ensure no category finishes dead last.

    Forces a closer by round 8 if SV == 0.
    Skips low-AVG hitters if team AVG is below the floor.
    """
    closer_count = _count_closers(tracker, board, full_board)
    current_round = tracker.current_round

    # Force at least one closer by deadline
    if closer_count == 0 and current_round >= NO_PUNT_SV_DEADLINE:
        result = _force_closer(board, tracker, full_board, config)
        if result:
            return result

    # Get recommendations, then filter for AVG floor
    recs = _get_recs(board, full_board, tracker, balance, config, n=10, **kwargs)
    if not recs:
        return None, None

    current_h = sum(h.get("h", 0) for h in balance._hitters)
    current_ab = sum(h.get("ab", 0) for h in balance._hitters)

    for rec in recs:
        if rec["player_type"] != "hitter":
            return rec["name"], _lookup_pid(board, rec["name"])

        rows = board[board["name"] == rec["name"]]
        if rows.empty:
            continue
        player = rows.iloc[0]
        new_h = current_h + player.get("h", 0)
        new_ab = current_ab + player.get("ab", 0)
        projected_avg = new_h / new_ab if new_ab > 0 else 0

        if projected_avg >= NO_PUNT_AVG_FLOOR or current_ab == 0:
            return rec["name"], _lookup_pid(board, rec["name"])

    return recs[0]["name"], _lookup_pid(board, recs[0]["name"])


def pick_avg_anchor(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """Target a high-AVG hitter (.285+) in the first 3 hitter picks.

    Once the anchor is secured, falls back to default.
    """
    hitter_count = _count_hitters(tracker, board, full_board)

    # Check if we already have an AVG anchor
    has_anchor = False
    for pid in tracker.user_roster_ids:
        rows = board[board["player_id"] == pid]
        if rows.empty:
            rows = full_board[full_board["player_id"] == pid]
        if not rows.empty:
            p = rows.iloc[0]
            if p.get("player_type") == "hitter" and p.get("avg", 0) >= AVG_ANCHOR_MIN:
                has_anchor = True
                break

    # If no anchor and we're within the hitter deadline, prefer high-AVG hitters
    if not has_anchor and hitter_count < AVG_ANCHOR_DEADLINE_HITTER:
        recs = _get_recs(board, full_board, tracker, balance, config, n=15, **kwargs)
        if recs:
            # Try to find a high-AVG hitter in the recommendations
            for rec in recs:
                if rec["player_type"] != "hitter":
                    continue
                rows = board[board["name"] == rec["name"]]
                if rows.empty:
                    continue
                if rows.iloc[0].get("avg", 0) >= AVG_ANCHOR_MIN:
                    return rec["name"], _lookup_pid(board, rec["name"])

            # If none in recs, search the board for the best high-AVG hitter
            available = board[~board["player_id"].isin(tracker.drafted_ids)]
            anchors = available[
                (available["player_type"] == "hitter") &
                (available["avg"] >= AVG_ANCHOR_MIN)
            ].sort_values("var", ascending=False)
            filled = get_filled_positions(
                tracker.user_roster_ids, full_board,
                roster_slots=config.roster_slots,
            )
            for _, best in anchors.head(5).iterrows():
                if _can_roster_player(best, filled, config.roster_slots):
                    return best["name"], best["player_id"]

    # Fall back to default
    return pick_default(board, full_board, tracker, balance, config,
                        team_filled, **kwargs)


def pick_closers_avg(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """Combine three_closers + avg_anchor.

    Closer deadlines take priority. Between deadlines, try to land an
    AVG anchor in the first 3 hitter picks. Otherwise, default.
    """
    # Check closer deadlines first (highest priority)
    closer_count = _count_closers(tracker, board, full_board)
    current_round = tracker.current_round

    if closer_count < THREE_CLOSERS_TARGET:
        deadline_idx = closer_count
        if deadline_idx < len(THREE_CLOSERS_DEADLINES):
            deadline = THREE_CLOSERS_DEADLINES[deadline_idx]
            if current_round >= deadline:
                result = _force_closer(board, tracker, full_board, config)
                if result:
                    return result

    # Then try AVG anchor
    return pick_avg_anchor(board, full_board, tracker, balance, config,
                           team_filled, **kwargs)


def pick_balanced(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """Alternate hitter/pitcher picks to diversify risk.

    If pitchers lead hitters by more than BALANCED_MAX_SKEW, force a hitter.
    If hitters lead pitchers by more than BALANCED_MAX_SKEW, force a pitcher.
    """
    n_hitters = _count_hitters(tracker, board, full_board)
    n_pitchers = _count_pitchers(tracker, board, full_board)

    recs = _get_recs(board, full_board, tracker, balance, config, n=15, **kwargs)
    if not recs:
        return None, None

    force_type = None
    if n_pitchers - n_hitters > BALANCED_MAX_SKEW:
        force_type = "hitter"
    elif n_hitters - n_pitchers > BALANCED_MAX_SKEW:
        force_type = "pitcher"

    if force_type:
        for rec in recs:
            if rec["player_type"] == force_type:
                return rec["name"], _lookup_pid(board, rec["name"])

    # No imbalance — take the best recommendation
    return recs[0]["name"], _lookup_pid(board, recs[0]["name"])


def pick_anti_fragile(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """Prefer durable mid-tier pitchers over fragile aces.

    Applies a VAR discount to pitchers with high IP projections,
    then re-ranks recommendations.
    """
    recs = _get_recs(board, full_board, tracker, balance, config, n=15, **kwargs)
    if not recs:
        return None, None

    # Re-score recommendations with durability discount
    scored = []
    for rec in recs:
        rows = board[board["name"] == rec["name"]]
        if rows.empty:
            scored.append((rec, rec.get("var", 0)))
            continue
        player = rows.iloc[0]
        var = player.get("var", 0)

        if player.get("player_type") == "pitcher":
            ip = player.get("ip", 0)
            if ip > ANTI_FRAGILE_IP_THRESHOLD:
                excess_ip = ip - ANTI_FRAGILE_IP_THRESHOLD
                penalty = (excess_ip / 30.0) * ANTI_FRAGILE_DISCOUNT
                var = var * (1.0 - penalty)

        scored.append((rec, var))

    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0]
    return best["name"], _lookup_pid(board, best["name"])


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
    "three_closers": pick_three_closers,
    "no_punt": pick_no_punt,
    "no_punt_opp": pick_no_punt_opp,
    "avg_anchor": pick_avg_anchor,
    "closers_avg": pick_closers_avg,
    "balanced": pick_balanced,
    "anti_fragile": pick_anti_fragile,
}
