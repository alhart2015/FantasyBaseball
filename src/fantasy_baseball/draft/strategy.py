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
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD
from fantasy_baseball.utils.positions import can_fill_slot
# Draft a closer by this round if you have none
CLOSER_DEADLINE_ROUND = 10
# Don't let team AVG fall below this
AVG_FLOOR = 0.255
# n-closers strategies: target counts and spaced deadlines
TWO_CLOSERS_TARGET = 2
TWO_CLOSERS_DEADLINES = [8, 14]
THREE_CLOSERS_TARGET = 3
THREE_CLOSERS_DEADLINES = [5, 9, 13]
FOUR_CLOSERS_TARGET = 4
FOUR_CLOSERS_DEADLINES = [5, 8, 12, 16]
# no_punt: AVG floor and dynamic SV danger zone
NO_PUNT_SV_DEADLINE = 9  # legacy fallback if team_rosters not available
NO_PUNT_AVG_FLOOR = 0.250
# How many teams with closers before we start worrying about SV rank
NO_PUNT_SV_MIN_TEAMS_WITH_CLOSERS = 3
# Bottom N in SV triggers a closer pick (2 = last or second-to-last)
NO_PUNT_SV_DANGER_ZONE = 2
# opportunistic: grab a closer if they've fallen past their ADP
# (effective_pick >= ADP = they "should" already be gone, someone else will grab them)
OPP_CLOSER_ADP_BUFFER = 0  # trigger only when actually past ADP
# no_punt_stagger: staggered closer deadlines + no_punt category protection
NO_PUNT_STAGGER_TARGET = 3
NO_PUNT_STAGGER_DEADLINES = [13, 17, 20]
# no_punt_cap3: staggered deadlines + hard cap at 3 closers
# Deadlines are late backstops — VONA handles urgency timing.
# These only fire if VONA somehow misses a closer run.
NO_PUNT_CAP3_TARGET = 3
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
    picks_until_next = getattr(tracker, "picks_until_next_turn", None)
    recs = get_recommendations(
        board, drafted=tracker.drafted_ids,
        user_roster=tracker.user_roster,
        n=10, filled_positions=filled,
        picks_until_next=picks_until_next,
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
    """No-punt with dynamic SV monitoring and opportunistic closer grabs.

    Watches projected SV standings across all teams.  When our team
    would finish in the bottom NO_PUNT_SV_DANGER_ZONE (default: last
    or second-to-last), forces a closer pick.  Also grabs closers
    opportunistically if they've fallen past their ADP.

    Requires ``team_rosters`` in kwargs (dict of team_num -> [player_ids]).
    Falls back to the legacy round-based deadline if not provided.
    """
    current_round = tracker.current_round
    current_pick = tracker.current_pick
    team_rosters = kwargs.get("team_rosters")

    # Dynamic SV check: are we in danger of finishing last in saves?
    need_closer = False
    if team_rosters:
        need_closer = _sv_in_danger(
            tracker, board, full_board, team_rosters, config.num_teams,
        )
    else:
        # Legacy fallback: force at least 1 closer by deadline
        closer_count = _count_closers(tracker, board, full_board)
        if closer_count == 0 and current_round >= NO_PUNT_SV_DEADLINE:
            need_closer = True

    if need_closer:
        result = _force_closer(board, tracker, full_board, config)
        if result:
            return result

    # Opportunistic: grab a closer falling past ADP, but only if we're
    # still in SV danger or haven't drafted any closers yet.
    closer_count = _count_closers(tracker, board, full_board)
    if need_closer or closer_count == 0:
        available = board[~board["player_id"].isin(tracker.drafted_ids)]
        closers = available[
            available.apply(lambda r: r.get("sv", 0) >= CLOSER_SV_THRESHOLD, axis=1)
        ]
        if not closers.empty:
            num_keepers = len(tracker.drafted_players) - (current_pick - 1)
            effective_pick = current_pick + num_keepers
            falling = closers[effective_pick >= closers["adp"] - OPP_CLOSER_ADP_BUFFER]
            if not falling.empty:
                falling = falling.sort_values("var", ascending=False)
                filled = get_filled_positions(
                    tracker.user_roster_ids, full_board,
                    roster_slots=config.roster_slots,
                )
                for _, best in falling.iterrows():
                    if _can_roster_player(best, filled, config.roster_slots):
                        return best["name"], best["player_id"]

    # Default with AVG floor
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


def _make_n_closers_strategy(target, deadlines):
    """Factory: create a strategy that drafts exactly N closers at spaced deadlines."""
    def pick_n_closers(
        board, full_board, tracker, balance, config, team_filled, **kwargs,
    ):
        closer_count = _count_closers(tracker, board, full_board)
        current_round = tracker.current_round

        need_closer = False
        if closer_count < target:
            deadline_idx = closer_count
            if deadline_idx < len(deadlines):
                deadline = deadlines[deadline_idx]
                if current_round >= deadline:
                    need_closer = True

        if need_closer:
            available = board[~board["player_id"].isin(tracker.drafted_ids)]
            closers = available[
                available.apply(lambda r: r.get("sv", 0) >= CLOSER_SV_THRESHOLD, axis=1)
            ]
            if not closers.empty:
                closers = closers.sort_values("var", ascending=False)
                filled = get_filled_positions(
                    tracker.user_roster_ids, full_board,
                    roster_slots=config.roster_slots,
                )
                for _, best in closers.iterrows():
                    if _can_roster_player(best, filled, config.roster_slots):
                        return best["name"], best["player_id"]

        return pick_default(board, full_board, tracker, balance, config,
                            team_filled, **kwargs)
    pick_n_closers.__doc__ = f"Draft exactly {target} closers at deadlines {deadlines}."
    return pick_n_closers


pick_two_closers = _make_n_closers_strategy(TWO_CLOSERS_TARGET, TWO_CLOSERS_DEADLINES)
pick_three_closers = _make_n_closers_strategy(THREE_CLOSERS_TARGET, THREE_CLOSERS_DEADLINES)
pick_four_closers = _make_n_closers_strategy(FOUR_CLOSERS_TARGET, FOUR_CLOSERS_DEADLINES)


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


def _sv_in_danger(tracker, board, full_board, team_rosters, num_teams):
    """Check if our projected SV would finish in the danger zone.

    Returns True if:
    - At least NO_PUNT_SV_MIN_TEAMS_WITH_CLOSERS teams have drafted closers, AND
    - Our team would finish in the bottom NO_PUNT_SV_DANGER_ZONE in SV.

    This avoids panic-triggering early when nobody has closers yet.
    """
    if not team_rosters:
        return False

    # Project SV for each team from their current roster
    team_sv = {}
    teams_with_closers = 0
    user_team = None
    for tn, pids in team_rosters.items():
        sv_total = 0
        for pid in pids:
            rows = board[board["player_id"] == pid]
            if rows.empty:
                rows = full_board[full_board["player_id"] == pid]
            if not rows.empty:
                sv_total += rows.iloc[0].get("sv", 0)
        team_sv[tn] = sv_total
        if sv_total >= CLOSER_SV_THRESHOLD:
            teams_with_closers += 1
        if pid in tracker.user_roster_ids:
            user_team = tn

    # Find user team if not found via pid matching
    if user_team is None:
        for tn, pids in team_rosters.items():
            if set(pids) & set(tracker.user_roster_ids):
                user_team = tn
                break

    if user_team is None or teams_with_closers < NO_PUNT_SV_MIN_TEAMS_WITH_CLOSERS:
        return False

    # Count how many teams have more SV than us
    our_sv = team_sv.get(user_team, 0)
    teams_above = sum(1 for tn, sv in team_sv.items() if sv > our_sv and tn != user_team)
    our_rank = teams_above + 1  # 1 = most SV, num_teams = least

    return our_rank > num_teams - NO_PUNT_SV_DANGER_ZONE


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


def _fallback_non_closer(board, tracker, full_board, config):
    """Pick the best available non-closer by VAR. Returns (name, pid) or (None, None).

    Searches the full board (not just top recs) for a non-closer who can
    fill an open roster slot.  This ensures the closer cap is respected
    even when the recommendation engine returns empty.
    """
    available = board[~board["player_id"].isin(tracker.drafted_ids)]
    non_closers = available[
        available.apply(lambda r: r.get("sv", 0) < CLOSER_SV_THRESHOLD, axis=1)
    ]
    filled = get_filled_positions(
        tracker.user_roster_ids, full_board,
        roster_slots=config.roster_slots,
    )
    if not non_closers.empty:
        for _, best in non_closers.sort_values("var", ascending=False).head(50).iterrows():
            if _can_roster_player(best, filled, config.roster_slots):
                return best["name"], best["player_id"]
    # Also check full_board for players not on the draft board
    # (e.g. low-projection players filtered during board construction)
    avail_full = full_board[~full_board["player_id"].isin(tracker.drafted_ids)]
    non_closers_full = avail_full[
        avail_full.apply(lambda r: r.get("sv", 0) < CLOSER_SV_THRESHOLD, axis=1)
    ]
    if not non_closers_full.empty:
        for _, best in non_closers_full.sort_values("var", ascending=False).head(50).iterrows():
            if _can_roster_player(best, filled, config.roster_slots):
                return best["name"], best["player_id"]
    return None, None


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
    # Use the tracker's snake-draft calculation for picks until next turn
    picks_until_next = getattr(tracker, "picks_until_next_turn", None)
    return get_recommendations(
        board, drafted=tracker.drafted_ids,
        user_roster=tracker.user_roster,
        n=n, filled_positions=filled,
        picks_until_next=picks_until_next,
        roster_slots=config.roster_slots,
        num_teams=config.num_teams,
        draft_leverage=leverage,
        scoring_mode=kwargs.get("scoring_mode", "var"),
    )


def pick_no_punt(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """Ensure no category finishes dead last.

    Watches projected SV standings — forces a closer when our team
    would finish in the danger zone.  Skips low-AVG hitters if team
    AVG is below the floor.

    Requires ``team_rosters`` in kwargs for dynamic SV monitoring.
    Falls back to the legacy round-based deadline if not provided.
    """
    team_rosters = kwargs.get("team_rosters")

    need_closer = False
    if team_rosters:
        need_closer = _sv_in_danger(
            tracker, board, full_board, team_rosters, config.num_teams,
        )
    else:
        closer_count = _count_closers(tracker, board, full_board)
        if closer_count == 0 and kwargs.get("current_round", tracker.current_round) >= NO_PUNT_SV_DEADLINE:
            need_closer = True

    if need_closer:
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


def pick_no_punt_stagger(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """No-punt with staggered closer deadlines.

    Combines no_punt's category protection (AVG floor, dynamic SV monitoring)
    with staggered closer deadlines to ensure adequate SV investment.
    Fixes no_punt's "one and done" closer bug by requiring multiple closers
    on a schedule, while also triggering early via SV danger monitoring.
    """
    current_round = tracker.current_round
    team_rosters = kwargs.get("team_rosters")

    # Count current closers
    closer_count = _count_closers(tracker, board, full_board)

    # Check staggered deadlines: force a closer if we're behind schedule
    need_closer = False
    if closer_count < NO_PUNT_STAGGER_TARGET:
        deadline_idx = closer_count  # 0th closer -> deadline[0], etc.
        if deadline_idx < len(NO_PUNT_STAGGER_DEADLINES):
            deadline = NO_PUNT_STAGGER_DEADLINES[deadline_idx]
            if current_round >= deadline:
                need_closer = True

    # Also check dynamic SV danger (if team_rosters available)
    if not need_closer and team_rosters and closer_count < NO_PUNT_STAGGER_TARGET:
        need_closer = _sv_in_danger(
            tracker, board, full_board, team_rosters, config.num_teams,
        )

    if need_closer:
        result = _force_closer(board, tracker, full_board, config)
        if result:
            return result

    # Get recommendations with AVG floor protection
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


def pick_no_punt_cap3(
    board, full_board, tracker, balance, config, team_filled, **kwargs,
):
    """No-punt with staggered closer deadlines and a hard 3-closer cap.

    Uses staggered deadlines to ensure we draft up to 3 closers, then
    filters closers from recommendations so the VONA engine can't
    over-draft them.  Keeps AVG floor and dynamic SV monitoring.
    """
    closer_count = _count_closers(tracker, board, full_board)
    current_round = tracker.current_round
    team_rosters = kwargs.get("team_rosters")

    # Force closers via staggered deadlines, up to the cap
    need_closer = False
    if closer_count < NO_PUNT_CAP3_TARGET:
        deadline_idx = closer_count
        if deadline_idx < len(NO_PUNT_STAGGER_DEADLINES):
            deadline = NO_PUNT_STAGGER_DEADLINES[deadline_idx]
            if current_round >= deadline:
                need_closer = True

        # Dynamic SV danger check
        if not need_closer and team_rosters:
            need_closer = _sv_in_danger(
                tracker, board, full_board, team_rosters, config.num_teams,
            )

    if need_closer:
        result = _force_closer(board, tracker, full_board, config)
        if result:
            return result

    # Get recommendations
    recs = _get_recs(board, full_board, tracker, balance, config, n=15, **kwargs)

    # If recs is empty (late-draft roster nearly full), fall back to
    # board search that respects the closer cap.
    if not recs:
        return _fallback_non_closer(
            board, tracker, full_board, config,
        ) if closer_count >= NO_PUNT_CAP3_TARGET else (None, None)

    current_h = sum(h.get("h", 0) for h in balance._hitters)
    current_ab = sum(h.get("ab", 0) for h in balance._hitters)

    for rec in recs:
        # Hard cap: skip closers once we have enough
        if closer_count >= NO_PUNT_CAP3_TARGET:
            rows = board[board["name"] == rec["name"]]
            if not rows.empty and rows.iloc[0].get("sv", 0) >= CLOSER_SV_THRESHOLD:
                continue

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

    # All recs filtered — respect closer cap in fallback
    if closer_count >= NO_PUNT_CAP3_TARGET:
        return _fallback_non_closer(board, tracker, full_board, config)
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
    "two_closers": pick_two_closers,
    "three_closers": pick_three_closers,
    "four_closers": pick_four_closers,
    "no_punt": pick_no_punt,
    "no_punt_opp": pick_no_punt_opp,
    "no_punt_stagger": pick_no_punt_stagger,
    "no_punt_cap3": pick_no_punt_cap3,
    "avg_anchor": pick_avg_anchor,
    "closers_avg": pick_closers_avg,
    "balanced": pick_balanced,
    "anti_fragile": pick_anti_fragile,
}
