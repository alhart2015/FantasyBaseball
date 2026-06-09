"""Draft strategy overlays for the unified recommend() seam.

Strategies are orthogonal overlays in the OVERLAYS registry. An overlay
receives a ranked list[RankedPick] and returns a RankedPick to force that
pick, or None to defer to recommend()'s slot-gated greedy selection.

STRATEGIES is an alias for OVERLAYS (backward-compatible name used by
scripts and config validation).

See src/fantasy_baseball/draft/CLAUDE.md for full strategy port status.
"""

import pandas as pd

from fantasy_baseball.draft.roster_state import RosterState, get_filled_positions
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD
from fantasy_baseball.utils.constants import safe_float as _safe
from fantasy_baseball.utils.positions import can_fill_slot
from fantasy_baseball.utils.rate_stats import calculate_avg

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


def build_player_lookup(
    board: pd.DataFrame,
    full_board: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Build a dict from player_id -> row for O(1) lookups.

    Checks board first, then full_board for players not on the filtered
    board (e.g. keepers removed by apply_keepers).
    """
    # full_board first, then board overrides (board has live VAR)
    lookup: dict[str, pd.Series] = dict(
        zip(full_board["player_id"], (row for _, row in full_board.iterrows()), strict=False)
    )
    lookup.update(dict(zip(board["player_id"], (row for _, row in board.iterrows()), strict=False)))
    return lookup


def select_from_ranked(ranked, open_starters, pick_rank):
    """Pick the ``pick_rank``-th item of ``ranked``, restricted to items that
    fill an open starter slot when any such remain (else the full list).

    Used by recommend() (var/vona arm) and the deltaRoto sim adapter so both
    arms gate the position-aware / k-th-choice selection identically. Items
    need only a ``.positions`` attribute (a ``Recommendation`` or a
    ``eroto_recs.RecRow``).
    """
    pool = ranked
    if open_starters:
        fillers = [r for r in ranked if any(can_fill_slot(r.positions, s) for s in open_starters)]
        if fillers:
            pool = fillers
    return pool[min(pick_rank, len(pool) - 1)] if pool else None


def _count_closers(tracker, board, full_board, player_lookup=None):
    """Count how many closers are on the user's roster."""
    if player_lookup is None:
        player_lookup = build_player_lookup(board, full_board)
    count = 0
    for pid in tracker.user_roster_ids:
        row = player_lookup.get(pid)
        if row is not None and _safe(row.get("sv")) >= CLOSER_SV_THRESHOLD:
            count += 1
    return count


def _count_hitters(tracker, board, full_board, player_lookup=None):
    """Count hitters on the user's roster."""
    if player_lookup is None:
        player_lookup = build_player_lookup(board, full_board)
    count = 0
    for pid in tracker.user_roster_ids:
        row = player_lookup.get(pid)
        if row is not None and row.get("player_type") == PlayerType.HITTER:
            count += 1
    return count


def _count_pitchers(tracker, board, full_board, player_lookup=None):
    """Count pitchers on the user's roster."""
    return len(tracker.user_roster) - _count_hitters(tracker, board, full_board, player_lookup)


def _sv_in_danger(tracker, board, full_board, team_rosters, num_teams, player_lookup=None):
    """Check if our projected SV would finish in the danger zone.

    Returns True if:
    - At least NO_PUNT_SV_MIN_TEAMS_WITH_CLOSERS teams have drafted closers, AND
    - Our team would finish in the bottom NO_PUNT_SV_DANGER_ZONE in SV.

    This avoids panic-triggering early when nobody has closers yet.
    """
    if not team_rosters:
        return False

    if player_lookup is None:
        player_lookup = build_player_lookup(board, full_board)

    # Project SV for each team from their current roster
    team_sv = {}
    teams_with_closers = 0
    user_team = None
    for tn, pids in team_rosters.items():
        sv_total = 0
        for pid in pids:
            row = player_lookup.get(pid)
            if row is not None:
                sv_total += row.get("sv", 0)
        team_sv[tn] = sv_total
        if sv_total >= CLOSER_SV_THRESHOLD:
            teams_with_closers += 1

    # Identify user team via set intersection
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


def _force_closer(board, tracker, full_board, config, player_lookup=None):
    """Pick the best available closer by VAR. Returns (name, pid) or None."""
    available = board[~board["player_id"].isin(tracker.drafted_ids)]
    closers = available[available["sv"].fillna(0) >= CLOSER_SV_THRESHOLD]
    if closers.empty:
        return None
    closers = closers.sort_values("var", ascending=False)
    filled = get_filled_positions(
        tracker.user_roster_ids,
        full_board,
        roster_slots=config.roster_slots,
        player_lookup=player_lookup,
    )
    roster_state = RosterState.from_dicts(filled, config.roster_slots)
    for _, best in closers.iterrows():
        if roster_state.any_slot_open_for(best["positions"]):
            return best["name"], best["player_id"]
    return None


def _fallback_non_closer(board, tracker, full_board, config):
    """Pick the best available non-closer by VAR. Returns (name, pid) or (None, None).

    Searches the full board (not just top recs) for a non-closer who can
    fill an open roster slot.  This ensures the closer cap is respected
    even when the recommendation engine returns empty.
    """
    available = board[~board["player_id"].isin(tracker.drafted_ids)]
    non_closers = available[available["sv"].fillna(0) < CLOSER_SV_THRESHOLD]
    filled = get_filled_positions(
        tracker.user_roster_ids,
        full_board,
        roster_slots=config.roster_slots,
    )
    roster_state = RosterState.from_dicts(filled, config.roster_slots)
    if not non_closers.empty:
        for _, best in non_closers.sort_values("var", ascending=False).head(50).iterrows():
            if roster_state.any_slot_open_for(best["positions"]):
                return best["name"], best["player_id"]
    # Also check full_board for players not on the draft board
    # (e.g. low-projection players filtered during board construction)
    avail_full = full_board[~full_board["player_id"].isin(tracker.drafted_ids)]
    non_closers_full = avail_full[avail_full["sv"].fillna(0) < CLOSER_SV_THRESHOLD]
    if not non_closers_full.empty:
        for _, best in non_closers_full.sort_values("var", ascending=False).head(50).iterrows():
            if roster_state.any_slot_open_for(best["positions"]):
                return best["name"], best["player_id"]
    return None, None


def _pick_with_avg_floor(recs, board, balance, avg_floor, player_lookup=None):
    """Select the first rec that keeps team AVG above the floor.

    Pitchers are always acceptable (they don't affect AVG).
    Returns (name, player_id).
    """
    current_h, current_ab = balance.get_avg_components()

    for rec in recs:
        if rec.player_type != PlayerType.HITTER:
            return rec.name, _lookup_pid(board, rec.name, player_lookup)

        rows = board[board["name"] == rec.name]
        if rows.empty:
            continue
        player = rows.iloc[0]
        new_h = current_h + player.get("h", 0)
        new_ab = current_ab + player.get("ab", 0)
        projected_avg = calculate_avg(new_h, new_ab)

        if projected_avg >= avg_floor or current_ab == 0:
            return rec.name, _lookup_pid(board, rec.name, player_lookup)

    return recs[0].name, _lookup_pid(board, recs[0].name, player_lookup)


def _lookup_pid(board, name, name_to_pid=None):
    if name_to_pid is not None and name in name_to_pid:
        return name_to_pid[name]
    rows = board[board["name"] == name]
    if not rows.empty:
        return rows.iloc[0]["player_id"]
    return name + "::unknown"


def overlay_default(ranked, *, roster_state=None, config=None, **kwargs):
    """No-constraint overlay; returns None to defer to recommend()'s slot-gate.

    Overlay contract: return a RankedPick to override slot-gated selection,
    or None to defer to it. This function always defers."""
    return None


def _save_projection(pick) -> float:
    """Extract save projection from a RankedPick.

    Uses dict.get with 0.0 default -- never `x or 0` so that a genuine
    SV=0.0 entry is not accidentally promoted.
    """
    return pick.per_category.get("SV", 0.0)


def _best_closer_from_ranked(ranked):
    """Return the highest-score RankedPick whose SV projection >= CLOSER_SV_THRESHOLD.

    Ranked is already ordered by score descending (the overlay receives it
    that way from recommend()), so the first qualifying entry is the best.
    """
    for pick in ranked:
        if _save_projection(pick) >= CLOSER_SV_THRESHOLD:
            return pick
    return None


def overlay_nonzero_sv(ranked, *, roster_state=None, config=None, **kwargs):
    """Force a closer (SV >= CLOSER_SV_THRESHOLD) by CLOSER_DEADLINE_ROUND.

    Overlay contract: return a RankedPick to override slot-gated selection,
    or None to defer to it.

    kwargs expected:
        current_round (int): the round currently being drafted.
        closer_count (int): closers already on the user's roster.
    """
    current_round = int(kwargs.get("current_round", 0))
    closer_count = int(kwargs.get("closer_count", 0))

    if closer_count == 0 and current_round >= CLOSER_DEADLINE_ROUND:
        return _best_closer_from_ranked(ranked)
    return None


def _make_n_closers_overlay(target, deadlines):
    """Factory: create a closer-count overlay for a target count and spaced deadlines.

    Mirrors _make_n_closers_strategy: deadline_idx == closer_count, so the
    Nth closer must arrive by deadlines[N-1].
    """

    def overlay(ranked, *, roster_state=None, config=None, **kwargs):
        current_round = int(kwargs.get("current_round", 0))
        closer_count = int(kwargs.get("closer_count", 0))

        if closer_count < target:
            deadline_idx = closer_count
            if deadline_idx < len(deadlines):
                deadline = deadlines[deadline_idx]
                if current_round >= deadline:
                    return _best_closer_from_ranked(ranked)
        return None

    overlay.__doc__ = (
        f"Force exactly {target} closers at deadlines {deadlines}. "
        "Returns the highest-score SV-positive RankedPick when a deadline "
        "is reached, None otherwise."
    )
    return overlay


overlay_two_closers = _make_n_closers_overlay(TWO_CLOSERS_TARGET, TWO_CLOSERS_DEADLINES)
overlay_three_closers = _make_n_closers_overlay(THREE_CLOSERS_TARGET, THREE_CLOSERS_DEADLINES)
overlay_four_closers = _make_n_closers_overlay(FOUR_CLOSERS_TARGET, FOUR_CLOSERS_DEADLINES)


# ---------------------------------------------------------------------------
# no_punt family -- FALLBACK / PARTIAL PORT
# ---------------------------------------------------------------------------


def overlay_no_punt(ranked, *, roster_state=None, config=None, **kwargs):
    """Documented FALLBACK -- defers to recommend()'s slot-gated selection.

    Missing signal: team's current AVG components (balance.get_avg_components()
    returns the team's accumulated H and AB totals, not available on RankedPick)
    and team_rosters for dynamic SV danger monitoring (_sv_in_danger requires
    the full roster dict for all teams, which is not passed through the overlay
    kwargs by the current recommend() caller).  Threading both signals would
    require changes to the recommend() call-site and the sim loop that exceed
    a small additive change, so this overlay defers until those are available.

    Original pick_no_punt behavior (src line ~481-529):
    - If team SV rank is in bottom NO_PUNT_SV_DANGER_ZONE (needs team_rosters) or
      no closer by round NO_PUNT_SV_DEADLINE (needs closer_count kwarg),
      force the best available closer.
    - Then filter hitters whose H+team_H / AB+team_AB < NO_PUNT_AVG_FLOOR (0.250)
      (needs team's accumulated H/AB not on RankedPick).
    """
    return None


def overlay_no_punt_opp(ranked, *, roster_state=None, config=None, **kwargs):
    """Documented FALLBACK -- defers to recommend()'s slot-gated selection.

    Missing signal: team_rosters (all-team roster dict for _sv_in_danger),
    effective_pick / ADP context for opportunistic grab logic, and team's
    accumulated H/AB for AVG floor filtering.  Opponent-relative SV standings
    (_sv_in_danger projection across all rosters) cannot be reconstructed from
    a single RankedPick's per_category or simple kwargs without threading the
    full team_rosters dict through recommend().

    Original pick_no_punt_opp behavior (src line ~206-279):
    - Dynamic SV danger check via _sv_in_danger (needs team_rosters).
    - Opportunistic closer grab when effective_pick >= ADP (needs current_pick
      and keeper count, not in overlay contract).
    - AVG floor filter via _pick_with_avg_floor (needs team H/AB totals).
    """
    return None


overlay_no_punt_stagger = _make_n_closers_overlay(NO_PUNT_STAGGER_TARGET, NO_PUNT_STAGGER_DEADLINES)
overlay_no_punt_stagger.__doc__ = """PARTIAL PORT -- staggered closer deadlines ported; AVG floor deferred.

    The staggered closer scheduling (NO_PUNT_STAGGER_DEADLINES = [13, 17, 20],
    target = 3) is faithfully ported via closer_count + current_round kwargs,
    identical to the n-closers overlay factory pattern.

    Missing signal: team's accumulated H/AB for AVG floor filtering
    (NO_PUNT_AVG_FLOOR = 0.250) and team_rosters for dynamic SV danger
    monitoring.  The dynamic SV check is omitted; only the deadline-based
    trigger fires.  The AVG floor pass that filters low-AVG hitters from recs
    is not applied because balance.get_avg_components() (team totals) is not
    threaded through the overlay contract.

    Original pick_no_punt_stagger behavior (src line ~532-584):
    - Staggered deadlines: force Nth closer by NO_PUNT_STAGGER_DEADLINES[N-1].
    - Dynamic SV danger check (needs team_rosters -- OMITTED here).
    - AVG floor filter (needs team H/AB -- OMITTED here).

    kwargs expected:
        current_round (int): the round currently being drafted.
        closer_count (int): closers already on the user's roster.
    """


def overlay_no_punt_cap3(ranked, *, roster_state=None, config=None, **kwargs):
    """PARTIAL PORT -- staggered deadlines + cap-3 logic ported; AVG floor deferred.

    The staggered closer scheduling (NO_PUNT_STAGGER_DEADLINES = [13, 17, 20],
    cap = NO_PUNT_CAP3_TARGET = 3) is faithfully ported.  When the cap is
    reached the overlay defers (no closer forced, no AVG filter applied) so
    that recommend()'s slot-gate picks the best non-closer.

    Missing signal: team's accumulated H/AB for AVG floor filtering and
    balance.get_avg_components() for the _pick_with_avg_floor call.  The cap
    also originally filtered closers out of the rec pool once 3 were drafted
    (source line ~653-655); that filter requires reading player SV from the
    board, not from per_category, so it is omitted here.

    Original pick_no_punt_cap3 behavior (src line ~587-675):
    - Same staggered deadlines + dynamic SV danger as no_punt_stagger.
    - After force, walks rec list skipping closers once cap reached.
    - AVG floor filter on each hitter (needs team H/AB -- OMITTED here).

    kwargs expected:
        current_round (int): the round currently being drafted.
        closer_count (int): closers already on the user's roster.
    """
    current_round = int(kwargs.get("current_round", 0))
    closer_count = int(kwargs.get("closer_count", 0))

    # Hard cap: if already at target, defer (let slot-gate pick non-closer).
    if closer_count >= NO_PUNT_CAP3_TARGET:
        return None

    deadline_idx = closer_count
    if deadline_idx < len(NO_PUNT_STAGGER_DEADLINES):
        deadline = NO_PUNT_STAGGER_DEADLINES[deadline_idx]
        if current_round >= deadline:
            return _best_closer_from_ranked(ranked)
    return None


# ---------------------------------------------------------------------------
# AVG family -- FALLBACK
# ---------------------------------------------------------------------------


def overlay_avg_hedge(ranked, *, roster_state=None, config=None, **kwargs):
    """Documented FALLBACK -- defers to recommend()'s slot-gated selection.

    Missing signal: the team's accumulated hit (H) and at-bat (AB) totals from
    balance.get_avg_components().  pick_avg_hedge calls _pick_with_avg_floor
    which computes projected_avg = (team_H + player_H) / (team_AB + player_AB)
    and filters out hitters that would push the team below AVG_FLOOR (0.255).
    The player's raw H and AB values come from board rows; per_category carries
    only marginal roto deltas (not absolute counting stats).

    Neither team_H/team_AB nor player_H/player_AB are available in the overlay
    contract without threading balance or raw projections through recommend().
    Threading balance would require adding it to every overlay call-site and
    the sim loop, which exceeds a small additive change.
    """
    return None


def overlay_avg_anchor(ranked, *, roster_state=None, config=None, **kwargs):
    """Documented FALLBACK -- defers to recommend()'s slot-gated selection.

    Missing signal: the candidate's absolute projected AVG (e.g. .285+).
    pick_avg_anchor checks board['avg'] >= AVG_ANCHOR_MIN (0.285) for each rec.
    RankedPick.per_category carries the marginal roto-delta for AVG (a float in
    roto-point units), not the player's season batting average projection.  These
    are dimensionally incompatible: an absolute AVG of .285 cannot be compared
    to a delta-roto contribution of e.g. +0.04 roto points.

    Additionally, hitter_count (how many hitters are already on the team) is not
    part of the overlay contract without threading it as a kwarg.

    Original pick_avg_anchor behavior (src line ~678-737):
    - While hitter_count < AVG_ANCHOR_DEADLINE_HITTER (3) and no anchor yet,
      scan recs for a hitter with board['avg'] >= AVG_ANCHOR_MIN (0.285).
    """
    return None


# ---------------------------------------------------------------------------
# closers_avg -- COMPOSED (closer scheduling ported; AVG anchor deferred)
# ---------------------------------------------------------------------------


def overlay_closers_avg(ranked, *, roster_state=None, config=None, **kwargs):
    """COMPOSED: three_closers closer gate applied; AVG anchor portion deferred.

    Mirrors pick_closers_avg (src line ~740-770): closer deadlines have highest
    priority, then fall through to avg_anchor.  The closer gate is faithfully
    ported (identical to overlay_three_closers).  The avg_anchor fallback is
    omitted because overlay_avg_anchor is a documented FALLBACK -- avg_anchor
    needs the candidate's absolute AVG projection (not available in per_category)
    so that tier simply defers.

    kwargs expected:
        current_round (int): the round currently being drafted.
        closer_count (int): closers already on the user's roster.
    """
    # Priority 1: closer scheduling -- delegate to overlay_three_closers.
    result = overlay_three_closers(ranked, roster_state=roster_state, config=config, **kwargs)
    if result is not None:
        return result

    # Priority 2: avg_anchor -- deferred (missing absolute AVG signal).
    return None


# ---------------------------------------------------------------------------
# balanced -- PORTED
# ---------------------------------------------------------------------------


def overlay_balanced(ranked, *, roster_state=None, config=None, **kwargs):
    """PORTED: force hitter or pitcher to cap positional skew.

    Mirrors pick_balanced (src line ~773-807): if pitchers outnumber hitters by
    more than BALANCED_MAX_SKEW (2), force the highest-score hitter from the
    ranked list.  If hitters outnumber pitchers by more than BALANCED_MAX_SKEW,
    force the highest-score pitcher.  Otherwise, defer.

    player_type is available on RankedPick directly; n_hitters/n_pitchers are
    passed by the caller as kwargs (cheap, identical to closer_count pattern).

    kwargs expected:
        n_hitters (int): hitters already on the user's roster.
        n_pitchers (int): pitchers already on the user's roster.
    """
    n_hitters = int(kwargs.get("n_hitters", 0))
    n_pitchers = int(kwargs.get("n_pitchers", 0))

    force_type = None
    if n_pitchers - n_hitters > BALANCED_MAX_SKEW:
        force_type = PlayerType.HITTER
    elif n_hitters - n_pitchers > BALANCED_MAX_SKEW:
        force_type = PlayerType.PITCHER

    if force_type is not None:
        for pick in ranked:
            if pick.player_type == force_type:
                return pick
    return None


# ---------------------------------------------------------------------------
# anti_fragile -- FALLBACK
# ---------------------------------------------------------------------------


def overlay_anti_fragile(ranked, *, roster_state=None, config=None, **kwargs):
    """Documented FALLBACK -- defers to recommend()'s slot-gated selection.

    Missing signal: the candidate's absolute innings-pitched (IP) projection.
    pick_anti_fragile (src line ~810-849) penalizes pitchers whose board['ip']
    exceeds ANTI_FRAGILE_IP_THRESHOLD (170), applying a 25% VAR penalty per
    30 IP above the threshold.  IP is a counting stat, not a roto category, so
    it is absent from per_category (which holds marginal roto-point deltas for
    the five pitching categories: W, K, ERA, WHIP, SV).  Re-scoring candidates
    by durability-adjusted VAR would require threading raw IP projections into
    the overlay, which is beyond a small additive change.
    """
    return None


OVERLAYS = {
    "default": overlay_default,
    "nonzero_sv": overlay_nonzero_sv,
    "two_closers": overlay_two_closers,
    "three_closers": overlay_three_closers,
    "four_closers": overlay_four_closers,
    "no_punt": overlay_no_punt,
    "no_punt_opp": overlay_no_punt_opp,
    "no_punt_stagger": overlay_no_punt_stagger,
    "no_punt_cap3": overlay_no_punt_cap3,
    "avg_hedge": overlay_avg_hedge,
    "avg_anchor": overlay_avg_anchor,
    "closers_avg": overlay_closers_avg,
    "balanced": overlay_balanced,
    "anti_fragile": overlay_anti_fragile,
}

# STRATEGIES is a backward-compatible alias for OVERLAYS.
# All callers (scripts, config validation, tests) that reference STRATEGIES
# now get the overlay functions.
STRATEGIES = OVERLAYS
