"""Simulate a full draft to evaluate strategy.

Usage:
    python scripts/simulate_draft.py [--strategy default|nonzero_sv|avg_hedge]

- Your team: picks according to the selected strategy (default: 'default').
- Other teams: take the highest-ADP available player they can legally roster.
- Roster limits are enforced for all teams.

Outputs projected roto standings at the end.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.data.db import get_connection
from fantasy_baseball.draft.balance import CategoryBalance
from fantasy_baseball.draft.board import apply_keepers, build_draft_board
from fantasy_baseball.draft.eroto_recs import is_reliever
from fantasy_baseball.draft.recommend import RecommendContext, recommend
from fantasy_baseball.draft.recommender import (
    compute_slot_scarcity_order,
    get_filled_positions,
    get_recommendations,
)
from fantasy_baseball.draft.recs_integration import (
    _build_replacements,
    build_adp_table,
    build_projected_standings,
    build_team_rosters,
    empirical_pitcher_replacements,
)
from fantasy_baseball.draft.roster_state import RosterState
from fantasy_baseball.draft.strategy import STRATEGIES, build_player_lookup
from fantasy_baseball.draft.tracker import DraftTracker
from fantasy_baseball.models.player import Player
from fantasy_baseball.scoring import build_team_sds, project_team_stats, score_roto_dict
from fantasy_baseball.utils.constants import ALL_CATEGORIES as ALL_CATS
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import can_fill_slot

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"

# Candidate pool size for the deltaRoto user pick -- must match sim_deltaroto.POOL_CAP
# so the rerouted user pick reproduces the pre-refactor deltaRoto pick sequence exactly.
_DELTAROTO_POOL_CAP = 200

# Set of scoring_mode strings that use the deltaRoto ranking engine.
_DELTAROTO_STRATEGY_NAMES = frozenset({"deltaroto_immediate", "deltaroto_vopn"})

# Static per-board inputs for the deltaRoto path: cache keyed on board id.
# Mirrors sim_deltaroto._STATIC_CACHE to avoid rebuilding replacements/ADP per pick.
_SIM_STATIC_CACHE: dict[int, tuple] = {}


def _sim_static_inputs(board, config):
    """Board-derived inputs constant across the draft for the deltaRoto sim path.

    Mirrors sim_deltaroto._static_inputs: replacements, ADP table, pid->Player map,
    VAR-ordered pids, and team names. Cached on board id so they are built once per
    simulation run (not per pick). Keyed separately from sim_deltaroto._STATIC_CACHE
    so the two scripts can coexist in the same process without aliasing.
    """
    key = id(board)
    cached = _SIM_STATIC_CACHE.get(key)
    if cached is not None:
        return cached
    board_by_id: dict[str, Player] = {}
    for row in board.to_dict("records"):
        p = Player.from_dict(row)
        if p.yahoo_id:
            board_by_id[p.yahoo_id] = p
    replacements = {
        **_build_replacements(board, config.roster_slots, config.num_teams),
        **empirical_pitcher_replacements(),
    }
    adp_table = build_adp_table(board)
    pool_sorted = board.sort_values("var", ascending=False) if "var" in board.columns else board
    ordered_pids = list(pool_sorted["player_id"])
    team_names = list(config.teams.values())
    cached = (board_by_id, replacements, adp_table, ordered_pids, team_names)
    _SIM_STATIC_CACHE[key] = cached
    return cached


def _build_deltaroto_rec_inputs(board, full_board, tracker, config, team_rosters, player_lookup):
    """Build a RecInputs object for the deltaRoto user pick.

    Mirrors make_deltaroto_pick from sim_deltaroto.py exactly so the rerouted
    recommend() call produces the same candidate pool, standings, and team_sds as
    the pre-refactor pick function -- a prerequisite for the golden tests to hold.

    The candidate list is: top _DELTAROTO_POOL_CAP undrafted, roster-legal players
    by VAR order (matching POOL_CAP=200 in sim_deltaroto).
    """
    from fantasy_baseball.draft.recs_integration import RecInputs
    from fantasy_baseball.draft.state import StateKey

    board_by_id, replacements, adp_table, ordered_pids, team_names = _sim_static_inputs(
        board, config
    )

    # Synthetic draft-state from live sim rosters (keepers already folded in by
    # run_simulation). Mirrors make_deltaroto_pick's state construction exactly.
    picks = [
        {"player_id": pid, "team": config.teams[num], "position": ""}
        for num, pids in team_rosters.items()
        for pid in pids
    ]
    state = {StateKey.KEEPERS.value: [], StateKey.PICKS.value: picks}
    rosters = build_team_rosters(state, board_by_id, team_names, config.roster_slots, replacements)
    standings = build_projected_standings(rosters)
    team_sds = build_team_sds(rosters, sd_scale=1.0)

    # Candidate pool: undrafted, roster-legal for the user team, top-N by VAR.
    # Mirrors make_deltaroto_pick candidate filtering so the candidate list is identical.
    drafted = set(tracker.drafted_ids)
    filled = get_filled_positions(
        tracker.user_roster_ids,
        full_board,
        roster_slots=config.roster_slots,
        player_lookup=player_lookup,
    )
    roster_state = RosterState.from_dicts(filled, config.roster_slots)
    candidates: list[Player] = []
    for pid in ordered_pids:
        if pid in drafted:
            continue
        p = board_by_id.get(pid)
        if p is None:
            continue
        row = player_lookup.get(pid)
        positions = row["positions"] if row is not None else p.positions
        if not roster_state.any_slot_open_for(positions):
            continue
        candidates.append(p)
        if len(candidates) >= _DELTAROTO_POOL_CAP:
            break

    user_rp_filled = sum(
        1
        for pid in tracker.user_roster_ids
        if (pp := board_by_id.get(pid)) is not None and is_reliever(pp)
    )
    rp_filled_by_team = {config.team_name: user_rp_filled}

    return RecInputs(
        candidates=candidates,
        replacements=replacements,
        projected_standings=standings,
        team_sds=team_sds,
        adp_table=adp_table,
        rp_filled_by_team=rp_filled_by_team,
    )


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
    hitter_slots = sum(v for k, v in roster_slots.items() if k not in ("P", "BN", "IL"))
    pitcher_slots = roster_slots.get("P", 9)
    return hitter_slots, pitcher_slots


def _select_active_players(hitters, pitchers, roster_slots, scarcity_order=None):
    """Return only the active-roster hitters and pitchers.

    Pitchers share one fungible ``P`` pool, so the top ``P`` by value
    (closers first) fill it. Hitters are slotted into their *position* slots
    by eligibility: highest counting-stat value first, each taking the
    scarcest open starting slot it can fill (via :func:`_assign_slot`), bench
    last. A hitter that fits no remaining starting slot is benched, and a
    position with no eligible player is left empty.

    Position-awareness keeps an imbalanced roster honest: the sole catcher
    always fills C rather than being benched behind better OF bats, and a
    pile of surplus OF sits instead of inflating the active lineup. The old
    top-N-by-counting-stats logic ignored slots, so it overstated
    position-imbalanced rosters and (mis)rewarded stat-piling over the
    positionally balanced rosters that VAR/VONA deliberately draft.
    """
    _, p_slots = _active_slot_counts(roster_slots)

    ranked_p = sorted(
        pitchers,
        key=lambda p: (p.get("sv", 0) >= 15, p.get("w", 0) + p.get("k", 0) + p.get("sv", 0)),
        reverse=True,
    )
    active_p = ranked_p[:p_slots]

    ranked_h = sorted(
        hitters,
        key=lambda h: h.get("r", 0) + h.get("hr", 0) + h.get("rbi", 0) + h.get("sb", 0),
        reverse=True,
    )
    filled: dict[str, int] = {}
    active_h = []
    for h in ranked_h:
        slot = _assign_slot(h["positions"], filled, roster_slots, scarcity_order)
        if slot is not None and slot not in ("BN", "IL"):
            active_h.append(h)
    return active_h, active_p


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


def _assign_slot(player_positions, filled, roster_slots, scarcity_order=None):
    """Assign a player to the best available slot, updating filled in place.

    If *scarcity_order* is provided (list of slots from most to least scarce),
    specific slots are tried in that order so multi-eligible players fill the
    scarcest open position first.
    """
    if scarcity_order:
        active = [s for s in scarcity_order if s not in ("BN", "IL", "IF", "UTIL")]
        flex = [s for s in scarcity_order if s in ("IF", "UTIL")]
    else:
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


def _score_roto(team_players, config, full_board, board, scarcity_order=None):
    """Project roto standings from team rosters. Returns (results, all_cats)."""
    # Build per-team stats using active roster selection + shared projection
    team_stats = {}
    team_meta = {}
    active_rosters = {}
    for tn in range(1, config.num_teams + 1):
        tname = config.teams.get(tn, f"Team {tn}")
        all_hitters = [p for p in team_players[tn] if p["player_type"] == "hitter"]
        all_pitchers = [p for p in team_players[tn] if p["player_type"] == "pitcher"]
        hitters, pitchers = _select_active_players(
            all_hitters,
            all_pitchers,
            config.roster_slots,
            scarcity_order=scarcity_order,
        )
        active = list(hitters) + list(pitchers)
        team_stats[tname] = project_team_stats(active).to_dict()
        team_meta[tname] = {"nh": len(hitters), "np": len(pitchers)}
        active_rosters[tname] = active

    # Score with the empirical per-category variance (team_sds), matching the
    # deltaRoto recommender and the dashboard. Without it score_roto collapses
    # to a hard-rank step function (a category leader gets exactly 10.0 instead
    # of ~8), which makes standings -- and keeper/strategy edges -- look far more
    # deterministic than the projections warrant. Preseason -> full-season
    # variance, sd_scale=1.0.
    team_sds = build_team_sds(active_rosters, sd_scale=1.0)
    roto = score_roto_dict(team_stats, team_sds=team_sds)

    # Convert to legacy list-of-dicts format expected by callers
    results = []
    for tname in team_stats:
        entry = {"team": tname, **team_stats[tname], **team_meta[tname]}
        for cat in ALL_CATS:
            entry[f"{cat.value}_p"] = roto[tname].get(f"{cat.value}_pts", 0)
        entry["tot"] = roto[tname]["total"]
        results.append(entry)

    results.sort(key=lambda x: x["tot"], reverse=True)
    return results, ALL_CATS


def _parse_opponent_strategies(opp_str):
    """Parse '1:default,2:three_closers' into {team_num: strategy_fn}."""
    opp_strategies = {}
    if not opp_str:
        return opp_strategies
    for pair in opp_str.split(","):
        tn_str, strat_name = pair.strip().split(":")
        tn = int(tn_str)
        if strat_name in STRATEGIES:
            opp_strategies[tn] = STRATEGIES[strat_name]
    return opp_strategies


DRAFT_ORDER_PATH = PROJECT_ROOT / "config" / "draft_order.json"


def _load_pick_order(config):
    """Load custom draft order and build a post-keeper pick-to-team-num mapping.

    Returns a list of team numbers (1-indexed), one per post-keeper pick,
    or None if no custom order file exists.
    """
    if not DRAFT_ORDER_PATH.exists():
        return None

    with open(DRAFT_ORDER_PATH) as f:
        data = json.load(f)

    # Build reverse mapping: team_name -> team_num
    name_to_num = {v: k for k, v in config.teams.items()}

    rounds = data["rounds"]
    keeper_rounds = len(config.keepers) // config.num_teams
    post_keeper_rounds = rounds[keeper_rounds:]

    pick_order = []
    for round_teams in post_keeper_rounds:
        for team_name in round_teams:
            team_num = name_to_num.get(team_name)
            if team_num is None:
                # Fuzzy match for truncated names
                for full_name, num in name_to_num.items():
                    if team_name in full_name or full_name in team_name:
                        team_num = num
                        break
            pick_order.append(team_num or 0)
    return pick_order


def build_board_and_context(config_path=None):
    """Build the draft board and all reusable context. Call once.

    Returns a dict with keys: config, full_board, board, scarcity_order,
    pick_order.
    """
    if config_path is None:
        config_path = CONFIG_PATH
    config = load_config(config_path)
    conn = get_connection()
    full_board = build_draft_board(
        conn=conn,
        roster_slots=config.roster_slots or None,
        num_teams=config.num_teams,
    )
    conn.close()
    board = apply_keepers(full_board, config.keepers)
    scarcity_order = compute_slot_scarcity_order(full_board, config.roster_slots)
    pick_order = _load_pick_order(config)
    return {
        "config": config,
        "full_board": full_board,
        "board": board,
        "scarcity_order": scarcity_order,
        "pick_order": pick_order,
    }


def _build_adp_boards(board, num_teams, adp_noise, rng):
    """Per-team ADP draft boards. Each team gets its OWN noise draw -- a fixed
    'opinion' of every player held for the whole draft -- so an elite player one
    team undervalues is still grabbed near his true ADP by another team. A single
    shared reshuffle (the old model) instead let one unlucky draw push a low-ADP
    player down for ALL teams at once, so he'd fall league-wide. Returns
    ``{team_num: board sorted by that team's noised adp}``.
    """
    base = board.copy()
    if "adp" not in base.columns:
        base["adp"] = range(len(base))
    boards = {}
    for tn in range(1, num_teams + 1):
        bt = base.copy()
        if adp_noise > 0:
            bt["adp"] = bt["adp"].to_numpy() + rng.normal(0, adp_noise, size=len(bt))
        boards[tn] = bt.sort_values("adp", ascending=True)
    return boards


def _first_undrafted(adp_board, drafted_set):
    """Best-ADP player on the board not yet drafted, as (name, pid), or
    (None, "") if exhausted. The plain best-available fallback shared by the
    user and strategy-opponent pick paths."""
    for _, row in adp_board.iterrows():
        if row["player_id"] not in drafted_set:
            return row["name"], row["player_id"]
    return None, ""


def run_simulation(
    ctx,
    strategy_name="default",
    scoring_mode="var",
    adp_noise=0.0,
    strategy_noise=0.0,
    seed=None,
    opponent_strategies_str=None,
    position_aware=False,
    field_noise=False,
):
    """Run a single draft simulation and return results.

    *position_aware*: when True, the recommender-routed pick gates to "fill a
    starter slot, bench last" (``strategy._choose_rec`` and the deltaRoto
    adapter). ADP opponents always fill a starter slot first regardless. Only
    strategies that defer to ``_choose_rec`` (pick_default and the closer
    strategies that fall through to it) honor it; strategies that select their
    top recommendation directly are unaffected. *field_noise*: when True, each
    recommender-routed pick takes the k-th choice from its own ranking via a
    one-sided normal draw -- z<1 -> top (~84%), 1-2 -> 2nd, 2-3 -> 3rd, >=3 ->
    4th -- so the strategic field varies across seeds. Same routing caveat as
    above: forced-closer picks and direct-top-pick strategies ignore it.

    *ctx* is the dict returned by ``build_board_and_context()``.

    Returns dict with keys: pts, rank, results (full standings list).
    """
    config = ctx["config"]
    full_board = ctx["full_board"]
    board = ctx["board"]
    scarcity_order = ctx["scarcity_order"]
    pick_order = ctx.get("pick_order")  # custom draft order (or None)

    rng = np.random.default_rng(seed)
    # Per-team ADP boards: each team drafts off its OWN noised view of ADP, so
    # elite players don't fall league-wide (see _build_adp_boards).
    adp_boards = _build_adp_boards(board, config.num_teams, adp_noise, rng)

    # Initialize tracker — IL is not a draftable slot
    user_keepers = [k for k in config.keepers if k.get("team") == config.team_name]
    draftable_slots = sum(v for k, v in config.roster_slots.items() if k != "IL")
    rounds = draftable_slots - len(user_keepers)
    tracker = DraftTracker(
        num_teams=config.num_teams,
        user_position=config.draft_position,
        rounds=rounds,
    )
    balance = CategoryBalance()
    team_filled = {i: {} for i in range(1, config.num_teams + 1)}
    # Track player IDs per team so strategies can monitor league-wide stats
    team_rosters = {i: [] for i in range(1, config.num_teams + 1)}

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
                    tracker.draft_player(best["name"], is_user=is_user, player_id=best["player_id"])
                    _assign_slot(
                        best["positions"], team_filled[num], config.roster_slots, scarcity_order
                    )
                    team_rosters[num].append(best["player_id"])
                break

    # Parse opponent strategies
    opp_strategies = _parse_opponent_strategies(opponent_strategies_str)
    opp_balances = {}
    opp_rosters = {}
    opp_roster_names = {}
    for tn in opp_strategies:
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

    # Run draft
    user_team_num = config.draft_position
    player_lookup = build_player_lookup(board, full_board)
    drafted_set = set(tracker.drafted_ids)
    while tracker.current_pick <= tracker.total_picks:
        pick_idx = tracker.current_pick - 1
        if pick_order and pick_idx < len(pick_order):
            team_num = pick_order[pick_idx]
        else:
            team_num = tracker.picking_team
        is_user = team_num == user_team_num

        # Field variance: the k-th choice from this pick's own ranking, drawn
        # from a one-sided normal. Passed to every algorithm-driven pick so
        # each seed yields a different draft.
        pick_rank = 0
        if field_noise:
            z = rng.normal(0.0, 1.0)
            pick_rank = 0 if z < 1.0 else min(3, int(z))

        if is_user:
            # Route the user pick through the unified recommend() seam.
            # Determine effective scoring mode for the RecommendContext: deltaRoto
            # strategies are identified by strategy_name regardless of the scoring_mode
            # arg (sim_deltaroto passes scoring_mode="var" but strategy_name="deltaroto_*").
            _eff_scoring = (
                strategy_name if strategy_name in _DELTAROTO_STRATEGY_NAMES else scoring_mode
            )

            if _eff_scoring in _DELTAROTO_STRATEGY_NAMES:
                # deltaRoto path: build RecInputs per-pick from live sim state,
                # mirroring make_deltaroto_pick in sim_deltaroto.py exactly.
                _rec_inputs = _build_deltaroto_rec_inputs(
                    board, full_board, tracker, config, team_rosters, player_lookup
                )
                _ctx = RecommendContext(
                    scoring_mode=_eff_scoring,
                    team_name=config.team_name,
                    picks_until_next=tracker.picks_until_next_turn,
                    inputs=_rec_inputs,
                )
            else:
                # var/vona path: build RecommendContext from board + tracker state.
                _filled = get_filled_positions(
                    tracker.user_roster_ids,
                    full_board,
                    roster_slots=config.roster_slots,
                    player_lookup=player_lookup,
                )
                _ctx = RecommendContext(
                    scoring_mode=_eff_scoring,
                    team_name=config.team_name,
                    picks_until_next=tracker.picks_until_next_turn,
                    board=board,
                    drafted=list(tracker.drafted_ids),
                    filled_positions=_filled,
                    config=config,
                )

            # Compute open_starters for position-aware gating (matches _choose_rec).
            if position_aware:
                _filled_pa = get_filled_positions(
                    tracker.user_roster_ids,
                    full_board,
                    roster_slots=config.roster_slots,
                    player_lookup=player_lookup,
                )
                _open_starters = RosterState.from_dicts(
                    _filled_pa, config.roster_slots
                ).unfilled_starter_slots()
            else:
                _open_starters = set()

            # Closer count for the closer-family overlays.
            _closer_count = sum(
                1
                for _pid in tracker.user_roster_ids
                if ((_row := player_lookup.get(_pid)) is not None and (_row.get("sv") or 0) >= 15)
            )

            _pick = recommend(
                _ctx,
                strategy=strategy_name,
                open_starters=_open_starters,
                pick_rank=pick_rank,
                current_round=tracker.current_round,
                closer_count=_closer_count,
            )
            if _pick is not None:
                pick_name = _pick.name
                pid = _pick.player_id
            else:
                pick_name = None
                pid = ""

            # Strategy noise: sometimes take the 2nd or 3rd rec instead.
            # Normal distribution: ~68% take #1, ~27% take #2, ~4% take #3.
            if strategy_noise > 0 and pick_name is not None:
                skip = min(abs(round(rng.normal(0, strategy_noise))), 4)  # cap at 5th-best
                if skip > 0:
                    filled = get_filled_positions(
                        tracker.user_roster_ids,
                        full_board,
                        roster_slots=config.roster_slots,
                    )
                    recs = get_recommendations(
                        board,
                        drafted=tracker.drafted_ids,
                        user_roster=tracker.user_roster,
                        n=skip + 3,
                        filled_positions=filled,
                        picks_until_next=getattr(tracker, "picks_until_next_turn", None),
                        roster_slots=config.roster_slots,
                        num_teams=config.num_teams,
                        scoring_mode=scoring_mode,
                    )
                    if len(recs) > skip:
                        alt = recs[skip]
                        rows = board[board["name"] == alt.name]
                        if not rows.empty:
                            pick_name = alt.name
                            pid = rows.iloc[0]["player_id"]

            if pick_name is None:
                pick_name, pid = _first_undrafted(adp_boards[team_num], drafted_set)

            if pick_name:
                tracker.draft_player(pick_name, is_user=True, player_id=pid)
                drafted_set.add(pid)
                p = player_lookup.get(pid)
                if p is not None:
                    balance.add_player(p)
                    _assign_slot(
                        p["positions"], team_filled[team_num], config.roster_slots, scarcity_order
                    )
                team_rosters[team_num].append(pid)
            else:
                pick_name = "(no pick)"

        elif team_num in opp_strategies:
            opp_fn = opp_strategies[team_num]
            proxy = TeamTrackerProxy(
                tracker,
                opp_roster_names[team_num],
                opp_rosters[team_num],
            )
            pick_name, pid = opp_fn(
                board,
                full_board,
                proxy,
                opp_balances[team_num],
                config,
                team_filled,
                total_rounds=rounds,
                scoring_mode=scoring_mode,
                player_lookup=player_lookup,
                pick_rank=pick_rank,
                position_aware=position_aware,
            )
            if pick_name is None:
                pick_name, pid = _first_undrafted(adp_boards[team_num], drafted_set)

            if pick_name:
                tracker.draft_player(pick_name, is_user=False, player_id=pid)
                drafted_set.add(pid)
                p = player_lookup.get(pid)
                if p is not None:
                    opp_balances[team_num].add_player(p)
                    opp_rosters[team_num].append(pid)
                    opp_roster_names[team_num].append(pick_name)
                    _assign_slot(
                        p["positions"], team_filled[team_num], config.roster_slots, scarcity_order
                    )
                team_rosters[team_num].append(pid)
            else:
                pick_name = "(no pick)"
        else:
            pick_name = None
            pid = ""

            # ADP opponents fill an active starter slot first, then fall back to
            # any rosterable slot (bench) below. This is basic sane ADP drafting
            # and is independent of position_aware (which gates only the
            # strategy/recommender side); it matches pre-position-aware behavior
            # so compare_strategies/CLI opponents are unchanged.
            for _, row in adp_boards[team_num].iterrows():
                if row["player_id"] in drafted_set:
                    continue
                positions = row["positions"]
                if _can_fill_active_slot(positions, team_filled[team_num], config.roster_slots):
                    pick_name = row["name"]
                    pid = row["player_id"]
                    tracker.draft_player(pick_name, is_user=False, player_id=pid)
                    _assign_slot(
                        positions, team_filled[team_num], config.roster_slots, scarcity_order
                    )
                    drafted_set.add(pid)
                    team_rosters[team_num].append(pid)
                    break

            if pick_name is None:
                for _, row in adp_boards[team_num].iterrows():
                    if row["player_id"] in drafted_set:
                        continue
                    positions = row["positions"]
                    if _can_roster(positions, team_filled[team_num], config.roster_slots):
                        pick_name = row["name"]
                        pid = row["player_id"]
                        tracker.draft_player(pick_name, is_user=False, player_id=pid)
                        drafted_set.add(pid)
                        _assign_slot(
                            positions, team_filled[team_num], config.roster_slots, scarcity_order
                        )
                        team_rosters[team_num].append(pid)
                        break

            if pick_name is None:
                tracker.advance()
                continue

        tracker.advance()

    # Reconstruct rosters and score
    team_players = {i: [] for i in range(1, config.num_teams + 1)}

    for keeper in config.keepers:
        for num, name in config.teams.items():
            if name == keeper["team"]:
                norm = normalize_name(keeper["name"])
                matches = full_board[full_board["name_normalized"] == norm]
                if not matches.empty:
                    team_players[num].append(matches.loc[matches["var"].idxmax()])
                break

    num_keepers = len(config.keepers)
    draft_entries = list(
        zip(
            tracker.drafted_players[num_keepers:],
            tracker.drafted_ids[num_keepers:],
            strict=False,
        )
    )
    for pick_num, (_name, pid) in enumerate(draft_entries):
        if pick_order and pick_num < len(pick_order):
            team = pick_order[pick_num]
        else:
            # Standard snake fallback
            rnd = pick_num // config.num_teams + 1
            pos = pick_num % config.num_teams + 1
            team = pos if rnd % 2 == 1 else config.num_teams - pos + 1
        p = player_lookup.get(pid)
        if p is not None:
            team_players[team].append(p)

    results, _all_cats = _score_roto(
        team_players, config, full_board, board, scarcity_order=scarcity_order
    )

    hart = next(t for t in results if t["team"] == config.team_name)
    rank = next(i + 1 for i, t in enumerate(results) if t["team"] == config.team_name)

    return {
        "pts": hart["tot"],
        "rank": rank,
        "results": results,
        "user_roster": list(tracker.user_roster),
        "user_roster_ids": list(tracker.user_roster_ids),
        "tracker": tracker,
        "team_players": team_players,
        "config": config,
    }


def save_simulation_output(
    result, strategy_name, scoring_mode, opponent_strategies_str=None, run_timestamp=None
):
    """Save complete simulation output for later re-analysis.

    Writes all team rosters, standings, and draft log to a JSON file
    in data/sim_results/ with a timestamped filename.
    """
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    out_dir = PROJECT_ROOT / "data" / "sim_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{run_timestamp}_{strategy_name}_{scoring_mode}.json"
    out_path = out_dir / filename

    config = result["config"]
    team_players = result["team_players"]

    # Build rosters dict keyed by team name
    rosters = {}
    for team_num, players in team_players.items():
        team_name = config.teams.get(team_num, f"Team {team_num}")
        roster = []
        for p in players:
            entry = {
                "name": str(p.get("name", "")),
                "player_id": str(p.get("player_id", "")),
                "player_type": str(p.get("player_type", "")),
                "positions": [str(x) for x in p.get("positions", [])],
                "var": round(float(p.get("var", 0)), 2),
                "total_sgp": round(float(p.get("total_sgp", 0)), 2),
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
            if "adp" in p.index if hasattr(p, "index") else "adp" in p:
                entry["adp"] = round(float(p.get("adp", 0)), 1)
            roster.append(entry)
        rosters[team_name] = roster

    # Build standings
    standings = []
    all_cats = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
    for i, t in enumerate(result["results"], 1):
        entry = {
            "rank": i,
            "team": t["team"],
            "total_pts": int(t["tot"]),
            "categories": {},
        }
        for cat in all_cats:
            entry["categories"][cat] = {
                "value": round(float(t[cat]), 4),
                "points": int(t[f"{cat}_p"]),
            }
        standings.append(entry)

    # Build draft log from tracker
    tracker = result["tracker"]
    num_keepers = len(config.keepers)
    draft_log = []
    draft_entries = list(
        zip(
            tracker.drafted_players[num_keepers:],
            tracker.drafted_ids[num_keepers:],
            strict=False,
        )
    )
    for pick_num, (name, pid) in enumerate(draft_entries, 1):
        rnd = (pick_num - 1) // config.num_teams + 1
        pos = (pick_num - 1) % config.num_teams + 1
        team_num = pos if rnd % 2 == 1 else config.num_teams - pos + 1
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

    output = {
        "metadata": {
            "timestamp": run_timestamp,
            "strategy": strategy_name,
            "scoring_mode": scoring_mode,
            "opponent_strategies": opponent_strategies_str or "",
            "pts": int(result["pts"]),
            "rank": int(result["rank"]),
        },
        "standings": standings,
        "rosters": rosters,
        "draft_log": draft_log,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    return str(out_path)


def run_user_pick_sequence(*, scoring_mode, strategy, seed, strategy_noise=0.0):
    """Return the user team's drafted player_ids in pick order for one deterministic draft.

    Calls the real pre-refactor pick path with adp_noise=0.0, strategy_noise as given,
    and field_noise=False so opponents draft off clean ADP (fully deterministic when
    strategy_noise=0.0 too).  Used by the golden-master test to pin picks before the
    Phase 3/4 seam refactor.

    Parameters
    ----------
    scoring_mode : str
        "var" or "vona"
    strategy : str
        Key from STRATEGIES (e.g. "default", "two_closers").
    seed : int
        RNG seed (passed to run_simulation; determinism is guaranteed by noise=0 but
        the seed is threaded through for traceability).
    strategy_noise : float
        Fraction of picks that deviate from top rec.  0.0 = fully deterministic.

    Returns
    -------
    list[str]
        player_id strings in the order the user team drafted them.
    """
    ctx = build_board_and_context()
    result = run_simulation(
        ctx,
        strategy_name=strategy,
        scoring_mode=scoring_mode,
        adp_noise=0.0,
        strategy_noise=strategy_noise,
        seed=seed,
        field_noise=False,
    )
    return list(result["user_roster_ids"])


def main():
    parser = argparse.ArgumentParser(description="Simulate a fantasy baseball draft")
    parser.add_argument(
        "--strategy",
        "-s",
        choices=list(STRATEGIES.keys()),
        default="no_punt_cap3",
        help="Draft strategy for your team (default: %(default)s)",
    )
    parser.add_argument(
        "--closer-deadlines",
        type=str,
        default=None,
        help="Comma-separated closer deadline rounds for three_closers strategy (e.g. 4,8,12)",
    )
    parser.add_argument(
        "--no-punt-deadline",
        type=int,
        default=None,
        help="Override the no_punt closer deadline round (default: 8)",
    )
    parser.add_argument(
        "--adp-noise",
        type=float,
        default=0.0,
        help="Std dev of noise added to opponent ADP (e.g. 20 = +/- ~20 ADP spots)",
    )
    parser.add_argument(
        "--strategy-noise",
        type=float,
        default=0.0,
        help="Pick uncertainty: ~68%% take #1 rec, ~27%% take #2, ~4%% take #3 (default: 0 = always #1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for ADP noise",
    )
    parser.add_argument(
        "--opponent-strategies",
        type=str,
        default=None,
        help="Assign strategies to opponents: '3:default,5:three_closers' (team_num:strategy)",
    )
    parser.add_argument(
        "--scoring",
        choices=["var", "vona"],
        default="vona",
        help="Scoring mode: 'var' (Value Above Replacement) or 'vona' (Value Over Next Available)",
    )
    parser.add_argument(
        "--monte-carlo",
        type=int,
        default=0,
        metavar="N",
        help="Run N Monte Carlo simulations on the drafted rosters (injuries + variance)",
    )
    parser.add_argument(
        "--mc-seed",
        type=int,
        default=None,
        help="Random seed for Monte Carlo simulations",
    )
    args = parser.parse_args()

    # Apply custom closer deadlines if provided
    import fantasy_baseball.draft.strategy as strat_mod

    if args.closer_deadlines:
        deadlines = [int(r.strip()) for r in args.closer_deadlines.split(",")]
        strat_mod.THREE_CLOSERS_DEADLINES = deadlines
    if args.no_punt_deadline is not None:
        strat_mod.NO_PUNT_SV_DEADLINE = args.no_punt_deadline

    print("Building draft board...")
    ctx = build_board_and_context()
    config = ctx["config"]

    print(f"Simulating draft | {config.team_name} at position {config.draft_position}")
    print(f"Strategy: {args.strategy} | Scoring: {args.scoring}")
    print(f"League: {config.num_teams} teams, {sum(config.roster_slots.values())} roster slots")
    print(f"Draft pool: {len(ctx['board'])} players")
    print()

    result = run_simulation(
        ctx,
        strategy_name=args.strategy,
        scoring_mode=args.scoring,
        adp_noise=args.adp_noise,
        strategy_noise=args.strategy_noise,
        seed=args.seed,
        opponent_strategies_str=args.opponent_strategies,
    )

    # Save state for post-analysis
    tracker = result["tracker"]
    state_path = PROJECT_ROOT / "data" / "sim_state.json"
    state_data = {
        "current_pick": tracker.current_pick,
        "current_round": tracker.current_round,
        "drafted_players": list(tracker.drafted_players),
        "drafted_ids": list(tracker.drafted_ids),
        "user_roster": result["user_roster"],
        "user_roster_ids": result["user_roster_ids"],
        "roster_slots": dict(config.roster_slots),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state_data, f, indent=2)
    print(f"State saved to {state_path}")

    # Save full simulation output for re-analysis
    sim_path = save_simulation_output(
        result,
        args.strategy,
        args.scoring,
        args.opponent_strategies,
    )
    print(f"Full sim output saved to {sim_path}")

    # === Results ===
    print()
    print("=" * 80)
    print("DRAFT COMPLETE")
    print("=" * 80)

    full_board = ctx["full_board"]
    print(f"\n{config.team_name} ROSTER:")
    for name in result["user_roster"]:
        rows = full_board[full_board["name"] == name]
        if not rows.empty:
            r = rows.iloc[0]
            print(
                f"  {r['name']:<25} {'/'.join(r['positions'][:3]):<12} "
                f"{'hitter' if r['player_type'] == 'hitter' else 'pitcher'}"
            )

    results = result["results"]
    all_cats = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]

    # Print standings
    print("\nPROJECTED ROTO STANDINGS")
    print("=" * 132)
    print(
        f"{'Rk':<3} {'Team':<32} {'Pts':>4}  "
        f"{'R':>5} {'HR':>4} {'RBI':>5} {'SB':>4} {'AVG':>6}  "
        f"{'W':>4} {'K':>5} {'SV':>4} {'ERA':>5} {'WHIP':>6}  "
        f"{'H':>2}/{'P':>2}"
    )
    print("-" * 132)
    for i, t in enumerate(results, 1):
        m = " <<<" if t["team"] == config.team_name else ""
        print(
            f"{i:<3} {t['team']:<32} {t['tot']:>4}  "
            f"{t['R']:>5.0f} {t['HR']:>4.0f} {t['RBI']:>5.0f} "
            f"{t['SB']:>4.0f} {t['AVG']:>6.3f}  "
            f"{t['W']:>4.0f} {t['K']:>5.0f} {t['SV']:>4.0f} "
            f"{t['ERA']:>5.2f} {t['WHIP']:>6.3f}  "
            f"{t['nh']:>2}/{t['np']:>2}{m}"
        )

    # Category breakdown
    print("\nROTO POINTS BY CATEGORY (10=best, 1=worst)")
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
    rank = result["rank"]
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

    # Monte Carlo simulation on drafted rosters
    if args.monte_carlo > 0:
        from fantasy_baseball.simulation import simulate_season

        h_slots = sum(v for k, v in config.roster_slots.items() if k not in ("P", "BN", "IL"))
        p_slots = config.roster_slots.get("P", 9)

        team_players = result["team_players"]
        mc_rng = np.random.default_rng(args.mc_seed)
        n = args.monte_carlo

        print()
        print("=" * 80)
        print(f"MONTE CARLO ({n} simulations)")
        print("=" * 80)

        mc_totals = {tn: [] for tn in team_players}
        mc_wins = {tn: 0 for tn in team_players}
        mc_cat_pts = {tn: {c: [] for c in all_cats} for tn in team_players}
        user_best = None
        user_worst = None

        for _ in range(n):
            sim_stats, _sim_injuries = simulate_season(
                team_players,
                mc_rng,
                h_slots,
                p_slots,
            )
            sim_roto = score_roto_dict(sim_stats)
            ranked = sorted(sim_roto.items(), key=lambda x: x[1]["total"], reverse=True)
            for rk, (tn, pts) in enumerate(ranked, 1):
                total = pts["total"]
                mc_totals[tn].append(total)
                if rk == 1:
                    mc_wins[tn] += 1
                for c in all_cats:
                    mc_cat_pts[tn][c].append(pts.get(f"{c}_pts", 0))

            # Track best/worst for user team
            user_num = next(
                (num for num, name in config.teams.items() if name == config.team_name),
                None,
            )
            if user_num and user_num in sim_roto:
                total = sim_roto[user_num]["total"]
                if user_best is None or total > user_best:
                    user_best = total
                if user_worst is None or total < user_worst:
                    user_worst = total

        print(f"\n{'Team':<32} {'Med':>4} {'P10':>4} {'P90':>4}  {'Win%':>5}")
        print("-" * 60)
        team_order = sorted(
            team_players.keys(),
            key=lambda tn: np.median(mc_totals[tn]),
            reverse=True,
        )
        for tn in team_order:
            tname = config.teams.get(tn, f"Team {tn}")
            totals = np.array(mc_totals[tn])
            med = np.median(totals)
            p10 = np.percentile(totals, 10)
            p90 = np.percentile(totals, 90)
            win_pct = mc_wins[tn] / n * 100
            marker = " <<<" if tname == config.team_name else ""
            print(f"{tname:<32} {med:>4.0f} {p10:>4.0f} {p90:>4.0f}  {win_pct:>4.1f}%{marker}")

        # Category risk for user team
        if user_num:
            print(f"\nCategory risk — {config.team_name}:")
            print(f"  {'Cat':>4} {'Med':>4} {'P10':>4} {'P90':>4}")
            print("  " + "-" * 20)
            for c in all_cats:
                pts = mc_cat_pts[user_num][c]
                med = np.median(pts)
                p10 = np.percentile(pts, 10)
                p90 = np.percentile(pts, 90)
                print(f"  {c:>4} {med:>4.0f} {p10:>4.0f} {p90:>4.0f}")
            if user_best is not None:
                print(f"\n  Best sim: {user_best:.0f} pts  |  Worst sim: {user_worst:.0f} pts")


if __name__ == "__main__":
    main()
