"""Multi-player trade evaluator.

Generalizes the 1-for-1 trade math in trades.evaluate to arbitrary N-for-M
swaps with optional drops on either side and optional waiver pickups on
the user's side. Reports delta-roto for the user's team only.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from fantasy_baseball.models.player import Player
from fantasy_baseball.models.positions import BENCH_SLOTS, IL_SLOTS
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.scoring import score_roto_dict
from fantasy_baseball.trades.evaluate import (
    aggregate_player_stats,
    apply_swap_delta,
)
from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category

logger = logging.getLogger(__name__)


@dataclass
class TradeProposal:
    """A multi-player trade proposal submitted from the UI.

    All player identifiers are ``"<name>::<player_type>"`` keys (the same
    format used in rankings caches).
    """

    opponent: str
    send: list[str] = field(default_factory=list)
    receive: list[str] = field(default_factory=list)
    my_drops: list[str] = field(default_factory=list)
    opp_drops: list[str] = field(default_factory=list)
    my_adds: list[str] = field(default_factory=list)
    my_active_ids: set[str] = field(default_factory=set)
    opp_active_ids: set[str] = field(default_factory=set)


@dataclass
class CategoryDelta:
    """Per-category before/after/delta for a single roto stat."""

    before: float
    after: float
    delta: float  # roto points change (e.g. +0.5 = half a category point)


@dataclass
class CategoryView:
    """Per-category before/after/delta tuple, used in roto / eROTO / stat-totals views."""

    before: float
    after: float
    delta: float


@dataclass
class ViewBlock:
    """One delta view (roto, eROTO, or stat totals) - total + per-category."""

    delta_total: float  # 0.0 for stat_totals (no scalar total is meaningful)
    categories: dict[str, CategoryView]


@dataclass
class MultiTradeResult:
    """Output of :func:`evaluate_multi_trade`."""

    legal: bool
    reason: str | None
    delta_total: float
    categories: dict[str, CategoryDelta]
    roto: ViewBlock = field(default_factory=lambda: ViewBlock(delta_total=0.0, categories={}))
    ev_roto: ViewBlock = field(default_factory=lambda: ViewBlock(delta_total=0.0, categories={}))
    stat_totals: ViewBlock = field(
        default_factory=lambda: ViewBlock(delta_total=0.0, categories={})
    )
    band: dict[str, float | str] | None = None
    # --- Opponent-side parallel fields -------------------------------------
    opp_delta_total: float = 0.0
    opp_categories: dict[str, CategoryDelta] = field(default_factory=dict)
    opp_roto: ViewBlock = field(default_factory=lambda: ViewBlock(delta_total=0.0, categories={}))
    opp_ev_roto: ViewBlock = field(
        default_factory=lambda: ViewBlock(delta_total=0.0, categories={})
    )
    opp_stat_totals: ViewBlock = field(
        default_factory=lambda: ViewBlock(delta_total=0.0, categories={})
    )
    opp_band: dict[str, float | str] | None = None


def player_key(player: Player) -> str:
    """Canonical player identifier: ``name::player_type`` (hitter|pitcher)."""
    return f"{player.name}::{player.player_type}"


def _non_il_size(roster: list[Player]) -> int:
    return sum(1 for p in roster if p.selected_position not in IL_SLOTS)


def _index_roster(roster: list[Player]) -> dict[str, Player]:
    return {player_key(p): p for p in roster}


def _resolve_keys(keys: list[str], index: dict[str, Player]) -> list[Player]:
    missing = [k for k in keys if k not in index]
    if missing:
        raise KeyError(f"Unknown player key(s): {missing}")
    return [index[k] for k in keys]


def _current_active_set(roster: list[Player]) -> set[str]:
    """Keys of roster players not currently on BN, IL, IL+, DL, or DL+."""
    return {player_key(p) for p in roster if p.selected_position not in BENCH_SLOTS}


def evaluate_multi_trade(
    *,
    proposal: TradeProposal,
    hart_name: str,
    hart_roster: list[Player],
    opp_rosters: dict[str, list[Player]],
    waiver_pool: dict[str, Player],
    projected_standings: ProjectedStandings,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    roster_slots: dict[str, int],
    fraction_remaining: float,
) -> MultiTradeResult:
    """Evaluate an arbitrary N-for-M trade with optional drops and adds.

    See docs/superpowers/specs/2026-04-20-multi-player-trade-evaluator-design.md
    for the full math.  Returns ``MultiTradeResult`` with per-category and
    total delta-roto for ``hart_name``'s team.
    """
    # --- 1. Resolve keys -----------------------------------------------------
    if proposal.opponent not in opp_rosters:
        return MultiTradeResult(
            legal=False,
            reason=f"Unknown opponent: {proposal.opponent}",
            delta_total=0.0,
            categories={},
        )

    my_idx = _index_roster(hart_roster)
    opp_idx = _index_roster(opp_rosters[proposal.opponent])

    try:
        sent = _resolve_keys(proposal.send, my_idx)
        received = _resolve_keys(proposal.receive, opp_idx)
        # Resolve drops to validate keys exist (KeyError on unknown); objects
        # themselves are unused since we pass the key lists to _can_roster_after.
        _resolve_keys(proposal.my_drops, my_idx)
        _resolve_keys(proposal.opp_drops, opp_idx)
        my_adds = _resolve_keys(proposal.my_adds, waiver_pool)
    except KeyError as exc:
        return MultiTradeResult(
            legal=False,
            reason=str(exc),
            delta_total=0.0,
            categories={},
        )

    # --- 2. Legality ---------------------------------------------------------
    my_removals = proposal.send + proposal.my_drops
    my_additions = received + my_adds
    my_ok, my_reason = _can_roster_after(
        hart_roster,
        my_removals,
        my_additions,
        roster_slots,
    )
    if not my_ok:
        return MultiTradeResult(
            legal=False,
            reason=f"My team: {my_reason}",
            delta_total=0.0,
            categories={},
        )

    opp_removals = proposal.receive + proposal.opp_drops
    opp_additions = sent
    opp_ok, opp_reason = _can_roster_after(
        opp_rosters[proposal.opponent],
        opp_removals,
        opp_additions,
        roster_slots,
    )
    if not opp_ok:
        return MultiTradeResult(
            legal=False,
            reason=f"Opponent: {opp_reason}",
            delta_total=0.0,
            categories={},
        )

    # --- 3. Build active-set deltas ------------------------------------------
    # MY side: use proposal.my_active_ids as the post-trade active set.
    # When my_active_ids is empty (legacy callers), this treats my whole
    # current active roster as "leaving" -- callers are expected to provide
    # an active set when they want a meaningful delta.
    all_mine_by_key = {
        **my_idx,
        **{player_key(p): p for p in received},
        **{player_key(p): p for p in my_adds},
    }
    before_mine = _current_active_set(hart_roster)
    after_mine = set(proposal.my_active_ids)

    mine_leaving = [all_mine_by_key[k] for k in before_mine - after_mine if k in all_mine_by_key]
    mine_entering = [all_mine_by_key[k] for k in after_mine - before_mine if k in all_mine_by_key]
    my_loses = aggregate_player_stats(mine_leaving)
    my_gains = aggregate_player_stats(mine_entering)

    # OPP side: prefer active-set delta when opp_active_ids is provided;
    # otherwise fall back to the roster-level computation (today's behavior).
    all_opp_by_key = {
        **opp_idx,
        **{player_key(p): p for p in sent},
    }
    before_opp = _current_active_set(opp_rosters[proposal.opponent])

    if proposal.opp_active_ids:
        after_opp = set(proposal.opp_active_ids)
        opp_leaving = [all_opp_by_key[k] for k in before_opp - after_opp if k in all_opp_by_key]
        opp_entering = [all_opp_by_key[k] for k in after_opp - before_opp if k in all_opp_by_key]
        opp_loses = aggregate_player_stats(opp_leaving)
        opp_gains = aggregate_player_stats(opp_entering)
    else:
        # Legacy fallback: no opp_active_ids provided. Assume opp_drops
        # vacate active slots and sent slides in. Bench-only drops are
        # treated as no-ops on opp's stat line (consistent with the band's
        # view, since bench players never contribute to projected_standings).
        received_keys = {player_key(p) for p in received}
        sent_keys = {player_key(p) for p in sent}
        opp_drop_keys = set(proposal.opp_drops)
        after_opp = (before_opp - received_keys - opp_drop_keys) | sent_keys
        opp_leaving = [all_opp_by_key[k] for k in before_opp - after_opp if k in all_opp_by_key]
        opp_entering = [all_opp_by_key[k] for k in after_opp - before_opp if k in all_opp_by_key]
        opp_loses = aggregate_player_stats(opp_leaving)
        opp_gains = aggregate_player_stats(opp_entering)

    # --- 4. Apply deltas to baseline and score -------------------------------
    if not any(e.team_name == hart_name for e in projected_standings.entries):
        return MultiTradeResult(
            legal=False,
            reason=f"Team {hart_name} missing from projected_standings",
            delta_total=0.0,
            categories={},
        )

    before_stats = {e.team_name: e.stats.to_dict() for e in projected_standings.entries}
    after_stats = dict(before_stats)
    after_stats[hart_name] = apply_swap_delta(before_stats[hart_name], my_loses, my_gains)
    after_stats[proposal.opponent] = apply_swap_delta(
        before_stats[proposal.opponent], opp_loses, opp_gains
    )

    # Roto Points (integer ranks)
    roto_before = score_roto_dict(before_stats)
    roto_after = score_roto_dict(after_stats)

    # eROTO (fractional EV-based) — falls back to roto when team_sds is None
    ev_roto_before = score_roto_dict(before_stats, team_sds=team_sds)
    ev_roto_after = score_roto_dict(after_stats, team_sds=team_sds)

    def _build_view(team_name: str, before_pts, after_pts) -> ViewBlock:
        cats: dict[str, CategoryView] = {}
        total = 0.0
        for cat in ALL_CATEGORIES:
            b = before_pts[team_name][f"{cat.value}_pts"]
            a = after_pts[team_name][f"{cat.value}_pts"]
            cats[cat.value] = CategoryView(before=b, after=a, delta=a - b)
            total += a - b
        return ViewBlock(delta_total=total, categories=cats)

    def _build_stat_totals(team_name: str) -> ViewBlock:
        cats: dict[str, CategoryView] = {}
        for cat in ALL_CATEGORIES:
            b = float(before_stats[team_name][cat.value])
            a = float(after_stats[team_name][cat.value])
            cats[cat.value] = CategoryView(before=b, after=a, delta=a - b)
        return ViewBlock(delta_total=0.0, categories=cats)

    def _build_categories(team_name: str) -> dict[str, CategoryDelta]:
        cats: dict[str, CategoryDelta] = {}
        for cat in ALL_CATEGORIES:
            b = ev_roto_before[team_name][f"{cat.value}_pts"]
            a = ev_roto_after[team_name][f"{cat.value}_pts"]
            cats[cat.value] = CategoryDelta(before=b, after=a, delta=a - b)
        return cats

    roto_view = _build_view(hart_name, roto_before, roto_after)
    ev_roto_view = _build_view(hart_name, ev_roto_before, ev_roto_after)
    stat_totals_view = _build_stat_totals(hart_name)
    categories = _build_categories(hart_name)
    total_delta = ev_roto_view.delta_total

    opp_name = proposal.opponent
    opp_roto_view = _build_view(opp_name, roto_before, roto_after)
    opp_ev_roto_view = _build_view(opp_name, ev_roto_before, ev_roto_after)
    opp_stat_totals_view = _build_stat_totals(opp_name)
    opp_categories = _build_categories(opp_name)
    opp_total_delta = opp_ev_roto_view.delta_total

    # --- 5. Monte-Carlo confidence bands (my + opp) --------------------------
    from fantasy_baseball.lineup.delta_roto import compute_delta_roto_band

    field_stats = projected_standings.field_stats(hart_name)
    before_players = [my_idx[k] for k in before_mine if k in my_idx]
    after_players = [all_mine_by_key[k] for k in after_mine if k in all_mine_by_key]
    band_result = compute_delta_roto_band(
        before_players,
        after_players,
        field_stats,
        hart_name,
        fraction_remaining,
        projected_standings=projected_standings,
        team_sds=team_sds,
    )

    try:
        opp_field_stats = projected_standings.field_stats(opp_name)
        opp_before_players = [opp_idx[k] for k in before_opp if k in opp_idx]
        opp_after_players = [all_opp_by_key[k] for k in after_opp if k in all_opp_by_key]
        opp_band_result = compute_delta_roto_band(
            opp_before_players,
            opp_after_players,
            opp_field_stats,
            opp_name,
            fraction_remaining,
            projected_standings=projected_standings,
            team_sds=team_sds,
        )
        opp_band_dict = opp_band_result.to_dict()
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        # Opp band is best-effort. Math-domain failures (e.g. zero variance for
        # the team) fall back to None and let the UI show the plain delta.
        # Programming errors are intentionally not caught so they surface.
        logger.warning("opp band failed for %s: %s", opp_name, exc)
        opp_band_dict = None

    return MultiTradeResult(
        legal=True,
        reason=None,
        delta_total=total_delta,
        categories=categories,
        roto=roto_view,
        ev_roto=ev_roto_view,
        stat_totals=stat_totals_view,
        band=band_result.to_dict(),
        opp_delta_total=opp_total_delta,
        opp_categories=opp_categories,
        opp_roto=opp_roto_view,
        opp_ev_roto=opp_ev_roto_view,
        opp_stat_totals=opp_stat_totals_view,
        opp_band=opp_band_dict,
    )


def _target_size(roster_slots: dict[str, int]) -> int:
    """Total active + bench slots (excludes IL)."""
    return sum(v for k, v in roster_slots.items() if k != "IL")


def _can_roster_after(
    roster: list[Player],
    removals: list[str],
    additions: list[Player],
    roster_slots: dict[str, int],
) -> tuple[bool, str | None]:
    """Size-only legality check for a multi-player proposal.

    ``roster`` is the current roster including IL players.
    ``removals`` is a list of ``player_key()`` strings for players leaving
    (traded away or dropped).  ``additions`` is a list of Player objects
    coming in (traded in or picked up from waivers).

    The roster is considered legal iff
    ``non_il_count - |removals| + |additions| == target_size``,
    where ``target_size = sum(roster_slots) - roster_slots["IL"]``
    (23 in this league: 12 active hitters + 9 pitchers + 2 bench).
    IL-listed players neither count in the baseline nor are removed by
    this trade.

    Returns ``(True, None)`` if legal, otherwise ``(False, reason)``.
    """
    target = _target_size(roster_slots)
    non_il = _non_il_size(roster)
    new_size = non_il - len(removals) + len(additions)
    if new_size != target:
        return False, (
            f"Roster would have {new_size} non-IL players; league requires exactly {target}"
        )
    return True, None


def build_waiver_pool(
    hart_roster: list[Player],
    opp_rosters: dict[str, list[Player]],
    ros_projections: dict[str, list[dict[str, Any]]],
) -> dict[str, Player]:
    """Build a keyed player pool of everyone with ROS projections who is
    not on any roster.

    ``ros_projections`` is the cached ROS projection dict: ``{"hitters":
    [{...}], "pitchers": [{...}]}``. Each entry is in the format accepted
    by :meth:`Player.from_dict`.

    Returned dict is keyed by :func:`player_key` (``"name::player_type"``).
    """
    rostered = {player_key(p) for p in hart_roster}
    for roster in opp_rosters.values():
        rostered |= {player_key(p) for p in roster}

    pool: dict[str, Player] = {}
    for bucket, player_type in (("hitters", "hitter"), ("pitchers", "pitcher")):
        for d in ros_projections.get(bucket, []):
            payload = dict(d)
            payload.setdefault("player_type", player_type)
            player = Player.from_dict(payload)
            key = player_key(player)
            if key in rostered:
                continue
            pool[key] = player
    return pool
