"""Multi-player trade evaluator.

Generalizes the 1-for-1 trade math in trades.evaluate to arbitrary N-for-M
swaps with optional drops on either side and optional waiver pickups on
the user's side. Reports delta-roto for the user's team only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from fantasy_baseball.models.player import Player
from fantasy_baseball.models.positions import BENCH_SLOTS, IL_SLOTS
from fantasy_baseball.scoring import score_roto_dict
from fantasy_baseball.trades.evaluate import (
    aggregate_player_stats,
    apply_swap_delta,
)
from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category


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


@dataclass
class CategoryDelta:
    """Per-category before/after/delta for a single roto stat."""

    before: float
    after: float
    delta: float  # roto points change (e.g. +0.5 = half a category point)


@dataclass
class MultiTradeResult:
    """Output of :func:`evaluate_multi_trade`."""

    legal: bool
    reason: str | None
    delta_total: float
    categories: dict[str, CategoryDelta]


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
    projected_standings: list[dict],
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    roster_slots: dict,
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
        opp_drops = _resolve_keys(proposal.opp_drops, opp_idx)
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
    # Combine all players that may appear in `after_mine`: current roster
    # (my_idx), incoming trade pieces (received), and waiver adds (my_adds).
    # Omitting `received` would silently drop incoming players from my_gains.
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

    # Opp: treat all non-IL as active. They lose received+opp_drops, gain sent.
    opp_loses = aggregate_player_stats(received + opp_drops)
    opp_gains = aggregate_player_stats(sent)

    # --- 4. Apply deltas to baseline and score -------------------------------
    if not any(t["name"] == hart_name for t in projected_standings):
        return MultiTradeResult(
            legal=False,
            reason=f"Team {hart_name} missing from projected_standings",
            delta_total=0.0,
            categories={},
        )

    post = []
    for t in projected_standings:
        if t["name"] == hart_name:
            post.append(
                {"name": t["name"], "stats": apply_swap_delta(t["stats"], my_loses, my_gains)}
            )
        elif t["name"] == proposal.opponent:
            post.append(
                {"name": t["name"], "stats": apply_swap_delta(t["stats"], opp_loses, opp_gains)}
            )
        else:
            post.append(t)

    before_roto = score_roto_dict(
        {t["name"]: t["stats"] for t in projected_standings},
        team_sds=team_sds,
    )
    after_roto = score_roto_dict(
        {t["name"]: t["stats"] for t in post},
        team_sds=team_sds,
    )

    categories: dict[str, CategoryDelta] = {}
    total_delta = 0.0
    for cat in ALL_CATEGORIES:
        before_pts = before_roto[hart_name][f"{cat.value}_pts"]
        after_pts = after_roto[hart_name][f"{cat.value}_pts"]
        delta = after_pts - before_pts
        categories[cat.value] = CategoryDelta(
            before=before_pts,
            after=after_pts,
            delta=delta,
        )
        total_delta += delta

    return MultiTradeResult(
        legal=True,
        reason=None,
        delta_total=total_delta,
        categories=categories,
    )


def _target_size(roster_slots: dict) -> int:
    """Total active + bench slots (excludes IL)."""
    return sum(v for k, v in roster_slots.items() if k != "IL")


def _can_roster_after(
    roster: list[Player],
    removals: list[str],
    additions: list[Player],
    roster_slots: dict,
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
    ros_projections: dict,
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
