"""Multi-player trade evaluator.

Generalizes the 1-for-1 trade math in trades.evaluate to arbitrary N-for-M
swaps with optional drops on either side and optional waiver pickups on
the user's side. Reports delta-roto for the user's team only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fantasy_baseball.models.player import Player


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


def evaluate_multi_trade(*args, **kwargs) -> MultiTradeResult:  # pragma: no cover - placeholder
    raise NotImplementedError("evaluate_multi_trade is implemented in Task 3")


def player_key(player: Player) -> str:
    """Canonical player identifier: ``name::player_type`` (hitter|pitcher)."""
    return f"{player.name}::{player.player_type}"


def _non_il_size(roster: list[Player]) -> int:
    return sum(1 for p in roster if (getattr(p, "selected_position", None) or "") != "IL")


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
