"""Multi-player trade evaluator.

Generalizes the 1-for-1 trade math in trades.evaluate to arbitrary N-for-M
swaps with optional drops on either side and optional waiver pickups on
the user's side. Reports delta-roto for the user's team only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
