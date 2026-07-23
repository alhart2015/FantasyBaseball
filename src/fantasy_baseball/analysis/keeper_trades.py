"""Keeper-trade generator: keeper-mutual consolidation trades. Pure math -- the
2026 guardrail is injected as a callable. See
docs/superpowers/specs/2026-07-23-keeper-trade-generator-design.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class RosterPlayer:
    player_id: str  # Player.player_key = "name::player_type"
    name: str
    keeper_value: float  # discounted multi-year VAR at the chosen discount


@dataclass(frozen=True)
class GuardrailResult:
    legal: bool
    delta_total: float  # Hart's projected 2026 roto-point change
    ok: bool  # legal AND delta_total >= -threshold


Guardrail = Callable[[Sequence["RosterPlayer"], "RosterPlayer"], GuardrailResult]


@dataclass(frozen=True)
class TradeSuggestion:
    target_team: str
    acquire: RosterPlayer
    give: tuple[RosterPlayer, ...]
    variant: str  # "minimal" | "sweetened"
    my_top3_before: float
    my_top3_after: float
    my_gain: float
    their_top3_before: float
    their_top3_after: float
    their_gain: float
    guardrail: GuardrailResult


def top3_sum(players: Iterable[RosterPlayer]) -> float:
    return float(sum(sorted((p.keeper_value for p in players), reverse=True)[:3]))


def keeper_viable_packages(
    acquire: RosterPlayer,
    opp_roster: Sequence[RosterPlayer],
    giveable: Sequence[RosterPlayer],
    opp_top3_before: float,
    max_give: int,
) -> Iterator[tuple[RosterPlayer, ...]]:
    """Packages (subsets of giveable) that strictly lift the opponent's trio once
    `acquire` leaves and the package arrives. Ordered fewest-players, then least
    total keeper_value given (protect Hart's better surplus)."""
    opp_without = [p for p in opp_roster if p.player_id != acquire.player_id]
    candidates: list[tuple[int, float, tuple[RosterPlayer, ...]]] = []
    for size in range(1, max_give + 1):
        for combo in combinations(giveable, size):
            if top3_sum([*opp_without, *combo]) > opp_top3_before:
                candidates.append((size, sum(p.keeper_value for p in combo), combo))
    candidates.sort(key=lambda c: (c[0], c[1]))  # fewest players, then least kv given
    for _size, _cost, combo in candidates:
        yield combo
