"""Keeper-trade generator: keeper-mutual consolidation trades. Pure math -- the
2026 guardrail is injected as a callable. See
docs/superpowers/specs/2026-07-23-keeper-trade-generator-design.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from fantasy_baseball.models.player import Player
from fantasy_baseball.trades.multi_trade import TradeProposal, _current_active_set


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


def generate_consolidation_trades(
    my_team: str,
    rosters: Mapping[str, Sequence[RosterPlayer]],
    guardrail: Guardrail,
    *,
    max_give: int = 3,
    sweetener: bool = True,
) -> list[TradeSuggestion]:
    me = sorted(rosters[my_team], key=lambda p: p.keeper_value, reverse=True)
    if len(me) < 3:
        return []
    my_top2, my_third = me[:2], me[2].keeper_value
    my_top3_before = top3_sum(me)
    protect = {me[0].player_id, me[1].player_id}
    giveable = [p for p in me if p.player_id not in protect]

    out: list[TradeSuggestion] = []
    for team, roster in rosters.items():
        if team == my_team:
            continue
        opp_top3_before = top3_sum(roster)
        for g in roster:
            if g.keeper_value <= my_third:
                continue
            my_gain = g.keeper_value - my_third
            my_top3_after = my_top2[0].keeper_value + my_top2[1].keeper_value + g.keeper_value
            for pkg in keeper_viable_packages(g, roster, giveable, opp_top3_before, max_give):
                verdict = guardrail(pkg, g)
                if not verdict.ok:
                    continue
                out.append(
                    _suggestion(
                        team,
                        g,
                        pkg,
                        "minimal",
                        my_top3_before,
                        my_top3_after,
                        my_gain,
                        roster,
                        opp_top3_before,
                        verdict,
                    )
                )
                if sweetener:
                    extra = next((p for p in giveable if p not in pkg), None)
                    if extra is not None:
                        spkg = (*pkg, extra)
                        sv = guardrail(spkg, g)
                        if sv.ok:
                            out.append(
                                _suggestion(
                                    team,
                                    g,
                                    spkg,
                                    "sweetened",
                                    my_top3_before,
                                    my_top3_after,
                                    my_gain,
                                    roster,
                                    opp_top3_before,
                                    sv,
                                )
                            )
                break  # first passing minimal package wins for this (team, g)
    out.sort(key=lambda s: s.my_gain, reverse=True)
    return out


def _suggestion(
    team, g, pkg, variant, my_before, my_after, my_gain, roster, opp_before, verdict
) -> TradeSuggestion:
    their_after = top3_sum([p for p in roster if p.player_id != g.player_id] + list(pkg))
    return TradeSuggestion(
        target_team=team,
        acquire=g,
        give=tuple(pkg),
        variant=variant,
        my_top3_before=my_before,
        my_top3_after=my_after,
        my_gain=my_gain,
        their_top3_before=opp_before,
        their_top3_after=their_after,
        their_gain=their_after - opp_before,
        guardrail=verdict,
    )


def build_consolidation_proposal(
    *,
    opponent: str,
    hart_players: list[Player],  # list (not Sequence): _current_active_set expects list
    package_keys: Sequence[str],
    receive_key: str,
    my_adds_keys: Sequence[str],
    opp_drop_keys: Sequence[str],
) -> TradeProposal:
    """Roster-legal 1-for-N consolidation proposal. `my_active_ids` is the post-trade
    active set -- REQUIRED, or evaluate_multi_trade zeroes Hart's active roster
    (multi_trade.py:200-213). opp_active_ids stays empty (opp fallback handles it)."""
    current_active = _current_active_set(hart_players)
    my_active = (current_active - set(package_keys)) | {receive_key} | set(my_adds_keys)
    return TradeProposal(
        opponent=opponent,
        send=list(package_keys),
        receive=[receive_key],
        my_drops=[],
        opp_drops=list(opp_drop_keys),
        my_adds=list(my_adds_keys),
        my_active_ids=my_active,
        opp_active_ids=set(),
    )
