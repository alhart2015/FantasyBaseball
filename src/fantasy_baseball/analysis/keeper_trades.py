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
    my_keepers_after: tuple[RosterPlayer, ...]  # your post-trade top-3, highest first
    their_keepers_after: tuple[RosterPlayer, ...]  # their post-trade top-3, highest first
    guardrail: GuardrailResult


def top3(players: Iterable[RosterPlayer]) -> list[RosterPlayer]:
    """The three highest-keeper_value players (or all, if fewer), highest first."""
    return sorted(players, key=lambda p: p.keeper_value, reverse=True)[:3]


def top3_sum(players: Iterable[RosterPlayer]) -> float:
    return float(sum(p.keeper_value for p in top3(players)))


def keeper_viable_packages(
    acquire: RosterPlayer,
    opp_roster: Sequence[RosterPlayer],
    giveable: Sequence[RosterPlayer],
    opp_top3_before: float,
    max_give: int,
) -> Iterator[tuple[RosterPlayer, ...]]:
    """Packages (subsets of giveable) that strictly lift the opponent's trio once
    `acquire` leaves and the package arrives. Yielded fewest-players first, then
    least total keeper_value given (protect Hart's better surplus). Enumerated
    lazily by size so a consumer that stops at the first hit never builds the
    larger-size combinations."""
    opp_without = [p for p in opp_roster if p.player_id != acquire.player_id]
    for size in range(1, max_give + 1):
        viable = [
            combo
            for combo in combinations(giveable, size)
            if top3_sum([*opp_without, *combo]) > opp_top3_before
        ]
        viable.sort(key=lambda combo: sum(p.keeper_value for p in combo))  # least kv given first
        yield from viable


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
    # Exclude the protected top-2 by player_id (not a me[2:] slice): guards the rare
    # same-name collision CLAUDE.md warns about (e.g. two "Luis Garcia::pitcher" entries
    # on one roster), where a top-2 keeper could otherwise leak into the giveable pool.
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
            # Hart keeps his protected top-2 + the acquired stud (the package is all
            # below the top-2, so removing it can't change the post-trade top-3).
            my_keepers = tuple(top3([*my_top2, g]))
            my_top3_after = float(sum(p.keeper_value for p in my_keepers))
            for pkg in keeper_viable_packages(g, roster, giveable, opp_top3_before, max_give):
                verdict = guardrail(pkg, g)
                if not verdict.ok:
                    continue
                picks = [("minimal", pkg, verdict)]
                if sweetener:
                    extra = next((p for p in giveable if p not in pkg), None)
                    if extra is not None:
                        spkg = (*pkg, extra)
                        sv = guardrail(spkg, g)
                        if sv.ok:
                            picks.append(("sweetened", spkg, sv))
                for variant, package, v in picks:
                    out.append(
                        _suggestion(
                            team,
                            g,
                            package,
                            variant,
                            my_top3_before,
                            my_top3_after,
                            my_gain,
                            my_keepers,
                            roster,
                            opp_top3_before,
                            v,
                        )
                    )
                break  # first passing minimal package wins for this (team, g)
    out.sort(key=lambda s: s.my_gain, reverse=True)
    return out


def _suggestion(
    team, g, pkg, variant, my_before, my_after, my_gain, my_keepers, roster, opp_before, verdict
) -> TradeSuggestion:
    their_keepers = top3([p for p in roster if p.player_id != g.player_id] + list(pkg))
    their_after = float(sum(p.keeper_value for p in their_keepers))
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
        my_keepers_after=tuple(my_keepers),
        their_keepers_after=tuple(their_keepers),
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
