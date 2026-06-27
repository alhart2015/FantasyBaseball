"""Pure per-iteration bench-injury-fill allocation (hitters).

Given an EffectiveRoster's active bodies (each with its stochastic frac_missed)
and bench bodies (each with a sampled per-game counting line), allocate each
active body's missed games to eligible bench bodies (highest per-game value
first, one-body capacity), then replacement-level for any residual. Returns ONLY
the FILL contributions to add on top of the active bodies' own realized counting
(the caller adds that). PURE: no sampler import, no globals -- Phase 4 feeds the
sampled inputs. Pitcher fill is deferred (Phase 5)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fantasy_baseball.mc_roster import PA_PER_GAME, ActiveBody, BenchBody
from fantasy_baseball.scoring import _real_positions
from fantasy_baseball.utils.constants import HITTING_COUNTING


@dataclass(frozen=True)
class ActiveSample:
    body: ActiveBody
    frac_missed: float  # max(0, 1 - scale); stochastic shortfall fraction this iter


@dataclass(frozen=True)
class BenchSample:
    body: BenchBody
    per_game_counts: dict[str, float]  # sampled HITTING_COUNTING stats PER GAME


@dataclass(frozen=True)
class FillResult:
    fill_counts: dict[str, float]  # bench + replacement fill to add on top of own


def allocate_bench_fill(
    actives: list[ActiveSample],
    benches: list[BenchSample],
    replacement_for: Callable[[ActiveBody], dict[str, float]],
) -> FillResult:
    """Allocate missed games to bench (value-ordered, capped) then replacement.

    games_missed = frac_missed * g_ros_adj (the reduced baseline -- the cap;
    NEVER g_ros_full). Largest shortfalls first; per shortfall pick the highest
    per_game_value position-eligible bench body with remaining capacity, assign
    min(shortfall, remaining), decrement both; residual -> replacement per-game
    (replacement total / PA_PER_GAME). One bench body's total assigned games
    <= its g_ros_full. Tie-break: higher per_game_value, then player-id ascending.
    """
    fill: dict[str, float] = {col: 0.0 for col in HITTING_COUNTING}

    # Remaining capacity per bench body (mutated as we allocate).
    remaining = {id(bs): bs.body.g_ros_full for bs in benches}

    # Shortfalls, largest games_missed first.
    shortfalls = [
        (a.frac_missed * a.body.g_ros_adj, a)
        for a in actives
        if a.frac_missed * a.body.g_ros_adj > 0.0
    ]
    shortfalls.sort(key=lambda t: t[0], reverse=True)

    for games_missed, a in shortfalls:
        need = games_missed
        active_pos = _real_positions(a.body.player)

        while need > 0.0:
            # Eligible bench bodies with remaining capacity.
            eligible = [
                bs
                for bs in benches
                if remaining[id(bs)] > 0.0 and (bs.body.eligible_positions & active_pos)
            ]
            if not eligible:
                break
            # Highest per-game value, then player-id ascending (deterministic).
            eligible.sort(key=lambda bs: (-bs.body.per_game_value, _pid(bs.body)))
            bs = eligible[0]
            assign = min(need, remaining[id(bs)])
            for col in HITTING_COUNTING:
                pg = bs.per_game_counts.get(col, 0.0)
                fill[col] += assign * (pg if pg is not None else 0.0)
            remaining[id(bs)] -= assign
            need -= assign

        if need > 0.0:
            repl = replacement_for(a.body)
            for col in HITTING_COUNTING:
                total = repl.get(col, 0.0)
                per_game = (total / PA_PER_GAME) if total is not None else 0.0
                fill[col] += per_game * need

    return FillResult(fill_counts=fill)


def _pid(b: BenchBody) -> str:
    """Player-id for the deterministic tie-break (ascending). Falls back to the
    name::player_type id when yahoo_id is absent (never bare name)."""
    yid = b.player.yahoo_id
    return str(yid) if yid is not None else f"{b.player.name}::{b.player.player_type}"
