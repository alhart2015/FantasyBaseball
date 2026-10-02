"""Pure per-iteration bench-injury-fill allocation (hitters and pitchers).

Given an EffectiveRoster's active bodies (each with its stochastic frac_missed)
and bench bodies (each with a sampled per-unit counting line), allocate each
active body's missed time to eligible bench bodies (highest value first, one-body
capacity), then replacement-level for any residual. Returns ONLY the FILL
contributions to add on top of the active bodies' own realized counting (the
caller adds that). PURE: no sampler import, no globals.

Hitters measure missed time in games and match by position. Pitchers (#393)
measure it in share of the remaining season and match by role: starters cover
starters, relievers cover relievers. Both run the same allocation core."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fantasy_baseball.mc_roster import PA_PER_GAME, ActiveBody, BenchBody, BenchPitcherBody
from fantasy_baseball.scoring import _real_positions
from fantasy_baseball.utils.constants import HITTING_COUNTING, PITCHING_COUNTING

# SV is not filled: the closer role mixture already models the save downside
# (same exemption as the sampler's built-in backfill).
PITCHER_FILL_COLS: tuple[str, ...] = tuple(c for c in PITCHING_COUNTING if c != "sv")


@dataclass(frozen=True)
class ActiveSample:
    body: ActiveBody
    frac_missed: float  # max(0, 1 - scale); stochastic shortfall fraction this iter


@dataclass(frozen=True)
class BenchSample:
    body: BenchBody | BenchPitcherBody
    # Sampled counting stats PER UNIT of cover: per game (hitters) or per share of
    # the remaining season (pitchers).
    per_game_counts: dict[str, float]
    # Units this body may cover THIS iteration: games (g_ros_full * sampled scale)
    # for hitters, season share (sampled scale) for pitchers.
    capacity: float


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
    (replacement total / (repl_ab / PA_PER_GAME) -- implied games, NOT
    / PA_PER_GAME). One bench body's total assigned games <= its per-iteration
    ``capacity`` (g_ros_full * sampled scale; CAN exceed g_ros_full when the body
    is sampled more available than projected). Tie-break: higher per_game_value,
    then player-id ascending.
    """
    shortfalls = [
        (a.frac_missed * a.body.g_ros_adj, a)
        for a in actives
        if a.frac_missed * a.body.g_ros_adj > 0.0
    ]

    def eligible(bench: BenchBody | BenchPitcherBody, a: ActiveSample) -> bool:
        assert isinstance(bench, BenchBody)
        return bool(bench.eligible_positions & _real_positions(a.body.player))

    def repl_per_game(a: ActiveSample) -> dict[str, float]:
        # The replacement line is a FULL-SEASON counting bundle with NO games
        # field, so convert to per-game by dividing each stat by the line's
        # IMPLIED games (ab / PA_PER_GAME -- the shared per-game heuristic),
        # NOT by PA_PER_GAME directly (that would treat a ~65-R full-season
        # total as a per-game rate, ~30x too high).
        repl = replacement_for(a.body)
        repl_ab = repl.get("ab", 0.0) or 0.0
        repl_games = (repl_ab / PA_PER_GAME) if repl_ab > 0 else 0.0
        out: dict[str, float] = {}
        for col in HITTING_COUNTING:
            total = repl.get(col, 0.0) or 0.0
            out[col] = (total / repl_games) if repl_games > 0 else 0.0
        return out

    fill = _allocate(shortfalls, benches, eligible, repl_per_game, HITTING_COUNTING)
    return FillResult(fill_counts=fill)


def _pid(b: BenchBody | BenchPitcherBody) -> str:
    """Player-id for the deterministic tie-break (ascending). Falls back to the
    name::player_type id when yahoo_id is absent (never bare name)."""
    yid = b.player.yahoo_id
    return str(yid) if yid is not None else b.player.player_key


def allocate_pitcher_fill(
    actives: list[ActiveSample],
    benches: list[BenchSample],
    role_of: Callable[[ActiveBody], str],
    replacement_per_share: Callable[[ActiveBody], dict[str, float]],
) -> FillResult:
    """Allocate injured pitchers' missed time to same-role bench arms, then replacement.

    Missed time is a SHARE of the remaining season: ``frac_missed * factor`` (the
    body's displacement factor is the share it was slated to pitch). A bench arm
    covers that time at his own sampled per-share rate, up to his own sampled
    availability (``capacity``), so a covering reliever brings reliever innings.
    A bench arm may cover an active pitcher only when ``BenchPitcherBody.role ==
    role_of(active)``: starters cover starters, relievers cover relievers. The
    residual goes to ``replacement_per_share`` (a replacement-level streamer of
    the same role). SV is never filled (``PITCHER_FILL_COLS``).
    """
    shortfalls = [
        (a.frac_missed * a.body.factor, a) for a in actives if a.frac_missed * a.body.factor > 0.0
    ]

    def eligible(bench: BenchBody | BenchPitcherBody, a: ActiveSample) -> bool:
        assert isinstance(bench, BenchPitcherBody)
        return bench.role == role_of(a.body)

    def repl(a: ActiveSample) -> dict[str, float]:
        return replacement_per_share(a.body)

    return FillResult(fill_counts=_allocate(shortfalls, benches, eligible, repl, PITCHER_FILL_COLS))


def _allocate(
    shortfalls: list[tuple[float, ActiveSample]],
    benches: list[BenchSample],
    eligible: Callable[[BenchBody | BenchPitcherBody, ActiveSample], bool],
    replacement_per_unit: Callable[[ActiveSample], dict[str, float]],
    cols: tuple[str, ...] | list[str],
) -> dict[str, float]:
    """Shared allocation core. Largest shortfalls first; per shortfall pick the
    highest-value eligible bench body with remaining capacity, assign
    min(shortfall, remaining), decrement both; residual -> replacement per unit.
    Tie-break: higher value, then player-id ascending."""
    fill: dict[str, float] = {col: 0.0 for col in cols}
    remaining = {id(bs): bs.capacity for bs in benches}
    ordered = sorted(shortfalls, key=lambda t: t[0], reverse=True)

    for need, a in ordered:
        while need > 0.0:
            candidates = [bs for bs in benches if remaining[id(bs)] > 0.0 and eligible(bs.body, a)]
            if not candidates:
                break
            candidates.sort(key=lambda bs: (-_value(bs.body), _pid(bs.body)))
            bs = candidates[0]
            assign = min(need, remaining[id(bs)])
            for col in cols:
                pg = bs.per_game_counts.get(col, 0.0)
                fill[col] += assign * (pg if pg is not None else 0.0)
            remaining[id(bs)] -= assign
            need -= assign

        if need > 0.0:
            per_unit = replacement_per_unit(a)
            for col in cols:
                fill[col] += per_unit.get(col, 0.0) * need

    return fill


def _value(b: BenchBody | BenchPitcherBody) -> float:
    return b.per_game_value if isinstance(b, BenchBody) else b.per_share_value
