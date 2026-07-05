"""MC setup: classification + IL displacement -> effective active set + bench fill pool.

Reuses ERoto's _classify_roster + _compute_displacement_factors so the MC's IL
handling agrees with ERoto by construction. Pure/deterministic -- runs once per
team at MC setup on the ROS means. Consumed by the per-iteration fill engine
(Phase 3) and the MC integration (Phase 4); nothing consumes it yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.scoring import (
    LeagueContext,
    _classify_roster,
    _compute_displacement_factors,
    _real_positions,
)
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.constants import Category

PA_PER_GAME: float = 4.3  # shared per-game constant (Phase 3 reuses; do not duplicate)


@dataclass(frozen=True)
class ActiveBody:
    player: Player
    factor: float  # displacement factor (1.0 if undisplaced)
    g_ros_adj: float  # factor * g_ros_full -- the games-missed multiplier / fill cap


@dataclass(frozen=True)
class BenchBody:
    player: Player
    g_ros_full: float  # ROS games (per-game-value denominator)
    per_game_value: float  # ROS SGP per ROS game
    eligible_positions: frozenset[Position]


@dataclass(frozen=True)
class EffectiveRoster:
    active: list[ActiveBody]  # active-slot + IL bodies, with factors
    bench: list[BenchBody]  # healthy-bench HITTER fill pool


def _g_ros_full(p: Player) -> float:
    """ROS games, with a PA-derived fallback when the projection lacks g.

    Never trusts a literal g==0 as 'plays zero games' (the falsy-zero footgun):
    derives from ROS PA via PA_PER_GAME. Pitchers fall back to their own g (now
    plumbed) or, absent that, are left at 0 -- pitcher bench-fill is deferred.
    """
    ros = p.rest_of_season
    if ros is None:
        return 0.0
    g_raw = getattr(ros, "g", 0)
    g = float(g_raw) if g_raw is not None else 0.0
    if g > 0:
        return g
    if p.player_type == PlayerType.HITTER:
        pa_raw = getattr(ros, "pa", 0)
        pa = float(pa_raw) if pa_raw is not None else 0.0
        return pa / PA_PER_GAME if pa > 0 else 0.0
    return 0.0


def build_effective_roster(
    roster: list[Player],
    league_context: LeagueContext,
    denoms: dict[Category, float] | None = None,
) -> EffectiveRoster:
    """Turn a roster + LeagueContext into the effective active set and bench fill pool.

    ``active`` carries active-slot bodies + IL bodies, each with its displacement
    factor (from ERoto's ``_compute_displacement_factors``) and ``g_ros_adj``
    (= factor * g_ros_full). ``bench`` is the healthy-bench HITTER fill pool;
    healthy bench pitchers are dropped (pitcher bench-fill is deferred to Phase 5).

    ``denoms`` are the resolved league SGP denominators used to score each
    bench body's ``per_game_value`` (the fill engine's ordering key); ``None``
    falls back to the library defaults.

    Factors come back name-keyed; they are re-keyed onto ``Player`` objects, with
    a guard that raises ``ValueError`` on a duplicate name within the active+IL
    set (a name-scoped factor cannot be re-keyed by identity unambiguously).
    """
    active, il, bench = _classify_roster([p for p in roster if isinstance(p, Player)])

    factors_by_name = _compute_displacement_factors(active, il, league_context=league_context)

    # Re-key factors onto Player objects; guard same-name collisions in active+il.
    bodies = [*il, *active]
    seen: set[str] = set()
    for b in bodies:
        if b.name in seen and b.name in factors_by_name:
            raise ValueError(
                f"Ambiguous displacement factor: duplicate name {b.name!r} in the "
                "active+IL set; cannot re-key a name-scoped factor by identity."
            )
        seen.add(b.name)

    active_bodies: list[ActiveBody] = []
    for b in bodies:
        factor = factors_by_name.get(b.name, 1.0)
        active_bodies.append(ActiveBody(player=b, factor=factor, g_ros_adj=factor * _g_ros_full(b)))

    bench_bodies: list[BenchBody] = []
    for b in bench:
        if b.player_type != PlayerType.HITTER:
            continue  # pitcher bench-fill deferred (Phase 5); healthy bench pitchers excluded
        gf = _g_ros_full(b)
        sgp = (
            calculate_player_sgp(b.rest_of_season, denoms) if b.rest_of_season is not None else 0.0
        )
        per_game = (sgp / gf) if gf > 0 else 0.0
        bench_bodies.append(
            BenchBody(
                player=b,
                g_ros_full=gf,
                per_game_value=per_game,
                eligible_positions=_real_positions(b),
            )
        )

    return EffectiveRoster(active=active_bodies, bench=bench_bodies)


def build_effective_rosters(
    team_rosters: dict[str, list[Player]],
    eos_baseline: dict,
    team_sds: dict,
    fraction_remaining: float,
    denoms: dict[Category, float] | None = None,
) -> dict[str, EffectiveRoster]:
    """Build each team's EffectiveRoster from the live Player rosters.

    For each team, constructs the SAME LeagueContext (baseline with this team
    excluded, team_sds, fraction_remaining) the standings build used, so the
    MC's IL handling agrees with ERoto by construction, then delegates to
    ``build_effective_roster`` (which classifies and filters the roster).
    ``denoms`` (resolved league SGP denominators, or ``None`` for defaults)
    is forwarded to the bench ``per_game_value`` scoring.
    """
    out: dict[str, EffectiveRoster] = {}
    for tname, roster in team_rosters.items():
        lc = LeagueContext(
            baseline_other_team_stats={t: s for t, s in eos_baseline.items() if t != tname},
            team_sds=team_sds,
            team_name=tname,
            fraction_remaining=fraction_remaining,
        )
        out[tname] = build_effective_roster(roster, lc, denoms)
    return out
