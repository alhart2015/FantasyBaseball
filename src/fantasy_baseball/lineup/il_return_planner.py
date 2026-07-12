"""IL return planner -- compute the optimal legal roster + transaction plan
when injured-list players are reactivated.

When IL players come off the IL they temporarily push the roster over the
active+bench body-count cap, forcing a drop plus an active/bench reshuffle.
Given the IL players a manager wants to activate, this module computes the
forced drops and returns the top transaction plans ranked by deltaRoto.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

from fantasy_baseball.lineup.delta_roto import band_reference_lineup, compute_delta_roto_band
from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS, Position
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.sgp.denominators import SgpOverrides, get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.constants import Category

logger = logging.getLogger(__name__)


@dataclass
class Move:
    """A single roster transaction for one player."""

    name: str
    player_type: str
    from_slot: str
    to_slot: str  # active slot label, "BN", or "DROP"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MovePlan:
    """One complete plan: the forced drop(s) plus the resulting move list."""

    drops: list[str]
    moves: list[Move]
    delta_roto: float
    band: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "drops": list(self.drops),
            "moves": [m.to_dict() for m in self.moves],
            "delta_roto": round(self.delta_roto, 2),
            "band": self.band,
        }


@dataclass
class IlReturnPlanResult:
    """All plans for activating a chosen set of IL players."""

    activating: list[str]
    capacity: int
    overflow: int
    plans: list[MovePlan] = field(default_factory=list)
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "activating": list(self.activating),
            "capacity": self.capacity,
            "overflow": self.overflow,
            "plans": [p.to_dict() for p in self.plans],
            "warning": self.warning,
        }


def roster_capacity(roster_slots: dict[str, int]) -> int:
    """Active + bench slot count -- every slot except IL slots.

    IL slots are exempt from Yahoo's active-roster size limit, so they do
    not count toward the body-count cap that forces a drop.
    """
    total = 0
    for key, count in roster_slots.items():
        pos = key if isinstance(key, Position) else Position.parse(key)
        if pos in IL_SLOTS:
            continue
        total += count
    return total


def _counts_against_cap(p: Player) -> bool:
    """True if this body counts against the active+bench cap.

    Slot-based, NOT status-based: a BN+IL-status player (Yahoo lets you
    stash an IL guy on the bench) still counts; only a true IL-slot body
    is exempt. This is why activating an IL-slot player is what forces a
    drop.
    """
    return p.selected_position not in IL_SLOTS


def _activate(p: Player) -> Player:
    """Return a copy with IL signals cleared so the optimizer treats the
    player as active-eligible. ``is_on_il()`` and the optimizers would
    otherwise keep excluding a returning IL player."""
    return dataclasses.replace(p, status="", selected_position=None)


def _build_pool(roster: list[Player], activating_il: list[Player]) -> list[Player]:
    """The set of players competing for active/bench slots after activation.

    = current counted bodies (active + healthy bench + any BN+IL-status
    players) UNION the activating players that were in true IL slots.
    Activating players get their IL signals cleared. Unchecked IL players
    stay parked and are excluded.
    """
    # Key on player_key, matching _make_plan/_build_moves: a two-way player is
    # two rows sharing a name, so activating his pitcher row must not clear the
    # IL signals on (or dedup away) his separate hitter row.
    activating_keys = {p.player_key for p in activating_il}
    counted = [p for p in roster if _counts_against_cap(p)]
    counted_keys = {p.player_key for p in counted}
    extra = [p for p in activating_il if p.player_key not in counted_keys]
    pool: list[Player] = []
    for p in counted + extra:
        pool.append(_activate(p) if p.player_key in activating_keys else p)
    return pool


def _solve_lineup(
    pool: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float,
):
    """Run both optimizers over ``pool``; return
    ``(hitter_assignments, pitcher_starters, pitcher_bench)``.

    Both optimizer calls skip per-starter band computation (the planner
    computes a plan-level band separately): the hitter call passes
    ``fraction_remaining=None`` (it needs no displacement context), and the
    pitcher call passes the real ``fraction_remaining`` with
    ``compute_bands=False`` so its pair-swap pool model sizes a returning IL
    pitcher's slot-share against the remaining season -- this planner runs
    mid-season specifically to evaluate IL returns, so that sizing is
    load-bearing.
    """
    hitters = [p for p in pool if p.player_type != PlayerType.PITCHER]
    pitchers = [p for p in pool if p.player_type == PlayerType.PITCHER]
    hitter_assignments = optimize_hitter_lineup(
        hitters=hitters,
        full_roster=pool,
        projected_standings=projected_standings,
        team_name=team_name,
        roster_slots=roster_slots,
        team_sds=team_sds,
        fraction_remaining=None,
    )
    pitcher_starters, pitcher_bench = optimize_pitcher_lineup(
        pitchers=pitchers,
        full_roster=pool,
        projected_standings=projected_standings,
        team_name=team_name,
        slots=roster_slots.get("P", 9),
        team_sds=team_sds,
        fraction_remaining=fraction_remaining,
        compute_bands=False,
    )
    return hitter_assignments, pitcher_starters, pitcher_bench


def _slot_value(p: Player) -> str:
    """The player's current slot label, defaulting to BN when unset."""
    return p.selected_position.value if p.selected_position is not None else "BN"


def _build_moves(
    roster: list[Player],
    pool: list[Player],
    hitter_assignments,
    pitcher_starters,
    dropped_keys: set[str],
) -> list[Move]:
    """Build the transaction list for one plan.

    ``from_slot`` is the player's CURRENT slot on ``roster`` (so a returning
    IL player reads as ``IL`` and Webb as ``BN``); ``to_slot`` is the
    assigned active slot, ``BN``, or ``DROP``. Only players whose slot
    changes get a move. Sorted by name for deterministic output.

    Keyed on :attr:`Player.player_key` so a two-way player's hitter and pitcher
    rows resolve their slots independently (bare name would collide).
    """
    orig_slot = {p.player_key: _slot_value(p) for p in roster}

    active_slot: dict[str, str] = {a.player.player_key: a.slot.value for a in hitter_assignments}
    for s in pitcher_starters:
        active_slot[s.player.player_key] = "P"

    moves: list[Move] = []
    for p in pool:
        key = p.player_key
        frm = orig_slot.get(key, "BN")
        if key in dropped_keys:
            to = "DROP"
        elif key in active_slot:
            to = active_slot[key]
        else:
            to = "BN"
        if frm != to:
            moves.append(
                Move(
                    name=p.name,
                    player_type=p.player_type.value,
                    from_slot=frm,
                    to_slot=to,
                )
            )
    moves.sort(key=lambda m: m.name)
    return moves


def _sgp(p: Player, denoms) -> float:
    if p.rest_of_season is None:
        return 0.0
    return calculate_player_sgp(p.rest_of_season, denoms)


def _make_plan(
    roster: list[Player],
    pool: list[Player],
    dropset: tuple[Player, ...],
    base_h,
    base_ps,
    before_active: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds,
    fraction_remaining: float,
    bn_slots: int,
    band_reference: list[Player] | None,
) -> MovePlan | None:
    """Solve one drop-set into a MovePlan, or None if infeasible.

    Only re-solves the side(s) the drop touches; the untouched side reuses
    the pre-drop baseline (the lineup there is identical). Feasibility:
    the benched survivors must fit in the BN slots.
    """
    # Key survivors/actives/moves on player_key: a two-way player is two rows
    # (hitter + pitcher) sharing a name, and dropping one row must not drop the
    # other. ``drops`` stays bare-name for display and the SGP tie-break lookup.
    drop_keys = {p.player_key for p in dropset}
    # Dedup names (a two-way dropset holds two rows sharing one name) so the
    # ``drops`` display list doesn't render one name twice -- matches the
    # pre-#190 set semantics now that the internal filtering keys on player_key.
    drop_names = sorted({p.name for p in dropset})
    survivors = [p for p in pool if p.player_key not in drop_keys]
    dropped_hitter = any(p.player_type != PlayerType.PITCHER for p in dropset)
    dropped_pitcher = any(p.player_type == PlayerType.PITCHER for p in dropset)

    # Re-solve only the side(s) the drop touches; reuse the pre-drop baseline
    # for the untouched side (its lineup is unchanged). An empty dropset
    # (overflow <= 0) touches neither and reuses both baselines.
    if dropped_hitter:
        h_assign = optimize_hitter_lineup(
            hitters=[p for p in survivors if p.player_type != PlayerType.PITCHER],
            full_roster=survivors,
            projected_standings=projected_standings,
            team_name=team_name,
            roster_slots=roster_slots,
            team_sds=team_sds,
            fraction_remaining=None,
        )
    else:
        h_assign = base_h

    if dropped_pitcher:
        ps, _ = optimize_pitcher_lineup(
            pitchers=[p for p in survivors if p.player_type == PlayerType.PITCHER],
            full_roster=survivors,
            projected_standings=projected_standings,
            team_name=team_name,
            slots=roster_slots.get("P", 9),
            team_sds=team_sds,
            fraction_remaining=fraction_remaining,
            compute_bands=False,
        )
    else:
        ps = base_ps

    active_keys = {a.player.player_key for a in h_assign} | {s.player.player_key for s in ps}
    benched = [p for p in survivors if p.player_key not in active_keys]
    if len(benched) > bn_slots:
        return None  # infeasible: can't bench everyone left over

    after_active = [a.player for a in h_assign] + [s.player for s in ps]
    try:
        # before_active is a re-optimized hypothetical, NOT the lineup the
        # cached standings row reflects -- anchor on the roster's current
        # actives per the contract on _ev_delta_and_stats (plans are RANKED
        # by this mean, so a wrong anchor can reorder drop sets).
        band = compute_delta_roto_band(
            before_active,
            after_active,
            projected_standings.field_stats(team_name),
            team_name,
            fraction_remaining,
            reference_players=band_reference,
            projected_standings=projected_standings,
            team_sds=team_sds,
        )
    except KeyError as exc:
        logger.warning("IL plan band failed for drop %s: %s", drop_names, exc)
        return None

    moves = _build_moves(roster, pool, h_assign, ps, drop_keys)
    return MovePlan(
        drops=drop_names,
        moves=moves,
        delta_roto=band.mean,
        band=band.to_dict(),
    )


def plan_il_returns(
    roster: list[Player],
    activating_il: list[Player],
    roster_slots: dict[str, int],
    *,
    projected_standings: ProjectedStandings,
    team_name: str,
    fraction_remaining: float,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
    max_plans: int = 5,
    sgp_overrides: SgpOverrides | None = None,
) -> IlReturnPlanResult:
    """Plan the roster moves to reactivate ``activating_il`` players.

    Returns up to ``max_plans`` plans ranked by deltaRoto descending. Each
    plan's deltaRoto is the cost of its forced drop relative to the pre-drop
    ideal lineup (which already includes the returning players, so the
    activation gain the standings already price is not double-counted).

    ``sgp_overrides`` (from ``config.sgp_overrides``) replaces individual
    SGP denominators with league-specific values; None keeps the code
    defaults.
    """
    capacity = roster_capacity(roster_slots)
    activating_names = [p.name for p in activating_il]

    if not activating_il:
        return IlReturnPlanResult(activating=[], capacity=capacity, overflow=0, plans=[])

    pool = _build_pool(roster, activating_il)
    overflow = len(pool) - capacity
    denoms = get_sgp_denominators(sgp_overrides)
    bn_slots = roster_slots.get("BN", 0)

    # Pre-drop ideal lineup -> the band baseline (returning players present here).
    base_h, base_ps, _ = _solve_lineup(
        pool, roster_slots, projected_standings, team_name, team_sds, fraction_remaining
    )
    before_active = [a.player for a in base_h] + [s.player for s in base_ps]
    # Anchor for every plan's band: the CURRENT lineup the cached standings
    # row reflects (before_active is a re-optimized hypothetical). Loop-
    # invariant across drop-sets, so computed once here.
    band_reference = band_reference_lineup(roster)

    if overflow <= 0:
        plan = _make_plan(
            roster,
            pool,
            (),
            base_h,
            base_ps,
            before_active,
            roster_slots,
            projected_standings,
            team_name,
            team_sds,
            fraction_remaining,
            bn_slots,
            band_reference,
        )
        plans = [plan] if plan is not None else []
        return IlReturnPlanResult(
            activating=activating_names, capacity=capacity, overflow=0, plans=plans
        )

    # Forced drops: enumerate drop-sets of size `overflow`. For overflow >= 3
    # (rare) restrict to the bottom-12 bodies by SGP to bound the combinatorics.
    droppable = pool
    if overflow >= 3:
        droppable = sorted(pool, key=lambda p: _sgp(p, denoms))[:12]

    # Pair each plan with the total SGP of the bodies it drops, computed from
    # the dropset itself (exact for a two-way player's two rows -- a bare-name
    # lookup keyed on the deduped display names would collide and undercount).
    scored: list[tuple[MovePlan, float]] = []
    for dropset in combinations(droppable, overflow):
        plan = _make_plan(
            roster,
            pool,
            dropset,
            base_h,
            base_ps,
            before_active,
            roster_slots,
            projected_standings,
            team_name,
            team_sds,
            fraction_remaining,
            bn_slots,
            band_reference,
        )
        if plan is not None:
            scored.append((plan, sum(_sgp(p, denoms) for p in dropset)))

    if not scored:
        return IlReturnPlanResult(
            activating=activating_names,
            capacity=capacity,
            overflow=overflow,
            plans=[],
            warning=f"No legal roster after dropping {overflow} player(s).",
        )

    # Rank by deltaRoto; tie-break by dropping the lower-SGP body.
    scored.sort(key=lambda item: (item[0].delta_roto, -item[1]), reverse=True)
    return IlReturnPlanResult(
        activating=activating_names,
        capacity=capacity,
        overflow=overflow,
        plans=[plan for plan, _ in scored[:max_plans]],
    )
