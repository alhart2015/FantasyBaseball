import dataclasses
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from fantasy_baseball.models.player import Player
from fantasy_baseball.models.positions import HITTER_ELIGIBLE, PITCHER_ELIGIBLE, Position
from fantasy_baseball.scoring import project_team_stats, score_roto_dict
from fantasy_baseball.utils.constants import DEFAULT_ROSTER_SLOTS
from fantasy_baseball.utils.positions import can_fill_slot


@dataclass
class HitterAssignment:
    slot: Position
    name: str
    player: Player
    roto_delta: float

    def to_dict(self) -> dict:
        return {
            "slot": self.slot.value,
            "name": self.name,
            "roto_delta": round(self.roto_delta, 2),
        }


@dataclass
class PitcherStarter:
    name: str
    player: Player
    roto_delta: float

    def to_dict(self) -> dict:
        return {"name": self.name, "roto_delta": round(self.roto_delta, 2)}


def _build_hitter_slot_positions(roster_slots: dict[str, int]) -> list[Position]:
    """Return the ordered list of active hitter slot Position values from config."""
    slots: list[Position] = []
    for pos_key, count in roster_slots.items():
        pos = pos_key if isinstance(pos_key, Position) else Position.parse(pos_key)
        if pos not in HITTER_ELIGIBLE:
            continue
        for _ in range(count):
            slots.append(pos)
    return slots


def _feasible_assignment(
    subset: list[Player],
    slot_positions: list[Position],
) -> list[Position] | None:
    """Return a list parallel to ``subset`` giving each player's assigned slot,
    or None if no valid assignment exists.

    Uses Hungarian on a 0/1 feasibility matrix (cost 0 for eligible, large cost
    for ineligible). A valid matching has zero total cost.
    """
    n_players = len(subset)
    n_slots = len(slot_positions)
    if n_players != n_slots:
        return None
    size = n_players
    cost = np.full((size, size), 1e9)
    for i, p in enumerate(subset):
        for j, slot in enumerate(slot_positions):
            if can_fill_slot(p.positions, slot.value):
                cost[i][j] = 0.0
    row_idx, col_idx = linear_sum_assignment(cost)
    assignments: list[Position | None] = [None] * n_players
    for r, c in zip(row_idx, col_idx):
        if cost[r][c] > 0.5:
            return None
        assignments[r] = slot_positions[c]
    return assignments  # type: ignore[return-value]


@dataclass
class _TeamContext:
    """Scoring-side inputs passed through every ERoto evaluation."""

    full_roster: list[Player]
    projected_standings: list[dict]
    team_name: str
    team_sds: dict[str, dict[str, float]] | None = None


def apply_lineup_to_roster(
    full_roster: list[Player],
    active_slots: dict[str, Position],
    bench_names: set[str],
) -> list[Player]:
    """Return a copy of ``full_roster`` with selected_position overridden.

    - Names in ``active_slots`` -> selected_position set to the mapped Position.
    - Names in ``bench_names`` -> selected_position set to Position.BN.
    - All other players (IL, opposite type) unchanged.
    """
    result: list[Player] = []
    for p in full_roster:
        if p.name in bench_names:
            result.append(dataclasses.replace(p, selected_position=Position.BN))
        elif p.name in active_slots:
            result.append(dataclasses.replace(p, selected_position=active_slots[p.name]))
        else:
            result.append(p)
    return result


def team_roto_total(hypothetical: list[Player], ctx: _TeamContext) -> float:
    my_stats = project_team_stats(hypothetical, displacement=True).to_dict()
    all_stats = {t["name"]: dict(t["stats"]) for t in ctx.projected_standings}
    all_stats[ctx.team_name] = my_stats
    return score_roto_dict(all_stats, team_sds=ctx.team_sds)[ctx.team_name]["total"]


def _score_hitter_subset(
    ctx: _TeamContext,
    subset: list[Player],
    assignment: list[Position],
    bench: list[Player],
) -> float:
    hypothetical = apply_lineup_to_roster(
        ctx.full_roster,
        active_slots={p.name: slot for p, slot in zip(subset, assignment)},
        bench_names={h.name for h in bench},
    )
    return team_roto_total(hypothetical, ctx)


def _pitcher_active_slots(subset: list[Player]) -> dict[str, Position]:
    return {
        p.name: next((pos for pos in p.positions if pos in PITCHER_ELIGIBLE), Position.P)
        for p in subset
    }


def _score_pitcher_subset(
    ctx: _TeamContext,
    subset: list[Player],
    bench: list[Player],
) -> float:
    hypothetical = apply_lineup_to_roster(
        ctx.full_roster,
        _pitcher_active_slots(subset),
        {p.name for p in bench},
    )
    return team_roto_total(hypothetical, ctx)


def optimize_hitter_lineup(
    hitters: list[Player],
    full_roster: list[Player],
    projected_standings: list[dict],
    team_name: str,
    roster_slots: dict[str, int] | None = None,
    team_sds: dict[str, dict[str, float]] | None = None,
) -> list[HitterAssignment]:
    """Return the ERoto-maximizing active hitter lineup."""
    if not hitters:
        return []
    slot_positions = _build_hitter_slot_positions(
        roster_slots if roster_slots is not None else DEFAULT_ROSTER_SLOTS
    )
    n_slots = len(slot_positions)
    ctx = _TeamContext(full_roster, projected_standings, team_name, team_sds)

    if n_slots == 0 or len(hitters) < n_slots:
        # Fewer hitters than slots — fall back to the best feasible partial lineup.
        partial_best: tuple[float, list[Player], list[Position]] | None = None
        for size in range(min(len(hitters), n_slots), 0, -1):
            for subset in combinations(hitters, size):
                assn = _feasible_assignment(list(subset), slot_positions[:size])
                if assn is None:
                    continue
                bench = [h for h in hitters if h not in subset]
                total = _score_hitter_subset(ctx, list(subset), assn, bench)
                if partial_best is None or total > partial_best[0]:
                    partial_best = (total, list(subset), assn)
            if partial_best is not None:
                break
        if partial_best is None:
            return []
        return [
            HitterAssignment(slot=slot, name=p.name, player=p, roto_delta=0.0)
            for p, slot in zip(partial_best[1], partial_best[2])
        ]

    best: tuple[float, list[Player], list[Position], list[Player]] | None = None
    for subset in combinations(hitters, n_slots):
        assn = _feasible_assignment(list(subset), slot_positions)
        if assn is None:
            continue
        bench = [h for h in hitters if h not in subset]
        total = _score_hitter_subset(ctx, list(subset), assn, bench)
        if best is None or total > best[0]:
            best = (total, list(subset), assn, bench)

    if best is None:
        return []

    best_total, active_subset, assignment, bench = best

    roto_deltas: dict[str, float] = {}
    for starter in active_subset:
        remaining_hitters = [h for h in hitters if h is not starter]
        alt_best = None
        for sub in combinations(remaining_hitters, n_slots):
            assn = _feasible_assignment(list(sub), slot_positions)
            if assn is None:
                continue
            sub_bench = [h for h in remaining_hitters if h not in sub] + [starter]
            t = _score_hitter_subset(ctx, list(sub), assn, sub_bench)
            if alt_best is None or t > alt_best:
                alt_best = t
        if alt_best is None:
            # No feasible full-size replacement lineup (the roster is too
            # thin to cover every slot without this starter). Counterfactual
            # is "starter benched, their slot left empty" — score the rest
            # of the optimal lineup without them.
            no_rep_subset = [p for p in active_subset if p is not starter]
            no_rep_assn = [a for p, a in zip(active_subset, assignment) if p is not starter]
            alt_best = _score_hitter_subset(ctx, no_rep_subset, no_rep_assn, bench + [starter])
        roto_deltas[starter.name] = best_total - alt_best

    return [
        HitterAssignment(
            slot=slot,
            name=p.name,
            player=p,
            roto_delta=roto_deltas.get(p.name, 0.0),
        )
        for p, slot in zip(active_subset, assignment)
    ]


def optimize_pitcher_lineup(
    pitchers: list[Player],
    full_roster: list[Player],
    projected_standings: list[dict],
    team_name: str,
    slots: int = 9,
    team_sds: dict[str, dict[str, float]] | None = None,
) -> tuple[list[PitcherStarter], list[Player]]:
    """Return (starters with roto_delta, bench) maximizing ERoto."""
    if not pitchers or slots <= 0:
        return [], list(pitchers)
    k = min(slots, len(pitchers))
    ctx = _TeamContext(full_roster, projected_standings, team_name, team_sds)

    best = None
    for subset in combinations(pitchers, k):
        bench = [p for p in pitchers if p not in subset]
        total = _score_pitcher_subset(ctx, list(subset), bench)
        if best is None or total > best[0]:
            best = (total, list(subset), bench)

    best_total, active_subset, bench = best  # type: ignore[misc]

    roto_deltas: dict[str, float] = {}
    for starter in active_subset:
        remaining = [p for p in pitchers if p is not starter]
        alt_best = None
        if len(remaining) >= k:
            for sub in combinations(remaining, k):
                sub_bench = [p for p in remaining if p not in sub] + [starter]
                t = _score_pitcher_subset(ctx, list(sub), sub_bench)
                if alt_best is None or t > alt_best:
                    alt_best = t
        if alt_best is None:
            # No feasible full-size replacement (roster has exactly k
            # pitchers, no bench). Counterfactual is "starter benched,
            # their slot left empty" — score the other k-1 starters alone.
            no_rep_subset = [p for p in active_subset if p is not starter]
            alt_best = _score_pitcher_subset(ctx, no_rep_subset, bench + [starter])
        roto_deltas[starter.name] = best_total - alt_best

    starters = [
        PitcherStarter(name=p.name, player=p, roto_delta=roto_deltas[p.name]) for p in active_subset
    ]
    return starters, bench


def combined_team_roto(
    roster: list[Player],
    hitters: list[Player],
    hitter_lineup: list[HitterAssignment],
    pitcher_starters: list[PitcherStarter],
    pitcher_bench: list[Player],
    projected_standings: list[dict],
    team_name: str,
    team_sds: dict[str, dict[str, float]] | None = None,
) -> float:
    """Score a combined hitter + pitcher lineup as a single ERoto total.

    The per-side optimizers score independently; this recomputes once on the
    combined lineup so the reported total reflects both sides together.
    """
    active_slots: dict[str, Position] = {a.name: a.slot for a in hitter_lineup}
    active_slots.update(_pitcher_active_slots([s.player for s in pitcher_starters]))
    bench_names = {h.name for h in hitters} - {a.name for a in hitter_lineup} | {
        p.name for p in pitcher_bench
    }
    hypothetical = apply_lineup_to_roster(roster, active_slots, bench_names)
    ctx = _TeamContext(roster, projected_standings, team_name, team_sds)
    return team_roto_total(hypothetical, ctx)
