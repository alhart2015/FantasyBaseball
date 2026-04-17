import dataclasses
import numpy as np
from dataclasses import dataclass
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import Position, HITTER_ELIGIBLE
from fantasy_baseball.utils.constants import DEFAULT_ROSTER_SLOTS
from fantasy_baseball.utils.positions import can_fill_slot
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.scoring import project_team_stats, score_roto


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


def _build_hitter_slots(roster_slots: dict[str, int]) -> list[str]:
    """Build the list of active hitter slot labels from roster config."""
    slots: list[str] = []
    for pos, count in roster_slots.items():
        if pos in ("P", "BN", "IL"):
            continue
        for _ in range(count):
            slots.append(pos)
    return slots


# Default for backward compatibility
HITTER_SLOTS: list[str] = _build_hitter_slots(DEFAULT_ROSTER_SLOTS)


def optimize_hitter_lineup(
    hitters: list[Player],
    leverage: dict[str, float],
    roster_slots: dict[str, int] | None = None,
) -> dict[str, str]:
    """Assign hitters to roster slots to maximize leverage-weighted SGP.

    Uses scipy's linear_sum_assignment (Hungarian algorithm).

    Returns:
        Dict of slot -> player name for optimal lineup.
    """
    if not hitters:
        return {}

    hitter_slots = _build_hitter_slots(roster_slots) if roster_slots else HITTER_SLOTS
    n_players = len(hitters)
    n_slots = len(hitter_slots)

    values = []
    for h in hitters:
        values.append(calculate_weighted_sgp(h.rest_of_season, leverage))

    # Build cost matrix (negative because we maximize)
    size = max(n_players, n_slots)
    cost = np.full((size, size), 1e9)

    for i, hitter in enumerate(hitters):
        positions = hitter.positions
        for j, slot in enumerate(hitter_slots):
            if can_fill_slot(positions, slot):
                cost[i][j] = -values[i]

    row_idx, col_idx = linear_sum_assignment(cost)

    lineup: dict[str, str] = {}
    assigned_slots: dict[str, int] = {}
    for r, c in zip(row_idx, col_idx):
        if r < n_players and c < n_slots and cost[r][c] < 1e8:
            slot = hitter_slots[c]
            player_name = hitters[r].name
            slot_key = slot
            count = assigned_slots.get(slot, 0)
            if count > 0:
                slot_key = f"{slot}_{count + 1}"
            assigned_slots[slot] = count + 1
            lineup[slot_key] = player_name

    return lineup


def optimize_pitcher_lineup(
    pitchers: list[Player],
    leverage: dict[str, float],
    slots: int = 9,
) -> tuple[list[dict], list[dict]]:
    """Select top pitchers by leverage-weighted SGP.

    All P slots are interchangeable, so just rank and start top N.

    Returns:
        Tuple of (starters, bench) as lists of dicts with name and wsgp.
    """
    scored = []
    for p in pitchers:
        wsgp = calculate_weighted_sgp(p.rest_of_season, leverage)
        scored.append({"name": p.name, "wsgp": wsgp, "player": p})

    scored.sort(key=lambda x: x["wsgp"], reverse=True)

    starters = scored[:slots]
    bench = scored[slots:]
    return starters, bench


# ---------------------------------------------------------------------------
# ERoto-based hitter optimizer (Task 3 — staged as _roto suffix)
# ---------------------------------------------------------------------------

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


def _hypothetical_roster_with_hitter_statuses(
    full_roster: list[Player],
    active_hitters: set[str],
    bench_hitters: set[str],
) -> list[Player]:
    """Return a copy of full_roster with hitter selected_position updated.

    - Active hitter names: set selected_position to their first eligible hitter
      slot (value doesn't matter to project_team_stats beyond "not BN/IL").
    - Bench hitter names: set selected_position to Position.BN.
    - Every other player (pitchers, IL hitters): unchanged.
    """
    result: list[Player] = []
    for p in full_roster:
        if p.name in bench_hitters:
            result.append(dataclasses.replace(p, selected_position=Position.BN))
        elif p.name in active_hitters:
            new_slot = next(
                (pos for pos in p.positions if pos in HITTER_ELIGIBLE),
                Position.UTIL,
            )
            result.append(dataclasses.replace(p, selected_position=new_slot))
        else:
            result.append(p)
    return result


def _team_total_after_hitter_swap(
    full_roster: list[Player],
    active_subset: list[Player],
    bench_subset: list[Player],
    projected_standings: list[dict],
    team_name: str,
    team_sds: dict[str, dict[str, float]] | None,
) -> float:
    hypothetical = _hypothetical_roster_with_hitter_statuses(
        full_roster,
        active_hitters={p.name for p in active_subset},
        bench_hitters={p.name for p in bench_subset},
    )
    my_stats = project_team_stats(hypothetical, displacement=True).to_dict()
    all_stats = {t["name"]: dict(t["stats"]) for t in projected_standings}
    all_stats[team_name] = my_stats
    return score_roto(all_stats, team_sds=team_sds)[team_name]["total"]


def optimize_hitter_lineup_roto(
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
    if n_slots == 0 or len(hitters) < n_slots:
        # Can't fill every slot — fall back to fitting as many as possible.
        # Pick the feasible subset with best ERoto among all sizes down to 0.
        best = None
        for size in range(min(len(hitters), n_slots), 0, -1):
            for subset in combinations(hitters, size):
                assn = _feasible_assignment(list(subset), slot_positions[:size])
                if assn is None:
                    continue
                bench = [h for h in hitters if h not in subset]
                total = _team_total_after_hitter_swap(
                    full_roster, list(subset), bench,
                    projected_standings, team_name, team_sds,
                )
                if best is None or total > best[0]:
                    best = (total, list(subset), assn)
            if best is not None:
                break
        if best is None:
            return []
        return [
            HitterAssignment(slot=slot, name=p.name, player=p, roto_delta=0.0)
            for p, slot in zip(best[1], best[2])
        ]

    best = None
    for subset in combinations(hitters, n_slots):
        assn = _feasible_assignment(list(subset), slot_positions)
        if assn is None:
            continue
        bench = [h for h in hitters if h not in subset]
        total = _team_total_after_hitter_swap(
            full_roster, list(subset), bench,
            projected_standings, team_name, team_sds,
        )
        if best is None or total > best[0]:
            best = (total, list(subset), assn, bench)

    if best is None:
        return []

    best_total, active_subset, assignment, bench = best

    # Compute roto_delta for each starter: drop them to bench, pick best remaining subset.
    roto_deltas: dict[str, float] = {}
    for starter in active_subset:
        remaining_hitters = [h for h in hitters if h is not starter]
        alt_best = None
        for sub in combinations(remaining_hitters, n_slots):
            assn = _feasible_assignment(list(sub), slot_positions)
            if assn is None:
                continue
            sub_bench = [h for h in remaining_hitters if h not in sub] + [starter]
            t = _team_total_after_hitter_swap(
                full_roster, list(sub), sub_bench,
                projected_standings, team_name, team_sds,
            )
            if alt_best is None or t > alt_best:
                alt_best = t
        # If no feasible replacement lineup exists, the starter is irreplaceable:
        # credit them with the full best_total (versus 0 for "no valid lineup").
        roto_deltas[starter.name] = best_total - (alt_best if alt_best is not None else 0.0)

    return [
        HitterAssignment(
            slot=slot, name=p.name, player=p,
            roto_delta=roto_deltas.get(p.name, 0.0),
        )
        for p, slot in zip(active_subset, assignment)
    ]


# ---------------------------------------------------------------------------
# ERoto-based pitcher optimizer (Task 4 — staged as _roto suffix)
# ---------------------------------------------------------------------------

def _hypothetical_roster_with_pitcher_statuses(
    full_roster: list[Player],
    active_pitchers: set[str],
    bench_pitchers: set[str],
) -> list[Player]:
    result: list[Player] = []
    for p in full_roster:
        if p.name in bench_pitchers:
            result.append(dataclasses.replace(p, selected_position=Position.BN))
        elif p.name in active_pitchers:
            new_slot = next(
                (pos for pos in p.positions if pos in {Position.SP, Position.RP, Position.P}),
                Position.P,
            )
            result.append(dataclasses.replace(p, selected_position=new_slot))
        else:
            result.append(p)
    return result


def _team_total_after_pitcher_swap(
    full_roster: list[Player],
    active_subset: list[Player],
    bench_subset: list[Player],
    projected_standings: list[dict],
    team_name: str,
    team_sds: dict[str, dict[str, float]] | None,
) -> float:
    hypothetical = _hypothetical_roster_with_pitcher_statuses(
        full_roster,
        active_pitchers={p.name for p in active_subset},
        bench_pitchers={p.name for p in bench_subset},
    )
    my_stats = project_team_stats(hypothetical, displacement=True).to_dict()
    all_stats = {t["name"]: dict(t["stats"]) for t in projected_standings}
    all_stats[team_name] = my_stats
    return score_roto(all_stats, team_sds=team_sds)[team_name]["total"]


def optimize_pitcher_lineup_roto(
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

    best = None
    for subset in combinations(pitchers, k):
        bench = [p for p in pitchers if p not in subset]
        total = _team_total_after_pitcher_swap(
            full_roster, list(subset), bench,
            projected_standings, team_name, team_sds,
        )
        if best is None or total > best[0]:
            best = (total, list(subset), bench)

    best_total, active_subset, bench = best  # type: ignore[misc]

    # roto_delta per starter: demote them to bench, pick best remaining subset.
    roto_deltas: dict[str, float] = {}
    for starter in active_subset:
        remaining = [p for p in pitchers if p is not starter]
        if len(remaining) < k:
            # Irreplaceable: no feasible replacement lineup of size k.
            # Credit the starter with the full best_total (versus 0 for no valid lineup).
            roto_deltas[starter.name] = best_total
            continue
        alt_best = None
        for sub in combinations(remaining, k):
            sub_bench = [p for p in remaining if p not in sub] + [starter]
            t = _team_total_after_pitcher_swap(
                full_roster, list(sub), sub_bench,
                projected_standings, team_name, team_sds,
            )
            if alt_best is None or t > alt_best:
                alt_best = t
        roto_deltas[starter.name] = best_total - (alt_best if alt_best is not None else 0.0)

    starters = [
        PitcherStarter(name=p.name, player=p, roto_delta=roto_deltas[p.name])
        for p in active_subset
    ]
    return starters, bench
