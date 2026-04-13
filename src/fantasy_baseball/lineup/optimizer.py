import numpy as np
from scipy.optimize import linear_sum_assignment
from fantasy_baseball.models.player import Player
from fantasy_baseball.utils.constants import DEFAULT_ROSTER_SLOTS
from fantasy_baseball.utils.positions import can_fill_slot
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp


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
