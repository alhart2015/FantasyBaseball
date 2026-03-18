import pandas as pd
from fantasy_baseball.utils.constants import DEFAULT_ROSTER_SLOTS, IF_ELIGIBLE
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import can_fill_slot, is_hitter
from fantasy_baseball.sgp.replacement import calculate_replacement_levels
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import compute_starters_per_position

REQUIRED_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF", "P"]


def get_recommendations(
    board: pd.DataFrame,
    drafted: list[str],
    user_roster: list[str],
    n: int = 5,
    filled_positions: dict[str, int] | None = None,
    picks_until_next: int | None = None,
    roster_slots: dict[str, int] | None = None,
    num_teams: int | None = None,
) -> list[dict]:
    """Get top draft pick recommendations.

    Recalculates replacement levels from the undrafted pool so that
    positional scarcity (e.g. a run on catchers) is reflected in VAR.
    """
    if roster_slots is None:
        roster_slots = DEFAULT_ROSTER_SLOTS
    available = board[~board["player_id"].isin(drafted)]

    # Recalculate replacement levels from the remaining pool so
    # positional scarcity is properly reflected in rankings.
    starters = compute_starters_per_position(roster_slots, num_teams)
    repl_levels = calculate_replacement_levels(available, starters)
    # Recompute VAR for available players using live replacement levels
    live_var = {}
    live_pos = {}
    for idx, row in available.iterrows():
        var, pos = calculate_var(row, repl_levels, return_position=True)
        live_var[idx] = var
        live_pos[idx] = pos
    available = available.copy()
    available["var"] = available.index.map(live_var)
    available["best_position"] = available.index.map(live_pos)
    available = available.sort_values("var", ascending=False)

    # Use a wider window for scarcity checks, narrower for rec candidates
    scarcity_pool = available.head(50)
    candidates = available.head(n * 3)
    if filled_positions is None:
        filled_positions = {}
    unfilled = _get_unfilled_positions(filled_positions, roster_slots)

    # Ensure the best available player at each unfilled position is included
    # so the user always sees their positional options, not just raw VAR.
    candidate_ids = set(candidates["player_id"])
    for slot in unfilled:
        for _, row in available.iterrows():
            if row["player_id"] in candidate_ids:
                continue
            if can_fill_slot(row["positions"], slot):
                candidates = pd.concat(
                    [candidates, row.to_frame().T], ignore_index=True
                )
                candidate_ids.add(row["player_id"])
                break  # only need the best one per slot

    recs = []
    for _, player in candidates.iterrows():
        rec = {
            "name": player["name"],
            "var": player["var"],
            "best_position": player["best_position"],
            "positions": player["positions"],
            "player_type": player["player_type"],
            "need_flag": False,
            "note": "",
        }
        positions = player["positions"]
        for slot in unfilled:
            if can_fill_slot(positions, slot):
                rec["need_flag"] = True
                rec["note"] = f"fills {slot} need"
                break
        if picks_until_next and picks_until_next > 8:
            pos = player["best_position"]
            remaining_at_pos = len(scarcity_pool[scarcity_pool["best_position"] == pos])
            if remaining_at_pos <= 3:
                scarcity = f"scarce position — only {remaining_at_pos} left in top tier"
                rec["note"] = f"{rec['note']}; {scarcity}" if rec["note"] else scarcity
        recs.append(rec)
    # Guarantee at least one player per unfilled position makes the final list.
    # Split into need-fills and pure-VAR, then merge.
    need_recs = []
    other_recs = []
    seen_need_slots: set[str] = set()
    # Sort all by VAR first
    recs.sort(key=lambda r: r["var"], reverse=True)
    for rec in recs:
        if rec["need_flag"] and rec["best_position"] not in seen_need_slots:
            need_recs.append(rec)
            seen_need_slots.add(rec["best_position"])
        else:
            other_recs.append(rec)
    # Fill remaining slots with best-VAR players
    result = need_recs + other_recs
    return result[:n]


def _get_unfilled_positions(
    filled: dict[str, int],
    roster_slots: dict[str, int],
) -> set[str]:
    unfilled = set()
    for pos, slots in roster_slots.items():
        if pos in ("BN", "IL"):
            continue
        current = filled.get(pos, 0)
        if current < slots:
            unfilled.add(pos)
    return unfilled


def get_filled_positions(
    user_roster_names: list[str],
    board: pd.DataFrame,
    roster_slots: dict[str, int] | None = None,
) -> dict[str, int]:
    """Count how many of each roster slot the user has filled.

    Greedily assigns each drafted player to their most specific open slot
    before falling back to flex slots (IF, UTIL), so multi-position players
    don't over-count a single position.
    """
    if roster_slots is None:
        roster_slots = DEFAULT_ROSTER_SLOTS

    # Build capacity: how many of each slot are available
    capacity: dict[str, int] = {
        pos: count for pos, count in roster_slots.items()
        if pos not in ("BN", "IL")
    }
    filled: dict[str, int] = {pos: 0 for pos in capacity}

    # Collect players with their positions
    players = []
    for name in user_roster_names:
        rows = board[board["name_normalized"] == normalize_name(name)]
        if rows.empty:
            continue
        player = rows.iloc[0]
        players.append(player)

    # Sort: assign players with fewer eligible slots first (most constrained)
    players.sort(key=lambda p: sum(
        1 for s in capacity if can_fill_slot(p["positions"], s)
    ))

    for player in players:
        positions = player["positions"]
        assigned = False
        # Try specific slots first (C, 1B, 2B, etc.), then flex (IF, UTIL)
        for slot in list(capacity.keys()):
            if slot in ("IF", "UTIL"):
                continue
            if filled[slot] < capacity[slot] and can_fill_slot(positions, slot):
                filled[slot] += 1
                assigned = True
                break
        if not assigned:
            # Try flex slots
            for slot in ("IF", "UTIL"):
                if slot in capacity and filled[slot] < capacity[slot] and can_fill_slot(positions, slot):
                    filled[slot] += 1
                    assigned = True
                    break

    # Remove zero entries for cleaner output
    return {pos: count for pos, count in filled.items() if count > 0}


def get_roster_by_position(
    user_roster_names: list[str],
    board: pd.DataFrame,
    roster_slots: dict[str, int] | None = None,
) -> dict[str, list[str]]:
    """Map roster slot -> list of player names for the user's roster.

    Uses the same greedy slot assignment as get_filled_positions.
    """
    if roster_slots is None:
        roster_slots = DEFAULT_ROSTER_SLOTS

    capacity: dict[str, int] = {
        pos: count for pos, count in roster_slots.items()
        if pos not in ("BN", "IL")
    }
    by_pos: dict[str, list[str]] = {pos: [] for pos in capacity}

    players = []
    for name in user_roster_names:
        rows = board[board["name_normalized"] == normalize_name(name)]
        if rows.empty:
            continue
        players.append(rows.iloc[0])

    players.sort(key=lambda p: sum(
        1 for s in capacity if can_fill_slot(p["positions"], s)
    ))

    for player in players:
        positions = player["positions"]
        assigned = False
        for slot in list(capacity.keys()):
            if slot in ("IF", "UTIL"):
                continue
            if len(by_pos[slot]) < capacity[slot] and can_fill_slot(positions, slot):
                by_pos[slot].append(player["name"])
                assigned = True
                break
        if not assigned:
            for slot in ("IF", "UTIL"):
                if slot in capacity and len(by_pos[slot]) < capacity[slot] and can_fill_slot(positions, slot):
                    by_pos[slot].append(player["name"])
                    assigned = True
                    break

    return {pos: names for pos, names in by_pos.items() if names}
