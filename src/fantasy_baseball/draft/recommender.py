import pandas as pd
from fantasy_baseball.utils.constants import DEFAULT_ROSTER_SLOTS, IF_ELIGIBLE
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import can_fill_slot, is_hitter
from fantasy_baseball.sgp.replacement import calculate_replacement_levels
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import compute_starters_per_position
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp

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
    draft_leverage: dict[str, float] | None = None,
) -> list[dict]:
    """Get top draft pick recommendations.

    Recalculates replacement levels from the undrafted pool so that
    positional scarcity (e.g. a run on catchers) is reflected in VAR.

    If *draft_leverage* is provided (category weights from balance
    analysis), candidates are scored by leverage-weighted SGP instead
    of raw VAR, steering picks toward categories the team needs most.
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

    if filled_positions is None:
        filled_positions = {}

    # Filter out players who have no open roster slot (including BN).
    # E.g. if all OF, UTIL, and BN spots are full, don't suggest more OFs.
    available = _filter_rosterable(available, filled_positions, roster_slots)
    available = available.sort_values("var", ascending=False)

    # Use a wider window for scarcity checks, narrower for rec candidates
    scarcity_pool = available.head(50)
    candidates = available.head(n * 3)
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
        # Use leverage-weighted SGP if available, otherwise raw VAR
        if draft_leverage:
            score = calculate_weighted_sgp(player, draft_leverage)
        else:
            score = player["var"]
        rec = {
            "name": player["name"],
            "var": player["var"],
            "score": round(score, 2),
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
    # Split into need-fills and pure-score, then merge.
    need_recs = []
    other_recs = []
    seen_need_slots: set[str] = set()
    # Sort all by score (leverage-weighted if available, else VAR)
    recs.sort(key=lambda r: r["score"], reverse=True)
    for rec in recs:
        if rec["need_flag"] and rec["best_position"] not in seen_need_slots:
            need_recs.append(rec)
            seen_need_slots.add(rec["best_position"])
        else:
            other_recs.append(rec)
    # Fill remaining slots with best-score players
    result = need_recs + other_recs
    return result[:n]


def _filter_rosterable(
    available: pd.DataFrame,
    filled: dict[str, int],
    roster_slots: dict[str, int],
) -> pd.DataFrame:
    """Remove players who cannot fit in any open roster slot (including BN)."""
    # Build open-slot counts (exclude IL — you don't draft to IL)
    open_slots: dict[str, int] = {}
    for pos, total in roster_slots.items():
        if pos == "IL":
            continue
        current = filled.get(pos, 0)
        if current < total:
            open_slots[pos] = total - current

    if not open_slots:
        return available.iloc[0:0]  # no room at all

    def has_open_slot(positions):
        for slot in open_slots:
            if can_fill_slot(positions, slot):
                return True
        return False

    mask = available["positions"].apply(has_open_slot)
    return available[mask]


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


def _collect_roster_entries(
    user_roster_ids: list[str], board: pd.DataFrame,
) -> list[pd.Series]:
    """Look up board entries for each roster player by player_id."""
    players: list[pd.Series] = []
    for pid in user_roster_ids:
        rows = board[board["player_id"] == pid]
        if rows.empty:
            # Fallback: try name match (for entries without player_id)
            name = pid.split("::")[0] if "::" in pid else pid
            rows = board[board["name_normalized"] == normalize_name(name)]
        if not rows.empty:
            players.append(rows.iloc[0])
    return players


def get_filled_positions(
    user_roster_ids: list[str],
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
        if pos != "IL"
    }
    filled: dict[str, int] = {pos: 0 for pos in capacity}

    players = _collect_roster_entries(user_roster_ids, board)

    # Sort: assign players with fewer eligible active slots first (most constrained)
    active_slots = {k: v for k, v in capacity.items() if k != "BN"}
    players.sort(key=lambda p: sum(
        1 for s in active_slots if can_fill_slot(p["positions"], s)
    ))

    for player in players:
        positions = player["positions"]
        assigned = False
        # Try specific slots first (C, 1B, 2B, etc.), then flex (IF, UTIL)
        for slot in list(active_slots.keys()):
            if slot in ("IF", "UTIL"):
                continue
            if filled[slot] < capacity[slot] and can_fill_slot(positions, slot):
                filled[slot] += 1
                assigned = True
                break
        if not assigned:
            for slot in ("IF", "UTIL"):
                if slot in active_slots and filled[slot] < capacity[slot] and can_fill_slot(positions, slot):
                    filled[slot] += 1
                    assigned = True
                    break
        if not assigned:
            filled["BN"] = filled.get("BN", 0) + 1

    # Remove zero entries for cleaner output
    return {pos: count for pos, count in filled.items() if count > 0}


def get_roster_by_position(
    user_roster_ids: list[str],
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
        if pos != "IL"
    }
    by_pos: dict[str, list[str]] = {pos: [] for pos in capacity}

    players = _collect_roster_entries(user_roster_ids, board)

    # Sort: assign players with fewer eligible active slots first (most constrained)
    active_slots = {k: v for k, v in capacity.items() if k != "BN"}
    players.sort(key=lambda p: sum(
        1 for s in active_slots if can_fill_slot(p["positions"], s)
    ))

    for player in players:
        positions = player["positions"]
        assigned = False
        # Try specific slots first (C, 1B, 2B, etc.), then flex (IF, UTIL)
        for slot in list(active_slots.keys()):
            if slot in ("IF", "UTIL"):
                continue
            if len(by_pos[slot]) < capacity[slot] and can_fill_slot(positions, slot):
                by_pos[slot].append(player["name"])
                assigned = True
                break
        if not assigned:
            for slot in ("IF", "UTIL"):
                if slot in active_slots and len(by_pos[slot]) < capacity[slot] and can_fill_slot(positions, slot):
                    by_pos[slot].append(player["name"])
                    assigned = True
                    break
        if not assigned:
            by_pos.setdefault("BN", []).append(player["name"])

    return {pos: names for pos, names in by_pos.items() if names}
