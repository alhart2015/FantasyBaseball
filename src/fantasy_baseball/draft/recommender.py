import pandas as pd
from fantasy_baseball.utils.constants import DEFAULT_ROSTER_SLOTS, IF_ELIGIBLE
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import can_fill_slot, is_hitter

REQUIRED_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF", "P"]


def get_recommendations(
    board: pd.DataFrame,
    drafted: list[str],
    user_roster: list[str],
    n: int = 5,
    filled_positions: dict[str, int] | None = None,
    picks_until_next: int | None = None,
    roster_slots: dict[str, int] | None = None,
) -> list[dict]:
    """Get top draft pick recommendations."""
    if roster_slots is None:
        roster_slots = DEFAULT_ROSTER_SLOTS
    available = board[~board["player_id"].isin(drafted)].head(n * 3)
    if filled_positions is None:
        filled_positions = {}
    unfilled = _get_unfilled_positions(filled_positions, roster_slots)
    recs = []
    for _, player in available.iterrows():
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
            remaining_at_pos = len(available[available["best_position"] == pos])
            if remaining_at_pos <= 3:
                scarcity = f"scarce position — only {remaining_at_pos} left in top tier"
                rec["note"] = f"{rec['note']}; {scarcity}" if rec["note"] else scarcity
        recs.append(rec)
    recs.sort(key=lambda r: r["var"], reverse=True)
    return recs[:n]


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
    user_roster_names: list[str], board: pd.DataFrame
) -> dict[str, int]:
    """Count how many of each position the user has filled."""
    filled: dict[str, int] = {}
    for name in user_roster_names:
        rows = board[board["name_normalized"] == normalize_name(name)]
        if rows.empty:
            continue
        player = rows.iloc[0]
        pos = player["best_position"]
        filled[pos] = filled.get(pos, 0) + 1
    return filled


def get_roster_by_position(
    user_roster_names: list[str], board: pd.DataFrame
) -> dict[str, list[str]]:
    """Map position -> list of player names for the user's roster."""
    by_pos: dict[str, list[str]] = {}
    for name in user_roster_names:
        rows = board[board["name_normalized"] == normalize_name(name)]
        if rows.empty:
            continue
        player = rows.iloc[0]
        pos = player["best_position"]
        by_pos.setdefault(pos, []).append(player["name"])
    return by_pos
