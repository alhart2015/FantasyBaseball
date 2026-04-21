import pandas as pd

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.replacement import calculate_replacement_levels
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import (
    CLOSER_SV_THRESHOLD,
    DEFAULT_ROSTER_SLOTS,
    compute_starters_per_position,
)
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import can_fill_slot


def compute_slot_scarcity_order(
    board: pd.DataFrame,
    roster_slots: dict[str, int] | None = None,
) -> list[str]:
    """Return roster slots ordered by positional scarcity (most scarce first).

    Scarcity = sum(SGP of all eligible players) / number of roster slots.
    Lower scarcity = fewer resources per slot = assign multi-eligible players
    here first so flex slots stay open for less flexible players.
    """
    if roster_slots is None:
        roster_slots = DEFAULT_ROSTER_SLOTS
    scarcity: dict[str, float] = {}
    for slot, n_slots in roster_slots.items():
        if slot in ("BN", "IL"):
            continue
        eligible = board[board["positions"].apply(lambda p, slot=slot: can_fill_slot(p, slot))]
        total_sgp = eligible["total_sgp"].sum() if "total_sgp" in eligible.columns else 0
        scarcity[slot] = total_sgp / n_slots if n_slots > 0 else float("inf")
    return sorted(scarcity.keys(), key=lambda s: scarcity[s])

def _player_bucket(player) -> str:
    """Classify a player into hitter / sp / closer for VONA."""
    if player.get("player_type") == PlayerType.HITTER:
        return "hitter"
    if player.get("sv", 0) >= CLOSER_SV_THRESHOLD:
        return "closer"
    return "sp"


def calculate_vona_scores(
    available: pd.DataFrame,
    picks_until_next: int | None = None,
) -> dict:
    """Compute Value Over Next Available for each player.

    For each player, estimates what the best player of the same bucket
    (hitter / SP / closer) will still be available after opponents make
    ``picks_until_next`` picks, drafting by ADP.

    Returns dict of player_id -> VONA score.

    Note: position-level VONA (per-position hitter buckets) was tested
    and regressed badly — it over-values positional scarcity, causing the
    recommender to reach for scarce positions at the expense of overall
    value.  The 3-bucket approach keeps hitter VONA balanced.
    """
    if picks_until_next is None or picks_until_next < 1:
        picks_until_next = 10  # sensible default

    # Sort the full pool by ADP — opponents draft roughly by ADP
    adp_sorted = available.sort_values("adp", ascending=True)

    # The next N picks (by ADP) are the ones opponents will take
    gone_ids = set(adp_sorted.head(picks_until_next)["player_id"])

    # What remains after opponents pick
    remaining = available[~available["player_id"].isin(gone_ids)]

    # Assign buckets vectorized
    is_hitter = remaining["player_type"] == PlayerType.HITTER
    sv = remaining["sv"].fillna(0) if "sv" in remaining.columns else pd.Series(0, index=remaining.index)
    remaining_buckets = pd.Series("sp", index=remaining.index)
    remaining_buckets[is_hitter] = "hitter"
    remaining_buckets[(~is_hitter) & (sv >= CLOSER_SV_THRESHOLD)] = "closer"

    sgp = remaining["total_sgp"].fillna(0)
    best_remaining = sgp.groupby(remaining_buckets).max().to_dict()
    for b in ("hitter", "sp", "closer"):
        best_remaining.setdefault(b, 0)

    # VONA = player SGP - best remaining in same bucket (vectorized)
    is_hitter_a = available["player_type"] == PlayerType.HITTER
    sv_a = available["sv"].fillna(0) if "sv" in available.columns else pd.Series(0, index=available.index)
    avail_buckets = pd.Series("sp", index=available.index)
    avail_buckets[is_hitter_a] = "hitter"
    avail_buckets[(~is_hitter_a) & (sv_a >= CLOSER_SV_THRESHOLD)] = "closer"

    avail_sgp = available["total_sgp"].fillna(0)
    best_for_bucket = avail_buckets.map(best_remaining)
    vona_series = avail_sgp - best_for_bucket

    return dict(zip(available["player_id"], vona_series, strict=False))



def get_recommendations(
    board: pd.DataFrame,
    drafted: list[str],
    user_roster: list[str],  # noqa: ARG001  (kept for API; used by callers positionally)
    n: int = 5,
    filled_positions: dict[str, int] | None = None,
    picks_until_next: int | None = None,
    roster_slots: dict[str, int] | None = None,
    num_teams: int | None = None,
    scoring_mode: str = "var",
) -> list[dict]:
    """Get top draft pick recommendations.

    Recalculates replacement levels from the undrafted pool so that
    positional scarcity (e.g. a run on catchers) is reflected in VAR.

    *scoring_mode*: "var" (default) uses Value Above Replacement for
    ranking; "vona" uses Value Over Next Available, which accounts for
    talent depth at each player type (hitter/SP/closer).
    """
    if roster_slots is None:
        roster_slots = DEFAULT_ROSTER_SLOTS
    available = board[~board["player_id"].isin(drafted)]

    # Recalculate replacement levels from the full remaining pool so
    # positional scarcity is properly reflected.
    starters = compute_starters_per_position(roster_slots, num_teams)
    repl_levels = calculate_replacement_levels(available, starters)

    # Only recompute VAR for top candidates (by pre-computed VAR).
    # The full pool sets replacement levels accurately, but iterating
    # all ~3000 players per pick is the main performance bottleneck.
    # 300 covers all draftable players (10 teams × 23 slots = 230) plus
    # padding for positional scarcity shifts at C/SS/1B.
    _VAR_CANDIDATE_LIMIT = 300
    candidates = available.nlargest(_VAR_CANDIDATE_LIMIT, "var")
    live_var = {}
    live_pos = {}
    for idx, row in candidates.iterrows():
        var, pos = calculate_var(row, repl_levels, return_position=True)
        live_var[idx] = var
        live_pos[idx] = pos
    available = candidates.copy()
    available["var"] = available.index.map(live_var)
    available["best_position"] = available.index.map(live_pos)

    # Compute VONA if requested.  Use the full undrafted pool (not the
    # top-150 VAR candidates) so that "best remaining in bucket" reflects
    # true talent depth.  VONA only needs total_sgp and adp, both static
    # columns, so iterating the full pool is cheap (~O(n)).
    vona_scores = None
    if scoring_mode == "vona":
        full_available = board[~board["player_id"].isin(drafted)]
        vona_scores = calculate_vona_scores(full_available, picks_until_next)
        available["vona"] = available["player_id"].map(vona_scores).fillna(0)

    # Sort by the active scoring mode
    sort_col = "vona" if scoring_mode == "vona" else "var"
    available = available.sort_values(sort_col, ascending=False)

    if filled_positions is None:
        filled_positions = {}

    # Filter out players who have no open roster slot (including BN).
    # E.g. if all OF, UTIL, and BN spots are full, don't suggest more OFs.
    available = _filter_rosterable(available, filled_positions, roster_slots)
    available = available.sort_values(sort_col, ascending=False)

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
        score = player.get("vona", 0) if scoring_mode == "vona" else player["var"]
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
        # Check specific positional slots before flex (IF/UTIL) so the note
        # shows "fills 3B need" rather than "fills IF need" when both are open.
        specific_unfilled = [s for s in unfilled if s not in ("IF", "UTIL")]
        flex_unfilled = [s for s in unfilled if s in ("IF", "UTIL")]
        for slot in specific_unfilled + flex_unfilled:
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
    # Sort all by score (VAR or VONA)
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
        return any(can_fill_slot(positions, slot) for slot in open_slots)

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
    player_lookup: dict | None = None,
) -> list[pd.Series]:
    """Look up board entries for each roster player by player_id."""
    if player_lookup is None:
        # Build a pid lookup dict for O(1) lookups instead of O(n) per player
        pid_index = {}
        for _idx, row in board.iterrows():
            pid_index[row["player_id"]] = row
    else:
        pid_index = player_lookup
    players: list[pd.Series] = []
    for pid in user_roster_ids:
        if pid in pid_index:
            players.append(pid_index[pid])
        else:
            # Fallback: try name match (for entries without player_id).
            # player_id may be fg_id::type or name::type, so try the
            # prefix as a name only if it looks like one (not numeric).
            prefix = pid.split("::")[0] if "::" in pid else pid
            if prefix.isdigit() or prefix.startswith("sa"):
                continue  # fg_id prefix, can't match by name
            rows = board[board["name_normalized"] == normalize_name(prefix)]
            if not rows.empty:
                players.append(rows.iloc[0])
    return players


_scarcity_cache: dict[int, list[str]] = {}


def get_filled_positions(
    user_roster_ids: list[str],
    board: pd.DataFrame,
    roster_slots: dict[str, int] | None = None,
    player_lookup: dict | None = None,
) -> dict[str, int]:
    """Count how many of each roster slot the user has filled.

    Assigns each drafted player to the most *scarce* open slot they're
    eligible for (by positional scarcity index), then falls back to flex
    slots (IF, UTIL), then bench.  This ensures multi-position players
    occupy their scarcest position so flex slots stay open.
    """
    if roster_slots is None:
        roster_slots = DEFAULT_ROSTER_SLOTS

    # Build capacity: how many of each slot are available
    capacity: dict[str, int] = {
        pos: count for pos, count in roster_slots.items()
        if pos != "IL"
    }
    filled: dict[str, int] = {pos: 0 for pos in capacity}

    players = _collect_roster_entries(user_roster_ids, board, player_lookup)

    # Sort: assign players with fewer eligible active slots first (most constrained)
    active_slots = {k: v for k, v in capacity.items() if k != "BN"}
    players.sort(key=lambda p: sum(
        1 for s in active_slots if can_fill_slot(p["positions"], s)
    ))

    # Slot assignment order: scarcity-based for specific slots, then flex.
    # Cache the scarcity order since it depends only on the board and slots.
    cache_key = id(board)
    if cache_key not in _scarcity_cache:
        _scarcity_cache.clear()  # keep only one entry
        _scarcity_cache[cache_key] = compute_slot_scarcity_order(board, roster_slots)
    scarcity_order = _scarcity_cache[cache_key]
    specific_slots = [s for s in scarcity_order if s not in ("IF", "UTIL")]
    flex_slots = [s for s in scarcity_order if s in ("IF", "UTIL")]

    for player in players:
        positions = player["positions"]
        assigned = False
        # Try specific slots in scarcity order (most scarce first)
        for slot in specific_slots:
            if filled[slot] < capacity[slot] and can_fill_slot(positions, slot):
                filled[slot] += 1
                assigned = True
                break
        if not assigned:
            for slot in flex_slots:
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

    # Slot assignment order: scarcity-based (cached)
    cache_key = id(board)
    if cache_key not in _scarcity_cache:
        _scarcity_cache.clear()
        _scarcity_cache[cache_key] = compute_slot_scarcity_order(board, roster_slots)
    scarcity_order = _scarcity_cache[cache_key]
    specific_slots = [s for s in scarcity_order if s not in ("IF", "UTIL")]
    flex_slots = [s for s in scarcity_order if s in ("IF", "UTIL")]

    for player in players:
        positions = player["positions"]
        assigned = False
        for slot in specific_slots:
            if len(by_pos[slot]) < capacity[slot] and can_fill_slot(positions, slot):
                by_pos[slot].append(player["name"])
                assigned = True
                break
        if not assigned:
            for slot in flex_slots:
                if slot in active_slots and len(by_pos[slot]) < capacity[slot] and can_fill_slot(positions, slot):
                    by_pos[slot].append(player["name"])
                    assigned = True
                    break
        if not assigned:
            by_pos.setdefault("BN", []).append(player["name"])

    return {pos: names for pos, names in by_pos.items() if names}
