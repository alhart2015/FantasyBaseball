"""Transaction analysis — pairing, scoring, and aggregation."""

HITTER_POSITIONS = {"C", "1B", "2B", "3B", "SS", "OF", "Util", "DH"}
PITCHER_POSITIONS = {"SP", "RP", "P"}

PAIRING_WINDOW_SECONDS = 86400  # 24 hours


def _parse_positions(pos_str):
    """Split comma-separated positions string into a set."""
    if not pos_str:
        return set()
    return {p.strip() for p in pos_str.split(",")}


def _is_hitter(positions):
    return bool(positions & HITTER_POSITIONS)


def _is_pitcher(positions):
    return bool(positions & PITCHER_POSITIONS)


def pair_standalone_moves(transactions: list[dict]) -> list[tuple[str, str]]:
    """Find pairs of standalone drop + add from the same team within 24h.

    Only considers transactions with type "drop" or "add" (not "add/drop").
    Skips transactions that already have a paired_with value.

    Matching priority:
    1. Exact position overlap between dropped and added player
    2. Same player type (hitter/pitcher)
    3. No match — left unpaired

    Args:
        transactions: List of transaction dicts with keys: transaction_id,
            team, type, timestamp, add_positions, drop_positions, paired_with.

    Returns:
        List of (drop_txn_id, add_txn_id) tuples.
    """
    drops = []
    adds = []
    for txn in transactions:
        if txn.get("paired_with"):
            continue
        if txn["type"] == "drop":
            drops.append(txn)
        elif txn["type"] == "add":
            adds.append(txn)

    paired_drop_ids = set()
    paired_add_ids = set()
    pairs = []

    for drop in drops:
        drop_ts = int(drop.get("timestamp", 0) or 0)
        drop_pos = _parse_positions(drop.get("drop_positions"))
        drop_is_hitter = _is_hitter(drop_pos)
        drop_is_pitcher = _is_pitcher(drop_pos)

        candidates = []
        for add in adds:
            if add["transaction_id"] in paired_add_ids:
                continue
            if add["team"] != drop["team"]:
                continue
            add_ts = int(add.get("timestamp", 0) or 0)
            if abs(add_ts - drop_ts) > PAIRING_WINDOW_SECONDS:
                continue
            candidates.append(add)

        if not candidates:
            continue

        best = None
        best_score = 0
        for add in candidates:
            add_pos = _parse_positions(add.get("add_positions"))
            overlap = drop_pos & add_pos
            if overlap:
                score = 2 + len(overlap)
            elif (drop_is_hitter and _is_hitter(add_pos)) or \
                 (drop_is_pitcher and _is_pitcher(add_pos)):
                score = 1
            else:
                score = 0

            if score > best_score:
                best_score = score
                best = add

        if best and best_score > 0:
            pairs.append((drop["transaction_id"], best["transaction_id"]))
            paired_drop_ids.add(drop["transaction_id"])
            paired_add_ids.add(best["transaction_id"])

    return pairs
