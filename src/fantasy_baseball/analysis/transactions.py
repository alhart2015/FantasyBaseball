"""Transaction analysis — pairing, scoring, and aggregation."""

from datetime import datetime

from fantasy_baseball.analysis.spoe import load_projections_for_date
from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.utils.name_utils import normalize_name

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


def _find_player_wsgp(name, positions_str, hitters_proj, pitchers_proj, leverage):
    """Look up a player in projections and compute wSGP."""
    if not name:
        return 0.0

    name_norm = normalize_name(name)
    positions = _parse_positions(positions_str)

    # Try hitter first
    if _is_hitter(positions) and not hitters_proj.empty:
        matches = hitters_proj[hitters_proj["_name_norm"] == name_norm]
        if not matches.empty:
            from fantasy_baseball.models.player import HitterStats
            row = matches.iloc[0]
            stats = HitterStats.from_dict(row.to_dict())
            return calculate_weighted_sgp(stats, leverage)

    # Try pitcher
    if _is_pitcher(positions) and not pitchers_proj.empty:
        matches = pitchers_proj[pitchers_proj["_name_norm"] == name_norm]
        if not matches.empty:
            from fantasy_baseball.models.player import PitcherStats
            row = matches.iloc[0]
            stats = PitcherStats.from_dict(row.to_dict())
            return calculate_weighted_sgp(stats, leverage)

    return 0.0


def score_transaction(conn, txn: dict, year: int) -> dict:
    """Compute wSGP for the add and drop sides of a transaction.

    Uses the team's leverage at the time of the transaction (from the
    nearest prior standings snapshot) and the nearest ROS projections.

    Args:
        conn: SQLite connection.
        txn: Transaction dict with team, timestamp, add_name, add_positions,
             drop_name, drop_positions.
        year: Season year.

    Returns:
        {"add_wsgp": float, "drop_wsgp": float, "value": float}
    """
    # Convert Unix timestamp to date string for DB lookups
    ts = int(txn.get("timestamp", 0) or 0)
    txn_date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else f"{year}-03-01"

    # Find nearest standings snapshot for leverage
    row = conn.execute(
        "SELECT MAX(snapshot_date) as best FROM standings "
        "WHERE year = ? AND snapshot_date <= ?",
        (year, txn_date),
    ).fetchone()
    standings_date = row["best"] if row and row["best"] else None

    if standings_date:
        standings_rows = conn.execute(
            "SELECT team, r, hr, rbi, sb, avg, w, k, sv, era, whip "
            "FROM standings WHERE year = ? AND snapshot_date = ?",
            (year, standings_date),
        ).fetchall()
        standings = [
            {"name": r["team"], "stats": {
                "R": r["r"], "HR": r["hr"], "RBI": r["rbi"], "SB": r["sb"],
                "AVG": r["avg"], "W": r["w"], "K": r["k"], "SV": r["sv"],
                "ERA": r["era"], "WHIP": r["whip"],
            }}
            for r in standings_rows
        ]
        leverage = calculate_leverage(standings, txn["team"])
    else:
        # No standings yet — equal weights
        leverage = {cat: 1.0 for cat in ["R", "HR", "RBI", "SB", "AVG",
                                          "W", "K", "SV", "ERA", "WHIP"]}

    # Load projections nearest to transaction date
    hitters_proj, pitchers_proj = load_projections_for_date(conn, year, txn_date)

    add_wsgp = _find_player_wsgp(
        txn.get("add_name"), txn.get("add_positions"),
        hitters_proj, pitchers_proj, leverage,
    )
    drop_wsgp = _find_player_wsgp(
        txn.get("drop_name"), txn.get("drop_positions"),
        hitters_proj, pitchers_proj, leverage,
    )

    return {
        "add_wsgp": round(add_wsgp, 2),
        "drop_wsgp": round(drop_wsgp, 2),
        "value": round(add_wsgp - drop_wsgp, 2),
    }
