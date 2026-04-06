"""Recency blending for Player objects.

Bridges the gap between the recency model (which works on rate dicts and
game-log lists) and the Player/Stats dataclasses used by the lineup and
waiver modules.
"""

from __future__ import annotations

from collections import defaultdict

from fantasy_baseball.utils.name_utils import normalize_name

# Hitter game log columns to extract from SQLite
_HITTER_LOG_COLS = ("date", "pa", "ab", "h", "r", "hr", "rbi", "sb")

# Pitcher game log columns to extract from SQLite (g synthesized as 1 per row)
_PITCHER_LOG_COLS = ("date", "ip", "k", "er", "bb", "h_allowed", "w", "sv", "gs")


def load_game_logs_by_name(
    conn,
    season: int,
) -> dict[str, list[dict]]:
    """Load per-game log entries from SQLite, keyed by normalized name.

    Returns {normalized_name: [game_dicts]} where each game dict has the
    fields expected by predict_reliability_blend.  Pitcher dicts include a
    synthesized ``g = 1`` field (each row is one game appearance).
    """
    logs: dict[str, list[dict]] = defaultdict(list)

    # Hitters
    rows = conn.execute(
        "SELECT name, date, pa, ab, h, r, hr, rbi, sb "
        "FROM game_logs WHERE season = ? AND player_type = 'hitter' "
        "ORDER BY date",
        (season,),
    ).fetchall()
    for row in rows:
        key = f"{normalize_name(row['name'])}::hitter"
        logs[key].append(
            {col: row[col] if col == "date" else (row[col] or 0) for col in _HITTER_LOG_COLS}
        )

    # Pitchers
    rows = conn.execute(
        "SELECT name, date, ip, k, er, bb, h_allowed, w, sv, gs "
        "FROM game_logs WHERE season = ? AND player_type = 'pitcher' "
        "ORDER BY date",
        (season,),
    ).fetchall()
    for row in rows:
        key = f"{normalize_name(row['name'])}::pitcher"
        entry = {col: row[col] if col == "date" else (row[col] or 0) for col in _PITCHER_LOG_COLS}
        entry["g"] = 1  # each row is one game appearance
        logs[key].append(entry)

    return dict(logs)
