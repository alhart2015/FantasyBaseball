"""Per-season position eligibility from the MLB Stats fielding leaderboard.

Pure and I/O-free like the rest of the normalization layer: it takes the raw
fielding frame (`keepers.mlb_stats.fetch_mlb_season(cache_dir, year, "fielding")`)
and returns, per MLBAM id, the base slots at which the player has >= 10 games that
season. That is the contemporaneous eligibility the scarcity measurement needs;
`keepers.positions` (the Yahoo map) has only the current season and is used for the
LIVE board, not for measuring history. See
`docs/superpowers/specs/2026-07-30-per-season-positions-design.md`.
"""

from __future__ import annotations

import pandas as pd

# Yahoo eligibility rule: >= 10 games at a position in the season. Outfield is
# combined -- the league rosters a single OF slot -- so LF/CF/RF all fold to OF and
# their games sum before the threshold. DH and any non-fielding token fold to no base
# slot, so a pure DH is absent here and the caller prices him at UTIL.
#
# A generic "OF" token is deliberately NOT mapped. MLB Stats' "OF" is an aggregate of
# the corners, so folding it in and summing would double-count (a real 7-game OF logged
# as OF=7 plus LF=4/CF=3 would read as 14 and clear the threshold falsely). No cached
# 2022-2025 season emits an "OF" token; if one ever does, resolve aggregate-vs-corner
# semantics before mapping it rather than summing blind.
GAMES_THRESHOLD = 10
POSITION_TO_SLOT: dict[str, str] = {
    "C": "C",
    "1B": "1B",
    "2B": "2B",
    "3B": "3B",
    "SS": "SS",
    "LF": "OF",
    "CF": "OF",
    "RF": "OF",
    "P": "P",
}
_REQUIRED = ("player.id", "position.abbreviation", "stat.games")


def season_eligibility(fielding: pd.DataFrame) -> dict[int, set[str]]:
    """MLBAM id -> base slots with >= 10 games this season."""
    missing = [c for c in _REQUIRED if c not in fielding.columns]
    if missing:
        raise KeyError(f"fielding frame missing {missing}; got {sorted(fielding.columns)}")
    frame = pd.DataFrame(
        {
            "pid": pd.to_numeric(fielding["player.id"], errors="coerce").astype("Int64"),
            "slot": fielding["position.abbreviation"].map(POSITION_TO_SLOT),
            "games": pd.to_numeric(fielding["stat.games"], errors="coerce"),
        }
    ).dropna(subset=["pid", "slot", "games"])
    grouped = frame.groupby(["pid", "slot"], sort=False)["games"].sum()
    out: dict[int, set[str]] = {}
    for (pid, slot), games in grouped.items():
        if games >= GAMES_THRESHOLD:
            out.setdefault(int(pid), set()).add(str(slot))
    return out
