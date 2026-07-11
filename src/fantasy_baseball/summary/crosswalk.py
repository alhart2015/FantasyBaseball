"""Name -> MLBAM crosswalk, namespaced by player type.

Unlike ``streaks.build_name_to_mlbam_map`` (hitter-only, bare-name keyed), this
reads both hitter and pitcher projection CSVs and keys by
``(normalized_name, player_type)`` so a same-name hitter and pitcher resolve to
their own MLBAM ids -- a bare-name map would first-write-win and return the
wrong player's box-score line. Deliberately imports no streaks code (that module
pulls in duckdb, which the Render process cannot load).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fantasy_baseball.utils.name_utils import normalize_name

_PITCHER_POSITIONS = {"SP", "RP", "P"}


def _read_name_id(path: Path) -> list[tuple[str, int]]:
    """Return (normalized_name, mlbam_id) pairs from one projection CSV.

    A CSV missing the ``Name``/``MLBAMID`` columns (a ``usecols`` mismatch ->
    ValueError) or an unreadable file is skipped, not fatal -- mirrors the
    guard in ``streaks.build_name_to_mlbam_map`` so one bad projection file
    cannot wipe out the whole crosswalk.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", usecols=["Name", "MLBAMID"])
    except (ValueError, FileNotFoundError):
        return []
    out: list[tuple[str, int]] = []
    for name, raw_id in zip(df["Name"], df["MLBAMID"], strict=True):
        if pd.isna(raw_id) or str(raw_id).strip() == "":
            continue
        try:
            mlbam = int(float(raw_id))
        except (ValueError, TypeError):
            continue
        out.append((normalize_name(str(name)), mlbam))
    return out


def build_typed_name_to_mlbam(projections_root: Path, *, season: int) -> dict[tuple[str, str], int]:
    """Build a ``{(normalized_name, player_type): mlbam_id}`` map.

    ``player_type`` is ``"hitter"`` (from ``*-hitters.csv``) or ``"pitcher"``
    (from ``*-pitchers.csv``). First-write-wins within each type namespace.
    """
    season_dir = projections_root / str(season)
    result: dict[tuple[str, str], int] = {}
    for path in sorted(season_dir.glob("*.csv")):
        name = path.name
        if "hitters" in name and "pitchers" not in name:
            player_type = "hitter"
        elif "pitchers" in name:
            player_type = "pitcher"
        else:
            continue
        for norm_name, mlbam in _read_name_id(path):
            result.setdefault((norm_name, player_type), mlbam)
    return result


def player_group(positions: list[str]) -> list[str]:
    """Map Yahoo eligible positions to the game-log groups to read.

    A pitcher-eligible player reads ``"pitching"``; a hitter-eligible player
    reads ``"hitting"``; a two-way player (both) reads both.
    """
    groups: list[str] = []
    if any(p in _PITCHER_POSITIONS for p in positions):
        groups.append("pitching")
    if any(p not in _PITCHER_POSITIONS for p in positions):
        groups.append("hitting")
    return groups or ["hitting"]
