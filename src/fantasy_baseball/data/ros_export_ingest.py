"""Ingest FanGraphs one-click member exports into a dated ROS snapshot dir.

FanGraphs prohibits automated extraction (scraping/API/web query); the supported
path is a human's one-click member export. This module only handles the CSV
files the user has already exported -- it never contacts FanGraphs. The guided
loop walks the user through the 5 systems x {hitters, pitchers}, stages each
freshly-downloaded + type-validated CSV under our naming convention, and reports
which systems are complete so the caller can blend + push to prod.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fantasy_baseball.data.fangraphs import parse_hitting_csv, parse_pitching_csv

PLAYER_TYPES: tuple[str, str] = ("hitters", "pitchers")


def export_steps(systems: list[str]) -> list[tuple[str, str]]:
    """(system, player_type) pairs in prompt order: each system hitters then pitchers."""
    return [(system, ptype) for system in systems for ptype in PLAYER_TYPES]


def find_newest_csv(source_dir: Path, since_ts: float) -> Path | None:
    """Newest ``*.csv`` in ``source_dir`` with mtime >= ``since_ts``; ``None`` if none.

    ``since_ts`` is captured just before prompting the user, so this picks the file
    they just exported rather than a stale prior download.
    """
    candidates = [p for p in Path(source_dir).glob("*.csv") if p.stat().st_mtime >= since_ts]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def validate_export_type(path: Path, player_type: str) -> bool:
    """True if ``path`` parses as a FanGraphs export of ``player_type``.

    Reuses the production parsers, which raise ``ValueError`` when the required
    hitter/pitcher columns are absent -- so a "wrong page" export (e.g. a hitters
    CSV offered as pitchers) is rejected.
    """
    try:
        if player_type == "hitters":
            parse_hitting_csv(path)
        else:
            parse_pitching_csv(path)
    except Exception:
        return False
    return True


def stage_export(
    source_dir: Path, since_ts: float, system: str, player_type: str, dest_dir: Path
) -> Path | None:
    """Stage the newest valid export for ``(system, player_type)`` into ``dest_dir``.

    Returns the staged path ``dest_dir/{system}-{player_type}.csv``, or ``None``
    when no ``*.csv`` newer than ``since_ts`` exists or the newest one fails type
    validation (caller re-prompts; nothing is staged on ``None``).
    """
    src = find_newest_csv(Path(source_dir), since_ts)
    if src is None or not validate_export_type(src, player_type):
        return None
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{system}-{player_type}.csv"
    shutil.copy(src, dest)
    return dest
