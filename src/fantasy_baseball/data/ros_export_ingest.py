"""Ingest FanGraphs one-click member exports into a dated ROS snapshot dir.

FanGraphs prohibits automated extraction (scraping/API/web query); the supported
path is a human's one-click member export. This module only handles the CSV
files the user has already exported -- it never contacts FanGraphs. The guided
loop walks the user through the 5 systems x {hitters, pitchers}, stages each
freshly-downloaded + type-validated CSV under our naming convention, and reports
which systems are complete so the caller can blend + push to prod.
"""

from __future__ import annotations

from pathlib import Path

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
