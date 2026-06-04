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
from collections.abc import Callable
from dataclasses import dataclass, field
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


@dataclass
class IngestResult:
    """Outcome of a guided ingest run."""

    staged: dict[tuple[str, str], Path] = field(default_factory=dict)
    skipped_systems: set[str] = field(default_factory=set)
    aborted: bool = False

    def complete_systems(self, systems: list[str]) -> list[str]:
        """Systems with BOTH hitters and pitchers staged (push-eligible)."""
        return [
            s for s in systems if (s, "hitters") in self.staged and (s, "pitchers") in self.staged
        ]


def run_guided_ingest(
    systems: list[str],
    source_dir: Path,
    dest_dir: Path,
    *,
    prompt_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    now_fn: Callable[[], float],
) -> IngestResult:
    """Walk the user through exporting each (system, player_type) and stage each.

    ``prompt_fn(message) -> response`` returns the user's reply: ``""`` (Enter =
    "I exported it"), ``"s"`` (skip this system), or ``"q"`` (abort). ``output_fn``
    prints progress. ``now_fn`` supplies the timestamp captured just before each
    prompt so :func:`stage_export` can pick the just-exported file. I/O is injected
    so the loop is unit-testable without real stdin/clock.
    """
    result = IngestResult()
    for system, ptype in export_steps(systems):
        if system in result.skipped_systems:
            continue
        while True:
            since = now_fn()
            resp = (
                prompt_fn(
                    f"Export {system} {ptype} from FanGraphs, then press Enter "
                    f"(s=skip system, q=abort): "
                )
                .strip()
                .lower()
            )
            if resp == "q":
                result.aborted = True
                return result
            if resp == "s":
                result.skipped_systems.add(system)
                output_fn(f"  skipped {system}")
                break
            staged = stage_export(source_dir, since, system, ptype, dest_dir)
            if staged is None:
                output_fn(
                    f"  no new valid {ptype} export found in {source_dir} -- "
                    f"export it and press Enter again"
                )
                continue
            result.staged[(system, ptype)] = staged
            output_fn(f"  staged {staged.name}")
            break
    return result
