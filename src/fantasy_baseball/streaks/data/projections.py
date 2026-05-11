"""Read FanGraphs preseason projection CSVs and blend per-PA rates.

Reads every ``<system>-hitters*.csv`` under ``data/projections/{season}/``,
filters to rows with PA >= ``PROJECTION_PA_FLOOR`` (drops org filler / NRI
spring rows), coerces MLBAMID to int, and computes per-system rates for
all five fantasy categories:

- ``hr_per_pa`` = HR / PA
- ``sb_per_pa`` = SB / PA
- ``r_per_pa`` = R / PA
- ``rbi_per_pa`` = RBI / PA
- ``avg`` = projected AVG (already a rate in FanGraphs CSVs; no PA divisor)

Returns one ``HitterProjectionRate`` per (player_id, season) with the
simple arithmetic mean across the systems that included that player.
Old-style CSVs lacking R/RBI/AVG columns produce ``None`` for those rates
(rather than crashing) — caller decides whether to use them.

Filename pattern variation: 2025's CSVs are named ``<system>-hitters-2025.csv``;
other years use ``<system>-hitters.csv``. The discovery glob matches both.

This module reads flat CSVs only — no imports from ``web/`` or ``lineup/``,
preserving the streaks package's hard isolation from the production stack.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_baseball.streaks.models import HitterProjectionRate

logger = logging.getLogger(__name__)

PROJECTION_PA_FLOOR = 200

_DENSE_CAT_SOURCE_COLS: tuple[tuple[str, str, bool], ...] = (
    # (source_csv_col, output_rate_col, is_per_pa_count)
    # is_per_pa_count=True: divide by PA. False: already a rate (AVG).
    ("R", "r_per_pa", True),
    ("RBI", "rbi_per_pa", True),
    ("AVG", "avg", False),
)


def discover_projection_files(projections_root: Path, *, season: int) -> list[Path]:
    """Return all ``<system>-hitters*.csv`` files under ``{projections_root}/{season}/``.

    Pitcher files and any non-hitter files are excluded. Order is filesystem-
    dependent; the blender doesn't care.
    """
    season_dir = projections_root / str(season)
    if not season_dir.is_dir():
        return []
    return [
        p
        for p in season_dir.iterdir()
        if p.is_file() and p.suffix == ".csv" and "hitters" in p.name and "pitchers" not in p.name
    ]


def _load_one_system(path: Path) -> pd.DataFrame:
    """Load one system's projection CSV.

    Output columns: MLBAMID, PA, hr_per_pa, sb_per_pa, r_per_pa, rbi_per_pa, avg.
    Dense-cat rate columns are NaN if the source CSV is missing R/RBI/AVG.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"MLBAMID", "PA", "HR", "SB"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("File %s missing required columns %s; skipping", path, sorted(missing))
        return pd.DataFrame(
            columns=["MLBAMID", "PA", "hr_per_pa", "sb_per_pa", "r_per_pa", "rbi_per_pa", "avg"]
        )
    df["MLBAMID"] = pd.to_numeric(df["MLBAMID"], errors="coerce")
    df = df.dropna(subset=["MLBAMID"])
    df["MLBAMID"] = df["MLBAMID"].astype(int)
    df = df[df["PA"] >= PROJECTION_PA_FLOOR].copy()
    df["hr_per_pa"] = df["HR"] / df["PA"]
    df["sb_per_pa"] = df["SB"] / df["PA"]
    for src_col, out_col, is_per_pa_count in _DENSE_CAT_SOURCE_COLS:
        if src_col in df.columns:
            df[out_col] = (df[src_col] / df["PA"]) if is_per_pa_count else df[src_col]
        else:
            df[out_col] = np.nan
    return df[["MLBAMID", "PA", "hr_per_pa", "sb_per_pa", "r_per_pa", "rbi_per_pa", "avg"]]


def load_projection_rates(projections_root: Path, *, season: int) -> list[HitterProjectionRate]:
    """Load and blend projection rates for one season.

    Returns one ``HitterProjectionRate`` per (player_id, season). Players who
    appear in only one system are emitted with ``n_systems=1`` (caller decides
    whether to filter). Players who appear in no system are not emitted.

    Dense-cat fields (r_per_pa, rbi_per_pa, avg) are ``None`` when no system
    that included the player carried that source column (old-style CSVs).
    """
    files = discover_projection_files(projections_root, season=season)
    if not files:
        logger.warning("No projection files found for season %d at %s", season, projections_root)
        return []
    logger.info("Season %d: %d projection files found", season, len(files))

    frames: list[pd.DataFrame] = []
    for path in files:
        sub = _load_one_system(path)
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return []
    stacked = pd.concat(frames, ignore_index=True)

    # NaN-aware mean: a player missing R from one system still gets the other
    # system's R contribution. Pandas mean() skips NaN by default.
    blended = stacked.groupby("MLBAMID", as_index=False).agg(
        hr_per_pa=("hr_per_pa", "mean"),
        sb_per_pa=("sb_per_pa", "mean"),
        r_per_pa=("r_per_pa", "mean"),
        rbi_per_pa=("rbi_per_pa", "mean"),
        avg=("avg", "mean"),
        n_systems=("PA", "count"),
    )

    out: list[HitterProjectionRate] = []
    for r in blended.itertuples(index=False):
        out.append(
            HitterProjectionRate(
                player_id=int(r.MLBAMID),
                season=season,
                hr_per_pa=float(r.hr_per_pa),
                sb_per_pa=float(r.sb_per_pa),
                r_per_pa=None if pd.isna(r.r_per_pa) else float(r.r_per_pa),
                rbi_per_pa=None if pd.isna(r.rbi_per_pa) else float(r.rbi_per_pa),
                avg=None if pd.isna(r.avg) else float(r.avg),
                n_systems=int(r.n_systems),
            )
        )
    return out


def load_projection_rates_for_seasons(
    projections_root: Path, seasons: Iterable[int]
) -> list[HitterProjectionRate]:
    """Convenience wrapper: concat per-season loads."""
    out: list[HitterProjectionRate] = []
    for s in seasons:
        out.extend(load_projection_rates(projections_root, season=s))
    return out
