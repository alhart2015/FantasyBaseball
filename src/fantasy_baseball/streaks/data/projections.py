"""Read FanGraphs preseason projection CSVs and blend per-PA rates.

Reads every `<system>-hitters*.csv` under `data/projections/{season}/`,
filters to rows with PA >= ``PROJECTION_PA_FLOOR`` (drops org filler / NRI
spring rows), coerces MLBAMID to int, and computes per-system HR/PA and
SB/PA. Returns one ``HitterProjectionRate`` per (player_id, season) with
the simple arithmetic mean across the systems that included that player.

Filename pattern variation: 2025's CSVs are named ``<system>-hitters-2025.csv``;
other years use ``<system>-hitters.csv``. The discovery glob matches both.

This module reads flat CSVs only — no imports from ``web/`` or ``lineup/``,
preserving the streaks package's hard isolation from the production stack.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from fantasy_baseball.streaks.models import HitterProjectionRate

logger = logging.getLogger(__name__)

# Drops org-filler rows (Steamer projects ~3,500-4,500 hitters/year, mostly
# 1-50 PA blowouts of MiLB depth charts; ZiPS does the same with ~1,700-2,000).
# The floor matches what we use for the draft pipeline elsewhere; revisit
# only if it bites a real player who genuinely projects below it.
PROJECTION_PA_FLOOR = 200


def discover_projection_files(projections_root: Path, *, season: int) -> list[Path]:
    """Return all ``<system>-hitters*.csv`` files under ``{projections_root}/{season}/``.

    Pitcher files and any non-hitter files are excluded. Order is filesystem-
    dependent; the blender doesn't care.
    """
    season_dir = projections_root / str(season)
    if not season_dir.is_dir():
        return []
    files = [
        p
        for p in season_dir.iterdir()
        if p.is_file() and p.suffix == ".csv" and "hitters" in p.name and "pitchers" not in p.name
    ]
    return files


def _load_one_system(path: Path) -> pd.DataFrame:
    """Load one system's projection CSV into a 4-column frame keyed by MLBAMID.

    Drops rows missing MLBAMID and rows below ``PROJECTION_PA_FLOOR`` PA.
    Computes hr_per_pa and sb_per_pa per row.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "MLBAMID" not in df.columns:
        logger.warning("File %s has no MLBAMID column; skipping", path)
        return pd.DataFrame(columns=["MLBAMID", "PA", "hr_per_pa", "sb_per_pa"])
    df["MLBAMID"] = pd.to_numeric(df["MLBAMID"], errors="coerce")
    df = df.dropna(subset=["MLBAMID"])
    df["MLBAMID"] = df["MLBAMID"].astype(int)
    df = df[df["PA"] >= PROJECTION_PA_FLOOR].copy()
    df["hr_per_pa"] = df["HR"] / df["PA"]
    df["sb_per_pa"] = df["SB"] / df["PA"]
    return df[["MLBAMID", "PA", "hr_per_pa", "sb_per_pa"]]


def load_projection_rates(projections_root: Path, *, season: int) -> list[HitterProjectionRate]:
    """Load and blend projection rates for one season.

    Returns one ``HitterProjectionRate`` per (player_id, season). Players who
    appear in only one system are emitted with ``n_systems=1`` (caller decides
    whether to filter). Players who appear in no system are not emitted.
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

    blended = stacked.groupby("MLBAMID", as_index=False).agg(
        hr_per_pa=("hr_per_pa", "mean"),
        sb_per_pa=("sb_per_pa", "mean"),
        n_systems=("PA", "count"),
    )

    return [
        HitterProjectionRate(
            player_id=int(r.MLBAMID),
            season=season,
            hr_per_pa=float(r.hr_per_pa),
            sb_per_pa=float(r.sb_per_pa),
            n_systems=int(r.n_systems),
        )
        for r in blended.itertuples(index=False)
    ]


def load_projection_rates_for_seasons(
    projections_root: Path, seasons: Iterable[int]
) -> list[HitterProjectionRate]:
    """Convenience wrapper: concat per-season loads."""
    out: list[HitterProjectionRate] = []
    for s in seasons:
        out.extend(load_projection_rates(projections_root, season=s))
    return out
