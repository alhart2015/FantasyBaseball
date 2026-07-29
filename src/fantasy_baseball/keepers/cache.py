"""Fetch-on-miss CSV cache plumbing shared by the keepers data pulls.

Never overwrites a good cache with an empty/failed pull. Not calculation --
pure I/O plumbing, preserved verbatim from the old data/skill_luck.py.

A fetcher that transforms its response (`bref` repairs mojibake names, `savant`
tallies pitch outcomes) must pass `version`. Cache keys otherwise encode only
the source and year, so changing a transform leaves every existing cache
serving pre-transform data with no signal that anything is wrong -- which is
exactly how repaired names got written and then silently un-repaired on the
next run. Bump `version` whenever a fetcher's output shape or values change.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd


def _read_cached(path: Path) -> pd.DataFrame | None:
    if path.exists():
        df: pd.DataFrame = pd.read_csv(path)
        if not df.empty:
            return df
    return None


def versioned(path: Path, version: int) -> Path:
    """`foo.csv` -> `foo.v2.csv`. Version 1 keeps the bare name, so adding a
    version to an untransformed pull does not orphan its existing cache."""
    return path if version == 1 else path.with_suffix(f".v{version}{path.suffix}")


def fetch_or_cache(
    path: Path,
    fetcher: Callable[[], pd.DataFrame],
    *,
    tolerate_empty: bool = False,
    version: int = 1,
) -> pd.DataFrame:
    path = versioned(path, version)
    cached = _read_cached(path)
    if cached is not None:
        return cached
    df = fetcher()
    if df is None or df.empty:
        if tolerate_empty and df is not None:
            return df  # expected-empty (e.g. pre-2016 xHR); do not cache emptiness
        raise RuntimeError(
            f"fetch for {path.name} returned empty; refusing to overwrite/write cache"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
