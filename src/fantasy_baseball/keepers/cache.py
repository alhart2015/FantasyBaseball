"""Fetch-on-miss CSV cache plumbing shared by the keepers data pulls.

Never overwrites a good cache with an empty/failed pull. Not calculation --
pure I/O plumbing, preserved verbatim from the old data/skill_luck.py.
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


def fetch_or_cache(
    path: Path, fetcher: Callable[[], pd.DataFrame], *, tolerate_empty: bool = False
) -> pd.DataFrame:
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
