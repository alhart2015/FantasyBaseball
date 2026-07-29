"""Fetch-on-miss CSV cache plumbing shared by the keepers data pulls.

Never overwrites a good cache with an empty/failed pull. Not calculation --
pure I/O plumbing, preserved verbatim from the old data/skill_luck.py.

Two independent kinds of staleness, because they fail differently:

* `version` guards the CODE. A fetcher that transforms its response (`bref`
  repairs mojibake names, `savant` tallies pitch outcomes) puts the transform's
  version in the filename. Without it a cache key encodes only source and year,
  so changing a transform leaves every existing cache serving pre-transform
  data -- which is exactly how repaired names were written once and silently
  un-repaired on the next run. Bump it when a fetcher's output changes.
* `max_age` guards the DATA, and is the one that bites daily. These are
  season-to-date pulls; a cache written in June is not an answer in July, and
  nothing about that announces itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pandas as pd


def _read_cached(path: Path, max_age: timedelta | None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    # Epoch arithmetic, not local wall clock: differencing two naive datetimes
    # shifts the age by an hour across a DST boundary. `>=` so max_age=0 (the
    # --refresh path) always refetches -- st_mtime resolves finer than
    # time.time()'s tick, so a just-written file can compute a negative age.
    if max_age is not None and time.time() - path.stat().st_mtime >= max_age.total_seconds():
        return None
    df: pd.DataFrame = pd.read_csv(path)
    return df if not df.empty else None


def _versioned(path: Path, version: int | None) -> Path:
    """`foo.csv` -> `foo.v2.csv`. `None` keeps the bare name, so declaring no
    version does not orphan an existing unversioned cache."""
    return path if version is None else path.with_suffix(f".v{version}{path.suffix}")


def fetch_or_cache(
    path: Path,
    fetcher: Callable[[], pd.DataFrame],
    *,
    tolerate_empty: bool = False,
    version: int | None = None,
    max_age: timedelta | None = None,
) -> pd.DataFrame:
    path = _versioned(path, version)
    cached = _read_cached(path, max_age)
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
