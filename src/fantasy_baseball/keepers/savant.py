"""Raw Baseball Savant pulls (expected stats + barrels via pybaseball; park-adjusted
xHR via a direct leaderboard CSV). Returned fully raw -- no rename, no percent->share
conversion, no merge. pybaseball is imported locally (heavy) so the module stays
import-safe.
"""

from __future__ import annotations

import io
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from fantasy_baseball.keepers.cache import fetch_or_cache

_SAVANT_HR_URL = (
    "https://baseballsavant.mlb.com/leaderboard/home-runs?type=batter&year={year}&min=1&csv=true"
)


def _savant_batter_expected(year: int) -> pd.DataFrame:
    from pybaseball import statcast_batter_expected_stats

    result: pd.DataFrame = statcast_batter_expected_stats(year, minPA=1)
    return result


def _savant_batter_barrels(year: int) -> pd.DataFrame:
    from pybaseball import statcast_batter_exitvelo_barrels

    result: pd.DataFrame = statcast_batter_exitvelo_barrels(year, minBBE=1)
    return result


def _savant_pitcher_expected(year: int) -> pd.DataFrame:
    from pybaseball import statcast_pitcher_expected_stats

    result: pd.DataFrame = statcast_pitcher_expected_stats(year, minPA=1)
    return result


def _savant_hr(year: int) -> pd.DataFrame:
    """Park-adjusted xHR leaderboard CSV (no pybaseball wrapper). Browser UA +
    utf-8-sig BOM. Pre-2016 returns a header-only body (empty frame)."""
    req = urllib.request.Request(
        _SAVANT_HR_URL.format(year=year),
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8-sig", "replace")
    result: pd.DataFrame = pd.read_csv(io.StringIO(body))
    return result


def fetch_batter_expected(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_batter_expected_{year}.csv",
        fetcher or (lambda: _savant_batter_expected(year)),
    )


def fetch_batter_barrels(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_batter_barrels_{year}.csv",
        fetcher or (lambda: _savant_batter_barrels(year)),
    )


def fetch_pitcher_expected(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_pitcher_expected_{year}.csv",
        fetcher or (lambda: _savant_pitcher_expected(year)),
    )


def fetch_savant_hr(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_hr_{year}.csv",
        fetcher or (lambda: _savant_hr(year)),
        tolerate_empty=True,
    )
