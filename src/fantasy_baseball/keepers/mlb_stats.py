"""Raw MLB Stats API season-stats leaderboard pull, keyed by MLBAM.

Returns the fully raw response: every split is json_normalized once across all
pages -- no column dropped, no rate derived. #266 selects/derives what it needs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from fantasy_baseball.keepers.cache import fetch_or_cache

_MLB_STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
_MLB_PAGE = 1000


def _fetch_mlb_season(
    group: str, year: int, *, get: Callable[..., Any] | None = None
) -> pd.DataFrame:
    """Paginated season leaderboard for `group` ('hitting'|'pitching'). Accumulates
    the raw splits across every page, then json_normalizes ONCE (consistent columns).
    `get` defaults to `requests.get` (local import keeps the module network-free at
    import time); tests inject a fake."""
    if get is None:
        import requests

        get = requests.get
    splits: list[dict[str, Any]] = []
    offset = 0
    while True:
        params: dict[str, str | int] = {
            "stats": "season",
            "group": group,
            "season": year,
            "sportId": 1,
            "playerPool": "all",
            "limit": _MLB_PAGE,
            "offset": offset,
        }
        resp = get(_MLB_STATS_URL, params=params, timeout=60)
        resp.raise_for_status()
        stats = resp.json().get("stats", [])
        page_splits = stats[0]["splits"] if stats else []
        if not page_splits:
            break
        splits.extend(page_splits)
        if len(page_splits) < _MLB_PAGE:
            break
        offset += _MLB_PAGE
    result: pd.DataFrame = pd.json_normalize(splits)
    return result


def fetch_mlb_season(
    cache_dir: Path,
    year: int,
    group: str,
    *,
    fetcher: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"mlb_{group}_{year}.csv",
        fetcher or (lambda: _fetch_mlb_season(group, year)),
    )
