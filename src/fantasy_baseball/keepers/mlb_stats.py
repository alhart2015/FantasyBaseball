"""Raw MLB Stats API season-stats leaderboard pull, keyed by MLBAM.

Returns the fully raw response: every split is json_normalized once across all
pages -- no column dropped, no rate derived. #266 selects/derives what it needs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from fantasy_baseball.keepers.cache import fetch_or_cache

_MLB_STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
_MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
_MLB_PAGE = 1000
# The `people` endpoint accepts at least 1000 ids in one `personIds` query, but the
# request is a GET and the id list rides in the URL, so batch well under whatever the
# server's line limit turns out to be. 500 ids is ~3.5KB of query string.
_PEOPLE_BATCH = 500


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


def _fetch_mlb_people(ids: Sequence[int], *, get: Callable[..., Any] | None = None) -> pd.DataFrame:
    """Batched `people` lookup: birth date, primary position, MLB debut date.

    The season leaderboards carry no birth date, and a player-season the API never
    returned (the injured/demoted seasons #291 represents explicitly) has no `age` to
    read. Deriving age needs the birth date, so it comes from here.

    Like `_fetch_mlb_season`, `json_normalize` runs ONCE over the accumulated batches
    so a field absent from one batch's players cannot shift the column set.
    """
    if get is None:
        import requests

        get = requests.get
    people: list[dict[str, Any]] = []
    unique = sorted({int(i) for i in ids})
    for start in range(0, len(unique), _PEOPLE_BATCH):
        batch = unique[start : start + _PEOPLE_BATCH]
        resp = get(
            _MLB_PEOPLE_URL,
            params={"personIds": ",".join(str(i) for i in batch)},
            timeout=60,
        )
        resp.raise_for_status()
        people.extend(resp.json().get("people", []))
    result: pd.DataFrame = pd.json_normalize(people)
    return result


def fetch_mlb_people(
    cache_dir: Path,
    ids: Sequence[int],
    tag: str,
    *,
    fetcher: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Cached `people` pull. `tag` names the id set (e.g. a year range), because the
    cache key cannot encode the id list itself -- a different `tag` for a different
    set of ids is the caller's responsibility."""
    return fetch_or_cache(
        cache_dir / f"mlb_people_{tag}.csv",
        fetcher or (lambda: _fetch_mlb_people(ids)),
    )
