"""Raw Baseball Reference season pulls via pybaseball. Returned fully raw -- no
rename, no unit conversion, no merge. pybaseball is imported locally (heavy) so
the module stays import-safe, matching `savant`.

This is the live source for the counting stats behind ERA-, FIP and K%. It also
publishes called- and swinging-strike rates as `StL`/`StS`, but rounded to two
decimals, which collapses CSW% to ~19 distinct values across a league of
pitchers; `skills` ignores those columns and takes the pitch-denominated rates
from `savant.fetch_pitcher_pitch_mix` instead.

FanGraphs would be the natural source for all of these and is NOT usable: as of
2026-07 the legacy leaderboard `pybaseball.batting_stats`/`pitching_stats`
scrape, the `/api/leaders/major-league/data` JSON endpoint, the `/leaders`
`__NEXT_DATA__` blob, and the Guts! constants page all return 403. Only
`/projections` still responds, which is why `data/fangraphs_fetch.py` works and
this module cannot use the same trick.

Note both frames report `IP` in baseball notation (`20.1` == 20 1/3), as a float
rather than a string -- see `actuals.innings_to_float`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from fantasy_baseball.keepers.cache import fetch_or_cache


def _bref_batting(year: int) -> pd.DataFrame:
    from pybaseball import batting_stats_bref

    result: pd.DataFrame = batting_stats_bref(year)
    return result


def _bref_pitching(year: int) -> pd.DataFrame:
    from pybaseball import pitching_stats_bref

    result: pd.DataFrame = pitching_stats_bref(year)
    return result


def fetch_bref_batting(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"bref_batting_{year}.csv",
        fetcher or (lambda: _bref_batting(year)),
    )


def fetch_bref_pitching(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"bref_pitching_{year}.csv",
        fetcher or (lambda: _bref_pitching(year)),
    )
