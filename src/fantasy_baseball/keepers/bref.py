"""Raw Baseball Reference season pulls via pybaseball. Returned fully raw -- no
rename, no unit conversion, no merge. pybaseball is imported locally (heavy) so
the module stays import-safe, matching `savant`.

The one transform applied is repairing `Name`, ~7% of which arrives with its
UTF-8 bytes spelled out as literal backslash escapes. That is decoding a
transport defect, not a derivation -- the same class of thing as `savant`
stripping a `utf-8-sig` BOM -- and it has to happen here rather than in a
caller: `fetch_or_cache` persists whatever the fetcher returns, so repairing
downstream would leave every cached CSV corrupt and every other consumer of
these public fetchers silently failing name joins.

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
from fantasy_baseball.utils.name_utils import repair_double_encoded


def _repair_names(frame: pd.DataFrame) -> pd.DataFrame:
    if "Name" in frame.columns:
        frame = frame.assign(Name=[repair_double_encoded(n) for n in frame["Name"]])
    return frame


def _bref_batting(year: int) -> pd.DataFrame:
    from pybaseball import batting_stats_bref

    return _repair_names(batting_stats_bref(year))


def _bref_pitching(year: int) -> pd.DataFrame:
    from pybaseball import pitching_stats_bref

    return _repair_names(pitching_stats_bref(year))


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
