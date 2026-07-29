"""Raw Baseball Savant pulls (expected stats + barrels via pybaseball; park-adjusted
xHR via a direct leaderboard CSV). Returned fully raw -- no rename, no percent->share
conversion, no merge. pybaseball is imported locally (heavy) so the module stays
import-safe.

`fetch_pitcher_pitch_mix` is the one deliberate exception: it counts pitch outcomes
per pitcher instead of returning the raw pitch table. That is a pivot, not a rate --
`skills` still owns every division -- and it is what keeps the cache a ~800-row CSV
rather than the ~700k-row season of pitches behind it. Persisting the raw pitch
table is the streaks pipeline's job (`streaks/data/statcast.py`), not this one's.
"""

from __future__ import annotations

import io
import urllib.request
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

from fantasy_baseball.keepers.cache import fetch_or_cache
from fantasy_baseball.utils.time_utils import local_today

_SAVANT_HR_URL = (
    "https://baseballsavant.mlb.com/leaderboard/home-runs?type=batter&year={year}&min=1&csv=true"
)

# Statcast `description` values. A missed bunt is a swing and a miss, and Savant
# counts it as one; a foul tip is contact, so it is a swing but NOT a whiff.
_WHIFF = frozenset({"swinging_strike", "swinging_strike_blocked", "missed_bunt"})
_CONTACT = frozenset({"foul", "foul_tip", "foul_bunt", "bunt_foul_tip", "hit_into_play"})
_SWING = _WHIFF | _CONTACT
_CALLED_STRIKE = "called_strike"

# Only the two columns the tally needs; statcast() returns 119.
_PITCH_COLUMNS = ["pitcher", "description"]


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


def tally_pitch_outcomes(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse a pitch-level Statcast frame to per-pitcher outcome counts.

    Returns one row per MLBAM `player_id` with `pitches`, `called_strikes`,
    `whiffs` and `swings` (`swings` includes whiffs). Pure, so the outcome
    classification is testable without a network pull.
    """
    if raw.empty:
        return pd.DataFrame()
    pitches = raw.loc[raw["pitcher"].notna(), _PITCH_COLUMNS]
    description = pitches["description"]
    tallies = pd.DataFrame(
        {
            "player_id": pitches["pitcher"].astype(int),
            "pitches": 1,
            "called_strikes": description.eq(_CALLED_STRIKE).astype(int),
            "whiffs": description.isin(_WHIFF).astype(int),
            "swings": description.isin(_SWING).astype(int),
        }
    )
    result: pd.DataFrame = tallies.groupby("player_id", as_index=False).sum()
    return result


def _savant_pitcher_pitch_mix(year: int) -> pd.DataFrame:
    """Per-pitcher pitch-outcome counts for `year`, from the pitch-level feed.

    Baseball Reference publishes the same rates rounded to two decimals, which
    collapses CSW% to ~19 distinct values across a league of pitchers and makes
    it useless as a ranking input; these counts carry full precision.

    Fetched in weekly chunks and tallied per chunk. `statcast()` would accept the
    whole range in one call, but it requests a day at a time internally and holds
    every day-frame before concatenating: a full season is ~513k pitches x 119
    columns, ~595 MB concatenated and over 1 GB at peak, to produce ~1.1k x 5.
    Folding each chunk down first keeps the peak near 40 MB. pybaseball's
    on-disk HTTP cache is enabled so a re-run only pays for genuinely new days.

    The window starts wide enough to cover any season's opening day and is
    capped at today, so an in-progress season does not query future dates.
    """
    # Local, like every other pull here: `streaks.data.statcast` imports
    # pybaseball at module scope, so importing it up top would cost this module
    # its import-safety.
    import pybaseball
    from pybaseball import statcast

    from fantasy_baseball.streaks.data.statcast import chunk_date_range

    pybaseball.cache.enable()
    start = date(year, 3, 1)
    end = min(date(year, 11, 30), local_today())
    tallies = []
    for chunk_start, chunk_end in chunk_date_range(start, end, days=7):
        raw = statcast(start_dt=chunk_start.isoformat(), end_dt=chunk_end.isoformat())
        if raw is None or raw.empty:
            continue
        tallies.append(tally_pitch_outcomes(raw))
    if not tallies:
        return pd.DataFrame()
    combined: pd.DataFrame = (
        pd.concat(tallies, ignore_index=True).groupby("player_id", as_index=False).sum()
    )
    return combined


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


def fetch_pitcher_pitch_mix(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_pitcher_pitch_mix_{year}.csv",
        fetcher or (lambda: _savant_pitcher_pitch_mix(year)),
    )


def fetch_savant_hr(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_hr_{year}.csv",
        fetcher or (lambda: _savant_hr(year)),
        tolerate_empty=True,
    )
