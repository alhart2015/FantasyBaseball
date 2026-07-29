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
import logging
import urllib.request
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from fantasy_baseball.keepers.cache import fetch_or_cache
from fantasy_baseball.utils.time_utils import local_today

logger = logging.getLogger(__name__)

_SAVANT_HR_URL = (
    "https://baseballsavant.mlb.com/leaderboard/home-runs?type=batter&year={year}&min=1&csv=true"
)

# Statcast `description` values. A missed bunt is a swing and a miss; a foul tip
# is contact, so it is a swing but NOT a whiff -- counting it as one would put
# SwStr% about 1.3 points above Baseball Reference's independent StS.
# Statcast split balls in play across three descriptions before 2020 and
# consolidated to `hit_into_play` after; all three are listed because this
# function is year-parameterized and dropping the old two would silently gut the
# swing denominator for an earlier season.
_WHIFF = frozenset({"swinging_strike", "swinging_strike_blocked", "missed_bunt"})
_CONTACT = frozenset(
    {
        "foul",
        "foul_tip",
        "foul_bunt",
        "bunt_foul_tip",
        "hit_into_play",
        "hit_into_play_score",
        "hit_into_play_no_out",
    }
)
_SWING = _WHIFF | _CONTACT

# Every pull here is season-to-date, so all of them go stale daily.
_MAX_AGE = timedelta(days=1)


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


def _tally_pitch_outcomes(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse a pitch-level Statcast frame to per-pitcher outcome counts.

    Returns one row per MLBAM `player_id` with `pitches`, `called_strikes`,
    `whiffs` and `swings` (`swings` includes whiffs). Pure, so the outcome
    classification is testable without a network pull.

    Regular season only. `statcast()` also serves spring ("S") and postseason
    ("P") and its date window cannot exclude them; spring whiff rates run high
    against non-roster hitters, and dropping them puts these rates on the same
    sample as the BBRef stats they sit beside, so comparing a pitcher's K% to
    his SwStr% stays honest.
    """
    if raw.empty:
        return pd.DataFrame()
    keep = raw["pitcher"].notna() & raw["game_type"].eq("R")
    pitches = raw.loc[keep, ["pitcher", "description"]]
    if pitches.empty:
        return pd.DataFrame()
    description = pitches["description"]
    tallies = pd.DataFrame(
        {
            "player_id": pitches["pitcher"].astype(int),
            "pitches": 1,
            "called_strikes": description.eq("called_strike").astype(int),
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

    Fetched in chunks and folded per chunk. `statcast()` would take the whole
    range in one call, but it requests a day at a time and holds every day-frame
    before concatenating -- a season is ~475k pitches x 119 columns, hundreds of
    MB concatenated, to produce ~760 x 5. Chunking caps that. Seven days is
    measured-optimal: larger chunks buy no wall time (Savant's per-day latency
    dominates) and cost 2-3x peak memory.

    pybaseball's on-disk cache is deliberately NOT enabled: `cache.enable()`
    writes a machine-global config that every later pybaseball process reads at
    import -- including `streaks/data/statcast.py`, which never opted in -- and
    caches Statcast days for 365 days, so a day fetched mid-game would be served
    partial for a year.

    The window is deliberately wider than the season and capped at today. It
    costs ~9% wasted day-requests, and narrowing it to `league.yaml`'s
    `season_start` was tried and reverted: that is the FANTASY season start, and
    MLB played regular-season games three days before it in 2026 -- using it
    dropped 3,568 real pitches. The `game_type` filter is what defines the
    sample; the date window only needs to contain it.

    Empty chunks are counted and logged -- Savant answering with an empty body
    is indistinguishable from a genuine off day, and a truncated tally would
    still be cached and look authoritative.
    """
    # Local, like every other pull here: `streaks.data.statcast` imports
    # pybaseball at module scope, so importing it up top would cost this module
    # its import-safety.
    from pybaseball import statcast

    from fantasy_baseball.streaks.data.statcast import chunk_date_range

    window_start = date(year, 3, 1)
    window_end = min(date(year, 11, 30), local_today())
    tallies = []
    empty_chunks = 0
    for chunk_start, chunk_end in chunk_date_range(window_start, window_end, days=7):
        raw = statcast(start_dt=chunk_start.isoformat(), end_dt=chunk_end.isoformat())
        tally = _tally_pitch_outcomes(raw) if raw is not None else pd.DataFrame()
        # An all-spring or all-offseason chunk tallies to a 0x0 frame; concat-ing
        # those would lose the `player_id` column and blow up the groupby.
        if tally.empty:
            empty_chunks += 1
            continue
        tallies.append(tally)
    if empty_chunks:
        logger.info(
            "pitch mix %d: %d of %d weekly chunks had no regular-season pitches",
            year,
            empty_chunks,
            empty_chunks + len(tallies),
        )
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
    cache_dir: Path,
    year: int,
    *,
    max_age: timedelta | None = _MAX_AGE,
    fetcher: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_batter_expected_{year}.csv",
        fetcher or (lambda: _savant_batter_expected(year)),
        max_age=max_age,
    )


def fetch_batter_barrels(
    cache_dir: Path,
    year: int,
    *,
    max_age: timedelta | None = _MAX_AGE,
    fetcher: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_batter_barrels_{year}.csv",
        fetcher or (lambda: _savant_batter_barrels(year)),
        max_age=max_age,
    )


def fetch_pitcher_expected(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_pitcher_expected_{year}.csv",
        fetcher or (lambda: _savant_pitcher_expected(year)),
    )


# 2: spring training and postseason excluded. A v1 cache mixes them in.
_PITCH_MIX_VERSION = 2


def fetch_pitcher_pitch_mix(
    cache_dir: Path,
    year: int,
    *,
    max_age: timedelta | None = _MAX_AGE,
    fetcher: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_pitcher_pitch_mix_{year}.csv",
        fetcher or (lambda: _savant_pitcher_pitch_mix(year)),
        version=_PITCH_MIX_VERSION,
        max_age=max_age,
    )


def fetch_savant_hr(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_hr_{year}.csv",
        fetcher or (lambda: _savant_hr(year)),
        tolerate_empty=True,
    )
