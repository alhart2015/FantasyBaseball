"""Builders for the canonical raw frames the keeper loaders consume.

These were retyped ~27 times across the keeper test files. Adding one stat to the
canonical schema meant editing every copy; now it means editing one builder.

Each takes keyword overrides, so a test states only the field it is about:

    zips_hitters(PA=0, AB=0)          # the zero-denominator case
    mlb_hitting(**{"player.id": [1, None]})   # the drop-row case
"""

from pathlib import Path
from typing import Any

import pandas as pd

_ZIPS_HITTER = {
    "MLBAMID": 11,
    "PA": 600,
    "AB": 540,
    "H": 162,
    "R": 90,
    "HR": 30,
    "RBI": 100,
    "SB": 12,
}
_ZIPS_PITCHER = {"MLBAMID": 22, "IP": 180.0, "ER": 60, "BB": 45, "H": 150, "SO": 200, "W": 15}
_MLB_HITTING = {
    "player.id": 1,
    "stat.plateAppearances": 600,
    "stat.atBats": 540,
    "stat.hits": 162,
    "stat.runs": 90,
    "stat.homeRuns": 30,
    "stat.rbi": 100,
    "stat.stolenBases": 12,
}
_MLB_PITCHING = {
    "player.id": 7,
    "stat.inningsPitched": "180.0",
    "stat.earnedRuns": 60,
    "stat.baseOnBalls": 45,
    "stat.hits": 150,
    "stat.strikeOuts": 200,
    "stat.wins": 15,
}


_SAVANT_EXPECTED = {
    "player_id": 11,
    "pa": 600,
    "ba": 0.270,
    "est_ba": 0.265,
    "woba": 0.340,
    "est_woba": 0.350,
}
_SAVANT_BARRELS = {"player_id": 11, "brl_percent": 12.0, "brl_pa": 4.9}
_BREF_BATTING = {"mlbID": 11, "PA": 600, "R": 90, "H": 162, "HR": 30}
# Counts chosen so the derived rates are round: SwStr% 10.0, CSW% 26.0, Whiff% 20.0.
_SAVANT_PITCH_MIX = {
    "player_id": 22,
    "pitches": 2800,
    "called_strikes": 448,
    "whiffs": 280,
    "swings": 1400,
}
# IP is baseball notation carried as a float: 180.1 is 180 1/3 innings.
_BREF_PITCHING = {
    "mlbID": 22,
    "IP": 180.0,
    "ER": 80,
    "HR": 20,
    "BB": 45,
    "HBP": 5,
    "SO": 200,
    "BF": 740,
    "StL": 0.16,
    "StS": 0.10,
}


def _frame(defaults: dict[str, Any], overrides: dict[str, Any]) -> pd.DataFrame:
    """One-row frame from `defaults`, with `overrides` applied.

    An override may be a scalar or a list; a list makes the frame that long and
    broadcasts every un-overridden default across it.
    """
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise KeyError(f"not in the canonical schema: {sorted(unknown)}")
    merged = {**defaults, **overrides}
    rows = max((len(v) for v in merged.values() if isinstance(v, list)), default=1)
    return pd.DataFrame(merged, index=range(rows))


def zips_hitters(**overrides: Any) -> pd.DataFrame:
    """A raw ZiPS hitter export row (FanGraphs column names)."""
    return _frame(_ZIPS_HITTER, overrides)


def zips_pitchers(**overrides: Any) -> pd.DataFrame:
    """A raw ZiPS pitcher export row (FanGraphs column names)."""
    return _frame(_ZIPS_PITCHER, overrides)


def mlb_hitting(**overrides: Any) -> pd.DataFrame:
    """A raw MLB Stats API hitting split (`player.id` + `stat.*`)."""
    return _frame(_MLB_HITTING, overrides)


def mlb_pitching(**overrides: Any) -> pd.DataFrame:
    """A raw MLB Stats API pitching split. Innings arrive as a STRING in baseball
    notation, which is why the default is `"180.0"` and not `180.0`."""
    return _frame(_MLB_PITCHING, overrides)


def savant_expected(**overrides: Any) -> pd.DataFrame:
    """A raw Savant expected-stats row (`player_id` + est_* columns)."""
    return _frame(_SAVANT_EXPECTED, overrides)


def savant_barrels(**overrides: Any) -> pd.DataFrame:
    """A raw Savant exit-velo/barrels row. Rates are percentages on 0-100."""
    return _frame(_SAVANT_BARRELS, overrides)


def savant_pitch_mix(**overrides: Any) -> pd.DataFrame:
    """A per-pitcher Statcast pitch-outcome count row. `swings` includes whiffs."""
    return _frame(_SAVANT_PITCH_MIX, overrides)


def bref_batting(**overrides: Any) -> pd.DataFrame:
    """A raw Baseball Reference season batting row."""
    return _frame(_BREF_BATTING, overrides)


def bref_pitching(**overrides: Any) -> pd.DataFrame:
    """A raw Baseball Reference season pitching row. `StL`/`StS` are shares of
    total pitches (0-1), and `IP` is baseball notation as a float."""
    return _frame(_BREF_PITCHING, overrides)


def write_zips_vintage(directory: Path, **overrides: Any) -> None:
    """Write a hitters+pitchers ZiPS pair into `directory`, as `load_vintage` expects.

    Every override goes to BOTH builders, so only keys shared by both schemas are
    legal: `MLBAMID` and `H`. Anything else raises from `_frame` -- including
    `PA`, which is valid for hitters alone. Set one side's fields by calling
    `zips_hitters`/`zips_pitchers` directly and writing the CSVs yourself.

    Note `H` means hits for the hitter file and hits allowed for the pitcher
    file, so one override sets two different quantities.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    # Forwarding to both rather than routing by key keeps `_frame`'s unknown-key
    # guard live: a routed typo would match neither schema and apply to nothing.
    zips_hitters(**overrides).to_csv(path / "zips-hitters.csv", index=False)
    zips_pitchers(**overrides).to_csv(path / "zips-pitchers.csv", index=False)
