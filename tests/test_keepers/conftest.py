"""Builders for the four canonical raw frames the keeper loaders consume.

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
    return pd.DataFrame(
        {key: (val if isinstance(val, list) else [val] * rows) for key, val in merged.items()}
    )


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


def write_zips_vintage(directory: Path, **overrides: Any) -> None:
    """Write a hitters+pitchers ZiPS pair into `directory`, as `load_vintage` expects."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    zips_hitters(**{k: v for k, v in overrides.items() if k in _ZIPS_HITTER}).to_csv(
        path / "zips-hitters.csv", index=False
    )
    zips_pitchers(**{k: v for k, v in overrides.items() if k in _ZIPS_PITCHER}).to_csv(
        path / "zips-pitchers.csv", index=False
    )
