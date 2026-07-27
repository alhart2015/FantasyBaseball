"""Keepers module: raw MLB Stats API + Baseball Savant data pulls.

Fetchers return the upstream response fully raw -- no derivation, rename, or join.
Downstream keeper-value logic (#266) is built on top of these; nothing here computes.
"""

from __future__ import annotations

from fantasy_baseball.keepers.cache import fetch_or_cache
from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season
from fantasy_baseball.keepers.savant import (
    fetch_batter_barrels,
    fetch_batter_expected,
    fetch_pitcher_expected,
    fetch_savant_hr,
)

__all__ = [
    "fetch_batter_barrels",
    "fetch_batter_expected",
    "fetch_mlb_season",
    "fetch_or_cache",
    "fetch_pitcher_expected",
    "fetch_savant_hr",
]
