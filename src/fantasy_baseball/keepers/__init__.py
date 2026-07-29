"""Keepers: raw data pulls for keeper analysis.

**Ingest.** Fetchers return the upstream response fully raw: no derivation,
rename, or join.

* `cache` -- fetch-on-miss CSV plumbing shared by the pulls below.
* `mlb_stats` -- MLB Stats API season leaderboards, keyed by MLBAM.
* `savant` -- Baseball Savant expected stats (xBA/xSLG/xwOBA), exit-velocity and
  barrel rates, and the park-adjusted xHR leaderboard.

**Normalization.** Two pure, I/O-free helpers that reshape a raw frame to one
canonical rate/playing-time schema: `actuals` for an MLB season pull, `vintages`
for a ZiPS export.

The #266 fold -- `fold`, `coefficients`, `calibration` -- was removed along with
the `analysis/keeper_value.py` metric it was built to replace. The written
findings and the study's CSV artifacts are retained under `docs/superpowers/` and
`data/analysis/`; the code is recoverable from git history.
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
