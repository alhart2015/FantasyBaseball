"""Keepers: raw data pulls plus the #266 keeper-value derivation built on them.

Two layers, deliberately separated:

* **Ingest** -- `cache`, `mlb_stats`, `savant`. Fetchers return the upstream
  response fully raw: no derivation, rename, or join.
* **Derivation** (#266) -- `actuals` and `vintages` normalize both sides to one
  canonical rate/playing-time schema, `calibration` runs the study that measures
  how much of a season's surprise carries forward, `fold` applies it, and
  `coefficients` holds the shipped result.

`POLICIES` is the production entry point: it carries the fitted coefficients
together with the `n0`, gate and ramp width they are conditional on.
"""

from __future__ import annotations

from fantasy_baseball.keepers.cache import fetch_or_cache
from fantasy_baseball.keepers.coefficients import POLICIES, FoldPolicy
from fantasy_baseball.keepers.fold import fold_rates, gate_ramp, shrink
from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season
from fantasy_baseball.keepers.savant import (
    fetch_batter_barrels,
    fetch_batter_expected,
    fetch_pitcher_expected,
    fetch_savant_hr,
)

__all__ = [
    "POLICIES",
    "FoldPolicy",
    "fetch_batter_barrels",
    "fetch_batter_expected",
    "fetch_mlb_season",
    "fetch_or_cache",
    "fetch_pitcher_expected",
    "fetch_savant_hr",
    "fold_rates",
    "gate_ramp",
    "shrink",
]
