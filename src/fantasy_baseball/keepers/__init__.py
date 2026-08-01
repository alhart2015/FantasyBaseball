"""Keepers: raw data pulls for keeper analysis.

**Ingest.** Fetchers return the upstream response fully raw: no derivation,
rename, or join.

* `cache` -- fetch-on-miss CSV plumbing shared by the pulls below.
* `mlb_stats` -- MLB Stats API season leaderboards, keyed by MLBAM.
* `savant` -- Baseball Savant expected stats (xBA/xSLG/xwOBA), exit-velocity and
  barrel rates, per-pitcher pitch-outcome counts, and the park-adjusted xHR
  leaderboard.
* `bref` -- Baseball Reference season batting/pitching, the live source for the
  counting stats behind ERA-, FIP and K%. FanGraphs is 403 across the board.

**Normalization.** Pure, I/O-free helpers that reshape a raw frame to a canonical
schema: `actuals` for an MLB season pull, `vintages` for a ZiPS export, and
`skills` for the season-to-date true-talent rates (barrel rate, xwOBA, xBA,
wRC+; ERA-, FIP, K%, whiff rate, CSW%). `scripts/fetch_keeper_skills.py` is the
entry point that runs the pulls above through these and writes the skills cache.

**Lookup.** `positions` -- eligible slots per player. The one module here that
touches the network, since it prefers the live blob over the committed cache.

**There is deliberately no scoring layer in this package.** Ingest and
normalization only: nothing here decides what a keeper is worth.

Three successive attempts at that model (`analysis/keeper_value.py`, the #266
fold, and the `composite`/`projection`/`scarcity` family model) were each torn
out after drifting from the decision they were meant to inform -- the last one
shipped a family set that, by its own backtest, scored BELOW the residual it
replaced. All three are in git history; the findings and backtests are in
`docs/superpowers/`. Before adding scoring back, read
`docs/keeper-value-teardown-2026-08-01.md` for what went wrong and which
constraints any replacement has to satisfy.
"""

from __future__ import annotations

from fantasy_baseball.keepers.bref import fetch_bref_batting, fetch_bref_pitching
from fantasy_baseball.keepers.cache import fetch_or_cache
from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season
from fantasy_baseball.keepers.savant import (
    fetch_batter_barrels,
    fetch_batter_expected,
    fetch_pitcher_expected,
    fetch_pitcher_pitch_mix,
    fetch_savant_hr,
)
from fantasy_baseball.keepers.skills import (
    HITTER_SKILLS,
    PITCHER_SKILLS,
    normalize_hitter_skills,
    normalize_pitcher_skills,
)

__all__ = [
    "HITTER_SKILLS",
    "PITCHER_SKILLS",
    "fetch_batter_barrels",
    "fetch_batter_expected",
    "fetch_bref_batting",
    "fetch_bref_pitching",
    "fetch_mlb_season",
    "fetch_or_cache",
    "fetch_pitcher_expected",
    "fetch_pitcher_pitch_mix",
    "fetch_savant_hr",
    "normalize_hitter_skills",
    "normalize_pitcher_skills",
]
