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
wRC+; ERA-, FIP, K%, whiff rate, CSW%) that keeper comparisons rank on.

The #266 fold -- `fold`, `coefficients`, `calibration` -- was removed along with
the `analysis/keeper_value.py` metric it was built to replace. The written
findings and the study's CSV artifacts are retained under `docs/superpowers/` and
`data/analysis/`; the code is recoverable from git history.
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
