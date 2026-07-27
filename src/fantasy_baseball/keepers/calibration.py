"""Assemble ZiPS-vs-actual year pairs and measure the fit sample.

The one non-negotiable methodological constraint (spec 6.1): the base must be
ZiPS_Y, built knowing only through Y-1, so it has NOT already absorbed year Y --
mirroring production, where ZiPS 2027 has never seen 2026. Using ZiPS_{Y+1} as
the base would fit how much surprise ZiPS already absorbed and drive the
coefficient to zero by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fantasy_baseball.keepers.actuals import normalize_hitting, normalize_pitching
from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season
from fantasy_baseball.keepers.vintages import load_vintage

# Year Y of each usable (Y, Y+1) pair. 2025 needs a complete 2026 season; 2021 has
# no ZiPS vintage on disk (data/projections starts at 2022).
PAIR_YEARS = (2022, 2023, 2024)

PT_COL = {"hitter": "pa", "pitcher": "ip"}


@dataclass(frozen=True)
class YearPair:
    """One (Y, Y+1) observation set, already aligned on mlbam_id."""

    year: int
    base: pd.DataFrame  # ZiPS_Y rates
    residual: pd.DataFrame  # actual_Y rates - ZiPS_Y rates
    target: pd.DataFrame  # actual_{Y+1} rates
    realized_pt: pd.Series  # actual_Y playing time (drives shrink and gate)
    target_pt: pd.Series  # actual_{Y+1} playing time


LAST_COMPLETE_SEASON = 2025


def build_pairs(
    player_type: str,
    cache_dir: Path,
    projections_root: Path,
    years: tuple[int, ...] = PAIR_YEARS,
) -> list[YearPair]:
    """Assemble (Y, Y+1) observation sets.

    Two properties are load-bearing and were both wrong in an earlier draft:

    * The frames carry the PLAYING TIME column alongside the rates. PT is the
      twelfth coefficient, and spec requirement 12 -- the systematic mean of the
      PT residual -- is the single hardest constraint on the estimator. Stripping
      PT here would make that requirement unaddressable.
    * Membership is `zips INTERSECT actual_Y` ONLY. Intersecting year Y+1 as well would
      precondition the sample on having survived, inflating the measured survival
      rate by 7-9 points AND removing non-survivors before any estimator sees
      them -- making spec requirement 5 unmeasurable. Absentees get a NaN target
      and 0.0 target playing time, which is the honest encoding of "did not play".
    """
    if player_type not in {"hitter", "pitcher"}:
        raise ValueError(f"player_type must be 'hitter' or 'pitcher', got {player_type!r}")
    group = "hitting" if player_type == "hitter" else "pitching"
    normalize = normalize_hitting if player_type == "hitter" else normalize_pitching
    pairs: list[YearPair] = []
    for year in years:
        if year + 1 > LAST_COMPLETE_SEASON:
            # fetch_or_cache never invalidates, so a mid-season pull would freeze
            # permanently. Fail loud rather than cache an in-progress season.
            raise ValueError(
                f"pair {year}->{year + 1} needs a complete {year + 1} season; "
                f"last complete is {LAST_COMPLETE_SEASON}"
            )
        zips_h, zips_p = load_vintage(year, projections_root)
        zips = zips_h if player_type == "hitter" else zips_p
        act_y = normalize(fetch_mlb_season(cache_dir, year, group))
        act_next = normalize(fetch_mlb_season(cache_dir, year + 1, group))
        ids = zips.index.intersection(act_y.index)
        cols = list(zips.columns)  # rates AND the playing-time column
        pairs.append(
            YearPair(
                year=year,
                base=zips.loc[ids, cols],
                residual=act_y.loc[ids, cols] - zips.loc[ids, cols],
                target=act_next.reindex(ids)[cols],
                realized_pt=act_y.loc[ids, PT_COL[player_type]],
                target_pt=act_next.reindex(ids)[PT_COL[player_type]].fillna(0.0),
            )
        )
    return pairs


def survivorship(pairs: list[YearPair], threshold: float) -> pd.DataFrame:
    """Per pair: how many cleared `threshold` in year Y, and how many again in Y+1.

    Fitting on survivors alone measures persistence GIVEN continued play, which
    biases the playing-time coefficient upward. Spec 6.3 requires this measured on
    the actual fit sample, not on the wider MLB population.
    """
    rows = []
    for pair in pairs:
        in_year = pair.realized_pt >= threshold
        survived = in_year & (pair.target_pt >= threshold)
        n_in, n_sur = int(in_year.sum()), int(survived.sum())
        rows.append(
            {
                "year": pair.year,
                "n_matched": len(pair.base),
                "n_in_year": n_in,
                "n_survived": n_sur,
                "survival_rate": (n_sur / n_in) if n_in else float("nan"),
            }
        )
    return pd.DataFrame(rows)
