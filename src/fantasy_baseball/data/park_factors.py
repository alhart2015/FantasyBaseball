"""Park factor lookups for OPS and K% adjustments.

Park factors are venue effects: a multiplier on a stat reflecting how
much a ballpark inflates or deflates it relative to a neutral park.
1.00 means neutral, >1.00 inflates, <1.00 suppresses.

Values are multi-year averages (FanGraphs Guts!, 2022-2024). They shift
slowly year-to-year so this hardcoded snapshot is good enough for the
qualitative color signal on the lineup page. Team abbreviations match
the FanGraphs-style codes used elsewhere in the project (CHW, KCR,
SDP, SFG, TBR, WSN, ATH).

Two caveats for quantitative consumers (`keepers/skills.py` ranks on these):
the values are a 2022-24 average rather than the current season's, and only
`ops` and `k` exist -- there is no runs factor, so run-prevention stats like
ERA- use `ops` as a proxy. Both bias toward under-correction, which is the
safe direction for a ranking; neither is good enough to report as a precise
park-neutral figure.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

PARK_FACTORS: dict[str, dict[str, float]] = {
    "COL": {"ops": 1.13, "k": 0.95},
    "CIN": {"ops": 1.05, "k": 0.98},
    "BOS": {"ops": 1.04, "k": 0.98},
    "PHI": {"ops": 1.03, "k": 1.01},
    "TEX": {"ops": 1.03, "k": 0.99},
    "CHC": {"ops": 1.02, "k": 1.00},
    "KCR": {"ops": 1.02, "k": 1.00},
    "BAL": {"ops": 1.02, "k": 1.01},
    "ATL": {"ops": 1.01, "k": 1.00},
    "HOU": {"ops": 1.01, "k": 1.01},
    "TOR": {"ops": 1.01, "k": 0.99},
    "ARI": {"ops": 1.00, "k": 1.01},
    "LAA": {"ops": 1.00, "k": 1.00},
    "NYM": {"ops": 1.00, "k": 1.01},
    "WSN": {"ops": 1.00, "k": 1.00},
    "CHW": {"ops": 0.99, "k": 1.00},
    "MIN": {"ops": 0.99, "k": 1.00},
    "STL": {"ops": 0.99, "k": 1.01},
    "MIL": {"ops": 0.98, "k": 1.00},
    "TBR": {"ops": 0.98, "k": 1.01},
    "CLE": {"ops": 0.97, "k": 1.01},
    "DET": {"ops": 0.97, "k": 1.02},
    "LAD": {"ops": 0.97, "k": 1.01},
    "NYY": {"ops": 0.97, "k": 1.02},
    "ATH": {"ops": 0.96, "k": 1.02},
    "PIT": {"ops": 0.96, "k": 1.01},
    "MIA": {"ops": 0.94, "k": 1.02},
    "SEA": {"ops": 0.94, "k": 1.03},
    "SFG": {"ops": 0.94, "k": 1.02},
    "SDP": {"ops": 0.92, "k": 1.03},
}

NEUTRAL_FACTOR: dict[str, float] = {"ops": 1.00, "k": 1.00}


def get_park_factor(team_abbrev: str, stat: str) -> float:
    """Return the park factor for a team's home stadium.

    Falls back to 1.0 for unknown teams or unknown stat keys so callers
    never have to special-case a missing park.
    """
    return PARK_FACTORS.get(team_abbrev, NEUTRAL_FACTOR).get(stat, 1.0)


def _neutralize(value: Any, home_park_factor: Any) -> Any:
    """The 50/50 model itself, over scalars or Series. One expression, so the
    scalar and vectorized entry points cannot drift."""
    return value * 2.0 / (home_park_factor + 1.0)


def park_neutral_value(season_value: float, home_park_factor: float) -> float:
    """Estimate a team's park-neutral version of a season stat.

    Assumes a roughly 50/50 home/away schedule and that the team's
    visited away parks average to a neutral 1.00 park factor (true in
    expectation since each team plays a wide mix of road parks). Then

        season_value ~= neutral_value * (home_pf + 1) / 2

    so

        neutral_value = season_value * 2 / (home_pf + 1)

    Returns ``season_value`` unchanged when the model does not apply: a
    degenerate (<=0) park factor, or a non-positive value, where the multiplier
    inverts and a hitters' park would *help* the line instead of discounting it.
    """
    if home_park_factor <= 0 or season_value <= 0:
        return season_value
    return float(_neutralize(season_value, home_park_factor))


def park_neutral_series(season_values: pd.Series, home_park_factors: pd.Series) -> pd.Series:
    """Vectorized :func:`park_neutral_value`, with the same two escape hatches.

    Both live here rather than in a caller so the 50/50 model has one home; a
    caller that added its own non-positive guard would be a second definition.
    """
    factors = pd.to_numeric(home_park_factors, errors="coerce")
    usable = factors.where(factors > 0, 1.0)
    return _neutralize(season_values, usable).where(season_values > 0, season_values)
