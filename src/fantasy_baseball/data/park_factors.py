"""Park factor lookups for OPS and K% adjustments.

Park factors are venue effects: a multiplier on a stat reflecting how
much a ballpark inflates or deflates it relative to a neutral park.
1.00 means neutral, >1.00 inflates, <1.00 suppresses.

Values are multi-year averages (FanGraphs Guts!, 2022-2024). They shift
slowly year-to-year so this hardcoded snapshot is good enough for the
qualitative color signal on the lineup page. Team abbreviations match
the FanGraphs-style codes used elsewhere in the project (CHW, KCR,
SDP, SFG, TBR, WSN, ATH).
"""

from __future__ import annotations

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


def park_neutral_value(season_value: float, home_park_factor: float) -> float:
    """Estimate a team's park-neutral version of a season stat.

    Assumes a roughly 50/50 home/away schedule and that the team's
    visited away parks average to a neutral 1.00 park factor (true in
    expectation since each team plays a wide mix of road parks). Then

        season_value ~= neutral_value * (home_pf + 1) / 2

    so

        neutral_value = season_value * 2 / (home_pf + 1)

    Returns ``season_value`` unchanged if the park factor is degenerate
    (<=0), since dividing by something nonpositive would be nonsense.
    """
    if home_park_factor <= 0:
        return season_value
    return season_value * 2.0 / (home_park_factor + 1.0)
