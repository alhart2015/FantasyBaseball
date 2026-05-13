"""Unit tests for park factor data + helpers."""

from __future__ import annotations

import pytest

from fantasy_baseball.data.park_factors import (
    PARK_FACTORS,
    get_park_factor,
    park_neutral_value,
)


def test_all_thirty_teams_have_factors():
    """Catch silent abbreviation drift -- if a team is renamed/relocated
    upstream and the abbreviation changes, the rest of the matchup
    pipeline silently degrades to no-park-adjustment for that team."""
    assert len(PARK_FACTORS) == 30


def test_every_team_has_ops_and_k_factor():
    for team, factors in PARK_FACTORS.items():
        assert "ops" in factors, f"{team} missing OPS factor"
        assert "k" in factors, f"{team} missing K factor"


def test_factors_are_in_plausible_range():
    """Park factors should never be wildly out of the historically
    observed band. A typo like 11.3 instead of 1.13 should fail loudly."""
    for team, factors in PARK_FACTORS.items():
        assert 0.80 <= factors["ops"] <= 1.20, f"{team} OPS factor out of range"
        assert 0.85 <= factors["k"] <= 1.15, f"{team} K factor out of range"


def test_coors_is_the_strongest_hitter_park():
    """Sanity check on the data: Coors should always be the highest OPS
    park factor. If it isn't, something has been miskeyed."""
    max_team = max(PARK_FACTORS, key=lambda t: PARK_FACTORS[t]["ops"])
    assert max_team == "COL"


def test_petco_is_the_strongest_pitcher_park():
    """Sanity check on the data: Petco should always be the lowest OPS
    park factor (most pitcher-friendly)."""
    min_team = min(PARK_FACTORS, key=lambda t: PARK_FACTORS[t]["ops"])
    assert min_team == "SDP"


def test_get_park_factor_known_team():
    assert get_park_factor("COL", "ops") == PARK_FACTORS["COL"]["ops"]
    assert get_park_factor("SDP", "k") == PARK_FACTORS["SDP"]["k"]


def test_get_park_factor_unknown_team_returns_neutral():
    """Unknown teams must default to 1.0 so a missing entry never
    silently zeroes out the multiplier."""
    assert get_park_factor("XXX", "ops") == 1.0
    assert get_park_factor("XXX", "k") == 1.0


def test_get_park_factor_unknown_stat_returns_neutral():
    assert get_park_factor("COL", "bbpct") == 1.0


def test_park_neutral_recovers_neutral_team():
    """A team that plays in a 1.00 park should have park-neutral OPS
    equal to its season OPS."""
    assert park_neutral_value(0.740, 1.00) == pytest.approx(0.740)


def test_park_neutral_strips_coors_inflation():
    """Rockies season OPS includes 81 games at Coors. The park-neutral
    estimate should be lower than the season number."""
    season_ops = 0.740
    home_pf = 1.13
    neutral = park_neutral_value(season_ops, home_pf)
    assert neutral < season_ops
    # season ~= neutral * (1.13 + 1) / 2 = neutral * 1.065
    # so neutral ~= season / 1.065 ~= 0.695
    assert neutral == pytest.approx(0.740 * 2 / 2.13, rel=1e-6)


def test_park_neutral_lifts_petco_suppression():
    """Padres season OPS is dragged down by Petco. Park-neutral should
    be higher than the season number."""
    season_ops = 0.700
    home_pf = 0.92
    neutral = park_neutral_value(season_ops, home_pf)
    assert neutral > season_ops


def test_park_neutral_round_trip_at_extreme_parks():
    """Park-neutralizing then applying the SAME park factor should
    recover the original season value (within float precision). This
    is the self-consistency check that proves the formula is invertible."""
    season = 0.730
    for home_pf in [0.92, 1.00, 1.13]:
        neutral = park_neutral_value(season, home_pf)
        # Round-trip: a team that plays half home (home_pf) and half
        # at a neutral 1.0 road park produces the original season value.
        recovered = neutral * (home_pf + 1.0) / 2.0
        assert recovered == pytest.approx(season, rel=1e-9)


def test_park_neutral_handles_degenerate_factor():
    """A zero or negative park factor would blow up the math; the helper
    should fall back to the input rather than divide by zero."""
    assert park_neutral_value(0.700, 0.0) == 0.700
    assert park_neutral_value(0.700, -1.0) == 0.700
