import math

import pandas as pd
import pytest

from fantasy_baseball.keepers.skills import (
    HITTER_SKILLS,
    PITCHER_SKILLS,
    normalize_hitter_skills,
    normalize_pitcher_skills,
)
from tests.test_keepers.conftest import (
    bref_batting,
    bref_pitching,
    savant_barrels,
    savant_expected,
    savant_pitch_mix,
)


def _hitters(*, expected=None, barrels=None, batting=None, park_factor=None) -> pd.DataFrame:
    return normalize_hitter_skills(
        expected if expected is not None else savant_expected(),
        barrels if barrels is not None else savant_barrels(),
        batting if batting is not None else bref_batting(),
        park_factor=park_factor,
    )


def _pitchers(*, pitching=None, pitch_mix=None, park_factor=None) -> pd.DataFrame:
    return normalize_pitcher_skills(
        pitching if pitching is not None else bref_pitching(),
        pitch_mix if pitch_mix is not None else savant_pitch_mix(),
        park_factor=park_factor,
    )


# --- hitters ---------------------------------------------------------------


def test_indexed_by_mlbam_with_expected_columns():
    out = _hitters()
    assert out.index.name == "mlbam_id"
    assert out.index.tolist() == [11]
    assert list(out.columns) == ["pa", *HITTER_SKILLS]


def test_expected_stats_pass_through_unscaled():
    out = _hitters()
    assert out["xwoba"].iloc[0] == pytest.approx(0.350)
    assert out["xba"].iloc[0] == pytest.approx(0.265)
    assert out["barrel_pct"].iloc[0] == pytest.approx(12.0)
    assert out["barrel_pa_pct"].iloc[0] == pytest.approx(4.9)


def test_lone_hitter_is_league_average():
    """One hitter IS the league, so his wOBA is the league wOBA and wRC+ is 100.
    Pins the centring of the index independently of WOBA_SCALE."""
    assert _hitters()["wrc_plus"].iloc[0] == pytest.approx(100.0)


def test_wrc_plus_orders_by_woba():
    out = _hitters(
        expected=savant_expected(player_id=[1, 2], woba=[0.400, 0.280]),
        barrels=savant_barrels(player_id=[1, 2]),
        batting=bref_batting(mlbID=[1, 2]),
    )
    assert out.loc[1, "wrc_plus"] > 100.0 > out.loc[2, "wrc_plus"]


def test_hitter_park_factor_deflates_a_hitters_park():
    """Coors-like inflation must lower wRC+ relative to neutral, not raise it."""
    expected = savant_expected(player_id=[1, 2], woba=[0.400, 0.280])
    kwargs = dict(
        expected=expected,
        barrels=savant_barrels(player_id=[1, 2]),
        batting=bref_batting(mlbID=[1, 2]),
    )
    neutral = _hitters(**kwargs)
    adjusted = _hitters(**kwargs, park_factor=pd.Series({1: 1.13, 2: 1.00}))
    assert adjusted.loc[1, "wrc_plus"] < neutral.loc[1, "wrc_plus"]
    assert adjusted.loc[2, "wrc_plus"] == pytest.approx(neutral.loc[2, "wrc_plus"])


def test_missing_park_factor_falls_back_to_neutral():
    """A player absent from the team bridge keeps his rate, rather than dividing
    by NaN and vanishing from the ranking."""
    out = _hitters(park_factor=pd.Series({999: 1.13}))
    assert out["wrc_plus"].iloc[0] == pytest.approx(100.0)


def test_hitter_without_batted_balls_gets_nan_barrel_rate():
    """He is absent from the barrels leaderboard; NaN is the honest rate and 0.0
    would read downstream as a real observation of zero barrels."""
    out = _hitters(
        expected=savant_expected(player_id=[1, 2]),
        barrels=savant_barrels(player_id=[1]),
        batting=bref_batting(mlbID=[1, 2]),
    )
    assert not math.isnan(out.loc[1, "barrel_pct"])
    assert math.isnan(out.loc[2, "barrel_pct"])


def test_zero_league_pa_raises():
    with pytest.raises(ValueError, match="PA is zero"):
        _hitters(expected=savant_expected(pa=0))


def test_renamed_upstream_column_raises():
    """Silent all-NaN skills are the failure mode this guards."""
    frame = savant_expected().rename(columns={"est_woba": "xwoba"})
    with pytest.raises(KeyError, match="est_woba"):
        _hitters(expected=frame)


# --- pitchers --------------------------------------------------------------


def test_pitcher_columns_and_index():
    out = _pitchers()
    assert out.index.name == "mlbam_id"
    assert list(out.columns) == ["ip", *PITCHER_SKILLS]


def test_innings_use_baseball_notation():
    """180.1 is 180 1/3 innings, not 180.1 -- treating it as decimal silently
    mis-scales every rate that divides by IP."""
    out = _pitchers(pitching=bref_pitching(IP=180.1))
    assert out["ip"].iloc[0] == pytest.approx(180 + 1 / 3)


def test_pitch_rates_come_from_counts_not_brefs_rounded_columns():
    """BBRef's StL/StS round to two decimals; the derivation must ignore them and
    divide the Statcast counts, or CSW% ties dozens of pitchers together."""
    out = _pitchers(
        pitching=bref_pitching(StL=0.99, StS=0.99),
        pitch_mix=savant_pitch_mix(pitches=2800, called_strikes=448, whiffs=280, swings=1400),
    )
    assert out["swstr_pct"].iloc[0] == pytest.approx(10.0)
    assert out["csw_pct"].iloc[0] == pytest.approx(26.0)


def test_whiff_and_swstr_use_different_denominators():
    """Whiff rate is per SWING and SwStr% is per PITCH -- conflating them would
    silently report the wrong stat."""
    out = _pitchers()
    assert out["whiff_pct"].iloc[0] == pytest.approx(20.0)  # 280/1400
    assert out["swstr_pct"].iloc[0] == pytest.approx(10.0)  # 280/2800


def test_pitcher_missing_from_pitch_mix_gets_nan_not_zero():
    out = _pitchers(
        pitching=bref_pitching(mlbID=[1, 2]),
        pitch_mix=savant_pitch_mix(player_id=[1]),
    )
    assert not math.isnan(out.loc[1, "csw_pct"])
    assert math.isnan(out.loc[2, "csw_pct"])
    assert not math.isnan(out.loc[2, "fip"])  # BBRef-sourced stats still resolve


def test_k_pct_uses_batters_faced():
    out = _pitchers(pitching=bref_pitching(SO=200, BF=800))
    assert out["k_pct"].iloc[0] == pytest.approx(25.0)


def test_lone_pitcher_is_league_average():
    assert _pitchers()["era_minus"].iloc[0] == pytest.approx(100.0)


def test_fip_constant_makes_lone_pitcher_fip_equal_his_era():
    """The constant is solved so league FIP == league ERA; with one pitcher that
    collapses to his own ERA. Pins the constant against a hardcoded 3.10."""
    out = _pitchers(pitching=bref_pitching(IP=180.0, ER=80))
    assert out["fip"].iloc[0] == pytest.approx(9.0 * 80 / 180.0)


def test_fip_rewards_strikeouts():
    out = _pitchers(
        pitching=bref_pitching(mlbID=[1, 2], SO=[250, 120]),
        pitch_mix=savant_pitch_mix(player_id=[1, 2]),
    )
    assert out.loc[1, "fip"] < out.loc[2, "fip"]


def test_era_minus_is_lower_for_better_pitchers():
    out = _pitchers(
        pitching=bref_pitching(mlbID=[1, 2], ER=[50, 110]),
        pitch_mix=savant_pitch_mix(player_id=[1, 2]),
    )
    assert out.loc[1, "era_minus"] < 100.0 < out.loc[2, "era_minus"]


def test_pitcher_park_factor_deflates_a_hitters_park():
    """Coors inflates runs allowed, so the same ERA there is a BETTER ERA-."""
    kwargs = dict(
        pitching=bref_pitching(mlbID=[1, 2], ER=[80, 80]),
        pitch_mix=savant_pitch_mix(player_id=[1, 2]),
    )
    neutral = _pitchers(**kwargs)
    adjusted = _pitchers(**kwargs, park_factor=pd.Series({1: 1.13, 2: 1.00}))
    assert adjusted.loc[1, "era_minus"] < neutral.loc[1, "era_minus"]
    assert adjusted.loc[2, "era_minus"] == pytest.approx(neutral.loc[2, "era_minus"])


def test_zero_innings_gives_nan_not_zero():
    out = _pitchers(
        pitching=bref_pitching(mlbID=[1, 2], IP=[180.0, 0.0]),
        pitch_mix=savant_pitch_mix(player_id=[1, 2]),
    )
    assert math.isnan(out.loc[2, "era_minus"])
    assert math.isnan(out.loc[2, "fip"])


def test_zero_league_innings_raises():
    with pytest.raises(ValueError, match="IP is zero"):
        _pitchers(pitching=bref_pitching(IP=0.0))
