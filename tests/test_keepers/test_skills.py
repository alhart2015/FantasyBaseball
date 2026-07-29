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


def _two_hitters(*, park_factor=None, **overrides) -> pd.DataFrame:
    """Two hitters with ids 1 and 2; `overrides` vary only the expected-stats side."""
    return _hitters(
        expected=savant_expected(player_id=[1, 2], **overrides),
        barrels=savant_barrels(player_id=[1, 2]),
        batting=bref_batting(mlbID=[1, 2]),
        park_factor=park_factor,
    )


def _two_pitchers(*, park_factor=None, **overrides) -> pd.DataFrame:
    """Two pitchers with ids 1 and 2; `overrides` vary only the BBRef side."""
    return _pitchers(
        pitching=bref_pitching(mlbID=[1, 2], **overrides),
        pitch_mix=savant_pitch_mix(player_id=[1, 2]),
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
    out = _two_hitters(woba=[0.400, 0.280])
    assert out.loc[1, "wrc_plus"] > 100.0 > out.loc[2, "wrc_plus"]


def test_hitter_park_factor_deflates_a_hitters_park():
    """Coors-like inflation must lower wRC+ relative to neutral, not raise it."""
    neutral = _two_hitters(woba=[0.400, 0.280])
    adjusted = _two_hitters(woba=[0.400, 0.280], park_factor=pd.Series({1: 1.13, 2: 1.00}))
    assert adjusted.loc[1, "wrc_plus"] < neutral.loc[1, "wrc_plus"]
    assert adjusted.loc[2, "wrc_plus"] == pytest.approx(neutral.loc[2, "wrc_plus"])


def test_park_adjustment_assumes_a_half_home_schedule():
    """Delegates to `park_neutral_value`'s 50/50 model: a hitter takes only half
    his PA at home, so the correction is value*2/(pf+1), NOT value/pf. Dividing
    by the raw factor would over-correct every Coors bat by ~6%.
    """
    adjusted = _two_hitters(woba=[0.400, 0.400], park_factor=pd.Series({1: 1.13, 2: 1.00}))
    assert adjusted.loc[1, "wrc_plus"] == pytest.approx(adjusted.loc[2, "wrc_plus"] * 2.0 / 2.13)


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
    out = _two_pitchers(SO=[250, 120])
    assert out.loc[1, "fip"] < out.loc[2, "fip"]


def test_era_minus_is_lower_for_better_pitchers():
    out = _two_pitchers(ER=[50, 110])
    assert out.loc[1, "era_minus"] < 100.0 < out.loc[2, "era_minus"]


def test_pitcher_park_factor_deflates_a_hitters_park():
    """Coors inflates runs allowed, so the same ERA there is a BETTER ERA-."""
    neutral = _two_pitchers(ER=[80, 80])
    adjusted = _two_pitchers(ER=[80, 80], park_factor=pd.Series({1: 1.13, 2: 1.00}))
    assert adjusted.loc[1, "era_minus"] < neutral.loc[1, "era_minus"]
    assert adjusted.loc[2, "era_minus"] == pytest.approx(neutral.loc[2, "era_minus"])


def test_zero_innings_gives_nan_not_zero():
    out = _two_pitchers(IP=[180.0, 0.0])
    assert math.isnan(out.loc[2, "era_minus"])
    assert math.isnan(out.loc[2, "fip"])


def test_zero_league_innings_raises():
    with pytest.raises(ValueError, match="IP is zero"):
        _pitchers(pitching=bref_pitching(IP=0.0))


def test_one_nan_count_does_not_rescale_every_other_pitcher():
    """Series.sum() skips NaN, so summing numerator and denominator separately
    would drop a pitcher's ER while keeping his IP -- halving league ERA and
    doubling everyone else's ERA-."""
    clean = _two_pitchers(ER=[80, 80])
    dirty = _two_pitchers(ER=[80, float("nan")])
    assert dirty.loc[1, "era_minus"] == pytest.approx(clean.loc[1, "era_minus"])
    assert dirty.loc[1, "fip"] == pytest.approx(clean.loc[1, "fip"])
    assert math.isnan(dirty.loc[2, "era_minus"])


def test_one_nan_woba_does_not_rescale_every_other_hitter():
    clean = _two_hitters(woba=[0.400, 0.320])
    dirty = _two_hitters(woba=[0.400, float("nan")])
    assert dirty.loc[1, "wrc_plus"] != pytest.approx(clean.loc[1, "wrc_plus"])  # league changed
    # ...but the survivor is still measured against a league built from HIS row
    # alone, so he is exactly average -- not inflated by a mismatched denominator.
    assert dirty.loc[1, "wrc_plus"] == pytest.approx(100.0)


def test_nan_runs_do_not_skew_league_r_per_pa():
    """R and PA are summed independently unless masked together."""
    clean = _two_hitters()
    dirty = _hitters(
        expected=savant_expected(player_id=[1, 2]),
        barrels=savant_barrels(player_id=[1, 2]),
        batting=bref_batting(mlbID=[1, 2], R=[90, float("nan")]),
    )
    assert dirty.loc[1, "wrc_plus"] == pytest.approx(clean.loc[1, "wrc_plus"])


def _heterogeneous_pitchers(**overrides):
    """Three pitchers with DIFFERENT rates. Identical ones make league-constant
    bugs cancel to exactly zero, which is how the first NaN test missed one."""
    return bref_pitching(
        mlbID=[1, 2, 3],
        IP=[200.0, 100.0, 100.0],
        ER=[150, 40, 60],
        HR=[25, 10, 20],
        BB=[45, 20, 30],
        HBP=[5, 2, 3],
        SO=[200, 120, 80],
        BF=[800, 400, 400],
        **overrides,
    )


def _league(out):
    """IP-weighted league FIP and ERA- over the rows that resolved."""
    ok = out[["ip", "fip", "era_minus"]].dropna()
    weight = ok["ip"].sum()
    return (ok["fip"] * ok["ip"]).sum() / weight, (ok["era_minus"] * ok["ip"]).sum() / weight


@pytest.mark.parametrize("blank", [None, "HR", "ER", "SO"])
def test_league_fip_equals_league_era_whatever_is_missing(blank):
    """The FIP constant is solved so league FIP == league ERA. That only holds
    if both constants are masked to the SAME pitchers -- masking each on its own
    numerator lets one blank count put a pitcher's innings in one denominator
    and not the other, and the constant silently absorbs the difference."""
    pitching = _heterogeneous_pitchers()
    if blank is not None:
        pitching = pitching.copy()
        pitching.loc[0, blank] = float("nan")
    out = normalize_pitcher_skills(pitching, savant_pitch_mix(player_id=[1, 2, 3]))

    lg_fip, lg_era_minus = _league(out)
    assert lg_era_minus == pytest.approx(100.0)
    assert lg_fip == pytest.approx(_expected_league_era(pitching))


def _expected_league_era(pitching):
    """League ERA over pitchers whose every FIP input is present."""
    frame = pitching.copy()
    cols = ["ER", "HR", "BB", "HBP", "SO"]
    complete = frame[cols].notna().all(axis=1)
    return 9.0 * frame.loc[complete, "ER"].sum() / frame.loc[complete, "IP"].sum()
