from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.keepers.playing_time import (
    FEATURES,
    HITTER_FEATURES,
    PITCHER_FEATURES,
    PlayingTimeCurve,
    build_features,
    carry_forward_role,
    fit_curve,
    lag_panel,
    normalize_to_full_season,
    per_appearance,
)


def _s(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_normalize_scales_a_short_season_to_162_games() -> None:
    out = normalize_to_full_season(_s([200.0, 600.0]), _s([60.0, 162.0]))
    assert out.iloc[0] == pytest.approx(540.0)  # 2020: 200 PA in 60 games
    assert out.iloc[1] == pytest.approx(600.0)


def test_role_is_a_rate_and_is_not_schedule_normalized() -> None:
    """Already per-appearance, so a 60-game 2020, a full season and a season still in
    progress all compare directly."""
    assert per_appearance(_s([359.0]), _s([84.0]), "hitter").iloc[0] == pytest.approx(
        4.274, abs=1e-3
    )
    assert per_appearance(_s([187.7]), _s([32.0]), "pitcher").iloc[0] == pytest.approx(
        5.866, abs=1e-3
    )


def test_role_separates_a_starter_from_a_reliever() -> None:
    """For a pitcher this term is not a nuance -- it IS the rotation/bullpen split."""
    starter = per_appearance(_s([180.0]), _s([31.0]), "pitcher").iloc[0]
    reliever = per_appearance(_s([65.0]), _s([64.0]), "pitcher").iloc[0]
    assert starter > 5.0 and reliever < 1.5


def test_role_treats_no_appearances_as_zero_not_missing() -> None:
    """'Did not play' is a real observation the curve should see, not a hole."""
    assert per_appearance(_s([0.0]), _s([0.0]), "hitter").iloc[0] == 0.0


def test_role_caps_an_absurd_small_sample_rate() -> None:
    """A two-game cameo can otherwise compute a nonsense rate off a rounding artifact."""
    assert per_appearance(_s([40.0]), _s([2.0]), "hitter").iloc[0] == pytest.approx(5.5)
    assert per_appearance(_s([27.0]), _s([2.0]), "pitcher").iloc[0] == pytest.approx(9.0)


def test_role_rejects_an_unknown_pool() -> None:
    with pytest.raises(ValueError, match="hitter"):
        per_appearance(_s([1.0]), _s([1.0]), "batter")


def test_build_features_emits_the_pools_own_columns_in_order() -> None:
    hit = build_features(_s([600.0]), _s([650.0]), _s([640.0]), _s([28.0]), _s([4.3]), "hitter")
    assert list(hit.columns) == list(HITTER_FEATURES)
    pit = build_features(
        _s([180.0]), _s([190.0]), _s([175.0]), _s([28.0]), _s([5.8]), "pitcher", _s([1.0])
    )
    assert list(pit.columns) == list(PITCHER_FEATURES)
    assert "start_share" in PITCHER_FEATURES and "start_share" not in HITTER_FEATURES


def test_pitcher_features_require_a_start_share() -> None:
    """Silently defaulting it would read every pitcher as a pure reliever."""
    with pytest.raises(ValueError, match="start_share"):
        build_features(_s([180.0]), _s([190.0]), _s([175.0]), _s([28.0]), _s([5.8]), "pitcher")


def test_vol2_is_an_input_but_not_a_feature() -> None:
    """The second lag earns no coefficient; it exists only to set the prior best that
    `shortfall` measures against."""
    assert "vol2" not in FEATURES["hitter"] and "vol2" not in FEATURES["pitcher"]
    out = build_features(_s([400.0]), _s([680.0]), _s([300.0]), _s([28.0]), _s([4.3]))
    assert out.loc[0, "shortfall"] == pytest.approx(280.0)  # off vol2, not vol3


def test_shortfall_is_one_sided_and_ignores_a_surplus() -> None:
    """A season ABOVE the prior norm is not the mirror image of one below it, so a
    breakout must not register as negative shortfall and quietly reverse the term."""
    out = build_features(_s([700.0]), _s([500.0]), _s([450.0]), _s([25.0]), _s([4.4]))
    assert out.loc[0, "shortfall"] == pytest.approx(0.0)


def test_a_missing_lag_counts_as_zero_playing_time_not_missing_data() -> None:
    out = build_features(_s([600.0]), pd.Series([np.nan]), _s([300.0]), _s([25.0]), _s([4.3]))
    assert out.loc[0, "vol3"] == pytest.approx(300.0)
    assert out.loc[0, "shortfall"] == pytest.approx(0.0)  # prior_best is the 300, below vol1


def test_role_separates_an_injured_regular_from_a_healthy_platoon_bat() -> None:
    """The whole point. Both took 360 PA; only the per-game rate says which was hurt
    and which is simply not an everyday player."""
    injured = build_features(_s([360.0]), _s([700.0]), _s([690.0]), _s([28.0]), _s([4.3]))
    platoon = build_features(_s([360.0]), _s([380.0]), _s([370.0]), _s([28.0]), _s([2.6]))
    assert injured.loc[0, "role"] > platoon.loc[0, "role"]
    assert injured.loc[0, "shortfall"] > platoon.loc[0, "shortfall"]


@pytest.mark.parametrize("kind", ["hitter", "pitcher"])
def test_fit_curve_recovers_planted_coefficients(kind: str) -> None:
    rng = np.random.default_rng(0)
    n = 600
    names = FEATURES[kind]
    features = pd.DataFrame({name: rng.uniform(1.0, 50.0, n) for name in names})
    truth = {name: 0.5 + 0.1 * i for i, name in enumerate(names)}
    target = 20.0 + sum(truth[c] * features[c] for c in names)
    curve = fit_curve(features, target, kind)
    assert curve.kind == kind
    assert curve.intercept == pytest.approx(20.0, abs=0.5)
    for name, beta in zip(names, curve.coefficients, strict=True):
        assert beta == pytest.approx(truth[name], abs=0.02)


def test_fit_curve_raises_rather_than_fitting_an_underdetermined_system() -> None:
    features = pd.DataFrame({name: [1.0, 2.0] for name in HITTER_FEATURES})
    with pytest.raises(ValueError, match="complete rows"):
        fit_curve(features, _s([1.0, 2.0]))


def test_predict_returns_nan_rather_than_clipping_a_negative_to_zero() -> None:
    """Changed deliberately: it used to clip to 0.0, which looks like a real forecast
    and prints a zeroed counting line. NaN is a signal, so the caller's per-player
    fallback to the gap model can actually fire."""
    curve = PlayingTimeCurve(
        kind="hitter", intercept=0.0, coefficients=(0.0, 0.0, -99.0, 0.0, 0.0), n=9, rmse=1.0
    )
    out = curve.predict(build_features(_s([0.0]), _s([600.0]), _s([600.0]), _s([40.0]), _s([0.0])))
    assert out.isna().all()
    assert not (out <= 0).any()  # never a hard zero masquerading as a forecast


def test_predict_rejects_a_frame_missing_a_feature() -> None:
    curve = PlayingTimeCurve(kind="hitter", intercept=0.0, coefficients=(1.0,) * 5, n=9, rmse=1.0)
    with pytest.raises(KeyError, match="shortfall"):
        curve.predict(pd.DataFrame({c: [1.0] for c in HITTER_FEATURES if c != "shortfall"}))


def test_as_dict_pairs_every_coefficient_with_its_feature() -> None:
    curve = PlayingTimeCurve(
        kind="pitcher", intercept=1.0, coefficients=(2.0, 3.0, 4.0, 5.0, 6.0), n=9, rmse=1.0
    )
    assert curve.as_dict() == {
        "intercept": 1.0,
        "vol1": 2.0,
        "vol3": 3.0,
        "age": 4.0,
        "role": 5.0,
        "start_share": 6.0,
    }


def test_shortfall_is_hitters_only() -> None:
    """Dropped for pitchers: on the exit-corrected fit it takes a NEGATIVE coefficient
    (it would subtract from a pitcher who fell short of his own norm) and buys nothing
    -- RMSE 48.95 with against 48.94 without."""
    assert "shortfall" in HITTER_FEATURES
    assert "shortfall" not in PITCHER_FEATURES


def _panel(kind: str = "hitter") -> pd.DataFrame:
    volume = "pa" if kind == "hitter" else "ip"
    rows = []
    for season in range(2018, 2024):
        for mlbam, vol, games in ((1, 600.0, 145.0), (2, 300.0, 120.0)):
            row = {
                "mlbam_id": mlbam,
                "season": season,
                volume: vol,
                "games": games,
                "age": 24 + (season - 2018),
                "scheduled_games": 60 if season == 2020 else 162,
                "partial_season": season == 2023,
            }
            if kind == "pitcher":
                row["starts"] = games
            rows.append(row)
    return pd.DataFrame(rows)


def test_lag_panel_drops_in_progress_seasons_as_targets() -> None:
    """Training on a two-thirds-finished season as though complete would teach the
    curve that everybody collapses."""
    assert 2023 not in set(lag_panel(_panel())["season"])


def test_lag_panel_normalizes_volume_but_not_role() -> None:
    """Volume must be schedule-scaled so 2020 compares; role must NOT be, because 600
    PA over 145 games is the same job in any year."""
    out = lag_panel(_panel())
    row = out[(out["mlbam_id"] == 1) & (out["season"] == 2021)].iloc[0]
    assert row["vol1"] == pytest.approx(600.0 * 162 / 60)
    assert row["role"] == pytest.approx(600.0 / 145.0)


def test_lag_panel_builds_the_pitcher_feature_set() -> None:
    out = lag_panel(_panel("pitcher"), "pitcher")
    assert set(FEATURES["pitcher"]).issubset(out.columns)
    assert out["start_share"].max() == pytest.approx(1.0)  # fixture starts every game


def test_lag_panel_honours_the_recent_volume_floor() -> None:
    out = lag_panel(_panel(), min_recent=400.0)
    assert (out["vol1"] >= 400.0).all()
    # Checked on a target year whose first lag is an ordinary 162-game season: in the
    # 2021 row the lag is 2020, which normalizes 300 PA up to 810 and legitimately
    # clears the floor for the low-volume player too.
    assert set(out[out["season"] == 2022]["mlbam_id"]) == {1}


def test_lag_panel_raises_on_a_panel_missing_a_required_column() -> None:
    for col in ("age", "games"):
        with pytest.raises(KeyError, match=col):
            lag_panel(_panel().drop(columns=[col]))
    with pytest.raises(KeyError, match="starts"):
        lag_panel(_panel("pitcher").drop(columns=["starts"]), "pitcher")


def _panel_with_a_career_ending() -> pd.DataFrame:
    """Player 1 plays 2018-2023; player 2 RETIRES after 2020 and has no later rows."""
    rows = []
    for season in range(2018, 2024):
        rows.append(
            {
                "mlbam_id": 1,
                "season": season,
                "pa": 600.0,
                "games": 145.0,
                "age": 24 + (season - 2018),
                "scheduled_games": 162,
                "partial_season": False,
            }
        )
        if season <= 2020:
            rows.append(
                {
                    "mlbam_id": 2,
                    "season": season,
                    "pa": 500.0,
                    "games": 130.0,
                    "age": 34 + (season - 2018),
                    "scheduled_games": 162,
                    "partial_season": False,
                }
            )
    return pd.DataFrame(rows)


def test_include_exits_trains_the_career_ending_as_zero() -> None:
    """Without this the curve learns E[volume | he plays again] and over-forecasts the
    aging and marginal players a keeper decision has to price."""
    out = lag_panel(_panel_with_a_career_ending(), min_recent=300.0)
    exit_rows = out[(out["mlbam_id"] == 2) & (out["target"] == 0.0)]
    assert len(exit_rows) == 1
    assert exit_rows.iloc[0]["season"] == 2021  # the year after his last


def test_exactly_one_exit_row_per_career() -> None:
    """The season AFTER the exit has no lag row to derive an age from, so it drops --
    long-retired players must not contribute a tail of zeros."""
    out = lag_panel(_panel_with_a_career_ending(), min_recent=300.0)
    assert len(out[(out["mlbam_id"] == 2) & (out["target"] == 0.0)]) == 1
    assert 2022 not in set(out[out["mlbam_id"] == 2]["season"])


def test_the_exit_rows_age_is_derived_from_the_prior_season() -> None:
    out = lag_panel(_panel_with_a_career_ending(), min_recent=300.0)
    row = out[(out["mlbam_id"] == 2) & (out["season"] == 2021)].iloc[0]
    assert row["age"] == pytest.approx(37.0)  # 36 in his final 2020 season, plus one


def test_include_exits_false_drops_the_career_ending_entirely() -> None:
    out = lag_panel(_panel_with_a_career_ending(), min_recent=300.0, include_exits=False)
    assert 2021 not in set(out[out["mlbam_id"] == 2]["season"])


def test_a_still_active_player_gets_no_spurious_exit_row() -> None:
    """Player 1's last season is the panel's last, which is an unfinished career rather
    than an ending; inventing a zero for him would be a fabricated retirement."""
    out = lag_panel(_panel_with_a_career_ending(), min_recent=300.0)
    assert not ((out["mlbam_id"] == 1) & (out["target"] == 0.0)).any()


def test_carry_forward_returns_role_and_start_share_together() -> None:
    """They are ONE joint term (VIF ~35), so they must advance in lockstep. Carrying
    role forward while leaving start_share at the base year gave a rehabbing starter
    his full starter credit with none of the offsetting term -- about +29 IP."""
    absent = pd.Series([np.nan], index=[1])
    prior_ip, prior_g, prior_gs = _s([180.0]), _s([31.0]), _s([29.0])
    prior_ip.index = prior_g.index = prior_gs.index = [1]
    role, start_share = carry_forward_role(
        volumes=[absent, prior_ip],
        appearances=[absent, prior_g],
        kind="pitcher",
        starts=[absent, prior_gs],
    )
    assert role.loc[1] == pytest.approx(180.0 / 31.0)
    assert start_share is not None
    # The partner must come from the SAME season the role did, not default to 0.
    assert start_share.loc[1] == pytest.approx(29.0 / 31.0)


def test_carry_forward_prefers_the_most_recent_observed_season() -> None:
    base_ip = pd.Series([120.0], index=[1])
    older_ip = pd.Series([180.0], index=[1])
    g = pd.Series([30.0], index=[1])
    role, _ = carry_forward_role([base_ip, older_ip], [g, g], "pitcher", starts=[g, g])
    assert role.loc[1] == pytest.approx(120.0 / 30.0)  # base year wins


def test_carry_forward_falls_back_to_zero_when_nothing_is_observed() -> None:
    absent = pd.Series([np.nan], index=[1])
    role, start_share = carry_forward_role([absent], [absent], "pitcher", starts=[absent])
    assert role.loc[1] == 0.0
    assert start_share is not None and start_share.loc[1] == 0.0


def test_carry_forward_omits_start_share_for_hitters() -> None:
    role, start_share = carry_forward_role([_s([600.0])], [_s([145.0])], "hitter")
    assert start_share is None
    assert role.iloc[0] == pytest.approx(600.0 / 145.0)
