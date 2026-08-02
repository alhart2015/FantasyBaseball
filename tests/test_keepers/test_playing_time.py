from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.keepers.playing_time import (
    PA_FEATURES,
    PlayingTimeCurve,
    build_features,
    fit_curve,
    lag_panel,
    normalize_to_full_season,
    plate_appearances_per_game,
)


def _s(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_normalize_scales_a_short_season_to_162_games() -> None:
    out = normalize_to_full_season(_s([200.0, 600.0]), _s([60.0, 162.0]))
    assert out.iloc[0] == pytest.approx(540.0)  # 2020: 200 PA in 60 games
    assert out.iloc[1] == pytest.approx(600.0)


def test_pa_per_game_is_a_rate_and_is_not_schedule_normalized() -> None:
    """It is already per-game, so a 60-game season and a 162-game season compare
    directly -- and so does a season still in progress."""
    assert plate_appearances_per_game(_s([359.0]), _s([84.0])).iloc[0] == pytest.approx(
        4.274, abs=1e-3
    )
    assert plate_appearances_per_game(_s([600.0]), _s([140.0])).iloc[0] == pytest.approx(
        4.286, abs=1e-3
    )


def test_pa_per_game_treats_no_games_as_zero_not_missing() -> None:
    """'Did not play' is a real observation the curve should see, not a hole."""
    assert plate_appearances_per_game(_s([0.0]), _s([0.0])).iloc[0] == 0.0


def test_pa_per_game_caps_an_absurd_small_sample_rate() -> None:
    """A two-game cameo can otherwise compute a nonsense rate off a rounding artifact."""
    assert plate_appearances_per_game(_s([40.0]), _s([2.0])).iloc[0] == pytest.approx(5.5)


def test_build_features_emits_every_model_column_in_order() -> None:
    out = build_features(_s([600.0]), _s([650.0]), _s([640.0]), _s([28.0]), _s([4.3]))
    assert list(out.columns) == list(PA_FEATURES)


def test_pa2_is_an_input_but_not_a_feature() -> None:
    """The second lag earns no coefficient of its own; it exists only to set the prior
    best that `shortfall` measures against."""
    assert "pa2" not in PA_FEATURES
    out = build_features(_s([400.0]), _s([680.0]), _s([300.0]), _s([28.0]), _s([4.3]))
    assert out.loc[0, "shortfall"] == pytest.approx(280.0)  # measured off pa2, not pa3


def test_shortfall_is_the_drop_below_the_players_own_prior_best() -> None:
    out = build_features(_s([400.0]), _s([650.0]), _s([600.0]), _s([28.0]), _s([4.3]))
    assert out.loc[0, "shortfall"] == pytest.approx(250.0)


def test_shortfall_is_one_sided_and_ignores_a_surplus() -> None:
    """A season ABOVE the prior norm is not the mirror image of one below it, so a
    breakout must not register as negative shortfall and quietly reverse the term."""
    out = build_features(_s([700.0]), _s([500.0]), _s([450.0]), _s([25.0]), _s([4.4]))
    assert out.loc[0, "shortfall"] == pytest.approx(0.0)


def test_a_missing_lag_counts_as_zero_playing_time_not_missing_data() -> None:
    out = build_features(_s([600.0]), pd.Series([np.nan]), _s([300.0]), _s([25.0]), _s([4.3]))
    assert out.loc[0, "pa3"] == pytest.approx(300.0)
    assert out.loc[0, "shortfall"] == pytest.approx(0.0)  # prior_best is the 300, below pa1


def test_role_separates_an_injured_regular_from_a_healthy_platoon_bat() -> None:
    """The whole point of `ppg1`. Both took 360 PA; only the per-game rate says which
    was hurt and which is simply not an everyday player."""
    injured = build_features(_s([360.0]), _s([700.0]), _s([690.0]), _s([28.0]), _s([4.3]))
    platoon = build_features(_s([360.0]), _s([380.0]), _s([370.0]), _s([28.0]), _s([2.6]))
    assert injured.loc[0, "ppg1"] > platoon.loc[0, "ppg1"]
    assert injured.loc[0, "shortfall"] > platoon.loc[0, "shortfall"]


def test_fit_curve_recovers_planted_coefficients() -> None:
    rng = np.random.default_rng(0)
    n = 500
    features = pd.DataFrame(
        {
            "pa1": rng.uniform(300, 700, n),
            "pa3": rng.uniform(0, 700, n),
            "age": rng.uniform(21, 38, n),
            "ppg1": rng.uniform(2.0, 4.8, n),
            "shortfall": rng.uniform(0, 300, n),
        }
    )
    truth = {"pa1": 0.475, "pa3": 0.077, "age": -10.1, "ppg1": 114.3, "shortfall": 0.141}
    target = 13.7 + sum(truth[c] * features[c] for c in PA_FEATURES)
    curve = fit_curve(features, target)
    assert curve.intercept == pytest.approx(13.7, abs=1.0)
    for name, beta in zip(PA_FEATURES, curve.coefficients, strict=True):
        assert beta == pytest.approx(truth[name], abs=0.05)
    assert curve.rmse == pytest.approx(0.0, abs=1e-6)


def test_fit_curve_raises_rather_than_fitting_an_underdetermined_system() -> None:
    features = pd.DataFrame({name: [1.0, 2.0] for name in PA_FEATURES})
    with pytest.raises(ValueError, match="complete rows"):
        fit_curve(features, _s([1.0, 2.0]))


def test_predict_never_returns_negative_playing_time() -> None:
    """The linear form can go negative for an old player with no recent history, and a
    negative volume would flip the sign of every counting stat built on it."""
    curve = PlayingTimeCurve(intercept=0.0, coefficients=(0.0, 0.0, -99.0, 0.0, 0.0), n=9, rmse=1.0)
    out = curve.predict(build_features(_s([0.0]), _s([600.0]), _s([600.0]), _s([40.0]), _s([0.0])))
    assert (out >= 0).all()


def test_predict_rejects_a_frame_missing_a_feature() -> None:
    curve = PlayingTimeCurve(intercept=0.0, coefficients=(1.0,) * 5, n=9, rmse=1.0)
    with pytest.raises(KeyError, match="shortfall"):
        curve.predict(pd.DataFrame({c: [1.0] for c in PA_FEATURES if c != "shortfall"}))


def test_as_dict_pairs_every_coefficient_with_its_feature() -> None:
    curve = PlayingTimeCurve(intercept=1.0, coefficients=(2.0, 3.0, 4.0, 5.0, 6.0), n=9, rmse=1.0)
    assert curve.as_dict() == {
        "intercept": 1.0,
        "pa1": 2.0,
        "pa3": 3.0,
        "age": 4.0,
        "ppg1": 5.0,
        "shortfall": 6.0,
    }


def _panel() -> pd.DataFrame:
    rows = []
    for season in range(2018, 2024):
        for mlbam, pa, games in ((1, 600.0, 145.0), (2, 300.0, 120.0)):
            rows.append(
                {
                    "mlbam_id": mlbam,
                    "season": season,
                    "pa": pa,
                    "games": games,
                    "age": 24 + (season - 2018),
                    "seasons_since_debut": season - 2018,
                    "scheduled_games": 60 if season == 2020 else 162,
                    "partial_season": season == 2023,
                }
            )
    return pd.DataFrame(rows)


def test_lag_panel_drops_in_progress_seasons_as_targets() -> None:
    """Training on a two-thirds-finished season as though complete would teach the
    curve that everybody collapses."""
    assert 2023 not in set(lag_panel(_panel())["season"])


def test_lag_panel_normalizes_the_short_season_before_lagging() -> None:
    out = lag_panel(_panel())
    row = out[(out["mlbam_id"] == 1) & (out["season"] == 2021)].iloc[0]
    assert row["pa1"] == pytest.approx(600.0 * 162 / 60)  # 2020 scaled up


def test_lag_panel_carries_the_unnormalized_per_game_rate() -> None:
    """`ppg` must NOT be schedule-scaled -- 600 PA over 145 games is the same role in
    2020 as in any other year."""
    out = lag_panel(_panel())
    row = out[(out["mlbam_id"] == 1) & (out["season"] == 2021)].iloc[0]
    assert row["ppg1"] == pytest.approx(600.0 / 145.0)


def test_lag_panel_honours_the_recent_volume_floor() -> None:
    out = lag_panel(_panel(), min_recent=400.0)
    assert (out["pa1"] >= 400.0).all()
    # Checked on a target year whose first lag is an ordinary 162-game season: in the
    # 2021 row the lag is 2020, which normalizes 300 PA up to 810 and legitimately
    # clears the floor for the low-volume player too.
    assert set(out[out["season"] == 2022]["mlbam_id"]) == {1}


def test_lag_panel_raises_on_a_panel_missing_a_required_column() -> None:
    for col in ("age", "games"):
        with pytest.raises(KeyError, match=col):
            lag_panel(_panel().drop(columns=[col]))
