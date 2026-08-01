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
)


def _s(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_normalize_scales_a_short_season_to_162_games() -> None:
    out = normalize_to_full_season(_s([200.0, 600.0]), _s([60.0, 162.0]))
    assert out.iloc[0] == pytest.approx(540.0)  # 2020: 200 PA in 60 games
    assert out.iloc[1] == pytest.approx(600.0)


def test_build_features_emits_every_model_column_in_order() -> None:
    out = build_features(_s([600.0]), _s([650.0]), _s([640.0]), _s([28.0]), _s([6.0]))
    assert list(out.columns) == list(PA_FEATURES)


def test_shortfall_is_the_drop_below_the_players_own_prior_best() -> None:
    out = build_features(_s([400.0]), _s([650.0]), _s([600.0]), _s([28.0]), _s([6.0]))
    assert out.loc[0, "shortfall"] == pytest.approx(250.0)


def test_shortfall_is_one_sided_and_ignores_a_surplus() -> None:
    """A season ABOVE the prior norm is not the mirror image of one below it, so a
    breakout must not register as negative shortfall and quietly reverse the term."""
    out = build_features(_s([700.0]), _s([500.0]), _s([450.0]), _s([25.0]), _s([4.0]))
    assert out.loc[0, "shortfall"] == pytest.approx(0.0)


def test_a_missing_lag_counts_as_zero_playing_time_not_missing_data() -> None:
    out = build_features(_s([600.0]), pd.Series([np.nan]), _s([300.0]), _s([25.0]), _s([5.0]))
    assert out.loc[0, "pa2"] == pytest.approx(0.0)
    assert out.loc[0, "shortfall"] == pytest.approx(0.0)  # prior_best is the 300, below pa1


def test_has_flags_separate_did_not_play_from_did_not_exist() -> None:
    """A rookie's older lags are structural zeros. Without these flags the curve reads
    an everyday young player as an injury risk."""
    rookie = build_features(_s([600.0]), _s([0.0]), _s([0.0]), _s([22.0]), _s([1.0]))
    veteran = build_features(_s([600.0]), _s([0.0]), _s([0.0]), _s([32.0]), _s([9.0]))
    assert rookie.loc[0, "has2"] == 0.0 and rookie.loc[0, "has3"] == 0.0
    assert veteran.loc[0, "has2"] == 1.0 and veteran.loc[0, "has3"] == 1.0


def test_fit_curve_recovers_planted_coefficients() -> None:
    rng = np.random.default_rng(0)
    n = 500
    features = pd.DataFrame(
        {
            name: rng.uniform(0, 600, n) if name.startswith("pa") else rng.uniform(0, 5, n)
            for name in PA_FEATURES
        }
    )
    truth = {
        "pa1": 0.7,
        "pa2": 0.05,
        "pa3": 0.1,
        "age": -11.0,
        "has2": -35.0,
        "has3": -6.0,
        "shortfall": 0.25,
    }
    target = 340.0 + sum(truth[c] * features[c] for c in PA_FEATURES)
    curve = fit_curve(features, target)
    assert curve.intercept == pytest.approx(340.0, abs=1.0)
    for name, beta in zip(PA_FEATURES, curve.coefficients, strict=True):
        assert beta == pytest.approx(truth[name], abs=0.02)
    assert curve.rmse == pytest.approx(0.0, abs=1e-6)


def test_fit_curve_raises_rather_than_fitting_an_underdetermined_system() -> None:
    features = pd.DataFrame({name: [1.0, 2.0] for name in PA_FEATURES})
    with pytest.raises(ValueError, match="complete rows"):
        fit_curve(features, _s([1.0, 2.0]))


def test_predict_never_returns_negative_playing_time() -> None:
    """The linear form can go negative for an old player with no recent history, and a
    negative volume would flip the sign of every counting stat built on it."""
    curve = PlayingTimeCurve(intercept=0.0, coefficients=(0.0,) * 6 + (-99.0,), n=9, rmse=1.0)
    out = curve.predict(build_features(_s([0.0]), _s([600.0]), _s([600.0]), _s([40.0]), _s([12.0])))
    assert (out >= 0).all()


def test_predict_rejects_a_frame_missing_a_feature() -> None:
    curve = PlayingTimeCurve(intercept=0.0, coefficients=(1.0,) * 7, n=9, rmse=1.0)
    with pytest.raises(KeyError, match="shortfall"):
        curve.predict(pd.DataFrame({c: [1.0] for c in PA_FEATURES if c != "shortfall"}))


def _panel() -> pd.DataFrame:
    rows = []
    for season in range(2018, 2024):
        for mlbam, base in ((1, 600.0), (2, 300.0)):
            rows.append(
                {
                    "mlbam_id": mlbam,
                    "season": season,
                    "pa": base,
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
    out = lag_panel(_panel())
    assert 2023 not in set(out["season"])


def test_lag_panel_normalizes_the_short_season_before_lagging() -> None:
    out = lag_panel(_panel())
    row = out[(out["mlbam_id"] == 1) & (out["season"] == 2021)].iloc[0]
    assert row["pa1"] == pytest.approx(600.0 * 162 / 60)  # 2020 scaled up


def test_lag_panel_honours_the_recent_volume_floor() -> None:
    out = lag_panel(_panel(), min_recent=400.0)
    assert (out["pa1"] >= 400.0).all()
    # Checked on a target year whose first lag is an ordinary 162-game season: in the
    # 2021 row the lag is 2020, which normalizes 300 PA up to 810 and legitimately
    # clears the floor for the low-volume player too.
    season_2022 = out[out["season"] == 2022]
    assert set(season_2022["mlbam_id"]) == {1}


def test_lag_panel_raises_on_a_panel_missing_a_required_column() -> None:
    with pytest.raises(KeyError, match="age"):
        lag_panel(_panel().drop(columns=["age"]))
