import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros.baselines import (
    MARCEL_WEIGHTS,
    REGRESS_BF,
    baseline_predictions,
    league_rates,
    marcel,
)
from fantasy_baseball.pitcher_ros.stats import TARGETS, rates_from_counts

from .conftest import game, league_games


def test_rates_from_counts():
    counts = pd.DataFrame(
        [
            {"bf": 100, "outs": 75, "k": 30, "w": 2, "sv": 1, "er": 10, "h": 20, "bb": 5},
            {"bf": 0, "outs": 0, "k": 0, "w": 0, "sv": 0, "er": 0, "h": 0, "bb": 0},
        ]
    )
    r = rates_from_counts(counts)
    assert r.loc[0, "k"] == pytest.approx(0.3)
    assert r.loc[0, "era"] == pytest.approx(9 * 10 / 25)  # 25 IP
    assert r.loc[0, "whip"] == pytest.approx(25 / 25)
    assert r.loc[1].isna().all()  # nothing to divide by: NaN, not inf or 0


def test_league_rates_come_from_the_three_seasons_before():
    games = pd.DataFrame(
        [*league_games(), game(900, "2021-06-01", bf=500, outs=300, k=500, er=300)]
    )
    lg = league_rates(games, 2025)  # 2022-2024; 2021 is out
    assert lg["k"] == pytest.approx(0.25) and lg["sv"] == pytest.approx(0.005)
    assert lg["era"] == pytest.approx(27 * 100 / 700) and lg["whip"] == pytest.approx(1.2)
    assert lg["outs_per_bf"] == pytest.approx(0.7)
    with pytest.raises(ValueError):
        league_rates(games, 2021)  # nothing before 2021


def test_marcel_with_no_history_is_league_average():
    games = pd.DataFrame(league_games())
    lg = league_rates(games, 2025)
    out = marcel(pd.Index([7], name="player_id"), games, 2025, pd.Timestamp(2025, 1, 1), lg)
    for s in TARGETS:
        assert out.loc[7, s] == pytest.approx(lg[s])


def test_marcel_weights_seasons_and_only_sees_games_before_the_date():
    games = pd.DataFrame(
        [
            *league_games(),
            game(1, "2024-05-01", bf=600, outs=420, k=210, w=6, er=40, h=100, bb=30),
            game(1, "2022-05-01", bf=400, outs=280, k=80, w=2, er=40, h=90, bb=40),
            game(1, "2025-04-10", bf=100, outs=70, k=40, w=1, er=5, h=15, bb=5),
            # After the projection date: must not move the projection.
            game(1, "2025-06-01", bf=500, outs=350, k=0, w=0, er=200, h=300, bb=100),
        ]
    )
    lg = league_rates(games, 2025)
    before = pd.Timestamp("2025-05-01")
    out = marcel(pd.Index([1], name="player_id"), games, 2025, before, lg).loc[1]
    w = MARCEL_WEIGHTS
    bf = w["std"] * 100 + w["p1"] * 600 + w["older"] * 400
    k = w["std"] * 40 + w["p1"] * 210 + w["older"] * 80
    assert out["k"] == pytest.approx((k + REGRESS_BF * lg["k"]) / (bf + REGRESS_BF))
    outs = w["std"] * 70 + w["p1"] * 420 + w["older"] * 280
    er = w["std"] * 5 + w["p1"] * 40 + w["older"] * 40
    regress_outs = REGRESS_BF * lg["outs_per_bf"]
    expected_era = 27 * (er + regress_outs * lg["era"] / 27) / (outs + regress_outs)
    assert out["era"] == pytest.approx(expected_era)


def test_baseline_predictions_cover_every_player_asked_for():
    games = pd.DataFrame([*league_games(), game(1, "2024-05-01", bf=50, outs=30, k=20)])
    players = pd.Index([1, 2], name="player_id")
    preds = baseline_predictions(players, games, 2025, pd.Timestamp(2025, 1, 1))
    assert set(preds) == {"marcel", "league_avg"}
    for p in preds.values():
        assert list(p.index) == [1, 2] and list(p.columns) == list(TARGETS)
        assert not np.isnan(p.to_numpy()).any()
