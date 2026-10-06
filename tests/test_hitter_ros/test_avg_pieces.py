"""#433 step 2: AVG as its pieces -- K/AB, HR/AB and BABIP."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.features import (
    PIECES,
    TARGETS,
    avg_from_pieces,
    column_horizon,
    horizon_columns,
    input_frame,
    league_answer_rates,
    league_reference,
    piece_rates,
    target_frame,
    target_stat,
)
from fantasy_baseball.hitter_ros.table import build_table
from tests.test_hitter_ros.test_table import _season, _write


def _with_strikeouts(season):
    """test_table's season, with one strikeout in every game of 3+ AB, and a steal for
    the leadoff hitter (a league rate of 0 can't be a relative target's denominator)."""
    lineups, pitches = season
    for row in lineups:
        row["k"] = 1 if row["ab"] >= 3 else 0
        row["sb"] = 1 if row["ab"] >= 4 else 0
    return lineups, pitches


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    seasons = {2024: _season(2024, date(2024, 4, 1), 10), 2025: _season(2025, date(2025, 4, 1), 21)}
    _write(root, {year: _with_strikeouts(s) for year, s in seasons.items()})
    return build_table(root)


def test_pieces_rebuild_avg_exactly():
    rng = np.random.default_rng(0)
    ab = rng.integers(1, 600, size=200)
    k = rng.integers(0, ab // 3 + 1)
    hr = rng.integers(0, ab // 10 + 1)
    h = hr + rng.integers(0, ab - k - hr + 1)
    counts = pd.DataFrame({"ab": ab, "h": h, "hr": hr, "k": k})
    p = piece_rates(counts)
    assert list(p.columns) == list(PIECES)
    np.testing.assert_allclose(avg_from_pieces(p.k_ab, p.hr_ab, p.babip), h / ab)
    # No AB: no pieces. No balls in play: no BABIP, but K/AB and HR/AB are known.
    empty = piece_rates(pd.DataFrame({"ab": [0, 2], "h": [0, 1], "hr": [0, 1], "k": [0, 1]}))
    assert empty.iloc[0].isna().all()
    assert np.isnan(empty.babip[1]) and empty.k_ab[1] == 0.5 and empty.hr_ab[1] == 0.5


def test_piece_column_names():
    assert target_stat("n25_k_ab") == "k_ab" and target_stat("hr_ab") == "hr_ab"
    assert column_horizon("n250_babip") == "n250" and column_horizon("k_ab") == "ros"
    assert target_stat("n25_hr") == "hr" and column_horizon("n100_avg") == "n100"
    assert horizon_columns((25,), PIECES) == [*PIECES, "n25_k_ab", "n25_hr_ab", "n25_babip"]


def test_targets_with_pieces(table):
    plain_rates, plain_weights = target_frame(table, (25, 100))
    rates, weights = target_frame(table, (25, 100), pieces=True)
    cols = [*horizon_columns((25, 100)), *horizon_columns((25, 100), PIECES)]
    assert list(rates.columns) == cols == list(weights.columns)
    pd.testing.assert_frame_equal(plain_rates, rates[list(plain_rates.columns)])
    pd.testing.assert_frame_equal(plain_weights, weights[list(plain_weights.columns)])
    # HITTER's next 25 PA from 2025's first date: 7 games of 4 AB, 2 H, 1 HR, 1 K.
    row = (table.player_id == 1) & (table.season == 2025) & (table.week == 0)
    got = rates.loc[row].iloc[0]
    assert got["n25_k_ab"] == pytest.approx(7 / 28) and got["n25_hr_ab"] == pytest.approx(7 / 28)
    assert got["n25_babip"] == pytest.approx(7 / 14)
    assert weights.loc[row, "n25_hr_ab"].iloc[0] == 28
    assert weights.loc[row, "n25_babip"].iloc[0] == 14  # balls in play: AB - K - HR
    assert avg_from_pieces(got["n25_k_ab"], got["n25_hr_ab"], got["n25_babip"]) == pytest.approx(
        got["n25_avg"]
    )
    assert np.isnan(got["n100_babip"]) and weights.loc[row, "n100_babip"].iloc[0] == 0


def test_league_rates_with_pieces(table):
    for rates in (league_answer_rates, league_reference):
        plain, both = rates(table), rates(table, pieces=True)
        assert list(both.columns) == [*TARGETS, *PIECES]
        pd.testing.assert_frame_equal(plain, both[list(TARGETS)])
    answer = league_answer_rates(table, pieces=True)
    rows = (table.season == 2025) & (table.week == 0)
    sums = table.loc[rows, ["ros_ab", "ros_h", "ros_hr", "ros_k"]].sum()
    expected = piece_rates(pd.DataFrame([sums.rename(lambda c: c.removeprefix("ros_"))]))
    pd.testing.assert_series_equal(
        answer.loc[rows, list(PIECES)].iloc[0], expected.iloc[0], check_names=False
    )
    # The league AVG from its pieces is the league AVG.
    a = answer.loc[rows].iloc[0]
    assert avg_from_pieces(a.k_ab, a.hr_ab, a.babip) == pytest.approx(a.avg)


def test_pieces_are_on_by_default():
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig

    assert NetConfig().avg_pieces == "extra"


def test_fit_season_refuses_targets_without_pieces(table):
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig
    from scripts.train_hitter_ros import fit_season

    y_all, w_all = target_frame(table, (25,))
    with pytest.raises(ValueError, match="pieces"):
        fit_season(table, input_frame(table), y_all, w_all, 2025, NetConfig(avg_pieces="extra"))
    y_all, w_all = target_frame(table, (25,), pieces=True)
    with pytest.raises(ValueError, match="head_layers"):
        fit_season(table, input_frame(table), y_all, w_all, 2025, NetConfig(head_layers=8))


def test_piece_config_and_loss():
    torch = pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig, count_loss, device

    with pytest.raises(ValueError):
        NetConfig(avg_pieces="all")
    with pytest.raises(ValueError):
        NetConfig(avg_pieces="derived", avg_loss="binomial")
    stats = [*TARGETS, *PIECES]
    y = np.full((10, len(stats)), 0.25)
    y[:5] = 0.2
    w = np.ones_like(y)
    spec = count_loss(NetConfig(avg_pieces="extra"), y, w, device(), stats)
    assert spec.binomial.tolist() == [s in PIECES for s in stats]
    assert not (spec.mask & spec.binomial).any()
    assert isinstance(spec.scale, torch.Tensor)


@pytest.mark.parametrize("mode", ["extra", "derived"])
def test_fit_season_with_pieces(table, mode):
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig
    from scripts.train_hitter_ros import fit_season

    config = NetConfig(
        hidden=[4],
        max_epochs=2,
        probes="none",
        milb="none",
        parks="none",
        steal_inputs=False,
        val_frac=0.0,
        avg_pieces=mode,
    )
    # The test seasons are too short to reach 100 PA: next 25 PA only.
    y_all, w_all = target_frame(table, (25,), pieces=True)
    preds, _ = fit_season(table, input_frame(table), y_all, w_all, 2025, config)
    assert list(preds.columns) == ["player_id", "season", "week", "as_of", *y_all.columns]
    known = preds.dropna(subset=list(PIECES))
    assert not known.empty
    for tag in ("", "n25_"):
        k = preds[[f"{tag}{p}" for p in PIECES]]
        assert ((k.stack() > 0) & (k.stack() < 1)).all()  # rates from log-odds
        derived = avg_from_pieces(*(preds[f"{tag}{p}"] for p in PIECES))
        if mode == "derived":
            pd.testing.assert_series_equal(preds[f"{tag}avg"], derived, check_names=False)
        else:
            assert not np.allclose(preds[f"{tag}avg"], derived)  # its own output
