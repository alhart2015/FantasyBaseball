import numpy as np
import pandas as pd
import pytest

from fantasy_baseball import blend_fit
from fantasy_baseball.draft import our_hitters as oh
from fantasy_baseball.hitter_ros import final
from fantasy_baseball.hitter_ros.evaluate import squared_error


def _fg():
    # Three board hitters: 1 and 2 ours projects, 3 it doesn't.
    return pd.DataFrame(
        {
            "fg_id": ["a", "b", "c"],
            "name": ["One", "Two", "Three"],
            "pa": [600.0, 400.0, 500.0],
            "ab": [540.0, 360.0, 450.0],
            "h": [162.0, 90.0, 117.0],
            "r": [90.0, 50.0, 60.0],
            "hr": [30.0, 10.0, 15.0],
            "rbi": [95.0, 45.0, 55.0],
            "sb": [10.0, 25.0, 5.0],
            "avg": [0.300, 0.250, 0.260],
        }
    )


def _ours():
    return pd.DataFrame(
        {
            "r": [0.12, 0.14],
            "hr": [0.06, 0.02],
            "rbi": [0.15, 0.10],
            "sb": [0.01, 0.08],
            "avg": [0.280, 0.270],
        },
        index=pd.Index([10, 20], name="player_id"),
    )


def test_ours_reshapes_the_rates_and_fangraphs_keeps_the_playing_time():
    fg = _fg()
    out = oh.blend_board_hitters(fg, _ours(), pd.Series({"a": 10, "b": 20}))
    w = final.WEIGHTS
    hr = w["hr"] * 0.06 + (1 - w["hr"]) * 30 / 600
    assert out.loc[0, "hr"] == pytest.approx(hr * 600)
    avg = w["avg"] * 0.280 + (1 - w["avg"]) * 162 / 540
    assert out.loc[0, "avg"] == pytest.approx(avg) and out.loc[0, "h"] == pytest.approx(avg * 540)
    sb = w["sb"] * 0.08 + (1 - w["sb"]) * 25 / 400
    assert out.loc[1, "sb"] == pytest.approx(sb * 400)
    assert out["pa"].tolist() == fg["pa"].tolist() and out["ab"].tolist() == fg["ab"].tolist()
    # A hitter ours doesn't project keeps the FanGraphs line, and the input is untouched.
    pd.testing.assert_series_equal(
        out.loc[2], fg.loc[2].astype(out.loc[2].dtype), check_names=False
    )
    pd.testing.assert_frame_equal(fg, _fg())


def test_an_unmatched_board_is_fangraphs_as_is():
    out = oh.blend_board_hitters(_fg(), _ours(), pd.Series({"x": 10}))
    pd.testing.assert_frame_equal(out, _fg())


def test_two_board_rows_on_one_mlbam_id_are_refused():
    with pytest.raises(ValueError, match="share an MLBAM id"):
        oh.blend_board_hitters(_fg(), _ours(), pd.Series({"a": 10, "b": 10}))


def _scored(w_true, n=400, seed=0):
    """Hitter scored rows whose actuals are exactly ``w_true`` x ours + rest x FanGraphs."""
    rng = np.random.default_rng(seed)
    pa = rng.uniform(300, 700, n)
    ab = pa * 0.9
    rows = []
    for stat, base in (("r", 0.13), ("hr", 0.04), ("rbi", 0.13), ("sb", 0.02), ("avg", 0.26)):
        ours = base * rng.uniform(0.7, 1.3, n)
        fg = base * rng.uniform(0.7, 1.3, n)
        actual = w_true[stat] * ours + (1 - w_true[stat]) * fg
        for system, proj in ((blend_fit.OURS, ours), (blend_fit.FG, fg)):
            rows.append(
                pd.DataFrame(
                    {
                        "player_id": range(n),
                        "season": 2025,
                        "system": system,
                        "stat": stat,
                        "projected": proj,
                        "actual": actual,
                        "pa": pa,
                        "ab": ab,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_hitter_fit_finds_the_mix_the_actuals_were_made_from():
    w_true = {"r": 0.55, "hr": 0.4, "rbi": 0.65, "sb": 0.15, "avg": 0.45}
    got = blend_fit.best_weights(_scored(w_true), final.SCORING)
    assert got.to_dict() == pytest.approx(w_true)


def test_hitter_error_weight_is_what_squared_error_puts_on_each_squared_miss():
    rows = blend_fit.wide(_scored(dict.fromkeys(("r", "hr", "rbi", "sb", "avg"), 0.5)))
    err = (rows[blend_fit.OURS] - rows["actual"]).to_numpy()
    sq = squared_error(rows.assign(projected=rows[blend_fit.OURS]), ["stat"])["sq"].to_numpy()
    assert sq == pytest.approx(final.error_weight(rows) * err**2)


def test_the_weights_in_use_cover_every_stat_on_the_grid():
    assert set(final.WEIGHTS) == {"r", "hr", "rbi", "sb", "avg"}
    assert all(w in blend_fit.GRID for w in final.WEIGHTS.values())
