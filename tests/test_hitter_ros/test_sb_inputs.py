"""#451: SB predicted by a second net that reads only box-score inputs."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.features import (
    BLEND_PA,
    BOX_RATES,
    WINDOWS,
    box_score_inputs,
    input_frame,
    target_frame,
)
from fantasy_baseball.hitter_ros.table import build_table
from tests.test_hitter_ros.test_sb_pieces import _with_steals
from tests.test_hitter_ros.test_table import _season, _write


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    seasons = {2024: _season(2024, date(2024, 4, 1), 10), 2025: _season(2025, date(2025, 4, 1), 21)}
    _write(root, {year: _with_steals(s) for year, s in seasons.items()})
    return build_table(root)


def test_box_score_inputs_are_blends_rates_and_volumes_only(table):
    x = box_score_inputs(table)
    blends = [c for c in x.columns if c.startswith("bl")]
    assert len(blends) == 12 * len(BLEND_PA)
    rates = [f"{w}_{name}" for _, _, name in BOX_RATES for w in WINDOWS]
    volumes = [f"{w}_log_{n}" for w in WINDOWS for n in ("pa", "ab")]
    assert list(x.columns) == [*blends, *rates, *volumes, "week", "frac_season_left", "age"]
    # None of the full net's batted-ball, speed or team inputs.
    full = input_frame(table, steal=True)
    for c in ("std_xba_con", "p1_sprint_speed", "std_team_r_pa"):
        assert c in full.columns and c not in x.columns, c


def test_box_score_rates_are_counts_over_trials(table):
    x = box_score_inputs(table)
    t = table
    played = t.std_pa > 0
    assert played.any() and (~played).any()
    np.testing.assert_allclose(x.loc[played, "std_hr"], (t.std_hr / t.std_pa)[played])
    batted = t.p1_ab > 0
    np.testing.assert_allclose(x.loc[batted, "p1_avg"], (t.p1_h / t.p1_ab)[batted])
    np.testing.assert_allclose(x["car_log_pa"], np.log1p(t.car_pa))
    assert x.loc[~played, "std_hr"].isna().all()  # no PA: unknown, not 0


def test_box_score_inputs_ignore_every_ros_column(table):
    changed = table.copy()
    for c in changed.columns:
        if c.startswith("ros_"):
            changed[c] = changed[c] * 7 + 3
    pd.testing.assert_frame_equal(box_score_inputs(table), box_score_inputs(changed))


def test_sb_inputs_setting():
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig

    assert NetConfig(sb_inputs="box").sb_inputs == "box"
    with pytest.raises(ValueError, match="sb_inputs"):
        NetConfig(sb_inputs="simple")


def test_with_sb_from_takes_every_sb_column_from_the_sb_net():
    pytest.importorskip("torch")
    from scripts.train_hitter_ros import with_sb_from

    idx = pd.Index([3, 5, 8])
    cols = ["player_id", "r", "sb", "avg", "n25_sb", "n25_r", "opp_pa", "n25_sb_att", "k_ab"]
    main = pd.DataFrame(1.0, index=idx, columns=cols)
    sb_net = pd.DataFrame(2.0, index=idx, columns=cols)
    got = with_sb_from(main, sb_net)
    from_sb = {"sb", "n25_sb", "opp_pa", "n25_sb_att"}
    for c in cols:
        assert (got[c] == (2.0 if c in from_sb else 1.0)).all(), c
    assert (main == 1.0).all().all()  # the main net's frame is left alone
    with pytest.raises(ValueError, match="different rows"):
        with_sb_from(main, sb_net.iloc[:2])


def test_an_sb_net_fits_on_box_score_inputs(table):
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig
    from scripts.train_hitter_ros import fit_season

    config = NetConfig(
        hidden=[4],
        max_epochs=2,
        probes="none",
        milb="none",
        parks="none",
        val_frac=0.0,
    )
    # The test seasons are too short to reach 100 PA: next 25 PA only.
    y_all, w_all = target_frame(table, (25,), pieces=True)
    preds, _ = fit_season(table, box_score_inputs(table), y_all, w_all, 2025, config)
    assert len(preds) == (table.season == 2025).sum()
    assert preds[["sb", "n25_sb"]].notna().all().all() and (preds["sb"] >= 0).all()
