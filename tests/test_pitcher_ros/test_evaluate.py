import logging

import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros.evaluate import (
    blend,
    load_fangraphs_pitchers,
    load_systems,
    mse_bootstrap,
    mse_table,
    scored_players,
)
from fantasy_baseball.pitcher_ros.stats import TARGETS

HEADER = "Name,W,SV,IP,TBF,ER,H,BB,SO,ERA,WHIP,MLBAMID"


def _csv(path, rows):
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def test_load_fangraphs_pitchers_rates_per_batter_faced(tmp_path):
    p = _csv(tmp_path / "steamer-pitchers.csv", ["A,10,0,180,750,70,150,50,200,3.5,1.11,1"])
    row = load_fangraphs_pitchers(p).loc[1]
    assert row["k"] == pytest.approx(200 / 750) and row["w"] == pytest.approx(10 / 750)
    assert row["era"] == 3.5 and row["whip"] == 1.11  # the file's own
    assert row["bf"] == 750 and row["ip"] == 180


def test_load_systems_leaves_out_files_missing_a_stat(tmp_path, caplog):
    _csv(tmp_path / "steamer-pitchers.csv", ["A,10,0,180,750,70,150,50,200,3.5,1.11,1"])
    _csv(tmp_path / "zips-pitchers.csv", ["A,10,,180,750,70,150,50,200,3.5,1.11,1"])  # no SV
    _csv(tmp_path / "atc-pitchers.csv", [])  # failed download
    with caplog.at_level(logging.WARNING):
        systems = load_systems(tmp_path)
    assert set(systems) == {"steamer"}
    assert "no sv projections" in caplog.text


def test_load_systems_refuses_a_rest_of_season_file_as_preseason(tmp_path):
    _csv(tmp_path / "steamer-pitchers.csv", ["A,4,0,80,330,30,70,20,90,3.4,1.12,1"])
    with pytest.raises(ValueError, match="rest-of-season"):
        load_systems(tmp_path, preseason=True)
    assert set(load_systems(tmp_path)) == {"steamer"}


def _proj(**rates):
    base = {"k": 0.25, "w": 0.01, "sv": 0.0, "era": 4.0, "whip": 1.2}
    return pd.DataFrame([{**base, **rates}], index=pd.Index([1], name="player_id"))


def test_blend_averages_rates():
    b = blend({"a": _proj(k=0.2), "b": _proj(k=0.3, era=3.0)})
    assert b.loc[1, "k"] == pytest.approx(0.25) and b.loc[1, "era"] == pytest.approx(3.5)


def _actual(rows):
    return pd.DataFrame(rows).set_index("player_id")


def test_mse_is_totals_on_actual_bf_and_ip_weighted_era():
    actual = _actual(
        [
            {"player_id": 1, "k": 0.3, "w": 0.01, "sv": 0.0, "era": 5.0, "whip": 1.2,
             "bf": 400.0, "ip": 100.0},
            {"player_id": 2, "k": 0.25, "w": 0.01, "sv": 0.0, "era": 3.0, "whip": 1.2,
             "bf": 200.0, "ip": 50.0},
        ]
    )  # fmt: skip
    proj = pd.DataFrame(
        {"k": [0.25, 0.25], "w": [0.01] * 2, "sv": [0.0] * 2, "era": [4.0, 4.0], "whip": [1.2] * 2},
        index=pd.Index([1, 2], name="player_id"),
    )
    t = mse_table(scored_players({"x": proj}, actual, min_bf=100))
    # K: player 1 off by .05 x 400 = 20 K, player 2 exact.
    assert t.loc["x", "k"] == pytest.approx((20**2 + 0) / 2)
    # ERA: both off by 1 run; weighted by 100 and 50 IP.
    assert t.loc["x", "era"] == pytest.approx((100 * 1 + 50 * 1) / 150)
    assert t.loc["x", "n"] == 2


def test_scored_players_keeps_pitchers_every_system_covers_with_enough_bf():
    actual = _actual(
        [
            {"player_id": p, "k": 0.2, "w": 0.0, "sv": 0.0, "era": 4.0, "whip": 1.3,
             "bf": bf, "ip": bf / 4}
            for p, bf in ((1, 300.0), (2, 50.0), (3, 300.0))
        ]
    )  # fmt: skip
    proj = pd.DataFrame(
        {s: [0.2, 0.2, 0.2] for s in TARGETS}, index=pd.Index([1, 2, 3], name="player_id")
    )
    partial = proj.drop(index=3)
    scored = scored_players({"a": proj, "b": partial}, actual, min_bf=100)
    assert set(scored["player_id"]) == {1}  # 2: too few BF; 3: b has no projection


def test_mse_bootstrap_signs():
    rows = [
        {"player_id": p, "k": 0.2 + p / 100, "w": 0.01, "sv": 0.0, "era": 4.0, "whip": 1.3,
         "bf": 300.0, "ip": 70.0}
        for p in range(1, 41)
    ]  # fmt: skip
    actual = _actual(rows)
    good = actual[list(TARGETS)] + 0.001
    bad = actual[list(TARGETS)] + 0.05
    b = mse_bootstrap(scored_players({"good": good, "bad": bad}, actual, 100), "good", "bad")
    assert b.loc["k", "diff"] < 0 and b.loc["k", "sure"] > 0.95
    assert b.loc["k", "lo"] <= b.loc["k", "diff"] <= b.loc["k", "hi"]
