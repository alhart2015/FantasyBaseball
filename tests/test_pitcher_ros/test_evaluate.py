import logging

import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros.evaluate import (
    blend,
    level_free,
    level_free_bootstrap,
    level_free_table,
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


def test_load_systems_refuses_two_files_for_one_system(tmp_path):
    _csv(tmp_path / "zips-pitchers.csv", ["A,10,0,180,750,70,150,50,200,3.5,1.11,1"])
    _csv(tmp_path / "zips-pitchers-old.csv", ["A,9,0,170,720,70,150,50,190,3.7,1.15,1"])
    with pytest.raises(ValueError, match="more than one zips"):
        load_systems(tmp_path)


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
    assert b.loc["k", "diff"] < 0 and b.loc["k", "sure"] > 0.975
    assert b.loc["k", "lo"] <= b.loc["k", "diff"] <= b.loc["k", "hi"]


def _two_seasons(systems, groups=None):
    """Five pitchers scored by ``systems`` (name -> function of the actual rates) in
    2024 and 2025; ``groups`` tags each pitcher vet/rookie."""
    rows = [
        {"player_id": p, "k": 0.15 + p / 100, "w": 0.01, "sv": 0.0, "era": 3.0 + p / 10,
         "whip": 1.3, "bf": 100.0 * p, "ip": 25.0 * p}
        for p in range(1, 7)
    ]  # fmt: skip
    actual = _actual(rows)
    projs = {name: f(actual[list(TARGETS)]) for name, f in systems.items()}
    scored = pd.concat(
        [scored_players(projs, actual, 100).assign(season=y) for y in (2024, 2025)],
        ignore_index=True,
    )
    if groups is not None:
        scored["group"] = scored["player_id"].map(groups)
    return scored


def test_level_free_is_the_best_score_with_the_league_level_put_right():
    scored = level_free(
        _two_seasons(
            {
                "level": lambda a: a * 1.1,  # right about everyone but the league's level
                "flat": lambda a: a * 0 + a.mean(),  # the same rate for everyone
                "exact": lambda a: a,
            }
        )
    )
    raw, lf = mse_table(scored), level_free_table(scored)
    assert raw.loc["level", "k"] > 0 and lf.loc["level", "k"] == pytest.approx(0)
    assert lf.loc["level", "era"] == pytest.approx(0)
    assert lf.loc["flat", "k"] > 0  # a ranking miss stays
    assert (lf[list(TARGETS)] <= raw[list(TARGETS)] + 1e-12).all().all()  # never above raw
    # The factor minimizes MSE: nudging it either way scores worse.
    flat = scored[scored.system == "flat"]
    for nudge in (0.98, 1.02):
        worse = mse_table(flat.assign(projected=flat.lf_projected * nudge))
        assert worse.loc["flat", "k"] > lf.loc["flat", "k"]


def test_a_subgroup_level_free_score_keeps_its_own_level_miss():
    """The level is fitted over everyone scored in a season: vets 10% high and rookies
    10% low cancel league-wide, so neither group's own miss is taken out."""
    vets = {1: "vet", 2: "rookie", 3: "vet", 4: "rookie", 5: "vet", 6: "rookie"}
    scored = level_free(
        _two_seasons(
            {"split": lambda a: a.mul([1.1, 0.9, 1.1, 0.9, 1.1, 0.9], axis=0)}, groups=vets
        )
    )
    vet_rows = scored[scored.group == "vet"]
    assert level_free_table(vet_rows).loc["split", "k"] > 0
    # Fitted on the vets alone, their level miss would vanish.
    assert level_free_table(level_free(vet_rows)).loc["split", "k"] == pytest.approx(0)


def test_level_free_bootstrap_refits_the_level_and_matches_the_table():
    scored = level_free(
        _two_seasons(
            {
                "level": lambda a: a * 1.1,
                "noisy": lambda a: a.mul([1.2, 0.8, 1.1, 0.9, 1.0, 1.05], axis=0),
            }
        )
    )
    vets = scored["player_id"].isin([1, 3, 5])
    b = level_free_bootstrap(scored, "level", "noisy", vets, n_boot=200)
    sub = level_free_table(scored[vets])
    assert b.loc["k", "a"] == pytest.approx(sub.loc["level", "k"])
    assert b.loc["k", "b"] == pytest.approx(sub.loc["noisy", "k"])
    assert b.loc["k", "lo"] <= b.loc["k", "diff"] <= b.loc["k", "hi"]
    assert b.loc["k", "diff"] < 0 and 0 <= b.loc["k", "sure"] <= 1


def test_level_free_refuses_rows_with_no_season():
    scored = _two_seasons({"exact": lambda a: a})
    scored.loc[0, "season"] = None
    with pytest.raises(ValueError, match="no season"):
        level_free(scored)
