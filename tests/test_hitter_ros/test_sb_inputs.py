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


def _league_steals_scaled(table, factor):
    """The table with every SB count, the player's and the league's, in every window
    times ``factor``: a league-wide jump in steals (the 2023 rules)."""
    changed = table.copy()
    for c in changed.columns:
        if c.endswith("_sb") and not c.startswith("ros_") and "team" not in c:
            changed[c] = changed[c].astype(float) * factor
    return changed


def test_relative_box_inputs_ignore_a_league_wide_jump_in_steals(table):
    jumped = _league_steals_scaled(table, 1.4)
    sb_cols = [
        c for c in box_score_inputs(table).columns if c.endswith("_sb") or c.endswith("sb_pa")
    ]
    assert len(sb_cols) == len(WINDOWS) + len(BLEND_PA)
    raw, raw_jumped = box_score_inputs(table)[sb_cols], box_score_inputs(jumped)[sb_cols]
    rel = box_score_inputs(table, relative=True)[sb_cols]
    rel_jumped = box_score_inputs(jumped, relative=True)[sb_cols]
    known = rel.notna() & (raw > 0)
    assert known.to_numpy().any()
    # Raw rates rise with the league; league-relative ones don't move.
    np.testing.assert_allclose(raw_jumped[known], 1.4 * raw[known])
    np.testing.assert_allclose(rel_jumped[known], rel[known])


def test_relative_box_rates_are_the_players_over_the_leagues(table):
    x = box_score_inputs(table, relative=True)
    t = table
    played = t.std_pa > 0
    league = t.lg_std_hr / t.lg_std_pa
    np.testing.assert_allclose(x.loc[played, "std_hr"], (t.std_hr / t.std_pa / league)[played])
    # Multi-season windows: the league over his own seasons (lgp), not the pool.
    known = t.car_pa > 0
    np.testing.assert_allclose(
        x.loc[known, "car_sb"], (t.car_sb / t.car_pa / (t.lgp_car_sb / t.lgp_car_pa))[known]
    )
    batted = t.p1_ab > 0
    np.testing.assert_allclose(
        x.loc[batted, "p1_avg"], (t.p1_h / t.p1_ab / (t.lg_p1_h / t.lg_p1_ab))[batted]
    )
    # Same columns, and the volumes and context are untouched.
    raw = box_score_inputs(table)
    assert list(x.columns) == list(raw.columns)
    for c in ("car_log_pa", "week", "age"):
        pd.testing.assert_series_equal(x[c], raw[c])


def test_relative_blends_are_one_for_a_league_average_hitter(table):
    """A hitter whose every window matches his league's rates blends to exactly 1."""
    avg = table.copy()
    for w, lg in (("std", "lg_std"), ("p1", "lg_p1"), ("p3", "lgp_p3")):
        for n in ("h", "r", "hr", "rbi", "sb", "bb", "k", "cs", "steal_opp2", "steal_opp3"):
            per = "ab" if n in ("h", "k") else "pa"
            avg[f"{w}_{n}"] = avg[f"{lg}_{n}"] / avg[f"{lg}_{per}"] * avg[f"{w}_{per}"]
    x = box_score_inputs(avg, relative=True)
    blends = x[[f"bl{k}_{n}" for k in BLEND_PA for n in ("r_pa", "hr_pa", "sb_pa", "avg")]]
    known = blends.notna()
    assert known.to_numpy().any()
    np.testing.assert_allclose(blends[known].to_numpy()[known.to_numpy()], 1.0)


# Per-PA league rates for the synthetic world below (SB set per season).
_RATES = {
    "ab": 0.9,
    "h": 0.22,
    "r": 0.12,
    "hr": 0.03,
    "rbi": 0.115,
    "bb": 0.08,
    "k": 0.22,
    "cs": 0.005,
    "steal_opp2": 0.1,
    "steal_opp3": 0.03,
}
_COUNTS = ("pa", *_RATES, "sb")


def _season_counts(pa, sb_rate, sb_factor=1.0):
    return {"pa": pa, **{c: r * pa for c, r in _RATES.items()}, "sb": sb_factor * sb_rate * pa}


def _step_world(jump):
    """One hitter's season-2026 row (week 10), built from season totals. He debuted in
    2024 and steals at 2x the league's rate every season. The league steals 0.02 per PA
    from 2024 on; with ``jump``, it stole 0.013 before that (a 2023-style rules jump
    that came before he arrived), else 0.02 throughout."""
    seasons = range(2020, 2026)
    league = {y: _season_counts(180_000, 0.013 if jump and y < 2024 else 0.02) for y in seasons}
    his = {y: _season_counts(600, 0.02, sb_factor=2.0) for y in (2024, 2025)}
    std, lg_std = _season_counts(300, 0.02, sb_factor=2.0), _season_counts(90_000, 0.02)
    windows = {"p1": [2025], "p3": [2023, 2024, 2025], "car": list(seasons)}
    row = {"week": 10.0, "frac_season_left": 0.6, "age": 25.0}
    for c in _COUNTS:
        row[f"std_{c}"], row[f"lg_std_{c}"] = std[c], lg_std[c]
        for w, ys in windows.items():
            row[f"{w}_{c}"] = sum(his[y][c] for y in ys if y in his)
            row[f"lg_{w}_{c}"] = sum(league[y][c] for y in ys)
            row[f"lgp_{w}_{c}"] = sum(
                his[y]["pa"] / league[y]["pa"] * league[y][c] for y in ys if y in his
            )
    return pd.DataFrame([row])


def test_relative_inputs_ignore_a_league_jump_before_his_seasons():
    """The review's case (#413): every one of his seasons came after the jump, so his
    inputs must not depend on it. The pooled multi-season league (lg_p3, lg_car) still
    holds the pre-jump seasons and would read him as beating it by more than 2x."""
    before, after = (
        box_score_inputs(_step_world(False), True),
        box_score_inputs(_step_world(True), True),
    )
    sb = [c for c in before.columns if c.endswith("_sb") or c.endswith("sb_pa")]
    pd.testing.assert_frame_equal(before[sb], after[sb])
    for w in WINDOWS:
        assert after.loc[0, f"{w}_sb"] == pytest.approx(2.0), w
    pooled = _step_world(True)
    pooled_car = (pooled.car_sb / pooled.car_pa) / (pooled.lg_car_sb / pooled.lg_car_pa)
    assert pooled_car.iloc[0] > 2.5  # what the pooled league read


def test_box_relative_needs_a_relative_target(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    from scripts import train_hitter_ros

    monkeypatch.setattr(train_hitter_ros, "RUNS", tmp_path / "runs")
    monkeypatch.setattr("sys.argv", ["train", "--name", "x", "--relative-target", "none"])
    with pytest.raises(SystemExit):
        train_hitter_ros.main()
    assert "--sb-inputs box" in capsys.readouterr().err


def test_sb_inputs_setting():
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig

    assert NetConfig(sb_inputs="box").sb_inputs == "box"
    assert NetConfig(sb_inputs="box_relative").sb_inputs == "box_relative"
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


def test_a_sequence_run_needs_sb_inputs_full(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    from scripts import train_hitter_ros

    tokens = tmp_path / "tokens.parquet"
    tokens.write_text("")
    monkeypatch.setattr(train_hitter_ros, "TOKENS", tokens)
    monkeypatch.setattr(train_hitter_ros, "RUNS", tmp_path / "runs")
    monkeypatch.setattr("sys.argv", ["train", "--name", "x", "--seq", "gru"])
    with pytest.raises(SystemExit):
        train_hitter_ros.main()
    assert "--sb-inputs full" in capsys.readouterr().err


def test_sb_parts_are_the_targets_definitions(table):
    from fantasy_baseball.hitter_ros.features import (
        piece_parts,
        piece_rates,
        sb_piece_parts,
        sb_piece_rates,
    )

    counts = table[[f"std_{c}" for c in ("pa", "ab", "h", "hr", "k", "sb", "cs")]]
    counts = counts.join(table[["std_steal_opp2", "std_steal_opp3"]])
    counts.columns = [c.removeprefix("std_") for c in counts.columns]
    opp = counts.steal_opp2 + counts.steal_opp3
    parts = sb_piece_parts(counts)
    pd.testing.assert_series_equal(parts["att_opp"][0], counts.sb + counts.cs, check_names=False)
    pd.testing.assert_series_equal(parts["att_opp"][1], opp.astype(float), check_names=False)
    rates = sb_piece_rates(counts)
    known = opp > 0
    assert known.any()
    np.testing.assert_allclose(rates.loc[known, "att_opp"], ((counts.sb + counts.cs) / opp)[known])
    bip = counts.ab - counts.k - counts.hr
    pd.testing.assert_series_equal(
        piece_parts(counts)["babip"][1], bip.astype(float), check_names=False
    )
    hit = bip > 0
    np.testing.assert_allclose(
        piece_rates(counts).loc[hit, "babip"], ((counts.h - counts.hr) / bip)[hit]
    )
