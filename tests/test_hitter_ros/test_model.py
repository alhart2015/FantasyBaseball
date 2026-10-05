from datetime import date

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.evaluate import (
    blend,
    load_fangraphs_hitters,
    mae_table,
    scored_players,
)
from fantasy_baseball.hitter_ros.features import Standardizer, input_frame, target_frame
from fantasy_baseball.hitter_ros.table import build_table
from tests.test_hitter_ros.test_table import _season, _write


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    _write(
        root, {2024: _season(2024, date(2024, 4, 1), 10), 2025: _season(2025, date(2025, 4, 1), 21)}
    )
    return build_table(root)


def test_inputs_ignore_every_ros_column(table):
    changed = table.copy()
    for c in changed.columns:
        if c.startswith("ros_"):
            changed[c] = changed[c] * 7 + 3
    pd.testing.assert_frame_equal(input_frame(table), input_frame(changed))


def test_input_rates(table):
    row = table[(table.player_id == 1) & (table.season == 2025) & (table.week == 1)]
    x = input_frame(row).iloc[0]
    assert x["std_hr_pa"] == pytest.approx(7 / 28)
    assert x["std_whiff_rate"] == pytest.approx(0.5)
    assert x["std_ev_mean"] == pytest.approx(100.0)
    assert x["std_ev_sd"] == pytest.approx(0.0)
    assert x["std_avg"] == pytest.approx(0.5)
    # Week 0 has no season to date: the rate is unknown, not zero.
    w0 = table[(table.player_id == 1) & (table.season == 2025) & (table.week == 0)]
    assert np.isnan(input_frame(w0).iloc[0]["std_hr_pa"])


def test_targets_and_weights(table):
    row = table[(table.player_id == 1) & (table.season == 2025) & (table.week == 1)]
    y, w = target_frame(row)
    assert y.iloc[0]["hr"] == pytest.approx(14 / 56)
    assert y.iloc[0]["avg"] == pytest.approx(0.5)
    assert w.iloc[0]["hr"] == 56 and w.iloc[0]["avg"] == 56


def test_standardizer_fills_unknowns_with_the_mean_and_flags_them():
    train = pd.DataFrame({"a": [1.0, 3.0, np.nan], "b": [2.0, 2.0, 2.0]})
    s = Standardizer().fit(train)
    z = s.transform(pd.DataFrame({"a": [np.nan, 3.0], "b": [2.0, 2.0]}))
    assert s.n_features == 3 == z.shape[1]
    assert z[0, 0] == 0.0 and z[0, 2] == 1.0  # unknown -> mean, flagged
    assert z[1, 0] > 0 and z[1, 2] == 0.0
    assert z[0, 1] == 0.0  # constant column does not divide by zero


def _rates(**cols):
    return pd.DataFrame(cols).set_index("player_id")


def test_score_uses_only_shared_players_with_enough_pa():
    actual = _rates(
        player_id=[1, 2, 3],
        r=[0.1, 0.1, 0.1],
        hr=[0.05, 0.05, 0.05],
        rbi=[0.1] * 3,
        sb=[0.0] * 3,
        avg=[0.300, 0.300, 0.300],
        pa=[500, 500, 50],
    )
    ours = _rates(
        player_id=[1, 2, 3],
        r=[0.11, 0.09, 0.5],
        hr=[0.05] * 3,
        rbi=[0.1] * 3,
        sb=[0.0] * 3,
        avg=[0.310, 0.290, 0.0],
    )
    theirs = _rates(
        player_id=[1, 3],
        r=[0.1, 0.1],
        hr=[0.06, 0.05],
        rbi=[0.1] * 2,
        sb=[0.0] * 2,
        avg=[0.300, 0.300],
    )
    t = mae_table(scored_players({"ours": ours, "theirs": theirs}, actual, min_pa=300))
    # Only player 1: covered by both, and player 3 is under the PA floor.
    assert t.loc["ours", "n"] == 1
    assert t.loc["ours", "r"] == pytest.approx(0.01 * 600)
    assert t.loc["ours", "avg"] == pytest.approx(10.0)
    assert t.loc["theirs", "hr"] == pytest.approx(0.01 * 600)


def test_fangraphs_loader_and_blend(tmp_path):
    csv = tmp_path / "steamer-hitters.csv"
    pd.DataFrame(
        {
            "Name": ["A", "B", "B dup", "No id"],
            "PA": [600, 300, 1, 100],
            "AB": [500, 250, 1, 90],
            "H": [150, 50, 0, 20],
            "R": [90, 30, 0, 10],
            "HR": [30, 6, 0, 2],
            "RBI": [90, 30, 0, 8],
            "SB": [12, 3, 0, 0],
            "MLBAMID": [11, 22, 22, None],
        }
    ).to_csv(csv, index=False, encoding="utf-8-sig")
    rates = load_fangraphs_hitters(csv)
    assert list(rates.index) == [11, 22]
    assert rates.loc[11, "hr"] == pytest.approx(30 / 600)
    assert rates.loc[11, "avg"] == pytest.approx(0.300)
    other = rates.copy()
    other["hr"] = other["hr"] * 3
    assert blend({"a": rates, "b": other}).loc[11, "hr"] == pytest.approx(2 * 30 / 600)


def test_net_learns_a_simple_rule_and_is_repeatable():
    torch = pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig, predict, train

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 4)).astype(np.float32)
    y = np.column_stack([x[:, 0] * 2, x[:, 1] - x[:, 2]]).astype(np.float32)
    w = np.ones_like(y)
    val = rng.random(2000) < 0.2
    # Synthetic rows have no season time; pin plain PA weighting (not the pre_mid default).
    config = NetConfig(
        hidden=[32],
        dropout=0.0,
        lr=1e-2,
        batch_size=256,
        max_epochs=60,
        patience=10,
        weighting="pa",
    )

    first = train(x, y, w, val, config)
    assert first.val_loss[first.best_epoch] < 0.05 * first.val_loss[0]
    again = train(x, y, w, val, config)
    if not torch.cuda.is_available():  # GPU kernels are not bit-for-bit deterministic
        np.testing.assert_allclose(predict(first.model, x), predict(again.model, x))


def test_standardizer_drops_inputs_never_seen_in_training():
    # Bat speed starts in 2023: training on earlier seasons never sees it.
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "bat_speed": [np.nan] * 3})
    s = Standardizer().fit(train)
    z = s.transform(pd.DataFrame({"a": [2.0], "bat_speed": [72.0]}))
    assert s.columns == ["a"] and s.missing_cols == []
    assert z.shape == (1, 1) and s.n_features == 1


def test_score_drops_players_any_projection_cannot_rate():
    actual = _rates(
        player_id=[1, 2],
        r=[0.1, 0.1],
        hr=[0.05] * 2,
        rbi=[0.1] * 2,
        sb=[0.0] * 2,
        avg=[0.3] * 2,
        pa=[500, 500],
    )
    ours = _rates(
        player_id=[1, 2], r=[0.11, 0.2], hr=[0.05] * 2, rbi=[0.1] * 2, sb=[0.0] * 2, avg=[0.3] * 2
    )
    # A 0-PA projection loads as NaN rates: that player leaves the shared sample.
    theirs = _rates(
        player_id=[1, 2],
        r=[0.1, np.nan],
        hr=[0.05, np.nan],
        rbi=[0.1, np.nan],
        sb=[0.0, np.nan],
        avg=[0.3, np.nan],
    )
    t = mae_table(scored_players({"ours": ours, "theirs": theirs}, actual, min_pa=300))
    assert t.loc["ours", "n"] == 1
    assert t.loc["ours", "r"] == pytest.approx(0.01 * 600)


def test_preseason_loader_refuses_a_rest_of_season_file(tmp_path):
    from fantasy_baseball.hitter_ros.evaluate import load_systems

    pd.DataFrame(
        {
            "PA": [120],
            "AB": [100],
            "H": [25],
            "R": [15],
            "HR": [4],
            "RBI": [14],
            "SB": [2],
            "MLBAMID": [1],
        }
    ).to_csv(tmp_path / "steamer-hitters.csv", index=False)
    assert set(load_systems(tmp_path)) == {"steamer"}
    with pytest.raises(ValueError, match="rest-of-season"):
        load_systems(tmp_path, preseason=True)


def test_net_fails_loudly_on_a_nan_loss():
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig, train

    x = np.full((50, 2), np.nan, dtype=np.float32)
    y = np.zeros((50, 1), dtype=np.float32)
    val = np.arange(50) < 10
    with pytest.raises(FloatingPointError, match="validation loss"):
        train(x, y, np.ones_like(y), val, NetConfig(hidden=[4], max_epochs=3, weighting="pa"))


def test_micro_batches_give_the_full_batch_gradient():
    torch = pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import MLP, accumulate_batch, weighted_mse

    torch.manual_seed(0)
    x = torch.randn(64, 5)
    y = torch.randn(64, 2)
    w = torch.rand(64, 2)
    idx = torch.arange(64)
    model = MLP(5, 2, [8], dropout=0.0)

    model.zero_grad()
    weighted_mse(model(x), y, w).backward()
    full = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    loss = accumulate_batch(model, idx, x, y, w, None, None, micro_batch=10)
    sliced = [p.grad.clone() for p in model.parameters()]

    assert loss == pytest.approx(weighted_mse(model(x), y, w).item(), rel=1e-5)
    for a, b in zip(full, sliced, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_balanced_weighting_gives_each_part_of_the_season_equal_weight():
    from fantasy_baseball.hitter_ros.features import SEASON_PARTS, balance_by_season_time

    frac_left = pd.Series([1.0, 0.95, 0.5, 0.1, 0.05])  # early, early, middle, late, late
    w = pd.DataFrame({"hr": [600.0, 500.0, 300.0, 40.0, 20.0], "avg": [5.0, 5.0, 5.0, 5.0, 5.0]})
    out = balance_by_season_time(w, frac_left)
    part = np.minimum(((1 - frac_left) * SEASON_PARTS).astype(int), SEASON_PARTS - 1)
    totals = out["hr"].groupby(part).sum()
    assert np.allclose(totals, totals.iloc[0])  # every season part present: same total
    assert out["hr"].sum() == pytest.approx(w["hr"].sum())
    # Within a part, PA still matters: the 40-PA row outweighs the 20-PA row 2:1.
    assert out["hr"][3] == pytest.approx(2 * out["hr"][4])


def test_balanced_weighting_is_applied_by_train_itself():
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig, train

    x = np.zeros((20, 2), dtype=np.float32)
    y = np.zeros((20, 1), dtype=np.float32)
    w = np.ones((20, 1), dtype=np.float32)
    val = np.arange(20) < 4
    with pytest.raises(ValueError, match="frac_season_left"):
        train(x, y, w, val, NetConfig(hidden=[4], weighting="balanced", max_epochs=1))
    train(
        x,
        y,
        w,
        val,
        NetConfig(hidden=[4], weighting="balanced", max_epochs=1),
        season_time=np.linspace(0, 1, 20),
    )


def test_era_inputs(table):
    row = table[(table.player_id == 1) & (table.season == 2025) & (table.week == 1)]
    x = input_frame(row, era="full").iloc[0]
    assert x["lg_std_hr_pa"] == pytest.approx(7 / 57)
    assert x["std_hr_pa_vs_lg"] == pytest.approx((7 / 28) / (7 / 57))
    assert x["rules_universal_dh"] == 1 and x["rules_2023"] == 1
    early = input_frame(table[(table.season == 2024) & (table.player_id == 1)], era="full").iloc[0]
    assert early["rules_universal_dh"] == 1 and early["rules_2023"] == 1


def test_era_modes_and_league_reference(table):
    from fantasy_baseball.hitter_ros.features import league_reference

    none = input_frame(table)
    rel = input_frame(table, era="relative")
    full = input_frame(table, era="full")
    assert not any(c.endswith("_vs_lg") for c in none.columns)
    assert "std_hr_pa_vs_lg" in rel.columns and "lg_std_hr_pa" not in rel.columns
    assert "lg_std_hr_pa" in full.columns and "rules_2023" in full.columns
    with pytest.raises(ValueError):
        input_frame(table, era="weird")

    row = table[(table.player_id == 1) & (table.season == 2025) & (table.week == 1)]
    ref = league_reference(row).iloc[0]
    # Last three seasons (only 2024 here: 81 PA, 10 HR) plus 2025 before Apr 8 (57 PA,
    # 7 HR), pooled.
    assert ref["hr"] == pytest.approx(17 / 138)


def test_era_league_formulas_match_the_player_rates(table):
    """The vs-league ratio must divide like by like: league formulas == player formulas."""
    from fantasy_baseball.hitter_ros.features import _ERA_RATES, WINDOWS

    x = input_frame(table)
    for w in WINDOWS:
        for name, rate in _ERA_RATES.items():
            mine = rate(lambda c, w=w: table[f"{w}_{c}"].astype(float))
            pd.testing.assert_series_equal(mine, x[f"{w}_{name}"], check_names=False)


def test_league_reference_is_unknown_in_the_first_store_season(table):
    from fantasy_baseball.hitter_ros.features import league_reference

    ref = league_reference(table)
    first = table["season"] == table["season"].min()
    assert ref[first].isna().all().all()
    assert ref[~first].notna().all().all()


def test_league_answer_rates_are_the_league_rest_of_season(table):
    from fantasy_baseball.hitter_ros.features import COUNTS, league_answer_rates, rates_from_counts

    answer = league_answer_rates(table)
    # Week 0's answer window is the whole season, which next season's rows also hold
    # (computed independently by the table build) as the league's previous season.
    wk0_2024 = answer[(table.season == 2024) & (table.week == 0)]
    p1 = table[table.season == 2025].iloc[0]
    expected = rates_from_counts(pd.DataFrame([{c: p1[f"lg_p1_{c}"] for c in COUNTS}])).iloc[0]
    for _, row in wk0_2024.iterrows():
        pd.testing.assert_series_equal(row, expected, check_names=False)
    # Same for every row of a season and week; changes with the week.
    week1 = answer[(table.season == 2025) & (table.week == 1)]
    assert (week1.nunique() == 1).all()
    week0 = answer[(table.season == 2025) & (table.week == 0)]
    assert not np.allclose(week0.iloc[0], week1.iloc[0])


def test_steal_inputs(table):
    plain = input_frame(table)
    x = input_frame(table, steal=True)
    assert set(plain.columns) < set(x.columns)
    assert "std_steal_opp_pa" not in plain.columns and "p1_team_steal_pa" in x.columns
    row = table[(table.player_id == 1) & (table.season == 2025) & (table.week == 1)]
    r = input_frame(row, steal=True).iloc[0]
    # 7 games, on first with second open once a game, 4 PA a game, starts in CF.
    assert r["std_steal_opp_pa"] == pytest.approx(7 / 28)
    assert r["std_start_share_cf"] == 1.0 and r["std_start_share_c"] == 0.0
    assert r["std_attempts_per_opp"] == 0.0 and np.isnan(r["std_sb_success"])  # never ran


def test_bolt_rate_counts_a_blank_as_zero_only_when_runs_are_known():
    from fantasy_baseball.hitter_ros.features import _steal_inputs

    row = pd.DataFrame(
        {
            "p1_sprint_runs": [50.0, 50.0, np.nan],
            "p1_bolts": [5.0, np.nan, np.nan],
            "p1_hp_to_1b": [4.2, 4.5, np.nan],
            "p2_sprint_runs": [np.nan] * 3,
            "p2_bolts": [np.nan] * 3,
            "p2_hp_to_1b": [np.nan] * 3,
        }
    )
    windows = {
        f"{w}_{c}": 1.0
        for w in ("std", "p1", "p3", "car")
        for c in (
            "steal_opp2",
            "steal_opp3",
            "sb",
            "cs",
            "pa",
            "starts",
            "starts_c",
            "starts_ss",
            "starts_cf",
            "starts_dh",
        )
    }
    teams = {f"{w}_team_{c}": 1.0 for w in ("std", "p1") for c in ("sb", "cs", "pa")}
    full = row.assign(**{k: v for k, v in {**windows, **teams}.items() if k not in row})
    out = pd.DataFrame(_steal_inputs(full))
    assert list(out["p1_bolt_rate"][:2]) == [0.1, 0.0]
    assert np.isnan(out["p1_bolt_rate"][2])  # no sprint data at all: unknown, not 0
    assert out["p1_hp_to_1b"][0] == 4.2
