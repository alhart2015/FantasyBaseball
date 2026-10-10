"""The run comparison reads a run folder's config and scored frames."""

import json

import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.features import TARGETS
from scripts import compare_hitter_ros_runs as cmp


def _scored(errors, **unit):
    rows = []
    for system, err in errors.items():
        for s in TARGETS:
            rows.append({"player_id": 1, "system": system, "stat": s, "abs_err": err, **unit})
    return pd.DataFrame(rows)


def _run(root, name, seed=0, pre=None, snap=None, config=None):
    run = root / name
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            {
                "config": {"seed": seed, **(config or {})},
                "seasons": [
                    {"test_season": 2024, "best_epoch": 2, "val_loss": [0.9, 0.8, 0.85]},
                    {"test_season": 2025, "best_epoch": 4, "val_loss": [0.7, 0.6]},
                ],
            }
        )
    )
    if pre is not None:
        pre.to_parquet(run / "scored_preseason.parquet")
    if snap is not None:
        snap.to_parquet(run / "scored_snapshots.parquet")
    (run / "summary.md").write_text("done\n")
    return run


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(cmp, "RUNS", tmp_path)
    return tmp_path


def test_run_row(runs):
    pre = pd.concat(
        [
            _scored({"ours": 10.0, "fg_blend": 9.0}, season=2024),
            _scored({"ours": 8.0, "fg_blend": 9.0}, season=2025),
        ]
    )
    snap = pd.concat(
        [
            _scored({"ours": 5.0, "fg_blend": 6.0}, season=2026, snapshot="2026-06-04"),
            _scored({"ours": 1.0, "fg_blend": 9.0}, season=2026, snapshot="2026-09-01"),
        ]
    )
    _run(runs, "r1", pre=pre, snap=snap)
    row = cmp.compare(["r1"], "2026-06-01", "2026-07-31").loc["r1"]
    assert row["best_epoch"] == 3 and row["val_loss"] == pytest.approx(0.7)
    assert row["seasons"] == "2024,2025"
    assert row["pre_hr"] == pytest.approx(9.0) and row["pre_gap_hr"] == pytest.approx(0.0)
    assert row["mid_snapshots"] == 1 and row["mid_gap_r"] == pytest.approx(-1.0)
    # No window: both snapshots count.
    assert cmp.compare(["r1"], None, None).loc["r1", "mid_gap_r"] == pytest.approx(-4.5)


def test_empty_window_and_missing_blend_do_not_crash(runs):
    pre = pd.concat(
        [
            _scored({"ours": 10.0, "fg_blend": 9.0}, season=2024),
            _scored({"ours": 8.0}, season=2025),  # only one FanGraphs system: no blend
        ]
    )
    snap = _scored({"ours": 5.0, "fg_blend": 6.0}, season=2026, snapshot="2026-06-04")
    _run(runs, "r1", pre=pre, snap=snap)
    row = cmp.compare(["r1"], "2026-09-20", "2026-09-30").loc["r1"]
    assert row["mid_snapshots"] == 0
    assert pd.isna(row.get("mid_gap_r"))
    # The gap uses only 2024, where both were scored; our own MAE uses both seasons.
    assert row["pre_gap_r"] == pytest.approx(1.0)
    assert row["pre_r"] == pytest.approx(9.0)


def test_warns_when_seeds_or_seasons_differ(runs):
    _run(runs, "a", seed=0)
    _run(runs, "b", seed=1)
    warnings = cmp.warnings_for(cmp.compare(["a", "b"], None, None))
    assert any("seeds differ" in w for w in warnings)


def test_warns_when_losses_differ(runs):
    _run(runs, "mse")  # a run from before #424: no loss setting, so MSE
    _run(runs, "rank", config={"loss": "rank"})
    df = cmp.compare(["mse", "rank"], None, None)
    assert list(df["loss"]) == ["mse", "rank"]
    assert any("losses differ" in w for w in cmp.warnings_for(df))


def test_warns_when_avg_losses_differ(runs):
    """#433: a binomial AVG loss puts a deviance, not a squared error, into val_loss."""
    _run(runs, "mse_avg", config={"avg_loss": "mse"})
    _run(runs, "binomial", config={"avg_loss": "binomial"})
    df = cmp.compare(["mse_avg", "binomial"], None, None)
    assert list(df["loss"]) == ["mse", "mse+avg_binomial"]
    assert any("losses differ" in w for w in cmp.warnings_for(df))


def test_warns_when_avg_pieces_differ(runs):
    """#433: AVG's pieces add binomial-deviance columns to val_loss."""
    _run(runs, "plain", config={"avg_pieces": "none"})
    _run(runs, "pieces", config={"avg_pieces": "extra"})
    df = cmp.compare(["plain", "pieces"], None, None)
    assert list(df["loss"]) == ["mse", "mse+avg_pieces_extra"]
    assert any("losses differ" in w for w in cmp.warnings_for(df))


def test_warns_when_weightings_differ(runs):
    _run(runs, "old")  # a run from before the weighting setting: PA weighting
    _run(runs, "pre_mid", config={"weighting": "pre_mid"})
    _run(runs, "split", config={"weighting": "pre_mid", "split": True})
    df = cmp.compare(["old", "pre_mid", "split"], None, None)
    assert list(df["weighting"]) == ["pa", "pre_mid", "pre_mid+split"]
    assert any("weightings differ" in w for w in cmp.warnings_for(df))
    same = cmp.compare(["pre_mid"], None, None)
    assert not any("weightings differ" in w for w in cmp.warnings_for(same))


def test_a_run_still_scoring_is_skipped(runs, capsys, monkeypatch):
    _run(runs, "done")
    scoring = runs / "scoring"
    scoring.mkdir()
    (scoring / "config.json").write_text("{}")  # config written, summary not yet
    monkeypatch.setattr("sys.argv", ["compare", "done", "scoring"])
    assert cmp.main() == 0
    assert "skipping scoring" in capsys.readouterr().out


def test_train_script_keeps_an_old_run_when_tokens_are_missing(tmp_path, monkeypatch):
    from scripts import train_hitter_ros

    old = tmp_path / "runs" / "keep-me"
    old.mkdir(parents=True)
    (old / "summary.md").write_text("old results\n")
    monkeypatch.setattr(train_hitter_ros, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(train_hitter_ros, "TOKENS", tmp_path / "missing.parquet")
    monkeypatch.setattr("sys.argv", ["train", "--name", "keep-me", "--seq", "gru", "--overwrite"])
    with pytest.raises(SystemExit):
        train_hitter_ros.main()
    assert (old / "summary.md").read_text() == "old results\n"


def test_train_script_rejects_zero_training_seasons(monkeypatch):
    from scripts import train_hitter_ros

    monkeypatch.setattr("sys.argv", ["train", "--name", "x", "--train-seasons", "0"])
    with pytest.raises(SystemExit):
        train_hitter_ros.main()


def test_pretrain_script_keeps_an_old_run_when_tokens_are_missing(tmp_path, monkeypatch):
    from scripts import pretrain_hitter_ros

    old = tmp_path / "p" / "keep"
    old.mkdir(parents=True)
    (old / "run.json").write_text("{}")
    monkeypatch.setattr(pretrain_hitter_ros, "PRETRAIN", tmp_path / "p")
    monkeypatch.setattr(pretrain_hitter_ros, "TOKEN_DIR", tmp_path / "no_tokens")
    monkeypatch.setattr("sys.argv", ["pretrain", "--name", "keep", "--overwrite"])
    with pytest.raises(SystemExit):
        pretrain_hitter_ros.main()
    assert (old / "run.json").exists()


def test_train_from_takes_the_later_limit():
    from scripts.train_hitter_ros import _train_from

    assert _train_from(2026, None, None) is None
    assert _train_from(2026, 4, None) == 2022
    assert _train_from(2026, None, 2015) == 2015
    assert _train_from(2018, 11, 2015) == 2015
    assert _train_from(2026, 4, 2015) == 2022


def test_train_script_asks_to_rebuild_a_table_without_era_columns(tmp_path, monkeypatch, capsys):
    from scripts import train_hitter_ros

    stale = tmp_path / "table.parquet"
    pd.DataFrame({"player_id": [1], "season": [2025]}).to_parquet(stale)
    monkeypatch.setattr(train_hitter_ros, "TABLE", stale)
    monkeypatch.setattr(train_hitter_ros, "RUNS", tmp_path / "runs")
    monkeypatch.setattr("sys.argv", ["train", "--name", "x", "--era", "relative"])
    with pytest.raises(SystemExit):
        train_hitter_ros.main()
    assert "build_hitter_ros_table" in capsys.readouterr().err


def test_run_row_reports_the_league_free_scores(runs):
    from fantasy_baseball.hitter_ros.evaluate import scored_players

    actual = pd.DataFrame({s: [0.1, 0.2, 0.3] for s in TARGETS}, index=[1, 2, 3])
    actual["pa"] = 500
    right_order = actual[list(TARGETS)] * 0.5  # a pure league-level miss
    wrong_order = actual[list(TARGETS)].iloc[::-1].set_axis(actual.index)
    pre = scored_players({"ours": right_order, "fg_blend": wrong_order}, actual, 1)
    _run(runs, "r1", pre=pre.assign(season=2025))
    row = cmp.compare(["r1"], None, None).loc["r1"]
    assert row["pre_lf_hr"] == pytest.approx(0.0, abs=1e-9)
    assert row["pre_pair_hr"] == pytest.approx(100.0)
    assert row["pre_pair_gap_hr"] == pytest.approx(100.0)
    assert row["pre_pairw_hr"] == pytest.approx(100.0)  # the main score
    assert row["pre_pairw_gap_hr"] == pytest.approx(100.0)
    assert row["pre_lf_gap_hr"] < 0  # ours ordered right, the blend backwards
    # Ours orders every pair right and the blend every pair wrong, on every resample.
    assert row["pre_pairw_sure_hr"] == pytest.approx(1.0)
    # Old runs (scored before #424) have no level-free column: their rows just lack it.
    _run(runs, "old", pre=_scored({"ours": 1.0, "fg_blend": 2.0}, season=2025))
    both = cmp.compare(["r1", "old"], None, None)
    assert pd.isna(both.loc["old", "pre_lf_hr"]) and pd.isna(both.loc["old", "pre_pairw_sure_hr"])


def test_run_row_reports_vets_and_rookies_separately(runs):
    from fantasy_baseball.hitter_ros.evaluate import scored_players

    actual = pd.DataFrame({s: [0.1, 0.2, 0.3] * 2 for s in TARGETS}, index=range(1, 7))
    actual["pa"] = 500
    right = actual[list(TARGETS)]
    backwards = right.copy()
    backwards.loc[[1, 2, 3]] = right.loc[[3, 2, 1]].to_numpy()
    backwards.loc[[4, 5, 6]] = right.loc[[6, 5, 4]].to_numpy()
    # Ours orders the vets (1-3) right and the rookies (4-6) backwards; the blend the opposite.
    ours = pd.concat([right.loc[[1, 2, 3]], backwards.loc[[4, 5, 6]]])
    blend = pd.concat([backwards.loc[[1, 2, 3]], right.loc[[4, 5, 6]]])
    pre = scored_players({"ours": ours, "fg_blend": blend}, actual, 1)
    pre = pre.assign(season=2025, group=pre.player_id.map(lambda p: "vet" if p <= 3 else "rookie"))
    _run(runs, "r1", pre=pre, snap=pre.assign(snapshot="2026-06-04"))
    row = cmp.compare(["r1"], None, None).loc["r1"]
    assert row["pre_vet_pairw_gap_hr"] == pytest.approx(100.0)
    assert row["pre_rookie_pairw_gap_hr"] == pytest.approx(-100.0)
    assert row["mid_vet_pairw_gap_hr"] == pytest.approx(100.0)
    assert row["mid_rookie_pairw_gap_hr"] == pytest.approx(-100.0)
    # Over everyone the two cancel out: a coin flip, whatever the luck of the draw.
    assert row["mid_pairw_gap_hr"] == pytest.approx(0.0)
    assert -1.0 < row["mid_pairw_sure_hr"] < 1.0
    # A run scored before the tag has no group rows.
    _run(runs, "old", pre=pre.drop(columns="group"))
    assert "pre_vet_pairw_gap_hr" not in cmp.compare(["old"], None, None).columns


def test_run_row_shows_where_sb_came_from(runs):
    _run(runs, "old")  # before #451: no setting, SB from the main net
    box = _run(runs, "box", config={"sb_inputs": "box"})
    meta = json.loads((box / "config.json").read_text())
    for season, (epoch, losses) in zip(
        meta["seasons"], [(5, [0.5, 0.4]), (7, [0.3, 0.2, 0.25])], strict=True
    ):
        season["sb_net"] = {"best_epoch": epoch, "val_loss": losses}
    (box / "config.json").write_text(json.dumps(meta))
    df = cmp.compare(["old", "box"], None, None)
    assert df.loc["old", "sb_inputs"] == "full" and pd.isna(df.loc["old", "sb_best_epoch"])
    assert df.loc["box", "sb_inputs"] == "box"
    assert df.loc["box", "sb_best_epoch"] == 6
    assert df.loc["box", "sb_val_loss"] == pytest.approx(0.3)
    assert df.loc["box", "val_loss"] == pytest.approx(0.7)  # the main net's, unchanged


def test_run_row_reports_mse_as_the_main_score(runs):
    from fantasy_baseball.hitter_ros.evaluate import mse_table, scored_players

    actual = pd.DataFrame({s: [0.1, 0.2, 0.3, 0.4] for s in TARGETS}, index=[1, 2, 3, 4])
    actual["pa"] = 500
    close = actual[list(TARGETS)] + 0.01
    far = actual[list(TARGETS)] + 0.05
    pre = scored_players({"ours": close, "fg_blend": far}, actual, 1).assign(
        season=2025, group="vet", relevant=True
    )
    _run(runs, "r1", pre=pre, snap=pre.assign(snapshot="2026-06-04"))
    row = cmp.compare(["r1"], None, None).loc["r1"]
    table = mse_table(pre)
    assert row["pre_mse_hr"] == pytest.approx(table.loc["ours", "hr"])  # 5 HR off, squared
    assert row["pre_mse_hr"] == pytest.approx(25.0)
    gap = table.loc["ours", "hr"] - table.loc["fg_blend", "hr"]
    assert gap < 0
    for col in ("pre_mse_gap_hr", "mid_mse_gap_hr", "pre_vet_mse_gap_hr", "pre_top_mse_gap_hr"):
        assert row[col] == pytest.approx(gap)
    # Ours is closer on every hitter, so on every resample.
    assert row["pre_mse_sure_hr"] == pytest.approx(1.0)
    assert row["mid_mse_sure_hr"] == pytest.approx(1.0)
    # Old test frames without projected/actual/pa just lack the MSE columns.
    _run(runs, "old", pre=_scored({"ours": 1.0, "fg_blend": 2.0}, season=2025))
    assert "pre_mse_hr" not in cmp.compare(["old"], None, None).columns
