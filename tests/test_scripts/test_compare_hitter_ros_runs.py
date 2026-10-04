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


def _run(root, name, seed=0, pre=None, snap=None):
    run = root / name
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            {
                "config": {"seed": seed},
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
    monkeypatch.setattr(pretrain_hitter_ros, "TOKENS", tmp_path / "missing.parquet")
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
    # Old runs (scored before #424) have no level-free column: their rows just lack it.
    _run(runs, "old", pre=_scored({"ours": 1.0, "fg_blend": 2.0}, season=2025))
    assert pd.isna(cmp.compare(["r1", "old"], None, None).loc["old", "pre_lf_hr"])
