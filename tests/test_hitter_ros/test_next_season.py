"""Projecting a season not played yet: preseason rows, predictions without scores."""

import json
from datetime import date

import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.table import build_options, build_table
from tests.test_hitter_ros.test_sb_pieces import _with_steals
from tests.test_hitter_ros.test_table import _season, _write


@pytest.fixture(scope="module")
def table_path(tmp_path_factory):
    """2024-2025 played, 2026 scheduled with no games yet."""
    root = tmp_path_factory.mktemp("store")
    seasons = {2024: _season(2024, date(2024, 4, 1), 10), 2025: _season(2025, date(2025, 4, 1), 21)}
    _write(root, {year: _with_steals(s) for year, s in seasons.items()})
    pd.DataFrame(
        {"season": [2026], "first_date": [date(2026, 3, 26)], "last_date": [date(2026, 9, 27)]}
    ).to_parquet(root / "schedule" / "2026.parquet", index=False)
    path = root / "table.parquet"
    build_table(root).to_parquet(path)
    path.with_suffix(".json").write_text(json.dumps(build_options()))
    return path


def _train(monkeypatch, table_path, tmp_path, *args):
    pytest.importorskip("torch")
    from scripts import train_hitter_ros

    monkeypatch.setattr(train_hitter_ros, "TABLE", table_path)
    monkeypatch.setattr(train_hitter_ros, "RUNS", tmp_path / "runs")
    argv = ["train", "--name", "x", "--test-seasons", "2026", "--hidden", "4", "--max-epochs"]
    argv += ["2", "--probes", "none", "--milb", "none", "--parks", "none", *args]
    monkeypatch.setattr("sys.argv", argv)
    return train_hitter_ros.main()


def test_an_unplayed_season_is_not_scored(table_path, tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _train(monkeypatch, table_path, tmp_path)
    assert "--predict-only" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "x").exists()


def test_predict_only_projects_next_season(table_path, tmp_path, monkeypatch):
    # The test seasons are too short to reach 100 PA: rest of season only.
    assert _train(monkeypatch, table_path, tmp_path, "--predict-only", "--no-horizons") == 0
    run = tmp_path / "runs" / "x"
    preds = pd.read_parquet(run / "predictions.parquet")
    table = pd.read_parquet(table_path)
    assert len(preds) == ((table.season == 2026) & (table.week == 0)).sum() > 0
    assert set(preds.week) == {0}
    assert preds[["r", "hr", "rbi", "sb", "avg"]].notna().all().all()
    assert "not scored" in (run / "summary.md").read_text()
    assert not (run / "scored_preseason.parquet").exists()
    meta = json.loads((run / "config.json").read_text())
    # Trained on the complete seasons before: both earlier seasons.
    assert meta["seasons"][0]["test_season"] == 2026 and meta["seasons"][0]["train_rows"] > 0


def _pretrain_run(tmp_path, monkeypatch, config=None):
    """A pretraining run p9 (2025, 2026) with today's default settings, and no token
    file anywhere the script would look."""
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.pretrain import PretrainConfig
    from scripts import pretrain_hitter_ros

    run = tmp_path / "p9"
    run.mkdir(parents=True)
    meta = {
        "config": PretrainConfig().to_dict() if config is None else config,
        "seasons": [2025, 2026],
        "complete": True,
    }
    (run / "run.json").write_text(json.dumps(meta))
    monkeypatch.setattr(pretrain_hitter_ros, "PRETRAIN", tmp_path)
    monkeypatch.setattr(pretrain_hitter_ros, "TOKEN_DIR", tmp_path / "no-tokens")
    return pretrain_hitter_ros, run


def _refused(script, monkeypatch, capsys, *argv):
    monkeypatch.setattr("sys.argv", ["pretrain", *argv])
    with pytest.raises(SystemExit):
        script.main()
    return capsys.readouterr().err


def test_pretrain_add_keeps_the_run_settings(tmp_path, monkeypatch, capsys):
    script, run = _pretrain_run(tmp_path, monkeypatch)
    lr = json.loads((run / "run.json").read_text())["config"]["lr"]
    for argv, message in (
        (["--add"], "needs --seasons"),
        (["--add", "--seasons", "2026"], "already has [2026]"),
        (["--add", "--seasons", "2027", "--lr", str(lr * 2)], "drop lr"),
        (["--add", "--seasons", "2027", "--zone", "fixed"], "drop zone"),
        (["--add", "--seasons", "2027", "--overwrite"], "can't also --overwrite"),
    ):
        assert message in _refused(script, monkeypatch, capsys, "--name", "p9", *argv), argv
    # The run's own value is fine to repeat: refused only later, for the missing tokens.
    err = _refused(
        script, monkeypatch, capsys, "--name", "p9", "--add", "--seasons", "2027", "--lr", str(lr)
    )
    assert "is missing" in err and "run's own settings" not in err
    err = _refused(script, monkeypatch, capsys, "--name", "nope", "--add", "--seasons", "2027")
    assert "no run.json" in err


def test_pretrain_add_refuses_leftovers_and_old_settings(tmp_path, monkeypatch, capsys):
    script, run = _pretrain_run(tmp_path, monkeypatch)
    (run / ".2027.tmp").mkdir()  # a crash's leftover
    err = _refused(script, monkeypatch, capsys, "--name", "p9", "--add", "--seasons", "2027")
    assert "left behind" in err and ".2027.tmp" in err

    old = json.loads((run / "run.json").read_text())["config"]
    old.pop("warmup_steps")  # a run from before a setting existed
    script, _ = _pretrain_run(tmp_path / "old", monkeypatch, config=old)
    err = _refused(script, monkeypatch, capsys, "--name", "p9", "--add", "--seasons", "2027")
    assert "missing ['warmup_steps']" in err


def test_a_season_missing_from_the_table_is_refused(table_path, tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _train(monkeypatch, table_path, tmp_path, "--predict-only", "--test-seasons", "2027")
    assert "no rows for [2027]" in capsys.readouterr().err


def test_split_skips_the_mid_season_model_with_nothing_to_predict(
    table_path, tmp_path, monkeypatch
):
    args = ("--predict-only", "--no-horizons", "--split")
    assert _train(monkeypatch, table_path, tmp_path, *args) == 0
    meta = json.loads((tmp_path / "runs" / "x" / "config.json").read_text())
    assert [s["weeks"] for s in meta["seasons"]] == ["pre"]


def test_rescoring_refuses_an_unplayed_season(table_path, tmp_path, monkeypatch, capsys):
    from scripts import score_hitter_ros_run

    run = tmp_path / "runs" / "proj"
    run.mkdir(parents=True)
    table = pd.read_parquet(table_path)
    table.loc[table.season == 2026, ["player_id", "season", "week"]].to_parquet(
        run / "predictions.parquet"
    )
    monkeypatch.setattr(score_hitter_ros_run, "TABLE", table_path)
    monkeypatch.setattr(score_hitter_ros_run, "RUNS", tmp_path / "runs")
    monkeypatch.setattr("sys.argv", ["score", "proj"])
    with pytest.raises(SystemExit):
        score_hitter_ros_run.main()
    assert "no games played yet in [2026]" in capsys.readouterr().err
