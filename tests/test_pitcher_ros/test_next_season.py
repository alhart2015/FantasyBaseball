"""Projecting a season not played yet: preseason rows, predictions without scores."""

import json
from datetime import date

import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros.table import build_table

from .test_table import _schedule, _seasons, _write


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """2024-2025 played, 2026 scheduled with no games yet."""
    root = tmp_path_factory.mktemp("store")
    _write(root, _seasons())
    _schedule(root, 2026, date(2026, 3, 26), date(2026, 9, 27))
    build_table(root).to_parquet(root / "table.parquet")
    return root


def _train(monkeypatch, store, tmp_path, *args):
    pytest.importorskip("torch")
    from scripts import train_pitcher_ros

    monkeypatch.setattr(train_pitcher_ros, "STORE", store)
    monkeypatch.setattr(train_pitcher_ros, "RUNS", tmp_path / "runs")
    argv = ["train", "--name", "x", "--table", str(store / "table.parquet"), "--seeds", "0"]
    # The fixture never gives up a HR or HBP: no league rate for ERA's pieces.
    argv += ["--seq", "none", "--parks", "none", "--stuff", "none", "--no-era-pieces", *args]
    monkeypatch.setattr("sys.argv", argv)
    return train_pitcher_ros.main()


def test_an_unplayed_season_is_not_scored(store, tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _train(monkeypatch, store, tmp_path, "--test-seasons", "2026")
    assert "--predict-only" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "x").exists()


def test_a_season_with_no_rows_is_refused_before_an_old_run_is_cleared(
    store, tmp_path, monkeypatch, capsys
):
    old = tmp_path / "runs" / "x"
    old.mkdir(parents=True)
    (old / "predictions.parquet").write_text("old run")
    with pytest.raises(SystemExit):
        _train(
            monkeypatch, store, tmp_path, "--test-seasons", "2027", "--predict-only", "--overwrite"
        )
    assert "no rows for [2027]" in capsys.readouterr().err
    assert (old / "predictions.parquet").read_text() == "old run"


def test_predict_only_projects_next_season(store, tmp_path, monkeypatch):
    assert _train(monkeypatch, store, tmp_path, "--test-seasons", "2026", "--predict-only") == 0
    run = tmp_path / "runs" / "x"
    preds = pd.read_parquet(run / "predictions.parquet")
    table = pd.read_parquet(store / "table.parquet")
    assert len(preds) == ((table.season == 2026) & (table.week == 0)).sum() > 0
    assert set(preds.season) == {2026} and set(preds.week) == {0}
    assert "not scored" in (run / "summary.md").read_text()
    assert not (run / "scores.md").exists()
    assert not list(run.glob("scored_*.parquet"))
    # Trained on the complete seasons before: both earlier seasons.
    info = json.loads((run / "info.json").read_text())
    assert [i["test_season"] for i in info] == [2026] and info[0]["train_rows"] > 0
