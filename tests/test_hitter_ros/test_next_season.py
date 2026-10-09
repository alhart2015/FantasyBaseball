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


def test_pretrain_add_keeps_the_run_settings(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    from scripts import pretrain_hitter_ros

    run = tmp_path / "p9"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"config": {}, "seasons": [2025, 2026]}))
    monkeypatch.setattr(pretrain_hitter_ros, "PRETRAIN", tmp_path)
    for argv, message in (
        (["--add"], "needs --seasons"),
        (["--add", "--seasons", "2026"], "already has [2026]"),
        (["--add", "--seasons", "2027", "--lr", "0.1"], "drop lr"),
        (["--add", "--seasons", "2027", "--overwrite"], "can't also --overwrite"),
    ):
        monkeypatch.setattr("sys.argv", ["pretrain", "--name", "p9", *argv])
        with pytest.raises(SystemExit):
            pretrain_hitter_ros.main()
        assert message in capsys.readouterr().err, argv
    monkeypatch.setattr("sys.argv", ["pretrain", "--name", "nope", "--add", "--seasons", "2027"])
    with pytest.raises(SystemExit):
        pretrain_hitter_ros.main()
    assert "no run.json" in capsys.readouterr().err
