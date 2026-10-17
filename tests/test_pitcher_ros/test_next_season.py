"""Projecting a season not played yet: preseason rows, predictions without scores."""

import json
from datetime import date

import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros.model import HORIZON_WEIGHTS
from fantasy_baseball.pitcher_ros.table import HORIZON_DAYS, build_table

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
    # The short horizons are on, and their loss weights are recorded with the run.
    config = json.loads((run / "config.json").read_text())
    assert config["horizons"] == list(HORIZON_DAYS)
    assert config["horizon_weights"] == HORIZON_WEIGHTS
    assert {f"d{n}_k" for n in HORIZON_DAYS} <= set(preds.columns)


def test_probes_read_from_older_pitches_than_the_store_are_refused(
    store, tmp_path, monkeypatch, capsys
):
    from fantasy_baseball.hitter_ros.features import ROW_KEYS
    from fantasy_baseball.pitcher_ros.probes import PROBE_FEATURES, probe_path

    table = pd.read_parquet(store / "table.parquet")
    probes = table[[*ROW_KEYS, "as_of"]].assign(**dict.fromkeys(PROBE_FEATURES, 0.0))
    path = probe_path(store, "old")
    probes.to_parquet(path)
    path.with_suffix(".json").write_text(json.dumps({"pitches_through": "2024-04-05"}))
    with pytest.raises(SystemExit):
        _train(monkeypatch, store, tmp_path, "--test-seasons", "2025", "--probes", "old")
    assert "probes_old.parquet read pitches through 2024-04-05" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "x").exists()


def test_an_encoder_pretrained_on_one_half_trains_only_on_the_other(
    store, tmp_path, monkeypatch, capsys
):
    pytest.importorskip("torch")
    from scripts import train_pitcher_ros
    from tests.test_hitter_ros.test_pretrain import _small

    run = tmp_path / "pretrain" / "q"
    run.mkdir(parents=True)
    meta = {"config": _small(window=8).to_dict(), "seasons": [2025], "tokens": {}, "half": 0}
    (run / "run.json").write_text(json.dumps(meta))
    monkeypatch.setattr(train_pitcher_ros, "PRETRAIN", tmp_path / "pretrain")
    tokens = tmp_path / "tokens.parquet"
    tokens.touch()  # only its presence is checked before the half
    ft = ["--seq", "transformer", "--tokens", str(tokens), "--finetune", "q"]
    ft += ["--finetune-len", "7", "--micro-batch", "4", "--test-seasons", "2025"]
    for half in ([], ["--train-half", "0"]):
        with pytest.raises(SystemExit):
            _train(monkeypatch, store, tmp_path, *ft, *half)
        assert "pass --train-half 1" in capsys.readouterr().err, half


def test_cross_fitting_trains_every_pitcher(store, tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    from scripts import train_pitcher_ros
    from tests.test_hitter_ros.test_pretrain import _small

    for half in (0, 1):
        run = tmp_path / "pretrain" / f"h{half}"
        run.mkdir(parents=True)
        meta = {"config": _small(window=8).to_dict(), "seasons": [2025], "tokens": {}}
        (run / "run.json").write_text(json.dumps({**meta, "half": half}))
    monkeypatch.setattr(train_pitcher_ros, "PRETRAIN", tmp_path / "pretrain")
    tokens = tmp_path / "tokens.parquet"
    tokens.touch()  # only its presence is checked before the runs
    ft = ["--seq", "transformer", "--tokens", str(tokens), "--finetune-len", "7"]
    ft += ["--micro-batch", "4", "--test-seasons", "2025"]
    for argv, message in (
        (["--finetune-cross", "h1"], "needs --finetune"),
        (["--finetune", "h0", "--finetune-cross", "h0"], "opposite halves"),
        (["--finetune", "h0", "--finetune-cross", "h1", "--train-half", "1"], "drop --train-half"),
    ):
        with pytest.raises(SystemExit):
            _train(monkeypatch, store, tmp_path, *ft, *argv)
        assert message in capsys.readouterr().err, argv


def test_probes_from_a_half_run_train_only_on_the_other_half(store, tmp_path, monkeypatch, capsys):
    from fantasy_baseball.hitter_ros.features import ROW_KEYS
    from fantasy_baseball.pitcher_ros.probes import PROBE_FEATURES, probe_path

    table = pd.read_parquet(store / "table.parquet")
    probes = table[[*ROW_KEYS, "as_of"]].assign(**dict.fromkeys(PROBE_FEATURES, 0.0))
    path = probe_path(store, "h0")
    probes.to_parquet(path)
    path.with_suffix(".json").write_text(json.dumps({"pitches_through": "2030-01-01", "half": 0}))
    for half in ([], ["--train-half", "0"]):
        with pytest.raises(SystemExit):
            _train(monkeypatch, store, tmp_path, "--test-seasons", "2025", "--probes", "h0", *half)
        assert "pass --train-half 1" in capsys.readouterr().err, half
    assert not (tmp_path / "runs" / "x").exists()


def test_finetuning_reads_the_pitcher_token_file_its_run_was_pretrained_on(
    store, tmp_path, monkeypatch, capsys
):
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.pitch_tokens import token_options
    from scripts import train_pitcher_ros
    from tests.test_hitter_ros.test_pretrain import _small

    run = tmp_path / "pretrain" / "q"
    run.mkdir(parents=True)
    meta = {"config": _small(window=8).to_dict(), "seasons": [2025], "tokens": {}}
    meta["token_options"] = token_options("statcast", "pitcher")
    (run / "run.json").write_text(json.dumps(meta))
    monkeypatch.setattr(train_pitcher_ros, "PRETRAIN", tmp_path / "pretrain")
    pitch_tokens = tmp_path / "pitch_tokens.parquet"
    pitch_tokens.touch()  # its recorded options are checked before it is read
    monkeypatch.setattr(train_pitcher_ros, "PITCH_TOKENS", pitch_tokens)
    tokens = tmp_path / "tokens.parquet"
    tokens.touch()
    ft = ["--seq", "transformer", "--tokens", str(tokens), "--finetune", "q"]
    ft += ["--finetune-len", "7", "--micro-batch", "4", "--test-seasons", "2025"]
    for options, message in (
        (token_options("statcast"), "holds hitter-side tokens"),
        (token_options("fixed", "pitcher"), "q was pretrained on"),
    ):
        pitch_tokens.with_suffix(".json").write_text(json.dumps(options))
        with pytest.raises(SystemExit):
            _train(monkeypatch, store, tmp_path, *ft)
        assert message in capsys.readouterr().err, options


def test_no_own_windows_needs_the_sequence(store, tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _train(monkeypatch, store, tmp_path, "--test-seasons", "2025", "--no-own-windows")
    assert "it needs --seq" in capsys.readouterr().err


def test_no_own_windows_also_leaves_out_the_stuff_windows(store, tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _train(
            monkeypatch,
            store,
            tmp_path,
            "--test-seasons",
            "2025",
            "--seq",
            "transformer",
            "--stuff",
            "s2-located",
            "--no-own-windows",
        )
    assert "add --stuff none" in capsys.readouterr().err
