"""The fetch script's summary must survive a store with lineup files from before #402."""

import pandas as pd

from scripts.fetch_pitch_data import print_summary


def _lineups(root, season, **extra):
    path = root / "lineups" / f"{season}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "game_pk": [1, 1],
        "game_date": [f"{season}-04-01"] * 2,
        "sub_index": [0, 0],
    }
    pd.DataFrame({**base, **extra}).to_parquet(path, index=False)


def test_summary_with_only_old_lineup_files(tmp_path, capsys):
    _lineups(tmp_path, 2016)
    print_summary(tmp_path)
    out = capsys.readouterr().out
    assert "2016" in out
    assert "rbi" not in out


def test_summary_with_old_and_new_lineup_files(tmp_path, capsys):
    _lineups(tmp_path, 2016)
    _lineups(tmp_path, 2025, pa=[4, 5], hr=[1, 0], r=[1, 2], rbi=[2, 0], sb=[0, 1])
    print_summary(tmp_path)
    lines = capsys.readouterr().out.splitlines()
    old = next(line for line in lines if line.startswith("2016"))
    new = next(line for line in lines if line.startswith("2025"))
    assert old.split()[-5:] == ["-"] * 5
    assert new.split()[-5:] == ["9", "1", "3", "2", "1"]
