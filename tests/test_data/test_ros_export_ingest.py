import os

from fantasy_baseball.data.ros_export_ingest import export_steps, find_newest_csv


def test_export_steps_orders_each_system_hitters_then_pitchers():
    steps = export_steps(["steamer", "zips"])
    assert steps == [
        ("steamer", "hitters"),
        ("steamer", "pitchers"),
        ("zips", "hitters"),
        ("zips", "pitchers"),
    ]


def test_find_newest_csv_returns_none_when_no_file_newer_than_since(tmp_path):
    old = tmp_path / "old.csv"
    old.write_text("x\n")
    os.utime(old, (1000.0, 1000.0))
    assert find_newest_csv(tmp_path, since_ts=2000.0) is None


def test_find_newest_csv_picks_most_recent_at_or_after_since(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("x\n")
    b.write_text("y\n")
    os.utime(a, (3000.0, 3000.0))
    os.utime(b, (3001.0, 3001.0))
    other = tmp_path / "note.txt"
    other.write_text("z\n")
    os.utime(other, (9999.0, 9999.0))
    assert find_newest_csv(tmp_path, since_ts=2999.0) == b
