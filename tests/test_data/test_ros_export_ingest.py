import os
import shutil
from pathlib import Path

from fantasy_baseball.data.ros_export_ingest import (
    export_steps,
    find_newest_csv,
    stage_export,
    validate_export_type,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _copy_fixture(name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES_DIR / name, dest)
    return dest


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


def test_validate_export_type_accepts_matching_type(tmp_path):
    h = _copy_fixture("steamer_hitters.csv", tmp_path / "h.csv")
    p = _copy_fixture("steamer_pitchers.csv", tmp_path / "p.csv")
    assert validate_export_type(h, "hitters") is True
    assert validate_export_type(p, "pitchers") is True


def test_validate_export_type_rejects_wrong_type(tmp_path):
    h = _copy_fixture("steamer_hitters.csv", tmp_path / "h.csv")
    assert validate_export_type(h, "pitchers") is False


def test_stage_export_stages_newest_valid_file(tmp_path):
    source = tmp_path / "dl"
    source.mkdir()
    _copy_fixture("steamer_hitters.csv", source / "FanGraphs Leaderboard.csv")
    import os

    os.utime(source / "FanGraphs Leaderboard.csv", (5000.0, 5000.0))
    dest_dir = tmp_path / "snap"
    staged = stage_export(source, 4000.0, "steamer", "hitters", dest_dir)
    assert staged == dest_dir / "steamer-hitters.csv"
    assert staged.exists()


def test_stage_export_returns_none_for_wrong_type(tmp_path):
    source = tmp_path / "dl"
    source.mkdir()
    f = _copy_fixture("steamer_hitters.csv", source / "x.csv")
    import os

    os.utime(f, (5000.0, 5000.0))
    dest_dir = tmp_path / "snap"
    assert stage_export(source, 4000.0, "steamer", "pitchers", dest_dir) is None
    assert not (dest_dir / "steamer-pitchers.csv").exists()


def test_stage_export_returns_none_when_no_new_file(tmp_path):
    source = tmp_path / "dl"
    source.mkdir()
    f = _copy_fixture("steamer_hitters.csv", source / "old.csv")
    import os

    os.utime(f, (1000.0, 1000.0))
    assert stage_export(source, 4000.0, "steamer", "hitters", tmp_path / "snap") is None
