import os
import shutil
from pathlib import Path

from fantasy_baseball.data.ros_export_ingest import (
    export_steps,
    find_newest_csv,
    run_guided_ingest,
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


def _seed(source: Path, name: str, fixture: str, ts: float) -> None:
    import os

    _copy_fixture(fixture, source / name)
    os.utime(source / name, (ts, ts))


def test_run_guided_ingest_stages_all_steps_and_reports_complete(tmp_path):
    source = tmp_path / "dl"
    source.mkdir()
    dest = tmp_path / "snap"
    systems = ["steamer"]
    clock = {"t": 100.0}

    def now_fn():
        clock["t"] += 10.0
        return clock["t"]

    pending = iter(
        [("steamer-h.csv", "steamer_hitters.csv"), ("steamer-p.csv", "steamer_pitchers.csv")]
    )

    def prompt_fn(_msg):
        name, fixture = next(pending)
        _seed(source, name, fixture, clock["t"] + 1.0)
        return ""

    outputs: list[str] = []
    result = run_guided_ingest(
        systems, source, dest, prompt_fn=prompt_fn, output_fn=outputs.append, now_fn=now_fn
    )
    assert result.aborted is False
    assert result.complete_systems(systems) == ["steamer"]
    assert (dest / "steamer-hitters.csv").exists()
    assert (dest / "steamer-pitchers.csv").exists()


def test_run_guided_ingest_skip_excludes_system(tmp_path):
    source = tmp_path / "dl"
    source.mkdir()
    result = run_guided_ingest(
        ["steamer"],
        source,
        tmp_path / "snap",
        prompt_fn=lambda _m: "s",
        output_fn=lambda _m: None,
        now_fn=lambda: 1.0,
    )
    assert result.complete_systems(["steamer"]) == []
    assert "steamer" in result.skipped_systems


def test_run_guided_ingest_abort_stops_immediately(tmp_path):
    result = run_guided_ingest(
        ["steamer", "zips"],
        tmp_path,
        tmp_path / "snap",
        prompt_fn=lambda _m: "q",
        output_fn=lambda _m: None,
        now_fn=lambda: 1.0,
    )
    assert result.aborted is True
    assert result.staged == {}


def test_run_guided_ingest_reprompts_on_missing_file(tmp_path):
    source = tmp_path / "dl"
    source.mkdir()
    dest = tmp_path / "snap"
    calls = {"n": 0}

    def prompt_fn(_msg):
        calls["n"] += 1
        if calls["n"] == 1:
            return ""  # hitters: no file yet -> stage_export None -> re-prompt
        if calls["n"] == 2:
            _seed(source, "h.csv", "steamer_hitters.csv", 1000.0)
            return ""  # hitters: now stages
        return "q"  # pitchers step: abort to end the run (avoids an infinite re-prompt)

    result = run_guided_ingest(
        ["steamer"],
        source,
        dest,
        prompt_fn=prompt_fn,
        output_fn=lambda _m: None,
        now_fn=lambda: 1.0,
    )
    assert calls["n"] >= 3  # 1 miss + 1 stage (hitters) + 1 abort (pitchers)
    assert ("steamer", "hitters") in result.staged
    assert result.aborted is True
