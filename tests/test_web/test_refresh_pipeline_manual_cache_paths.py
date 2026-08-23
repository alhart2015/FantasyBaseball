"""A manual run must not write the two git-TRACKED MLB cache files.

``_fetch_probable_starters`` hands two on-disk cache paths to code that WRITES them
on every successful fetch (``mlb_schedule.save_schedule_cache``,
``matchups.save_batting_stats_cache``). Both defaults --
``data/weekly_schedule.json`` and ``data/team_batting_stats.json`` -- are tracked by
git, and they sit outside the manual pipeline's isolation boundary (a whole separate
KV file named by ``FANTASY_LOCAL_KV_PATH``). A manual run writing them dirties the
working tree with hand-transcribed-run content, so the repo stops reflecting
Yahoo-mode state and a careless ``git add -A`` commits it.

Two opposite-facing properties, both pinned here:

  * Yahoo mode writes exactly where it has always written -- byte-identical paths;
  * manual mode writes under a directory derived from the resolved KV path, and
    touches neither tracked file.
"""

from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.mlb_schedule import save_schedule_cache
from fantasy_baseball.lineup import matchups as matchups_mod
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.web import refresh_pipeline as rp_mod

PROJECT_ROOT = Path(rp_mod.__file__).resolve().parents[3]

#: The two tracked files. Spelled out rather than imported so a change to either
#: default has to be made twice, deliberately.
TRACKED_SCHEDULE = PROJECT_ROOT / "data" / "weekly_schedule.json"
TRACKED_BATTING = PROJECT_ROOT / "data" / "team_batting_stats.json"


@pytest.fixture
def kv_isolation(tmp_path, monkeypatch):
    """Per-test isolated SQLite KV named like the real manual store."""
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "manual.db"))
    monkeypatch.delenv("RENDER", raising=False)
    kv_store._reset_singleton()
    yield tmp_path / "manual.db"
    kv_store._reset_singleton()


@pytest.fixture
def configured_test_env(monkeypatch):
    """Environment ``load_config`` expects (mirrors test_refresh_pipeline)."""
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "http://fake")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake-token")


def _run_probable_starters(monkeypatch, *, manual: bool) -> dict:
    """Drive ``_fetch_probable_starters`` and capture the two cache paths.

    Everything that would leave the process (the schedule fetch, the batting-stats
    fetch, the KV write) is stubbed; the paths are the subject.
    """
    from fantasy_baseball.data import mlb_schedule

    captured: dict = {}

    def fake_get_week_schedule(start_date, end_date, cache_path, lookback_days=0):
        captured["schedule_path"] = Path(cache_path)
        return {"probable_pitchers": [], "start_date": start_date, "end_date": end_date}

    def fake_get_team_batting_stats(cache_path, season=None):
        captured["batting_path"] = Path(cache_path)
        return {}

    monkeypatch.setattr(mlb_schedule, "get_week_schedule", fake_get_week_schedule)
    monkeypatch.setattr(matchups_mod, "get_team_batting_stats", fake_get_team_batting_stats)
    monkeypatch.setattr(matchups_mod, "get_probable_starters", lambda *a, **kw: [])
    monkeypatch.setattr(rp_mod, "write_cache", lambda *a, **kw: None)

    kwargs = {"free_agent_source": (lambda req: [])} if manual else {}
    run = rp_mod.RefreshRun(**kwargs)
    assert run.manual_mode is manual
    run.start_date = "2026-04-13"
    run.end_date = "2026-04-19"
    run.roster_players = [
        Player(
            name="Bryan Woo",
            player_type=PlayerType.PITCHER,
            positions=[Position.SP, Position.P],
            selected_position=Position.P,
            team="SEA",
        )
    ]
    run.pitchers_proj = pd.DataFrame([{"name": "Bryan Woo", "gs": 22, "ip": 130.0}])
    run._fetch_probable_starters()
    return captured


class TestYahooModeIsUnchanged:
    """Revert test: the default path must stay exactly where it is."""

    def test_default_paths_are_the_tracked_project_files(
        self, monkeypatch, kv_isolation, configured_test_env
    ):
        captured = _run_probable_starters(monkeypatch, manual=False)

        assert captured["schedule_path"] == TRACKED_SCHEDULE, (
            f"Yahoo mode must keep writing {TRACKED_SCHEDULE}; got {captured['schedule_path']}"
        )
        assert captured["batting_path"] == TRACKED_BATTING, (
            f"Yahoo mode must keep writing {TRACKED_BATTING}; got {captured['batting_path']}"
        )


class TestManualModeIsIsolated:
    def test_cache_paths_land_beside_the_manual_kv_file(
        self, monkeypatch, kv_isolation, configured_test_env
    ):
        captured = _run_probable_starters(monkeypatch, manual=True)

        expected_dir = kv_isolation.parent / "cache" / kv_isolation.stem
        assert captured["schedule_path"] == expected_dir / "weekly_schedule.json"
        assert captured["batting_path"] == expected_dir / "team_batting_stats.json"
        assert expected_dir.is_dir(), (
            "the directory has to exist before the cache writers open() into it"
        )

    def test_neither_tracked_file_is_the_target(
        self, monkeypatch, kv_isolation, configured_test_env
    ):
        captured = _run_probable_starters(monkeypatch, manual=True)

        for path in (captured["schedule_path"], captured["batting_path"]):
            assert path != TRACKED_SCHEDULE and path != TRACKED_BATTING
            assert PROJECT_ROOT not in path.parents, (
                f"a manual run must write outside the repo-tracked tree; got {path}"
            )

    def test_a_real_write_leaves_the_tracked_files_alone(
        self, monkeypatch, kv_isolation, configured_test_env
    ):
        """The end-to-end claim, with the real writer.

        ``get_week_schedule`` calls ``save_schedule_cache`` on every successful fetch,
        so the path handed in is a path written to. Do that write for real and check
        the two tracked files come out untouched -- same bytes, same mtime.
        """
        before = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (TRACKED_SCHEDULE, TRACKED_BATTING)
            if path.exists()
        }

        captured = _run_probable_starters(monkeypatch, manual=True)
        save_schedule_cache({"probable_pitchers": []}, captured["schedule_path"])

        assert captured["schedule_path"].exists(), "the isolated cache file was written"
        for path, (data, mtime) in before.items():
            assert path.read_bytes() == data, f"a manual run rewrote tracked {path}"
            assert path.stat().st_mtime_ns == mtime, f"a manual run touched tracked {path}"
