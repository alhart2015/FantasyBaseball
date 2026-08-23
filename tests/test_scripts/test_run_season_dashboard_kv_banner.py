"""Tests for the KV banner and the startup-sync guard in ``run_season_dashboard``.

``kv_sync.sync_remote_to_local()`` resolves its destination as ``local if
local is not None else get_kv()`` and then wipes it unconditionally --
``DELETE FROM kv; DELETE FROM hash_kv;`` -- before refilling it from Upstash.
The dashboard launcher is one of the callers that passes ``local=None``, so
launching it from a shell with ``FANTASY_LOCAL_KV_PATH=data/manual.db`` still
exported (the Yahoo-free manual pipeline sets exactly that) used to destroy the
hand-transcribed manual store and silently refill it with the Yahoo snapshot.

Two defences are pinned here:

  * the resolved absolute KV path is printed before anything else, so a
    terminal's mode is never ambiguous; and
  * the sync REFUSES to run against any store other than the default
    ``data/local.db``, exiting non-zero without deleting anything.

The default path -- no ``FANTASY_LOCAL_KV_PATH`` set -- must keep behaving
exactly as it did, which the "still syncs and serves" test exists to pin.

Note on fixtures: the real ``data/local.db`` is never opened here. Tests that
need the "this IS the baseline" verdict relocate ``_DEFAULT_LOCAL_DB`` into
``tmp_path`` and point the env var at the same file, so both the resolution and
the comparison are the real ones while the repo's baseline stays untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_season_dashboard as dash  # type: ignore[import-not-found]

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_season_dashboard.py"


@pytest.fixture(autouse=True)
def _fresh_kv_singleton():
    """Discard the process-wide KV singleton around every test.

    ``get_kv()`` caches its backend, so a store built against this test's
    ``tmp_path`` would otherwise outlive the directory and be handed to
    whatever runs next in the same worker.
    """
    from fantasy_baseball.data import kv_store

    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


class _FakeStats:
    def summary(self) -> str:
        return "0 string keys, 0 hashes"


class _FakeApp:
    """Stand-in for the Flask app: records ``run`` instead of serving."""

    def __init__(self) -> None:
        self.run_kwargs: list[dict] = []

    def run(self, **kwargs) -> None:
        self.run_kwargs.append(kwargs)


class _Harness:
    def __init__(self) -> None:
        self.sync_calls = 0
        self.app = _FakeApp()
        self.created_apps = 0


@pytest.fixture
def harness(monkeypatch):
    """Patch out the two side effects of ``main`` -- syncing and serving."""
    h = _Harness()

    def fake_sync(*args, **kwargs):
        h.sync_calls += 1
        return _FakeStats()

    def fake_create_app():
        h.created_apps += 1
        return h.app

    monkeypatch.setattr(dash, "sync_remote_to_local", fake_sync)
    monkeypatch.setattr(dash, "create_app", fake_create_app)
    monkeypatch.setattr(
        "fantasy_baseball.web.season_data.read_meta",
        lambda: {"last_refresh": "2026-08-17 09:00"},
    )
    monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
    return h


def _run(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["run_season_dashboard.py", *argv])
    return dash.main()


def _point_at(monkeypatch, path: Path) -> None:
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(path))


def _relocate_baseline(monkeypatch, path: Path) -> None:
    """Treat ``path`` as the Yahoo baseline for the duration of a test.

    Patched on ``kv_store``, the one module that defines it. The script used to
    carry its own from-import as well, which meant two names for one path and a
    test that had to know which of them the guard happened to read.
    """
    from fantasy_baseball.data import kv_store

    monkeypatch.setattr(kv_store, "_DEFAULT_LOCAL_DB", path)


# --------------------------------------------------------------------------
# Default behavior: no env var set must be exactly what it was.
# --------------------------------------------------------------------------


def test_unset_env_targets_the_repo_baseline():
    """The guard's allow-list is the same constant ``kv_store`` resolves.

    Re-deriving ``data/local.db`` anywhere would let the two drift, and a
    drifted guard would refuse the normal launch -- the one case that must
    always work.
    """
    from fantasy_baseball.data import kv_store

    assert dash.guard_sync_target(kv_store._DEFAULT_LOCAL_DB.resolve()) == dash.RC_OK


def test_default_kv_still_syncs_and_serves(monkeypatch, tmp_path, harness, capsys):
    baseline = tmp_path / "local.db"
    _relocate_baseline(monkeypatch, baseline)
    _point_at(monkeypatch, baseline)

    rc = _run(monkeypatch)

    assert rc == dash.RC_OK
    assert harness.sync_calls == 1
    assert harness.created_apps == 1
    assert harness.app.run_kwargs == [{"port": 5001, "debug": True}]
    out = capsys.readouterr().out
    assert "Syncing remote Upstash KV -> local SQLite..." in out
    assert "last_refresh: 2026-08-17 09:00" in out
    assert "REFUSING TO SYNC" not in out


def test_port_flag_still_reaches_the_server(monkeypatch, tmp_path, harness):
    baseline = tmp_path / "local.db"
    _relocate_baseline(monkeypatch, baseline)
    _point_at(monkeypatch, baseline)

    assert _run(monkeypatch, "--port", "5099") == dash.RC_OK
    assert harness.app.run_kwargs == [{"port": 5099, "debug": True}]


# --------------------------------------------------------------------------
# The banner.
# --------------------------------------------------------------------------


def test_banner_prints_the_resolved_absolute_path_first(monkeypatch, tmp_path, harness, capsys):
    manual = tmp_path / "manual.db"
    _point_at(monkeypatch, manual)

    _run(monkeypatch, "--no-sync")

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"KV store: {manual.resolve()}"
    assert Path(lines[0].split("KV store: ", 1)[1]).is_absolute()


def test_banner_precedes_the_sync(monkeypatch, tmp_path, harness, capsys):
    """The path must be on screen before anything touches the store."""
    baseline = tmp_path / "local.db"
    _relocate_baseline(monkeypatch, baseline)
    _point_at(monkeypatch, baseline)

    _run(monkeypatch)

    out = capsys.readouterr().out
    assert out.index("KV store: ") < out.index("Syncing remote Upstash KV")
    assert out.index("KV store: ") < out.index("Season dashboard:")


# --------------------------------------------------------------------------
# The guard.
# --------------------------------------------------------------------------


def test_refuses_to_sync_a_non_baseline_store(monkeypatch, tmp_path, harness, capsys):
    manual = tmp_path / "manual.db"
    _relocate_baseline(monkeypatch, tmp_path / "local.db")
    _point_at(monkeypatch, manual)

    rc = _run(monkeypatch)

    assert rc == dash.RC_REFUSED
    assert harness.sync_calls == 0, "a refused launch must not sync"
    assert harness.created_apps == 0, "a refused launch must not serve"
    out = capsys.readouterr().out
    assert "REFUSING TO SYNC" in out
    assert str(manual.resolve()) in out
    assert "--no-sync" in out


def test_refused_launch_leaves_the_manual_store_intact(monkeypatch, tmp_path, harness):
    """The regression this guard exists for: hand-typed rows must survive."""
    from fantasy_baseball.data.kv_store import SqliteKVStore

    manual = tmp_path / "manual.db"
    seeded = SqliteKVStore(manual)
    seeded.set("cache:standings", '{"source": "manual-transcription"}')
    seeded.hset("weekly_rosters_history", "2026-08-17", '{"teams": 10}')

    _relocate_baseline(monkeypatch, tmp_path / "local.db")
    _point_at(monkeypatch, manual)

    assert _run(monkeypatch) == dash.RC_REFUSED

    survivor = SqliteKVStore(manual)
    assert survivor.get("cache:standings") == '{"source": "manual-transcription"}'
    assert survivor.hgetall("weekly_rosters_history") == {"2026-08-17": '{"teams": 10}'}


def test_no_sync_opens_a_manual_store_without_refusing(monkeypatch, tmp_path, harness, capsys):
    """``--no-sync`` remains the way to view a manual store."""
    manual = tmp_path / "manual.db"
    _relocate_baseline(monkeypatch, tmp_path / "local.db")
    _point_at(monkeypatch, manual)

    rc = _run(monkeypatch, "--no-sync")

    assert rc == dash.RC_OK
    assert harness.sync_calls == 0
    assert harness.created_apps == 1
    assert "REFUSING TO SYNC" not in capsys.readouterr().out


def test_guard_refuses_a_store_with_no_local_file(capsys):
    """An Upstash / in-memory destination has no path and is never the baseline."""
    assert dash.guard_sync_target(None) == dash.RC_REFUSED
    assert "no local file" in capsys.readouterr().out


def test_reloader_child_still_skips_the_sync(monkeypatch, tmp_path, harness, capsys):
    """The guard sits inside the once-only branch, not in front of it."""
    baseline = tmp_path / "local.db"
    _relocate_baseline(monkeypatch, baseline)
    _point_at(monkeypatch, baseline)
    monkeypatch.setenv("WERKZEUG_RUN_MAIN", "true")

    assert _run(monkeypatch) == dash.RC_OK
    assert harness.sync_calls == 0
    assert "REFUSING TO SYNC" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# House rule.
# --------------------------------------------------------------------------


def test_script_is_ascii_only():
    """Every string here reaches ``print()`` on a cp1252 Windows console."""
    raw = SCRIPT_PATH.read_bytes()
    offenders = [
        (i + 1, line.decode("utf-8", "replace"))
        for i, line in enumerate(raw.split(b"\n"))
        if any(byte > 127 for byte in line)
    ]
    assert offenders == []
