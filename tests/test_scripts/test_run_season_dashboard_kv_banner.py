"""Tests for the KV banner and the startup-sync guard in ``run_season_dashboard``.

``kv_sync.sync_remote_to_local()`` resolves its destination as ``local if
local is not None else get_kv()`` and then wipes it unconditionally --
``DELETE FROM kv; DELETE FROM hash_kv;`` -- before refilling it from Upstash.
The dashboard launcher is one of the callers that passes ``local=None``, so
launching it from a shell with ``FANTASY_LOCAL_KV_PATH=data/manual.db`` still
exported (the Yahoo-free manual pipeline sets exactly that) used to destroy the
hand-transcribed manual store and silently refill it with the Yahoo snapshot.

Three defences are pinned here, in the order they now fire:

  * a launch without ``--manual`` CLEARS an inherited binding, so the sync
    resolves the baseline and the manual store is never the destination;
  * ``--manual`` never syncs at all, because the operation is a wipe-and-
    download from production rather than a re-derivation; and
  * the sync still REFUSES any store other than ``data/local.db`` -- now a
    backstop rather than the first line of defence, and still the live guard
    for ``scripts/refresh_remote.py``.

The resolved absolute KV path is printed before any of it, so a terminal's
mode is never ambiguous.

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
    """An inherited export no longer decides the store, so point the baseline
    itself at tmp_path: what is pinned here is that the FIRST line names the
    resolved absolute path, whatever that path is."""
    baseline = tmp_path / "local.db"
    _relocate_baseline(monkeypatch, baseline)
    _point_at(monkeypatch, baseline)

    _run(monkeypatch, "--no-sync")

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"KV store: {baseline.resolve()}"
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


def test_a_plain_launch_ignores_an_inherited_manual_binding(monkeypatch, tmp_path, harness, capsys):
    """Without ``--manual`` the launcher CLEARS an inherited export rather than
    refusing on it.

    This replaced a refusal test. The refusal still exists and still protects
    ``scripts/refresh_remote.py``, but the dashboard can no longer reach it:
    clearing the binding first means the sync resolves the Yahoo baseline and
    the manual store is not at the destination at all. Not reaching the cliff
    beats a guard rail at its edge -- so what is pinned now is the outcome
    (baseline synced, manual store never named) rather than the mechanism.
    """
    baseline = tmp_path / "local.db"
    manual = tmp_path / "manual.db"
    _relocate_baseline(monkeypatch, baseline)
    _point_at(monkeypatch, manual)

    rc = _run(monkeypatch)

    assert rc == dash.RC_OK
    assert harness.sync_calls == 1, "the baseline is a legitimate sync target"
    assert harness.created_apps == 1
    out = capsys.readouterr().out
    assert f"KV store: {baseline.resolve()}" in out
    assert "Ignored inherited FANTASY_LOCAL_KV_PATH" in out
    assert str(manual.resolve()) not in out, "the manual store must not be touched or named"


def test_a_plain_launch_leaves_the_manual_store_intact(monkeypatch, tmp_path, harness):
    """The regression this has always been about: hand-typed rows must survive
    a launch by someone who forgot they had exported the variable."""
    from fantasy_baseball.data.kv_store import SqliteKVStore

    manual = tmp_path / "manual.db"
    seeded = SqliteKVStore(manual)
    seeded.set("cache:standings", '{"source": "manual-transcription"}')
    seeded.hset("weekly_rosters_history", "2026-08-17", '{"teams": 10}')

    _relocate_baseline(monkeypatch, tmp_path / "local.db")
    _point_at(monkeypatch, manual)

    assert _run(monkeypatch) == dash.RC_OK

    survivor = SqliteKVStore(manual)
    assert survivor.get("cache:standings") == '{"source": "manual-transcription"}'
    assert survivor.hgetall("weekly_rosters_history") == {"2026-08-17": '{"teams": 10}'}


def test_the_sync_refusal_still_guards_a_deliberate_binding(monkeypatch, tmp_path):
    """`guard_sync_target` is not dead code -- `scripts/refresh_remote.py` still
    reaches it, and it is the backstop if the dashboard ever stops clearing."""
    manual = tmp_path / "manual.db"
    _relocate_baseline(monkeypatch, tmp_path / "local.db")

    assert dash.guard_sync_target(manual.resolve()) == dash.RC_REFUSED


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


# --------------------------------------------------------------------------
# sync_remote_to_local() must never run in manual mode.
# --------------------------------------------------------------------------


def test_manual_mode_never_syncs(monkeypatch, seeded_store, harness, capsys):
    """`sync_remote_to_local()` WIPES its destination and refills it from
    production Upstash. Against the manual store that deletes the hand-typed
    transcription and replaces it with the last Yahoo snapshot -- it is a
    download, not a re-derivation, and there is no circumstance in which it is
    wanted here. `--manual` must therefore never reach it.

    Re-deriving with a changed pipeline is `POST /api/refresh` (or
    `scripts/run_manual_refresh.py`), which runs the blend/score/audit against
    the store and is unaffected by this.
    """
    from fantasy_baseball.manual import environment as env

    manual = seeded_store()
    monkeypatch.setattr(env, "DEFAULT_MANUAL_KV_PATH", manual)

    rc = _run(monkeypatch, "--manual")

    assert rc == dash.RC_OK
    assert harness.sync_calls == 0, (
        "sync_remote_to_local wipes its destination; in manual mode that is the transcription"
    )
    assert "Manual mode" in capsys.readouterr().out


def test_only_two_call_sites_can_reach_the_sync():
    """Pin the blast radius. A third caller has to be a deliberate decision.

    Matches BOTH `sync_remote_to_local(...)` and `kv_sync.sync_remote_to_local(...)`
    -- the module-alias form is this repo's house style, so a guard blind to it
    would pass on the case it exists to catch.
    """
    import ast

    callers = set()
    for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
        for f in root.rglob("*.py"):
            try:
                tree = ast.parse(f.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = (
                    fn.id
                    if isinstance(fn, ast.Name)
                    else fn.attr
                    if isinstance(fn, ast.Attribute)
                    else None
                )
                if name == "sync_remote_to_local":
                    callers.add(f.relative_to(PROJECT_ROOT).as_posix())

    assert callers == {
        "scripts/refresh_remote.py",
        "scripts/run_season_dashboard.py",
    }, f"a new caller of sync_remote_to_local appeared: {sorted(callers)}"
