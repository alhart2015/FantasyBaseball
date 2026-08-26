"""Tests for ``run_season_dashboard.py --manual``.

Three things have to hold, and each fails silently if it does not:

* The store must be bound before anything resolves one, because ``get_kv()``
  caches on its first call.
* It must be a real, SEEDED manual store. ``SqliteKVStore.__init__`` creates
  the file on open, and an unstamped store reads as Yahoo mode -- so a bare
  path check would let the dashboard serve production rosters under a manual
  banner.
* The sync must not run, since it wipes its destination before refilling.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fantasy_baseball.data.cache_keys import MANUAL_PROVENANCE_KEY
from fantasy_baseball.manual.environment import (
    DEFAULT_MANUAL_KV_PATH,
    activate_manual_environment,
    deactivate_manual_environment,
    manual_store_refusal,
)


def _seeded_store(path: Path) -> Path:
    from fantasy_baseball.data.kv_store import SqliteKVStore

    SqliteKVStore(path).set(MANUAL_PROVENANCE_KEY, '{"seeded": "yes"}')
    return path


class TestManualStoreRefusal:
    """`manual_store_refusal` is the check that makes `--manual` mean a STORE."""

    def test_a_missing_store_is_refused_without_being_created(self, tmp_path):
        missing = tmp_path / "manual.db"
        refusal = manual_store_refusal(missing)
        assert refusal is not None
        assert "does not exist" in refusal
        assert "bootstrap_manual_kv" in refusal, "must say how to create it"
        assert not missing.exists(), "checking must not CREATE the store"

    def test_an_unstamped_store_is_refused(self, tmp_path):
        """The dangerous case: the file exists but was never seeded, so
        `rosters.manual_store_active()` reads it as Yahoo mode and
        `live_rosters()` would serve production Upstash."""
        from fantasy_baseball.data.kv_store import SqliteKVStore

        # Exactly what get_kv() would create: real schema, no provenance stamp.
        unstamped = tmp_path / "manual.db"
        SqliteKVStore(unstamped)

        refusal = manual_store_refusal(unstamped)

        assert refusal is not None
        assert MANUAL_PROVENANCE_KEY in refusal
        assert "production rosters" in refusal, "must name the actual consequence"

    def test_a_seeded_store_is_accepted(self, tmp_path):
        assert manual_store_refusal(_seeded_store(tmp_path / "manual.db")) is None

    def test_a_file_that_is_not_a_kv_store_is_refused_not_raised(self, tmp_path):
        junk = tmp_path / "manual.db"
        junk.write_bytes(b"not a sqlite database")
        refusal = manual_store_refusal(junk)
        assert refusal is not None
        assert "could not be read" in refusal

    def test_a_sqlite_file_without_the_kv_table_is_refused(self, tmp_path):
        """A stray .db at the manual path is not a KV store; say so rather
        than letting the missing table surface as a stack trace."""
        stray = tmp_path / "manual.db"
        sqlite3.connect(str(stray)).close()
        refusal = manual_store_refusal(stray)
        assert refusal is not None
        assert "could not be read" in refusal


class TestActivate:
    def test_binds_an_absolute_path(self, tmp_path, monkeypatch):
        """`kv_store` resolves the variable against the CWD, so a relative
        value silently creates a second, empty store."""
        monkeypatch.delenv("RENDER", raising=False)
        target = tmp_path / "manual.db"

        resolved = activate_manual_environment(target)

        assert Path(os.environ["FANTASY_LOCAL_KV_PATH"]).is_absolute()
        assert resolved == target.resolve()
        assert os.environ["FB_SKIP_YAHOO"] == "1"

    def test_defaults_to_the_repo_manual_store_without_binding_it(self, monkeypatch):
        """Asserts the CONSTANT, not the live singleton -- binding the real
        32 MB transcription store from a test would open and PRAGMA it."""
        assert DEFAULT_MANUAL_KV_PATH.name == "manual.db"
        assert DEFAULT_MANUAL_KV_PATH.parent.name == "data"

    def test_rebinds_a_singleton_something_else_already_built(self, tmp_path, monkeypatch):
        from fantasy_baseball.data import kv_store
        from fantasy_baseball.manual.seed import resolve_kv_path

        monkeypatch.delenv("RENDER", raising=False)
        monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "wrong.db"))
        kv_store.get_kv()  # bind to the wrong store first

        activate_manual_environment(tmp_path / "right.db")

        assert resolve_kv_path(kv_store.get_kv()) == (tmp_path / "right.db").resolve()

    def test_refuses_on_render_rather_than_mutating(self, monkeypatch):
        """The KV on Render is Upstash; no variable redirects it, and
        disabling Yahoo against production is not a side effect to have."""
        monkeypatch.setenv("RENDER", "true")
        monkeypatch.delenv("FB_SKIP_YAHOO", raising=False)

        with pytest.raises(RuntimeError, match="RENDER"):
            activate_manual_environment()

        assert "FB_SKIP_YAHOO" not in os.environ


class TestDeactivate:
    """The half that is easy to miss: these are EXPORTED variables, so they
    outlive the command that set them, and a later plain launch would inherit
    the last manual session."""

    def test_clears_the_store_binding(self, monkeypatch, tmp_path):
        monkeypatch.delenv("RENDER", raising=False)
        monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "manual.db"))

        cleared = deactivate_manual_environment()

        assert set(cleared) == {"FANTASY_LOCAL_KV_PATH"}
        assert "FANTASY_LOCAL_KV_PATH" not in os.environ

    def test_leaves_the_stale_data_switch_alone(self, monkeypatch, tmp_path):
        """`FB_SKIP_YAHOO` is a standalone stale-data mode against the ordinary
        `local.db` (docs/stale-data-refresh-runbook.md), and the manual runbook
        calls it a seatbelt for a stray Refresh click. With the Yahoo API
        unavailable, clearing it would re-arm live auth for someone who set it
        for entirely unrelated reasons."""
        monkeypatch.delenv("RENDER", raising=False)
        monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "manual.db"))
        monkeypatch.setenv("FB_SKIP_YAHOO", "1")

        deactivate_manual_environment()

        assert os.environ["FB_SKIP_YAHOO"] == "1"

    def test_rebinds_to_the_yahoo_baseline(self, monkeypatch, tmp_path):
        """Clearing the variable is not enough -- the singleton already built
        against it has to be discarded too."""
        from fantasy_baseball.data import kv_store
        from fantasy_baseball.manual.seed import resolve_kv_path

        monkeypatch.delenv("RENDER", raising=False)
        monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "manual.db"))
        kv_store.get_kv()

        deactivate_manual_environment()

        assert resolve_kv_path(kv_store.get_kv()).name == "local.db"

    def test_reports_nothing_when_already_clean(self, monkeypatch):
        monkeypatch.delenv("RENDER", raising=False)
        monkeypatch.delenv("FANTASY_LOCAL_KV_PATH", raising=False)
        assert deactivate_manual_environment() == {}

    def test_is_a_no_op_on_render(self, monkeypatch):
        monkeypatch.setenv("RENDER", "true")
        monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", "/whatever")
        assert deactivate_manual_environment() == {}
        assert os.environ["FANTASY_LOCAL_KV_PATH"] == "/whatever"


class TestGuardManualStore:
    """The script-level wrapper: identity first, then the store itself."""

    def test_refuses_a_store_that_is_not_the_manual_one(self, tmp_path, capsys):
        import run_season_dashboard  # type: ignore[import-not-found]

        rc = run_season_dashboard.guard_manual_store(tmp_path / "somewhere-else.db")

        assert rc == run_season_dashboard.RC_REFUSED
        assert "did not bind the manual store" in capsys.readouterr().out

    def test_render_refuses_with_an_exit_code_not_a_traceback(self, monkeypatch, capsys):
        """`activate_manual_environment` raises on Render. `main()` must turn
        that into RC_REFUSED (2) -- a wrapper distinguishes "refused, nothing
        happened" from "started and failed", which a traceback destroys."""
        import argparse

        import run_season_dashboard  # type: ignore[import-not-found]

        monkeypatch.setenv("RENDER", "true")
        args = argparse.Namespace(manual=True, no_sync=False, port=5001)

        rc = run_season_dashboard.enter_manual_mode(args)

        assert rc == run_season_dashboard.RC_REFUSED
        out = capsys.readouterr().out
        assert "Upstash" in out
        assert "local-only" in out

    def test_refuses_an_unseeded_manual_store(self, tmp_path, monkeypatch, capsys):
        """Identity passes, the store check does not -- the case that would
        otherwise serve production rosters."""
        import run_season_dashboard  # type: ignore[import-not-found]

        from fantasy_baseball.data.kv_store import SqliteKVStore

        fake = tmp_path / "manual.db"
        SqliteKVStore(fake)
        monkeypatch.setattr(run_season_dashboard._manual_env, "DEFAULT_MANUAL_KV_PATH", fake)

        rc = run_season_dashboard.guard_manual_store(fake.resolve())

        assert rc == run_season_dashboard.RC_REFUSED
        assert MANUAL_PROVENANCE_KEY in capsys.readouterr().out

    def test_accepts_a_seeded_manual_store(self, tmp_path, monkeypatch):
        import run_season_dashboard  # type: ignore[import-not-found]

        seeded = _seeded_store(tmp_path / "manual.db")
        monkeypatch.setattr(run_season_dashboard._manual_env, "DEFAULT_MANUAL_KV_PATH", seeded)

        rc = run_season_dashboard.guard_manual_store(seeded.resolve())

        assert rc == run_season_dashboard.RC_OK


class TestArgparseDecides:
    """The binding used to run at import off a `sys.argv` sniff, which could
    disagree with argparse's prefix matching (`--man` set `args.manual` while
    the store went unbound). It now runs inside `main()` after `parse_args()`,
    so there is one decision-maker and abbreviations cannot desync."""

    @pytest.mark.parametrize("arg", ["--manual", "--manua", "--manu", "--man"])
    def test_abbreviations_reach_args_manual(self, arg):
        import run_season_dashboard  # type: ignore[import-not-found]

        parser = run_season_dashboard.argparse.ArgumentParser()
        parser.add_argument("--no-sync", action="store_true")
        parser.add_argument("--manual", action="store_true")
        parser.add_argument("--port", type=int, default=5001)
        assert parser.parse_args([arg]).manual is True

    def test_importing_the_module_does_not_mutate_the_environment(self, monkeypatch):
        """Importing for its functions must not pop variables or discard the
        KV singleton -- a test that imports it mid-run would lose its own
        isolation."""
        import importlib
        import sys as _sys

        monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", "/sentinel/value")
        _sys.modules.pop("run_season_dashboard", None)
        importlib.import_module("run_season_dashboard")

        assert os.environ["FANTASY_LOCAL_KV_PATH"] == "/sentinel/value"


class TestTheTwoDefaultPathsCannotDrift:
    """`run_manual_refresh` cannot import the constant -- its module-level
    `fantasy_baseball` ban is an AST invariant -- so it spells the path itself.
    Pin the two together, the way `test_run_manual_refresh_guard` already pins
    `PROTECTED_DBS` against the literals it mirrors."""

    def test_the_refresh_script_default_matches_the_shared_constant(self):
        import run_manual_refresh  # type: ignore[import-not-found]

        assert run_manual_refresh.DEFAULT_KV_PATH == DEFAULT_MANUAL_KV_PATH

    def test_the_refresh_script_delegates_rather_than_reimplementing(self):
        """The activation sequence exists once. A second copy is how the two
        entry points drift on a step whose failure is silent."""
        import inspect

        import run_manual_refresh  # type: ignore[import-not-found]

        body = inspect.getsource(run_manual_refresh._activate_manual_environment)
        assert "activate_manual_environment(kv_path)" in body
        assert "os.environ[" not in body, "must not set the variables itself"


class TestTheFlagBindsTheStoreEndToEnd:
    """`activate_manual_environment()` with no argument is otherwise never
    exercised: every other test passes an explicit path. This is the only
    coverage that the DEFAULT binding -- the one `--manual` actually uses --
    reaches the singleton."""

    def test_the_default_binding_reaches_get_kv(self, monkeypatch, seeded_store):
        from fantasy_baseball.data import kv_store
        from fantasy_baseball.manual import environment as env
        from fantasy_baseball.manual.seed import resolve_kv_path

        store = seeded_store()
        monkeypatch.setattr(env, "DEFAULT_MANUAL_KV_PATH", store)
        monkeypatch.delenv("RENDER", raising=False)

        # Build a singleton against a DIFFERENT store first. Without that, the
        # conftest has already cleared it and get_kv() would rebuild from the
        # env var whether or not activate_manual_environment reset anything --
        # so the assertion below would hold for a broken implementation too.
        monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(seeded_store("other.db")))
        kv_store.get_kv()

        bound = env.activate_manual_environment()

        assert bound == store.resolve()
        assert resolve_kv_path(kv_store.get_kv()) == store.resolve(), (
            "the variable was set but get_kv() did not follow it"
        )
        assert os.environ["FB_SKIP_YAHOO"] == "1"

    def test_enter_manual_mode_binds_and_accepts_a_seeded_store(
        self, monkeypatch, seeded_store, capsys
    ):
        """The whole `--manual` path in one call: bind, force no-sync, guard."""
        import argparse

        import run_season_dashboard  # type: ignore[import-not-found]

        store = seeded_store()
        monkeypatch.setattr(run_season_dashboard._manual_env, "DEFAULT_MANUAL_KV_PATH", store)
        monkeypatch.delenv("RENDER", raising=False)
        args = argparse.Namespace(manual=True, no_sync=False, port=5001)

        rc = run_season_dashboard.enter_manual_mode(args)

        assert rc == run_season_dashboard.RC_OK
        assert args.no_sync is True, "--manual must force the sync off"
        out = capsys.readouterr().out
        assert f"KV store: {store.resolve()}" in out
        assert "Manual mode" in out
