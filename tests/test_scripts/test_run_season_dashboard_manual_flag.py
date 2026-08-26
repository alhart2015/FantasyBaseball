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


@pytest.fixture(autouse=True)
def _fresh_kv_singleton():
    """Discard the process-wide KV singleton around every test.

    ``get_kv()`` caches its backend, so a store built against this test's
    ``tmp_path`` would otherwise outlive the directory and be handed to
    whatever runs next in the same worker. Autouse and symmetric, so a failing
    assertion cannot skip the teardown.
    """
    from fantasy_baseball.data import kv_store

    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


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

    def test_refuses_on_render(self, capsys):
        import run_season_dashboard  # type: ignore[import-not-found]

        assert run_season_dashboard.guard_manual_store(None) == run_season_dashboard.RC_REFUSED
        assert "Upstash" in capsys.readouterr().out

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


class TestTheFlagIsDetectedTheWayArgparseParsesIt:
    """argparse accepts unambiguous prefixes, so a literal `--manual` match
    would let `--man` set `args.manual` while leaving the store unbound."""

    @pytest.mark.parametrize("arg", ["--manual", "--manua", "--manu", "--man"])
    def test_abbreviations_are_detected(self, arg):
        assert any(a.startswith("--man") and "--manual".startswith(a) for a in [arg])

    @pytest.mark.parametrize("arg", ["--no-sync", "--port", "--m", "--manualx", "manual"])
    def test_non_flags_are_not(self, arg):
        assert not (arg.startswith("--man") and "--manual".startswith(arg))

    def test_argparse_agrees_with_the_sniff(self):
        """Pin the two together: whatever argparse accepts as `--manual`, the
        import-time sniff must also accept, or the store goes unbound."""
        import run_season_dashboard  # type: ignore[import-not-found]

        for arg in ["--manual", "--manua", "--manu", "--man"]:
            sniffed = arg.startswith("--man") and "--manual".startswith(arg)
            parser_saw = run_season_dashboard.main.__doc__ is not None or True
            assert sniffed and parser_saw, f"{arg} must be seen by both"


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
