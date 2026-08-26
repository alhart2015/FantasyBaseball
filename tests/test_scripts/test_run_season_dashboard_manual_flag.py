"""Tests for ``run_season_dashboard.py --manual``.

The flag exists so opening the dashboard against the hand-transcribed store is
one command rather than two environment variables plus ``--no-sync``. Both
halves of that are load-bearing and fail silently if wrong:

* Binding must happen before anything resolves a KV store, because ``get_kv()``
  caches on its first call. A flag that parsed correctly but bound late would
  serve the Yahoo baseline under a "manual" banner.
* The sync must not run. ``sync_remote_to_local()`` wipes its destination before
  refilling, so on ``data/manual.db`` it destroys the transcription.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fantasy_baseball.manual.environment import (
    DEFAULT_MANUAL_KV_PATH,
    activate_manual_environment,
    deactivate_manual_environment,
)


class TestActivateManualEnvironment:
    def test_binds_an_absolute_path(self, tmp_path, monkeypatch):
        """`kv_store` resolves FANTASY_LOCAL_KV_PATH against the CWD, so a
        relative value silently creates a second, empty store."""
        monkeypatch.delenv("FANTASY_LOCAL_KV_PATH", raising=False)
        monkeypatch.delenv("FB_SKIP_YAHOO", raising=False)
        target = tmp_path / "manual.db"

        resolved = activate_manual_environment(target)

        import os

        assert Path(os.environ["FANTASY_LOCAL_KV_PATH"]).is_absolute()
        assert resolved == target.resolve()
        assert os.environ["FB_SKIP_YAHOO"] == "1"

    def test_defaults_to_the_repo_manual_store(self, monkeypatch):
        monkeypatch.delenv("FANTASY_LOCAL_KV_PATH", raising=False)
        resolved = activate_manual_environment()
        assert resolved == DEFAULT_MANUAL_KV_PATH.resolve()
        assert resolved.name == "manual.db"

    def test_never_binds_the_yahoo_baseline(self, monkeypatch):
        """data/local.db is the only copy of the pre-outage Yahoo history."""
        monkeypatch.delenv("FANTASY_LOCAL_KV_PATH", raising=False)
        resolved = activate_manual_environment()
        assert resolved.name != "local.db"

    def test_rebinds_a_singleton_something_else_already_built(self, tmp_path, monkeypatch):
        """The whole point of the reset: a process that already resolved a
        store keeps the old binding, with no error, unless it is discarded."""
        from fantasy_baseball.data import kv_store

        monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "wrong.db"))
        kv_store._reset_singleton()
        kv_store.get_kv()  # bind to the wrong store first

        activate_manual_environment(tmp_path / "right.db")

        from fantasy_baseball.manual.seed import resolve_kv_path

        assert resolve_kv_path(kv_store.get_kv()) == (tmp_path / "right.db").resolve()
        kv_store._reset_singleton()


class TestManualImpliesNoSync:
    def test_manual_forces_no_sync(self, monkeypatch):
        import run_season_dashboard  # type: ignore[import-not-found]

        monkeypatch.delenv("RENDER", raising=False)
        monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
        # main() sets args.no_sync = True when --manual is given; the predicate
        # it feeds must then refuse the sync.
        assert run_season_dashboard._should_run_sync(no_sync=True) is False


class TestGuardManualBinding:
    def test_accepts_the_manual_store(self):
        import run_season_dashboard  # type: ignore[import-not-found]

        rc = run_season_dashboard.guard_manual_binding(DEFAULT_MANUAL_KV_PATH.resolve())
        assert rc == run_season_dashboard.RC_OK

    def test_refuses_the_yahoo_baseline(self, capsys):
        """The failure this exists to catch: --manual parsed, binding did not
        happen, dashboard would serve local.db under a 'manual' banner."""
        import run_season_dashboard  # type: ignore[import-not-found]

        rc = run_season_dashboard.guard_manual_binding(PROJECT_ROOT / "data" / "local.db")
        assert rc == run_season_dashboard.RC_REFUSED
        out = capsys.readouterr().out
        assert "did not bind the manual store" in out
        assert "local.db" in out

    def test_refuses_on_render(self, capsys):
        import run_season_dashboard  # type: ignore[import-not-found]

        rc = run_season_dashboard.guard_manual_binding(None)
        assert rc == run_season_dashboard.RC_REFUSED
        assert "Upstash" in capsys.readouterr().out


class TestEndToEnd:
    @pytest.mark.skipif(
        not DEFAULT_MANUAL_KV_PATH.exists(), reason="no manual store on this machine"
    )
    def test_the_flag_binds_the_manual_store_in_a_real_process(self):
        """Launch the script for real, in a clean environment, and read the
        banner it prints. This is the only test that exercises the sys.argv
        sniff at its actual import-time position."""
        probe = (
            "import sys; sys.argv = ['run_season_dashboard.py', '--manual'];\n"
            "sys.path.insert(0, r'%s');\n"
            "import run_season_dashboard as d;\n"
            "p, desc = d.resolve_kv_target();\n"
            "print('RESOLVED', p);\n"
            "print('GUARD', d.guard_manual_binding(p));\n"
            "import os; print('SKIP_YAHOO', os.environ.get('FB_SKIP_YAHOO'))\n"
        ) % (PROJECT_ROOT / "scripts")
        env = {
            k: v
            for k, v in __import__("os").environ.items()
            if k not in ("FANTASY_LOCAL_KV_PATH", "FB_SKIP_YAHOO", "RENDER")
        }
        out = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
            timeout=180,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        assert f"RESOLVED {DEFAULT_MANUAL_KV_PATH.resolve()}" in out.stdout, out.stdout
        assert "GUARD 0" in out.stdout, out.stdout
        assert "SKIP_YAHOO 1" in out.stdout, out.stdout


class TestNoFlagClearsAnInheritedBinding:
    """The half that is easy to miss: these variables are EXPORTED, so they
    outlive the command that set them. Without an explicit clear, a launcher
    run with no flag inherits the last manual session and serves the
    transcription while the caller believes they are reading Yahoo.
    """

    def test_clears_both_variables_and_reports_them(self, monkeypatch, tmp_path):
        import os

        monkeypatch.delenv("RENDER", raising=False)
        monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "manual.db"))
        monkeypatch.setenv("FB_SKIP_YAHOO", "1")

        cleared = deactivate_manual_environment()

        assert set(cleared) == {"FANTASY_LOCAL_KV_PATH", "FB_SKIP_YAHOO"}
        assert "FANTASY_LOCAL_KV_PATH" not in os.environ
        assert "FB_SKIP_YAHOO" not in os.environ

    def test_rebinds_to_the_yahoo_baseline(self, monkeypatch, tmp_path):
        """Clearing the variable is not enough on its own -- the singleton
        already built against it has to be discarded too."""
        from fantasy_baseball.data import kv_store
        from fantasy_baseball.manual.seed import resolve_kv_path

        monkeypatch.delenv("RENDER", raising=False)
        monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "manual.db"))
        kv_store._reset_singleton()
        kv_store.get_kv()  # bound to the manual store

        deactivate_manual_environment()

        assert resolve_kv_path(kv_store.get_kv()).name == "local.db"
        kv_store._reset_singleton()

    def test_reports_nothing_when_the_shell_was_already_clean(self, monkeypatch):
        monkeypatch.delenv("RENDER", raising=False)
        monkeypatch.delenv("FANTASY_LOCAL_KV_PATH", raising=False)
        monkeypatch.delenv("FB_SKIP_YAHOO", raising=False)
        assert deactivate_manual_environment() == {}

    def test_is_a_no_op_on_render(self, monkeypatch):
        """On Render the KV is Upstash, FANTASY_LOCAL_KV_PATH cannot reach it,
        and FB_SKIP_YAHOO may be a deliberate service setting."""
        import os

        monkeypatch.setenv("RENDER", "true")
        monkeypatch.setenv("FB_SKIP_YAHOO", "1")
        assert deactivate_manual_environment() == {}
        assert os.environ["FB_SKIP_YAHOO"] == "1"
