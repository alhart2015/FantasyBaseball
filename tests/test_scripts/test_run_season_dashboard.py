"""Tests for the ``scripts/run_season_dashboard.py`` startup guard.

The dashboard syncs remote Upstash -> local SQLite once on startup. Under
Flask's debug reloader, ``main()`` runs in BOTH the supervisor and the
reloaded child, so an unguarded sync runs twice (and burns 2x the Upstash
command budget). ``_should_run_sync`` is the predicate that keeps it to
one run.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def test_runs_sync_off_render_outside_reloader(monkeypatch):
    import run_season_dashboard  # type: ignore[import-not-found]

    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
    assert run_season_dashboard._should_run_sync(no_sync=False) is True


def test_skips_sync_when_no_sync_flag(monkeypatch):
    import run_season_dashboard  # type: ignore[import-not-found]

    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
    assert run_season_dashboard._should_run_sync(no_sync=True) is False


def test_skips_sync_on_render(monkeypatch):
    import run_season_dashboard  # type: ignore[import-not-found]

    monkeypatch.setenv("RENDER", "true")
    monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
    assert run_season_dashboard._should_run_sync(no_sync=False) is False


def test_skips_sync_in_reloader_child(monkeypatch):
    """The Werkzeug reloader child carries WERKZEUG_RUN_MAIN=true; the
    supervisor has already synced, so the child must not sync again."""
    import run_season_dashboard  # type: ignore[import-not-found]

    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setenv("WERKZEUG_RUN_MAIN", "true")
    assert run_season_dashboard._should_run_sync(no_sync=False) is False
