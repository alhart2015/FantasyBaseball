"""The ``--with-keeper-board`` step on ``scripts/run_manual_refresh.py``.

The keeper (trajectory) board is NOT part of the refresh pipeline and cannot be --
its fit reads ``data/trajectory/*.csv`` and ``data/cache/keeper_skills``, both
gitignored and both absent on Render. So it does not move when the pipeline runs,
and on 2026-08-22 that produced a board blending a fresh 2026-08-22 ROS snapshot
with season-to-date actuals from a panel built on 2026-08-02. Nothing errored; the
board simply described a season three weeks behind the one being played.

The two properties pinned here are the two that failed that day:

1. A stale panel is DETECTED, by comparing the panel's own elapsed-season reading
   against the pipeline's, and rebuilt rather than silently reused.
2. The rebuild targets the panel actually in use. ``build_pt_panel.py`` defaults to
   ``--start 2010``, which writes a different filename than the ``_2000_2026`` file
   ``panel_path`` resolves -- so a default rebuild leaves the stale panel in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_manual_refresh as drv


class _Args:
    def __init__(self, **kw):
        self.with_keeper_board = kw.get("with_keeper_board", True)
        self.skip_panel_rebuild = kw.get("skip_panel_rebuild", False)


@pytest.fixture
def panel_dir(tmp_path, monkeypatch):
    """A panel directory holding BOTH spans, as a real repo does after a default rebuild."""
    d = tmp_path / "trajectory"
    d.mkdir()
    (d / "hitter_pt_panel_2000_2026.csv").write_text("x", encoding="utf-8")
    (d / "hitter_pt_panel_2010_2026.csv").write_text("x", encoding="utf-8")
    from fantasy_baseball.trajectory import panel as panel_mod

    monkeypatch.setattr(panel_mod, "DEFAULT_PANEL_DIR", d)
    return d


def test_panel_span_picks_the_widest_not_the_newest(panel_dir):
    """`panel_path` resolves the WIDEST span; a rebuild must target that same file.

    Ranking by filename string would put _2010_ above _2000_ and rebuild the wrong one.
    """
    span = drv._panel_span()
    assert span is not None
    path, start, end = span
    assert path.name == "hitter_pt_panel_2000_2026.csv"
    assert (start, end) == (2000, 2026)


def test_stale_panel_triggers_a_rebuild_of_the_span_in_use(panel_dir, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(drv, "_run_child", lambda label, argv: calls.append(argv) or 0)
    # panel says 69.8% played, pipeline says 80.0% -- the real 2026-08-22 drift.
    monkeypatch.setattr(drv, "_panel_drift", lambda season: (0.6975, 0.80))

    assert drv._keeper_board(_Args(), 2026) == 0

    assert len(calls) == 2, "expected a panel rebuild AND a board push"
    rebuild = calls[0]
    assert "build_pt_panel.py" in rebuild[0]
    # The whole point: the span in use, not build_pt_panel's 2010 default.
    assert "--start" in rebuild and rebuild[rebuild.index("--start") + 1] == "2000"
    assert "--end" in rebuild and rebuild[rebuild.index("--end") + 1] == "2026"
    assert "--refresh" in rebuild
    assert "push_trajectory_board.py" in calls[1][0]
    assert "--local" in calls[1], "the board must never be pushed to prod from here"


def test_current_panel_is_not_rebuilt(panel_dir, monkeypatch):
    """The two readings use different definitions, so a small gap is normal, not stale."""
    calls: list[list[str]] = []
    monkeypatch.setattr(drv, "_run_child", lambda label, argv: calls.append(argv) or 0)
    monkeypatch.setattr(drv, "_panel_drift", lambda season: (0.8086, 0.80))

    assert drv._keeper_board(_Args(), 2026) == 0
    assert len(calls) == 1
    assert "push_trajectory_board.py" in calls[0][0]


def test_skip_panel_rebuild_warns_instead_of_rebuilding(panel_dir, monkeypatch, capsys):
    calls: list[list[str]] = []
    monkeypatch.setattr(drv, "_run_child", lambda label, argv: calls.append(argv) or 0)
    monkeypatch.setattr(drv, "_panel_drift", lambda season: (0.6975, 0.80))

    assert drv._keeper_board(_Args(skip_panel_rebuild=True), 2026) == 0
    assert len(calls) == 1, "must not rebuild when the operator opted out"
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "stale" in out.lower()


def test_board_push_is_always_local(panel_dir, monkeypatch):
    """--local resolves through FANTASY_LOCAL_KV_PATH, which main() has already set.

    Without it the script's default target is PROD Upstash.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(drv, "_run_child", lambda label, argv: calls.append(argv) or 0)
    monkeypatch.setattr(drv, "_panel_drift", lambda season: None)

    drv._keeper_board(_Args(), 2026)
    assert all("--local" in c for c in calls if "push_trajectory_board.py" in c[0])


def test_missing_panel_fails_the_step_rather_than_pushing_a_board(tmp_path, monkeypatch):
    from fantasy_baseball.trajectory import panel as panel_mod

    monkeypatch.setattr(panel_mod, "DEFAULT_PANEL_DIR", tmp_path / "empty")
    calls: list[list[str]] = []
    monkeypatch.setattr(drv, "_run_child", lambda label, argv: calls.append(argv) or 0)

    assert drv._keeper_board(_Args(), 2026) == drv.RC_FAILED
    assert calls == [], "no board should be written without a panel to fit it on"
