from __future__ import annotations

import datetime as dt
from pathlib import Path

from scripts.build_pt_panel import (
    PROJECT_ROOT,
    _anchor,
    _captured_while_live,
    _default_end,
    _would_narrow,
)


def test_a_cache_written_mid_season_is_recognised_as_partial() -> None:
    """The calendar alone cannot tell: by the next run the season is over, so it is no
    longer 'live' and would be served from cache and trained as a COMPLETE year -- a
    fabricated league-wide collapse season."""
    assert _captured_while_live(2026, dt.date(2026, 8, 2)) is True


def test_a_cache_written_after_the_season_ended_is_complete() -> None:
    assert _captured_while_live(2026, dt.date(2026, 11, 15)) is False
    assert _captured_while_live(2026, dt.date(2027, 3, 1)) is False


def test_an_old_season_cached_long_afterwards_is_complete() -> None:
    assert _captured_while_live(2019, dt.date(2026, 8, 2)) is False


def test_a_relative_out_dir_anchors_to_the_repo_not_the_cwd() -> None:
    """The documented command uses a relative --out-dir; run from anywhere but the repo
    root, a cwd-relative Path would write panels no consumer ever reads."""
    assert _anchor(Path("data/trajectory")) == PROJECT_ROOT / "data" / "trajectory"


def test_an_absolute_out_dir_is_honoured_as_given() -> None:
    absolute = Path(PROJECT_ROOT.anchor) / "elsewhere" / "panels"
    assert _anchor(absolute) == absolute


def test_a_newer_but_narrower_panel_is_detected_as_narrowing(tmp_path) -> None:
    """_panel_path ranks on (end, -start), so 2010-2027 outranks 2000-2026 and silently
    retires the 2000-2009 comps the wider panel exists to hold."""
    (tmp_path / "hitter_pt_panel_2000_2026.csv").write_text("")
    assert _would_narrow(tmp_path, 2010, 2027) == (
        tmp_path / "hitter_pt_panel_2000_2026.csv",
        2000,
        2026,
    )


def test_rebuilding_the_same_span_is_not_narrowing(tmp_path) -> None:
    (tmp_path / "hitter_pt_panel_2000_2026.csv").write_text("")
    assert _would_narrow(tmp_path, 2000, 2027) is None
    assert _would_narrow(tmp_path, 2000, 2026) is None


def test_a_panel_that_does_not_outrank_is_not_narrowing(tmp_path) -> None:
    """It begins later but also ends earlier, so _panel_path keeps preferring the wide
    one and nothing is lost."""
    (tmp_path / "hitter_pt_panel_2000_2026.csv").write_text("")
    assert _would_narrow(tmp_path, 2010, 2020) is None


def test_unparseable_filenames_are_ignored_by_the_narrowing_check(tmp_path) -> None:
    (tmp_path / "hitter_pt_panel_backup.csv").write_text("")
    assert _would_narrow(tmp_path, 2010, 2027) is None


def test_a_cache_written_before_the_season_started_is_partial() -> None:
    """A file predating opening day cannot hold a finished season."""
    assert _captured_while_live(2026, dt.date(2026, 2, 1)) is True


def test_default_end_falls_back_before_opening_day() -> None:
    """Keeper work happens in the offseason, when the current year has no leaderboard."""
    assert _default_end(dt.date(2027, 2, 15)) == 2026
    assert _default_end(dt.date(2027, 3, 31)) == 2026
    assert _default_end(dt.date(2027, 4, 1)) == 2027
    assert _default_end(dt.date(2027, 9, 1)) == 2027
