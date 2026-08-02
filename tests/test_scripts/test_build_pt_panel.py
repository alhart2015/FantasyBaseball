from __future__ import annotations

import datetime as dt

from scripts.build_pt_panel import _captured_while_live, _default_end


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


def test_a_cache_written_before_the_season_started_is_partial() -> None:
    """A file predating opening day cannot hold a finished season."""
    assert _captured_while_live(2026, dt.date(2026, 2, 1)) is True


def test_default_end_falls_back_before_opening_day() -> None:
    """Keeper work happens in the offseason, when the current year has no leaderboard."""
    assert _default_end(dt.date(2027, 2, 15)) == 2026
    assert _default_end(dt.date(2027, 3, 31)) == 2026
    assert _default_end(dt.date(2027, 4, 1)) == 2027
    assert _default_end(dt.date(2027, 9, 1)) == 2027
