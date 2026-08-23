"""Block headline is the KEEP total, not the best-N total.

The By-team view deliberately shows MORE players than a team may keep -- opponents
without this board may keep someone outside their true best-N, and the rows past the
cut are the only way to see that. But summing those extra rows into the headline
counts players nobody can retain, and on 2026-08-22 that inverted the league ordering:
Hello Peanuts! led the best-5 measure 67.4 to 55.8 and TRAILED on best-3, 43.0 to 45.1.
"""

from __future__ import annotations

import pytest

from fantasy_baseball.web.trajectory_view import DEFAULT_KEEP, TeamBlock


def _block(totals, keep=DEFAULT_KEEP, team="T"):
    rows = [{"name": f"P{i}", "total": t, "rank_total": i + 1} for i, t in enumerate(totals)]
    return TeamBlock(team=team, rows=rows, scored=len(rows), unscored=[], is_mine=False, keep=keep)


def test_keep_total_sums_only_the_keepers():
    b = _block([10.0, 8.0, 6.0, 5.0, 4.0], keep=3)
    assert b.keep_total == pytest.approx(24.0)
    assert b.total == pytest.approx(33.0), "the depth figure still sums every row shown"


def test_the_headline_can_invert_the_depth_ordering():
    """The real 2026-08-22 numbers, rounded: depth says one team, keepers say the other."""
    peanuts = _block([15.46, 14.30, 13.25, 12.96, 11.41], keep=3, team="Hello Peanuts!")
    mine = _block([17.91, 15.38, 11.83, 5.41, 5.30], keep=3, team="Hart of the Order")

    assert peanuts.total > mine.total, "the fixture must reproduce the depth lead"
    assert mine.keep_total > peanuts.keep_total, "and the keeper lead must be the other way"


def test_stranded_counts_only_positive_value_past_the_cut():
    """A negative row past the cut costs nothing to lose -- crediting it would reward junk."""
    b = _block([10.0, 8.0, 6.0, 4.0, -3.0], keep=3)
    assert b.stranded == pytest.approx(4.0)


def test_stranded_is_zero_when_nothing_is_past_the_cut():
    assert _block([10.0, 8.0, 6.0], keep=3).stranded == pytest.approx(0.0)


def test_keep_dropoff_is_the_cost_of_keeping_the_wrong_player():
    """Near-zero means no exploitable mistake exists, however deep the team looks."""
    tight = _block([15.46, 14.30, 13.25, 12.96], keep=3)
    clear = _block([17.91, 15.38, 11.83, 5.41], keep=3)
    assert tight.keep_dropoff == pytest.approx(0.29, abs=0.01)
    assert clear.keep_dropoff == pytest.approx(6.42, abs=0.01)


def test_keep_dropoff_is_none_with_no_row_past_the_cut():
    """None, not 0.0 -- 'no fourth player' is a different fact from 'no gap'."""
    assert _block([10.0, 8.0, 6.0], keep=3).keep_dropoff is None


def test_keep_larger_than_the_rows_shown_degrades_to_the_block_total():
    b = _block([10.0, 8.0], keep=3)
    assert b.keep_total == pytest.approx(b.total)
    assert b.stranded == pytest.approx(0.0)
