from __future__ import annotations

import pandas as pd
import pytest

from fantasy_baseball.trajectory.comps import comp_trajectory
from fantasy_baseball.trajectory.value import best_floor, resolve_slots, to_var

LEVELS = {"RP": 7.42, "C": 7.70, "1B": 9.15, "SP": 9.29, "SS": 9.51, "OF": 9.96, "UTIL": 9.96}


def _trajectory() -> object:
    panel = pd.DataFrame(
        [(1, 2010, 27, 13.0), (1, 2011, 28, 11.0), (2, 2010, 27, 13.0), (2, 2011, 28, 9.0)],
        columns=["mlbam_id", "season", "age", "sgp"],
    )
    return comp_trajectory(panel, kind="hitter", age=27, sgp=13.0, band=1.0, horizons=(1,))


def test_a_multi_eligible_player_is_priced_at_his_scarcest_slot() -> None:
    """The roster hole he actually fills. Same rule calculate_var applies."""
    assert best_floor({"OF", "C"}, LEVELS) == ("C", 7.70)
    assert best_floor({"1B", "SS"}, LEVELS) == ("1B", 9.15)


def test_no_eligible_slot_falls_back_to_the_HIGHEST_floor() -> None:
    """A missing lookup must only ever understate a player, never invent value."""
    assert best_floor(set(), LEVELS) == ("UTIL", 9.96)
    assert best_floor({"nonsense"}, LEVELS) == ("UTIL", 9.96)


def test_a_two_way_players_bat_does_not_borrow_the_pitcher_floor() -> None:
    """The league scores him as two assets. Letting the fielding leaderboard's "P" reach
    the hitter side put Ohtani's BAT on the reliever floor -- 2.54 SGP a year his hitting
    never earned."""
    assert resolve_slots({"P"}, "hitter") == set()
    assert best_floor(resolve_slots({"P"}, "hitter"), LEVELS) == ("UTIL", 9.96)


def test_pitcher_role_comes_from_starts_not_from_the_leaderboard() -> None:
    # A closer: many appearances, no starts.
    assert resolve_slots({"P"}, "pitcher", starts=0.0, games=58.0) == {"RP"}
    # A starter: every appearance is a start.
    assert resolve_slots({"P"}, "pitcher", starts=31.0, games=31.0) == {"SP"}
    # Never routed off a hitter's games, and never left unresolved.
    assert resolve_slots(None, "pitcher", starts=0.0, games=0.0) == {"RP"}


def test_var_shifts_the_level_and_leaves_the_widths_alone() -> None:
    traj = _trajectory()
    raw = traj.path[0]
    scored, slot, floor = to_var(traj, LEVELS, {"C"})
    point = scored.path[0]
    assert (slot, floor) == ("C", 7.70)
    assert point.mean == pytest.approx(raw.mean - 7.70)
    assert point.median == pytest.approx(raw.median - 7.70)
    # A constant shift changes neither width nor support.
    assert point.se == raw.se
    assert point.spread == raw.spread
    assert (point.n, point.survivors, point.n_effective) == (
        raw.n,
        raw.survivors,
        raw.n_effective,
    )


def test_a_scarcer_slot_is_worth_more_on_the_same_projection() -> None:
    """The whole point: identical raw SGP, different value."""
    traj = _trajectory()
    catcher, _, _ = to_var(traj, LEVELS, {"C"})
    outfielder, _, _ = to_var(traj, LEVELS, {"OF"})
    assert catcher.path[0].mean - outfielder.path[0].mean == pytest.approx(9.96 - 7.70)


def test_an_unsupported_horizon_is_left_alone() -> None:
    """There is no estimate there to net, and shifting NaN would invent one."""
    panel = pd.DataFrame(
        [(1, 2010, 27, 13.0), (1, 2011, 28, 11.0)], columns=["mlbam_id", "season", "age", "sgp"]
    )
    traj = comp_trajectory(
        panel, kind="hitter", age=27, sgp=13.0, band=1.0, horizons=(1, 2), last_complete_season=2011
    )
    scored, _, _ = to_var(traj, LEVELS, {"C"})
    assert scored.path[1].n == 0
    assert scored.path[1] == traj.path[1]
