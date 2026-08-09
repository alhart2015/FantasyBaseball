from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.trajectory.shape import shape_trajectory
from fantasy_baseball.trajectory.value import best_floor, replacement_for, resolve_slots

LEVELS = {"RP": 7.42, "C": 7.70, "1B": 9.15, "SP": 9.29, "SS": 9.51, "OF": 9.96, "UTIL": 9.96}


def _panel(rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["mlbam_id", "season", "age", "sgp"])


# --------------------------------------------------------------------- floors


def test_a_multi_eligible_player_is_priced_at_his_scarcest_slot() -> None:
    """The roster hole he actually fills. Same rule calculate_var applies."""
    assert best_floor({"OF", "C"}, LEVELS) == ("C", 7.70)
    assert best_floor({"1B", "SS"}, LEVELS) == ("1B", 9.15)


def test_no_eligible_slot_falls_back_to_the_HIGHEST_floor() -> None:
    """A missing lookup must only ever understate a player, never invent value."""
    assert best_floor(set(), LEVELS) == ("UTIL", 9.96)
    assert best_floor({"nonsense"}, LEVELS) == ("UTIL", 9.96)


@pytest.mark.parametrize("position", ["SP", "RP", "1B", "OF", "SS", "C"])
def test_a_bare_STRING_is_refused_not_iterated(position: str) -> None:
    """A str is iterable, so `for s in "SP"` yields "S" and "P", matches no floor, and
    falls through to UTIL -- the feature silently disabled for every position except the
    one-character "C". The CLI passed a string here and nothing caught it, because every
    other test built the set by hand."""
    with pytest.raises(TypeError, match="set of slot names"):
        best_floor(position, LEVELS)
    # And the set form, which is what callers must pass, still works.
    assert best_floor({position}, LEVELS)[0] == position


def test_replacement_for_is_the_same_lookup() -> None:
    assert replacement_for({"C", "OF"}, LEVELS) == best_floor({"C", "OF"}, LEVELS)


# ---------------------------------------------------------------------- slots


def test_a_two_way_players_bat_does_not_borrow_the_pitcher_floor() -> None:
    """The league scores him as two assets. Letting the fielding leaderboard's "P" reach
    the hitter side put Ohtani's BAT on the reliever floor -- 2.54 SGP a year his hitting
    never earned."""
    assert resolve_slots({"P"}, "hitter") == set()
    assert best_floor(resolve_slots({"P"}, "hitter"), LEVELS) == ("UTIL", 9.96)


def test_pitcher_role_comes_from_starts_not_from_the_leaderboard() -> None:
    assert resolve_slots({"P"}, "pitcher", starts=0.0, games=58.0) == {"RP"}
    assert resolve_slots({"P"}, "pitcher", starts=31.0, games=31.0) == {"SP"}
    assert resolve_slots(None, "pitcher", starts=0.0, games=0.0) == {"RP"}


def test_a_short_schedule_starter_is_still_a_starter() -> None:
    """`panel._scale_short_schedules` scales BOTH counts, so the ratio survives. Scaling
    games alone capped 2020's attainable share at 0.37 and made every starter a
    reliever."""
    scale = 162 / 60
    assert resolve_slots({"P"}, "pitcher", starts=12 * scale, games=12 * scale) == {"SP"}


# ------------------------------------------------------- flooring at the source


def _cohort(forward: list[float]) -> pd.DataFrame:
    """One age-27 cohort at 13.0 SGP whose age-28 outcomes are `forward`."""
    return _panel(
        [(i, 2010, 27, 13.0) for i in range(len(forward))]
        + [(i, 2011, 28, v) for i, v in enumerate(forward) if v != 0.0]
    )


# ---------------------------------------------------------- VAR, after #325
#
# Four tests were DELETED here with the comp matchers, not transplanted:
#
#   test_a_departed_comp_is_worth_MINUS_the_floor
#   test_the_shift_reaches_median_spread_and_the_survivor_mean
#   test_survival_is_read_off_the_RAW_line
#   test_no_replacement_leaves_every_number_untouched
#
# Each drove `comp_trajectory` over a hand-built three-comp band to assert that VAR is
# the raw line minus the floor, unclamped, on every statistic. The invariant survives;
# the vehicle does not. `shape` has no hard band -- it kernel-weights the whole panel
# and fits -- so a three-row cohort gives it no predictor variance and it returns n=0.
# Transplanting them produced four tests that failed for a reason unrelated to what
# they assert, which is worse than not having them.
#
# The shape-side statement of the same contract lives in
# test_shape_invariants.test_var_is_the_raw_line_minus_the_floor_unclamped, over
# mean/median/p10/p90/mean_if_survived plus the widths that must NOT move, on a
# population with real variance. That is where a restored clamp gets caught.
#
# test_the_comps_frame_is_on_the_same_scale_as_the_path is gone outright: it asserted
# that `Trajectory.comps`, the frame `--show-comps` printed beneath a comp path, shared
# the aggregates' scale. `shape` populates no such frame, so the assertion has no
# surviving subject.


def test_shape_reports_a_collapsed_veteran_as_NEGATIVE_var() -> None:
    """REVERSED DELIBERATELY (#331, decision by Hart 2026-08-06).

    This asserted `mean >= 0` and `median >= 0`, clamping the prediction and the median
    at zero on top of the response clamp, because "a below-replacement player costs zero,
    not minus a floor -- you drop him and start the replacement".

    Rounding it away is what made a collapsed veteran render identically to a
    replacement-level one, which is the single most decision-relevant distinction on a
    keeper board.

    THE NEGATIVITY is all this asserts. That the shifted values equal `raw - floor` is
    test_mode_parity's contract and is not restated here.
    """
    rng = np.random.default_rng(5)
    rows = []
    for i in range(200):
        peak, down = float(rng.uniform(18, 26)), float(rng.uniform(0, 3))
        rows += [(i, 2010, 32, peak), (i, 2011, 33, down), (i, 2012, 34, float(rng.uniform(0, 4)))]
    panel = _panel(rows)
    kw = {
        "kind": "hitter",
        "age": 33,
        "sgp": 0.5,
        "prior_sgp": 24.0,
        "horizons": (1,),
        "prior_window": 60.0,
    }
    var, _ = shape_trajectory(panel, replacement=9.96, **kw)

    # This cohort produces 0-4 SGP against a floor of 9.96, so every one of them is a
    # negative -- the fixture cannot pass by accident on a clamp-free code path.
    assert var.path[0].mean < 0.0
    assert var.path[0].median < 0.0
    assert var.path[0].p10 < 0.0 and var.path[0].p90 < 0.0


@pytest.mark.parametrize(
    ("position", "pool", "ok"),
    [
        ("C", "hitter", True),
        ("RP", "pitcher", True),
        ("RP", "hitter", False),
        ("C", "pitcher", False),
        ("SP", "hitter", False),
        ("OF", "pitcher", False),
    ],
)
def test_a_position_cannot_price_the_wrong_pool(position: str, pool: str, ok: bool) -> None:
    """`--pool hitter --position RP` printed "RP floor 7.42" over a hitter, 2.54 SGP a
    year stated as fact, because argparse offers one flat list of slots."""
    from fantasy_baseball.trajectory.value import check_position

    assert (check_position(position, pool) is None) is ok
