from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.trajectory.comps import comp_trajectory
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


def test_a_departed_comp_is_worth_ZERO_not_minus_a_floor() -> None:
    """Subtracting a flat floor afterwards charged attrition the floor a second time:
    measured at age 30 it turned a +0.46 five-year value into -6.01. A man out of the
    league is worth 0 to the slot -- the manager starts the replacement he was already
    being measured against."""
    # One comp produces 12, one left the league (structural 0).
    panel = _cohort([12.0, 0.0])
    traj = comp_trajectory(
        panel, kind="hitter", age=27, sgp=13.0, band=1.0, horizons=(1,), replacement=10.0
    )
    # max(12-10,0)=2 and max(0-10,0)=0  ->  mean 1.0, NOT (6.0 - 10.0) = -4.0
    assert traj.path[0].mean == pytest.approx(1.0)


def test_flooring_reaches_median_spread_and_the_survivor_mean() -> None:
    """A post-hoc shift moved `mean` and left the other three level columns on the raw
    scale, so one printed row mixed VAR and SGP."""
    panel = _cohort([14.0, 12.0, 0.0, 11.0])
    raw = comp_trajectory(panel, kind="hitter", age=27, sgp=13.0, band=1.0, horizons=(1,))
    var = comp_trajectory(
        panel, kind="hitter", age=27, sgp=13.0, band=1.0, horizons=(1,), replacement=10.0
    )
    assert var.path[0].median == pytest.approx(np.median([4.0, 2.0, 0.0, 1.0]))
    assert var.path[0].mean_if_survived == pytest.approx(np.mean([4.0, 2.0, 1.0]))
    assert var.path[0].spread < raw.path[0].spread  # flooring compresses the low tail


def test_survival_is_read_off_the_RAW_line() -> None:
    """After flooring, a below-replacement season and a career ending are both 0. The
    survival column must still tell them apart."""
    panel = _cohort([14.0, 2.0, 0.0])  # one good, one below replacement, one departed
    traj = comp_trajectory(
        panel, kind="hitter", age=27, sgp=13.0, band=1.0, horizons=(1,), replacement=10.0
    )
    assert traj.path[0].survivors == 2
    assert traj.path[0].survival == pytest.approx(2 / 3)


def test_no_replacement_leaves_every_number_untouched() -> None:
    """`--scale sgp` is the default and must not move."""
    panel = _cohort([14.0, 12.0, 0.0])
    # `.path` rather than the whole Trajectory: it carries a DataFrame, which does not
    # compare elementwise to a bool.
    assert (
        comp_trajectory(panel, kind="hitter", age=27, sgp=13.0, band=1.0, horizons=(1,)).path
        == comp_trajectory(
            panel, kind="hitter", age=27, sgp=13.0, band=1.0, horizons=(1,), replacement=0.0
        ).path
    )


def test_shape_fits_on_var_when_given_a_floor() -> None:
    """Flooring the RESPONSE, not shifting the fitted mean, is what keeps a departed comp
    at 0 and keeps every derived statistic on one scale."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(300):
        prior, current = float(rng.uniform(12, 30)), float(rng.uniform(12, 30))
        # Real noise, or both spreads are ~1e-15 and comparing them says nothing.
        forward = 0.5 * current + 0.4 * prior + float(rng.normal(0, 2.0))
        rows += [(i, 2010, 26, prior), (i, 2011, 27, current), (i, 2012, 28, max(forward, 0.0))]
    panel = _panel(rows)
    kw = {
        "kind": "hitter",
        "age": 27,
        "sgp": 15.0,
        "prior_sgp": 15.0,
        "horizons": (1,),
        "prior_window": 60.0,
    }
    raw, _ = shape_trajectory(panel, **kw)
    var, _ = shape_trajectory(panel, replacement=8.0, **kw)
    # Every forward value here clears the floor, so VAR is exactly SGP minus it.
    assert var.path[0].mean == pytest.approx(raw.path[0].mean - 8.0, abs=0.25)
    # The widths are on the same scale, not left behind on the raw one.
    assert var.path[0].spread == pytest.approx(raw.path[0].spread, rel=0.15)


# ------------------------------------------- the class, not the instances


def test_a_trajectory_knows_its_own_scale() -> None:
    """Five review rounds each found a different consumer left on the raw scale -- the
    median, the survivor mean, the comps frame, the headers, the total. The scale rides
    ON the object now, so a reader cannot be wrong about what its numbers mean."""
    panel = _cohort([14.0, 12.0, 0.0])
    raw = comp_trajectory(panel, kind="hitter", age=27, sgp=13.0, band=1.0, horizons=(1,))
    var = comp_trajectory(
        panel,
        kind="hitter",
        age=27,
        sgp=13.0,
        band=1.0,
        horizons=(1,),
        replacement=10.0,
        slot="C",
    )
    assert (raw.scale, raw.slot, raw.floor) == ("sgp", None, 0.0)
    assert (var.scale, var.slot, var.floor) == ("var", "C", 10.0)


def test_the_comps_frame_is_on_the_same_scale_as_the_path() -> None:
    """`--show-comps` prints this frame directly beneath the path. Flooring the
    aggregates and not the frame listed raw SGP under a VAR table, so anyone checking
    the arithmetic got a different mean than the row above."""
    panel = _cohort([14.0, 12.0, 0.0])
    var = comp_trajectory(
        panel, kind="hitter", age=27, sgp=13.0, band=1.0, horizons=(1,), replacement=10.0
    )
    assert sorted(var.comps["h1"]) == [0.0, 2.0, 4.0]
    assert var.comps["h1"].mean() == pytest.approx(var.path[0].mean)


def test_shape_never_predicts_below_replacement() -> None:
    """comp_trajectory is structurally non-negative because it averages floored values;
    shape's prediction is an unconstrained extrapolation and printed -2.35 for a
    collapsed veteran. A below-replacement player costs zero, not minus a floor."""
    rng = np.random.default_rng(5)
    rows = []
    for i in range(200):
        peak, down = float(rng.uniform(18, 26)), float(rng.uniform(0, 3))
        rows += [(i, 2010, 32, peak), (i, 2011, 33, down), (i, 2012, 34, float(rng.uniform(0, 4)))]
    panel = _panel(rows)
    traj, _ = shape_trajectory(
        panel,
        kind="hitter",
        age=33,
        sgp=0.5,
        prior_sgp=24.0,
        horizons=(1,),
        prior_window=60.0,
        replacement=9.96,
    )
    assert traj.path[0].mean >= 0.0
    assert traj.path[0].median >= 0.0


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
