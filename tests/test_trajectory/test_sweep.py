from __future__ import annotations

from dataclasses import replace

import pytest

from fantasy_baseball.trajectory.board import BoardRow
from fantasy_baseball.trajectory.shape import shape_trajectory
from fantasy_baseball.trajectory.sweep import (
    SWEEP_DRAWS,
    add_ranks,
    from_payload,
    sweep_pool,
    to_payload,
    totals,
)
from tests._trajectory_panel import synthetic_panel


def _rows() -> list[BoardRow]:
    return [
        BoardRow(
            mlbam_id=9001,
            name="Above Replacement",
            pool="hitter",
            age=27,
            sgp=18.0,
            prior_sgp=17.0,
            slot="OF",
            floor=4.0,
        ),
        BoardRow(
            mlbam_id=9002,
            name="Below Replacement",
            pool="hitter",
            age=27,
            sgp=0.0,
            prior_sgp=0.0,
            slot="OF",
            floor=4.0,
        ),
    ]


def test_a_shorter_range_is_a_prefix_of_the_longest_sweep() -> None:
    """The claim the whole cached board rests on (#321).

    One sweep at the longest horizon serves every shorter end year, so the dropdown costs
    no refit. That is only true because the start is LOCKED at base+1: `horizons[0]` stays
    1, the comp mask does not move, and each horizon is fitted independently. If this ever
    fails, the board is quietly answering a different question at every timeframe.
    """
    panel = synthetic_panel()
    rows = _rows()
    long = sweep_pool(rows, panel, "hitter", (1, 2, 3))
    short = sweep_pool(rows, panel, "hitter", (1,))

    assert [p.name for p in long] == [p.name for p in short]
    for a, b in zip(long, short, strict=True):
        for scale in ("var", "sgp"):
            first_of_long = [y for y in a.points(scale) if y.horizon == 1]
            # Exact, not approx: nothing is re-derived, the longer tuple only fits more.
            assert first_of_long == list(b.points(scale))
        assert (a.support, a.extrapolated) == (b.support, b.extrapolated)


def test_a_range_total_is_the_prefix_sum_of_the_cached_years() -> None:
    swept = sweep_pool(_rows(), synthetic_panel(), "hitter", (1, 2, 3))
    one, three = totals(swept, (1,)), totals(swept, (1, 2, 3))
    by_name = {r["name"]: r for r in three}
    for row in one:
        player = next(p for p in swept if p.name == row["name"])
        var = player.points("var")
        assert row["total"] == pytest.approx(var[0].mean)
        assert row["years"] == 1
        assert by_name[row["name"]]["total"] == pytest.approx(sum(y.mean for y in var))


def test_var_is_the_raw_fit_minus_the_floor_on_every_column() -> None:
    """The invariant that replaced the second fit (#331).

    This was the exact opposite assertion: VAR fitted on `max(forward - floor, 0)`, so
    for a below-replacement player VAR read 0.0 and `VAR + floor` reconstructed his SGP
    as exactly the floor -- a number he had nothing to do with. The clamp is gone, so the
    two scales are one fit read twice, and the sweep no longer pays for the second.

    Asserted on the BAND as well as the mean. Shifting only the level and leaving p10/p90
    on the raw scale is the failure that has happened repeatedly on this feature, and it
    renders as a band that does not contain its own estimate.
    """
    swept = sweep_pool(_rows(), synthetic_panel(), "hitter", (1, 2, 3))
    below = next(p for p in swept if p.name == "Below Replacement")
    assert below.floor > 0, "fixture must net against a real floor"

    for raw, var in zip(below.points("sgp"), below.points("var"), strict=True):
        assert var.horizon == raw.horizon
        assert var.mean == pytest.approx(raw.mean - below.floor)
        assert var.p10 == pytest.approx(raw.p10 - below.floor)
        assert var.p90 == pytest.approx(raw.p90 - below.floor)
        # The fit itself did not move, so neither did what describes it.
        assert (var.n_effective, var.band_fell_back) == (raw.n_effective, raw.band_fell_back)

    # And the load-bearing half: he is genuinely under his floor, so an unclamped VAR
    # must be NEGATIVE. Without this the fixture would pass with the clamp restored.
    assert below.points("var")[0].mean < 0.0


def test_totals_serves_both_scales_off_the_one_stored_fit() -> None:
    """There is no longer a scale a sweep can fail to have fitted (#331), so neither
    `totals` nor `points` can hand back a silently empty or mis-scaled row."""
    swept = sweep_pool(_rows(), synthetic_panel(), "hitter", (1,))
    for scale in ("var", "sgp"):
        assert all(p.points(scale) for p in swept)
        assert len(totals(swept, (1,), scale)) == len(swept)


def test_points_rejects_a_scale_it_cannot_serve() -> None:
    swept = sweep_pool(_rows(), synthetic_panel(), "hitter", (1,))
    with pytest.raises(ValueError, match="scale must be one of"):
        swept[0].points("bogus")


def test_a_payload_round_trip_preserves_every_ranked_number() -> None:
    swept = sweep_pool(_rows(), synthetic_panel(), "hitter", (1, 2, 3))
    direct = totals(swept, (1, 2, 3))
    add_ranks(direct)
    restored = totals(from_payload(to_payload(swept, base_season=2026)), (1, 2, 3))
    add_ranks(restored)

    assert [r["name"] for r in direct] == [r["name"] for r in restored]
    for a, b in zip(direct, restored, strict=True):
        assert a["rank_total"] == b["rank_total"]
        assert a["rank_next"] == b["rank_next"]
        # Cached points are rounded, so this is approx -- but at a tolerance far below
        # anything the board prints or sorts on. See `_PRECISION`.
        assert a["total"] == pytest.approx(b["total"], abs=1e-5)
        assert a["p10"] == pytest.approx(b["p10"], abs=1e-5)


def test_a_payload_from_another_schema_is_refused_not_misread() -> None:
    """A compact point is a positional array, so a shape change does not fail loudly on
    its own -- it indexes to the wrong field and produces confident nonsense."""
    payload = to_payload(sweep_pool(_rows(), synthetic_panel(), "hitter", (1,)), base_season=2026)
    payload["version"] = 99
    with pytest.raises(ValueError, match="version"):
        from_payload(payload)


def test_a_two_way_player_keeps_one_line_per_pool() -> None:
    """One MLBAM id, two pools. Anything keyed on the bare id collapses them."""
    rows = [
        BoardRow(660271, "Shohei Ohtani", "hitter", 31, 16.6, 22.6, "UTIL", 4.0),
        BoardRow(660271, "Shohei Ohtani", "pitcher", 31, 13.2, 12.0, "SP", 3.0),
    ]
    swept = sweep_pool(rows[:1], synthetic_panel(), "hitter", (1,)) + sweep_pool(
        rows[1:], synthetic_panel(), "pitcher", (1,)
    )
    scored = totals(swept, (1,))
    assert len({r["id"] for r in scored}) == 1
    assert len({(r["id"], r["pool"]) for r in scored}) == 2


def test_ranks_run_over_the_whole_pool_and_break_ties_by_name() -> None:
    scored = [
        {"name": "B", "total": 5.0, "next": 1.0},
        {"name": "A", "total": 5.0, "next": 2.0},
        {"name": "C", "total": 9.0, "next": float("nan")},
    ]
    add_ranks(scored)
    by_name = {r["name"]: r for r in scored}
    assert by_name["C"]["rank_total"] == 1
    assert (by_name["A"]["rank_total"], by_name["B"]["rank_total"]) == (2, 3)
    # A NaN `next` sorts last rather than poisoning the comparison.
    assert by_name["C"]["rank_next"] == 3


def test_a_swept_row_matches_shape_trajectory_itself() -> None:
    """The one assertion grounding the sweep against the model it wraps.

    The refactor deleted `score()` from scripts/trajectory_board.py, whose row values
    were read straight off a `Trajectory` -- `traj.total`, `sum(p.p10 for p in
    traj.observable)`, `min(p.n_effective ...)`. The replacement re-derives them through
    `_points()` -> `YearPoint` -> `totals()`.

    Every other test here compares one sweep against another sweep -- prefix-of-longest,
    prefix-sum, payload round-trip -- so a wrong field mapping inside `_points` (`median`
    where `mean` was meant, or p10/p90 transposed in BOTH `_pack` and `_unpack`) stays
    perfectly self-consistent and passes all of them. This is the only check that would
    catch it, and scripts/trajectory_board.py has no test file of its own.
    """
    panel = synthetic_panel()
    row = BoardRow(1, "Grounded", "hitter", 27, 18.0, 16.0, "OF", 4.0)
    horizons = (1, 2, 3)

    swept = sweep_pool([row], panel, "hitter", horizons)
    got = totals(swept, horizons, scale="var")[0]

    traj, _ = shape_trajectory(
        panel,
        kind="hitter",
        age=row.age,
        sgp=row.sgp,
        prior_sgp=row.prior_sgp,
        horizons=horizons,
        replacement=row.floor,
        slot=row.slot,
        bootstrap_draws=SWEEP_DRAWS,
    )
    observable = traj.observable

    assert got["total"] == pytest.approx(traj.total, abs=1e-9)
    assert got["p10"] == pytest.approx(sum(p.p10 for p in observable), abs=1e-9)
    assert got["p90"] == pytest.approx(sum(p.p90 for p in observable), abs=1e-9)
    # NOT asserted against the fixture's own n_effective: the synthetic panel yields an
    # identical value at every horizon, so min/max/mean are indistinguishable there. The
    # conservative `min` is pinned separately below, on hand-set points.
    assert got["n_eff"] == pytest.approx(min(p.n_effective for p in observable), abs=1e-9)
    assert got["support"] == pytest.approx(traj.local_support, abs=1e-9)
    assert got["extrapolated"] == traj.extrapolated
    assert got["next"] == pytest.approx(
        next(p.mean for p in observable if p.horizon == 1), abs=1e-9
    )
    assert [c["mean"] for c in got["by_year"]] == [
        pytest.approx(p.mean, abs=1e-9) for p in observable
    ]


def test_the_band_flag_is_scoped_to_the_range_on_screen() -> None:
    """A fallback at +3 must not flag a 1-year board.

    Commit bde75ced exists to stop a latched trajectory-level flag marking a
    well-supported near year unreliable. test_shape.py covers `PathPoint.band_fell_back`
    inside `shape_trajectory`; this covers the consumer side -- `totals()` ORing it over
    the SELECTED horizons only. Reverting to an unfiltered `any(...)` over the whole path
    restores the exact regression, and nothing else would catch it.
    """
    panel = synthetic_panel()
    row = BoardRow(1, "Flagged", "hitter", 27, 18.0, 16.0, "OF", 4.0)
    swept = sweep_pool([row], panel, "hitter", (1, 2, 3))

    player = swept[0]
    # Force a fallback at the FAR horizon only, leaving the near years clean. Patched on
    # the stored RAW path; `points("var")` derives from it and carries the flag through.
    patched = replace(
        player,
        sgp=tuple(replace(p, band_fell_back=(p.horizon == 3)) for p in player.sgp),
    )

    assert totals([patched], (1,), scale="var")[0]["band_fell_back"] is False
    assert totals([patched], (1, 2), scale="var")[0]["band_fell_back"] is False
    assert totals([patched], (1, 2, 3), scale="var")[0]["band_fell_back"] is True


def test_n_eff_reports_the_worst_year_in_the_range() -> None:
    """`min`, deliberately -- support shrinks with the horizon on real data.

    The synthetic panel returns the SAME n_effective at every horizon, so a test built on
    it cannot tell min from max or mean. Hand-set points make the choice observable.
    """
    swept = sweep_pool(
        [BoardRow(1, "Solo", "hitter", 27, 18.0, 16.0, "OF", 4.0)],
        synthetic_panel(),
        "hitter",
        (1, 2, 3),
    )
    player = swept[0]
    thinning = replace(
        player,
        sgp=tuple(replace(pt, n_effective=float(100 - 10 * pt.horizon)) for pt in player.sgp),
    )

    assert [pt.n_effective for pt in thinning.points("var")] == [90.0, 80.0, 70.0]
    assert totals([thinning], (1, 2, 3), scale="var")[0]["n_eff"] == 70.0
    assert totals([thinning], (1, 2), scale="var")[0]["n_eff"] == 80.0
    assert totals([thinning], (1,), scale="var")[0]["n_eff"] == 90.0
