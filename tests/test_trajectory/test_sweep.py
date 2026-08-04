from __future__ import annotations

import pytest

from fantasy_baseball.trajectory.board import BoardRow
from fantasy_baseball.trajectory.sweep import (
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
        assert row["total"] == pytest.approx(player.var[0].mean)
        assert row["years"] == 1
        assert by_name[row["name"]]["total"] == pytest.approx(sum(y.mean for y in player.var))


def test_raw_sgp_is_a_separate_fit_not_var_plus_the_floor() -> None:
    """The reason the sweep pays for two fits per player.

    `shape_trajectory(replacement=floor)` fits on `max(forward - floor, 0)`, so the floor
    is baked into the response AND clamped. For a below-replacement player VAR is 0, and
    reconstructing his SGP as `VAR + floor` would report it as exactly the floor -- a
    number he has nothing to do with.
    """
    swept = sweep_pool(_rows(), synthetic_panel(), "hitter", (1,))
    below = next(p for p in swept if p.name == "Below Replacement")
    assert below.var[0].mean == 0.0, "expected the VAR fit to clamp at zero"
    # The wrong reconstruction lands exactly on the floor, which is the tell.
    assert below.var[0].mean + below.floor == pytest.approx(below.floor)
    assert below.sgp[0].mean > 0.0
    assert below.sgp[0].mean != pytest.approx(below.floor, abs=0.1)


def test_the_var_only_sweep_skips_the_second_fit() -> None:
    swept = sweep_pool(_rows(), synthetic_panel(), "hitter", (1,), scales=("var",))
    assert all(p.var for p in swept)
    assert all(p.sgp == () for p in swept)
    # A caller then asking for the scale that was never fitted gets nothing, not a
    # silently mis-scaled row.
    assert totals(swept, (1,), "sgp") == []


@pytest.mark.parametrize("scales", [(), ("sgp",), ("var", "bogus")])
def test_rejects_a_scale_set_it_cannot_serve(scales: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="scales must be"):
        sweep_pool(_rows(), synthetic_panel(), "hitter", (1,), scales=scales)


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
