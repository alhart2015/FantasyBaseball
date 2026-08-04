from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.trajectory.board import BoardRow
from fantasy_baseball.trajectory.sweep import sweep_pool, to_payload
from fantasy_baseball.web.trajectory_view import DEFAULT_TOP, build_board

BASE = 2026


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(160):
        level = float(rng.uniform(4.0, 22.0))
        for offset, season in enumerate(range(2010, 2019)):
            rows.append((i, season, 24 + offset, max(level + float(rng.normal(0, 2.0)), 0.0)))
    return pd.DataFrame(rows, columns=["mlbam_id", "season", "age", "sgp"])


@pytest.fixture(scope="module")
def payload() -> dict:
    """A four-player board -- two hitters, two pitchers -- swept to three years."""
    panel = _panel()
    hitters = [
        BoardRow(1, "Big Bat", "hitter", 27, 20.0, 19.0, "OF", 4.0),
        BoardRow(2, "Small Bat", "hitter", 27, 8.0, 7.0, "OF", 4.0),
        # Observable but extrapolated: a 24.0 season off a 5.0 prior matches the
        # low-prior cohort and is then evaluated far above their own current seasons,
        # so local_support is 0 and the row carries the (!) flag. On the live board
        # this shape lands in the top five (CJ Abrams, 6.4%), so a fixture without one
        # cannot exercise anything that treats flagged rows differently.
        BoardRow(9, "Thin Support", "hitter", 27, 24.0, 5.0, "OF", 4.0),
    ]
    pitchers = [
        BoardRow(3, "Big Arm", "pitcher", 27, 18.0, 17.0, "SP", 3.0),
        BoardRow(4, "Small Arm", "pitcher", 27, 6.0, 5.0, "RP", 1.0),
    ]
    swept = sweep_pool(hitters, panel, "hitter", (1, 2, 3)) + sweep_pool(
        pitchers, panel, "pitcher", (1, 2, 3)
    )
    return to_payload(
        swept,
        base_season=BASE,
        max_horizon=3,
        min_sgp=0.0,
        season_elapsed=0.7,
        generated_at="2026-08-04T09:00:00-04:00",
        panel_vintage={"hitter": "h.csv", "pitcher": "p.csv"},
        floors={"OF": 4.0},
        excluded={"low_sgp": 7, "no_current_line": 12, "total": 19},
    )


def test_the_start_year_is_locked_and_only_the_end_moves(payload: dict) -> None:
    board = build_board(payload, end=BASE + 3)
    assert board.end_years == [2027, 2028, 2029]
    assert board.span == "2027-29"
    assert board.year_columns == [2027, 2028, 2029]
    assert all(len(r["by_year"]) == 3 for r in board.rows)


def test_a_single_year_board_carries_no_per_year_columns(payload: dict) -> None:
    """They would repeat the total, column for column."""
    board = build_board(payload, end=BASE + 1)
    assert board.span == "2027"
    assert board.year_columns == []


def test_a_shorter_end_year_is_a_prefix_of_the_longer_one(payload: dict) -> None:
    one = {r["name"]: r for r in build_board(payload, end=BASE + 1).rows}
    three = {r["name"]: r for r in build_board(payload, end=BASE + 3).rows}
    for name, row in one.items():
        assert row["total"] == pytest.approx(three[name]["by_year"][0]["mean"], abs=1e-5)
        assert three[name]["total"] >= row["total"]


def test_ranks_are_league_wide_even_when_the_pool_is_filtered(payload: dict) -> None:
    """The rule #322 and #323 both depend on: a rank is a position among EVERYONE, so a
    filtered view can legitimately open at #2. Re-ranking inside a subset would make the
    top row of every slice read #1."""
    everyone = build_board(payload, end=BASE + 3)
    pitchers = build_board(payload, end=BASE + 3, pool="pitcher")

    league = {r["name"]: r["rank_total"] for r in everyone.rows}
    assert [r["name"] for r in pitchers.rows] == ["Big Arm", "Small Arm"]
    assert [r["rank_total"] for r in pitchers.rows] == [
        league["Big Arm"],
        league["Small Arm"],
    ]
    assert pitchers.rows[0]["rank_total"] != 1, "expected a league rank, not a within-pool one"


def test_the_scale_toggle_renumbers_the_rank(payload: dict) -> None:
    """The whole table is ONE scale, so the rank must be a rank on that scale.

    This replaces an earlier rule where `#` stayed a VAR rank under an SGP sort. That
    made sense when SGP was a second column beside VAR; with a toggle the reader is
    looking at one board, and a number that ranks by something not on screen is worse
    than no number. Raw SGP misranks catchers and relievers, which is why VAR remains
    the default -- but the SGP view now shows SGP ranks.
    """
    by_var = build_board(payload, end=BASE + 3, scale="var")
    by_sgp = build_board(payload, end=BASE + 3, scale="sgp")

    for board in (by_var, by_sgp):
        assert [r["rank_total"] for r in board.rows] == sorted(
            r["rank_total"] for r in board.rows
        ), "rows render in rank order"
        assert [r["total"] for r in board.rows] == sorted(
            (r["total"] for r in board.rows), reverse=True
        ), "and rank order is the displayed scale's order"

    assert {r["name"]: r["rank_total"] for r in by_var.rows} != {
        r["name"]: r["rank_total"] for r in by_sgp.rows
    }, "VAR and SGP disagree on this fixture, so the ranks must differ"


def test_now_is_rendered_on_the_selected_scale(payload: dict) -> None:
    """VAR is SGP minus the replacement level -- including for the current season.

    Deliberately NOT clamped at zero: a player already below his slot's waiver floor is
    exactly what a keeper reader needs to see, and clamping would render him identical
    to a replacement-level one.
    """
    by_var = {r["name"]: r for r in build_board(payload, end=BASE + 3, scale="var").rows}
    by_sgp = {r["name"]: r for r in build_board(payload, end=BASE + 3, scale="sgp").rows}

    for name, sgp_row in by_sgp.items():
        var_row = by_var[name]
        assert var_row["now"] == pytest.approx(sgp_row["now"] - var_row["floor"])
        assert var_row["floor"] > 0


@pytest.mark.parametrize(
    ("end", "expected"),
    [("nonsense", 2027), (1999, 2027), (2099, 2029), (None, 2027)],
)
def test_a_junk_or_out_of_range_end_year_falls_back_instead_of_500ing(
    payload: dict, end: object, expected: int
) -> None:
    """These arrive from a URL a reader can edit."""
    assert build_board(payload, end=end).end_year == expected


def test_top_all_shows_every_scored_row(payload: dict) -> None:
    board = build_board(payload, top="all")
    assert board.top == "all"
    assert len(board.rows) == board.scored == 5


def test_the_default_top_is_the_web_default_not_the_cli_one(payload: dict) -> None:
    assert build_board(payload).top == DEFAULT_TOP == 50


def test_extrapolated_rows_are_flagged_and_always_shown(payload: dict) -> None:
    """A thin-support row is the ambiguous keeper call, not noise to be filtered.

    These cluster at the TOP of the live board -- extrapolation inflates the estimate,
    so the rows the model is least sure about land where the decisions get made (James
    Wood #3, CJ Abrams #5 on the 2026 board). Dropping them leaves a board that looks
    authoritative and is missing exactly the players worth arguing about, so the board
    flags them and shows them. There is deliberately no filter to remove them.
    """
    board = build_board(payload, top="all")
    flagged = [r for r in board.rows if r["extrapolated"]]

    assert flagged, "fixture must contain an extrapolated row for this to mean anything"
    assert {r["name"] for r in flagged} == {"Thin Support"}
    assert board.scored == len(board.rows), "every scored row is rendered"

    with pytest.raises(TypeError):
        build_board(payload, hide_unsupported=True)  # type: ignore[call-arg]


def test_the_band_belongs_to_the_scale_on_screen(payload: dict) -> None:
    """One scale on screen, so one band -- and it must be that scale's own fit.

    The earlier layout showed both totals with a single VAR band beside them; measured
    on the live board, 1165 of 1169 rows had their SGP total fall outside the interval
    printed next to it, always optimistically. A single-scale table makes that
    structurally impossible, and this pins it.
    """
    for scale in ("var", "sgp"):
        board = build_board(payload, end=BASE + 3, scale=scale)
        assert board.rows
        for r in board.rows:
            assert r["p10"] <= r["total"] <= r["p90"], (
                f"{scale}: {r['name']} total {r['total']:.1f} outside its own band "
                f"{r['p10']:.1f}..{r['p90']:.1f}"
            )

    var_bands = {r["name"]: (r["p10"], r["p90"]) for r in build_board(payload, scale="var").rows}
    sgp_bands = {r["name"]: (r["p10"], r["p90"]) for r in build_board(payload, scale="sgp").rows}
    assert var_bands != sgp_bands, "the two scales are separate fits with separate bands"


def test_the_board_reports_who_it_left_out(payload: dict) -> None:
    """A shortened board reads as "these are the best players" when it is "these are the
    ones the model can price". The CLI already prints its drop count; the page must too.

    Two separate exclusions, and the second is the larger: players cut by the min-SGP
    gate, and players with no current-season line at all who were never candidates
    (measured on the 2026 board: 171 and 390).
    """
    board = build_board(payload)
    excluded = board.meta["excluded"]

    assert excluded["low_sgp"] == 7
    assert excluded["no_current_line"] == 12
    assert excluded["total"] == 19


def test_a_cache_written_by_another_schema_raises_rather_than_rendering_empty(
    payload: dict,
) -> None:
    stale = dict(payload, version=99)
    with pytest.raises(ValueError, match="version"):
        build_board(stale)
