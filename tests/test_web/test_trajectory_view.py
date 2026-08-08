from __future__ import annotations

import itertools
import math
import re
from typing import Any

import pytest

from fantasy_baseball.data.rosters import RosterSpot
from fantasy_baseball.trajectory.board import BoardRow
from fantasy_baseball.trajectory.sweep import (
    RANK_MOVE,
    chart_key,
    sweep_pool,
    to_chart_payload,
    to_payload,
)
from fantasy_baseball.web.trajectory_view import (
    DEFAULT_COMPS,
    DEFAULT_TOP,
    PlayerView,
    build_board,
    build_player_view,
    build_teams_board,
    find_players,
)
from tests._trajectory_panel import synthetic_panel

BASE = 2026
_HAND_SEQ = itertools.count()


def _spot(name: str, team: str, pool: str = "hitter", status: str = "") -> RosterSpot:
    """One roster spot. Module level so the seven tests below stop re-spelling the
    six-field constructor -- and the `normalized=name.lower()` convention -- by hand.
    Mirrors the helper of the same name in tests/test_trajectory/test_roster_join.py.
    """
    return RosterSpot(
        name=name,
        normalized=name.lower(),
        player_type=pool,
        team=team,
        yahoo_id="0",
        status=status,
    )


@pytest.fixture(scope="module")
def payload() -> dict:
    """A four-player board -- two hitters, two pitchers -- swept to three years."""
    panel = synthetic_panel()
    hitters = [
        BoardRow(1, "Big Bat", "hitter", 27, 20.0, 19.0, "OF", 4.0),
        BoardRow(2, "Small Bat", "hitter", 27, 8.0, 7.0, "OF", 4.0),
        # Observable but extrapolated: a 24.0 season off a 5.0 prior matches the
        # low-prior cohort and is then evaluated far above their own current seasons,
        # so local_support is 0 and the row carries the (!) flag. On the live board
        # this shape lands in the top five (CJ Abrams, 6.4%), so a fixture without one
        # cannot exercise anything that treats flagged rows differently.
        BoardRow(9, "Thin Support", "hitter", 27, 24.0, 5.0, "OF", 4.0),
        # Below his slot's floor, so `now` on the VAR scale is negative. Without him the
        # "deliberately not clamped" rule is unreachable and re-adding a clamp passes.
        BoardRow(10, "Under Water", "hitter", 27, 2.5, 3.0, "C", 6.0),
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
    """The one-year board must be the first term of the three-year one.

    It used to also assert the three-year total was the LARGER of the two, which held
    only while VAR was clamped at zero and every added year could therefore only add. A
    below-replacement year is now a negative one (#331), so a longer range legitimately
    scores lower -- that is the keeper signal, not a violation of the prefix rule.

    It also briefly asserted `three.total == sum(three.by_year means)`, which cannot
    fail: `totals` builds both from the same filtered `points` list, so the claim is one
    expression compared against itself. Every assertion here now spans SEPARATELY BUILT
    boards, which is the only way the range filter can be caught disagreeing with itself.
    """
    boards = {k: {r["name"]: r for r in build_board(payload, end=BASE + k).rows} for k in (1, 2, 3)}
    three = boards[3]

    for name, row in boards[1].items():
        assert row["total"] == pytest.approx(three[name]["by_year"][0]["mean"], abs=1e-5)

    for k in (1, 2, 3):
        for name, row in boards[k].items():
            # The k-year board is the first k years of the three-year one -- compared
            # across two independent `build_board` calls, not within one row.
            assert row["total"] == pytest.approx(
                sum(c["mean"] for c in three[name]["by_year"][:k]), abs=1e-5
            ), f"{name}: the {k}-year total is not the first {k} years of the three-year board"
            assert len(row["by_year"]) in (0, k), "a range rendered a different number of years"

    for k in (2, 3):
        for name, row in boards[k].items():
            # And each step adds exactly its own year. An off-by-one in the
            # `p.horizon in wanted` filter shows up here as a doubled or skipped year.
            assert row["total"] == pytest.approx(
                boards[k - 1][name]["total"] + three[name]["by_year"][k - 1]["mean"], abs=1e-5
            ), f"{name}: year {k} did not add its own mean"


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

    # The load-bearing half: a player BELOW his floor must read negative. Without such a
    # row in the fixture the clamp is unreachable and re-adding it would pass unnoticed.
    below = by_var["Under Water"]
    assert below["now"] < 0, "a below-replacement Now must not be clamped to zero"
    assert below["now"] == pytest.approx(by_sgp["Under Water"]["now"] - below["floor"])


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
    assert len(board.rows) == board.scored == 6


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
    assert var_bands != sgp_bands, "each scale's band is netted against its own floor"


def _hand_payload(players: list[tuple[str, list[float]]]) -> dict:
    """A payload with exact per-year means, so rank arithmetic can be pinned directly.

    Points are named keys, matching `_pack`; the band and support are irrelevant here
    and set wide/high so nothing else flags.

    The floor is ZERO so that `means` are the numbers the default (VAR) board reads. Only
    the raw fit is stored now and VAR is derived as `sgp - floor` (#331), so a non-zero
    floor here would silently shift every value these tests pin.
    """
    return {
        "base_season": BASE,
        "max_horizon": 3,
        # Unique per call: real payloads always carry one, and sharing it would let two
        # hand-built fixtures collide in the derived-state cache.
        "generated_at": f"hand-{next(_HAND_SEQ)}",
        "players": [
            {
                "id": i,
                "name": name,
                "pool": "hitter",
                "age": 27,
                "slot": "OF",
                "floor": 0.0,
                "now": 10.0,
                "prior": 10.0,
                "support": 0.9,
                "extrapolated": 0,
                "sgp": [
                    {
                        "horizon": h,
                        "mean": m,
                        "p10": m - 5,
                        "p90": m + 5,
                        "n_effective": 50.0,
                        "band_fell_back": 0,
                    }
                    for h, m in enumerate(means, start=1)
                ],
            }
            for i, (name, means) in enumerate(players, start=1)
        ],
    }


def test_the_toggle_cannot_reorder_players_who_share_a_slot(payload: dict) -> None:
    """The bug #331 was opened on, at the surface it was seen on.

    Flipping VAR/SGP visibly reordered same-slot players -- CJ Abrams above Elly De La
    Cruz on one scale and below him on the other, both shortstops, both netting the same
    floor. It happened because VAR was a SECOND FIT on a response clamped at zero, which
    flattened every sub-floor comp and so changed the fitted slope; the shift was per
    query rather than constant, and any pair closer together than the difference could
    swap. Pitchers reshuffled worst, their floors being high against a narrow spread.

    VAR is now the raw fit minus one number per slot, so this holds by construction. The
    test earns its place anyway: it is the reader-visible statement of the invariant, and
    it fails the moment anything nonlinear -- a clamp, a per-player floor, a second fit --
    comes back.
    """
    by_var = build_board(payload, end=BASE + 3, scale="var", top="all").rows
    by_sgp = build_board(payload, end=BASE + 3, scale="sgp", top="all").rows

    def order_within_slots(rows: list[dict]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in sorted(rows, key=lambda r: r["rank_total"]):
            out.setdefault(r["slot"], []).append(r["name"])
        return out

    var_order, sgp_order = order_within_slots(by_var), order_within_slots(by_sgp)
    shared = [slot for slot, names in var_order.items() if len(names) > 1]
    assert shared, "fixture must put more than one player in some slot, or this is vacuous"
    for slot in shared:
        assert var_order[slot] == sgp_order[slot], (
            f"{slot} players share a floor, so subtracting it cannot reorder them"
        )


def test_the_arrow_threshold_is_the_boundary_it_claims() -> None:
    """RANK_MOVE itself was unpinned -- deleting it drew an arrow on every 1-rank wobble.

    Two new tests appeared to cover `rank_move`, but one only exercised the all-zero
    guard and the other the NaN guard; the threshold that decides which real movements
    are worth drawing had no test anywhere in the repo. It is a tuned number -- 5 places
    was signal on a 25-row CLI dump and is plausibly noise across 1,169 rows -- so it is
    exactly the kind of constant someone will change.

    Built so one player's two rankings differ by precisely RANK_MOVE - 1 and RANK_MOVE.
    """

    def move_for(gap: int) -> int:
        # Subject tops the TOTAL ranking; `next` puts him `gap` places down.
        players = [("Subject", [1.0, 90.0, 90.0])]
        players += [(f"Peer {i}", [9.0 - i * 0.1, 0.0, 0.0]) for i in range(gap)]
        players += [(f"Tail {i}", [0.5, 0.0, 0.0]) for i in range(3)]
        board = build_board(_hand_payload(players), top="all", end=BASE + 3)
        row = next(r for r in board.rows if r["name"] == "Subject")
        assert row["rank_total"] == 1
        assert row["rank_next"] - row["rank_total"] == gap, "fixture must produce this gap"
        return row["rank_move"]

    assert move_for(RANK_MOVE - 1) == 0, "below the threshold the two rankings agree"
    assert move_for(RANK_MOVE) == RANK_MOVE, "at the threshold the arrow is drawn"

    # A TRIPWIRE, not a specification. Everything above is computed FROM RANK_MOVE, so it
    # pins the boundary's behaviour and is blind to the constant's VALUE -- retuning 5 to
    # 2 passes every assertion here and every other test in the repo, while silently
    # changing which players on a 1,169-row board are flagged hold-rather-than-start.
    # Retuning is legitimate; doing it without noticing is not, so this line makes it a
    # deliberate edit. If you are changing it on purpose, change this number too.
    assert RANK_MOVE == 5


def test_no_arrow_when_there_is_no_next_year_estimate() -> None:
    """`next` is NaN whenever horizon 1 is unobservable, and `add_ranks` sorts NaN last.

    That pairs a last-place `rank_next` with a real `rank_total`, which the arrow renders
    as the strongest HOLD signal on the board -- produced by the total absence of a
    next-year estimate rather than by any strength.
    """
    board = build_board(
        _hand_payload(
            [("Gapped", [0.0, 12.0, 12.0]), *[(f"Filler {i}", [5.0, 5.0, 5.0]) for i in range(8)]]
        ),
        top="all",
        end=BASE + 3,
    )
    # A REAL zero next year against a positive span is a genuine hold signal, and must
    # still draw. The guard is for rows with nothing to say, not for every zero.
    gapped = next(r for r in board.rows if r["name"] == "Gapped")
    assert gapped["total"] > 0 and gapped["next"] == 0.0
    assert gapped["rank_move"] > 0, "a real 0-next / positive-span row still holds"

    # A NaN row needs PEERS. Solo, rank_total and rank_next are both 1 and the raw move
    # is 0 regardless, so the guard is unreachable and deleting it would still pass.
    gapped_payload = _hand_payload(
        [
            ("Aaa Gapped", [0.0, 30.0, 30.0]),  # biggest total -> rank_total 1
            *[(f"Filler {i}", [9.0, 1.0, 1.0]) for i in range(8)],
        ]
    )
    gapped_payload["players"][0]["sgp"] = [
        p for p in gapped_payload["players"][0]["sgp"] if p["horizon"] != 1
    ]
    board2 = build_board(gapped_payload, top="all", end=BASE + 3)
    row = next(r for r in board2.rows if r["name"] == "Aaa Gapped")

    assert math.isnan(row["next"]), "fixture must produce a NaN next"
    assert row["rank_total"] == 1, "and a real rank on the total"
    assert abs(row["rank_next"] - row["rank_total"]) >= RANK_MOVE, (
        "add_ranks sorts NaN last, so the raw move must clear the threshold -- otherwise "
        "this proves nothing about the guard"
    )
    assert row["rank_move"] == 0, "no next-year estimate means no hold-vs-start claim"


def test_per_year_cells_are_placed_by_horizon_not_by_position() -> None:
    """Each fit is per-horizon, so `by_year` is not guaranteed to be a prefix.

    Rendering it positionally and padding at the tail would print a year-2 figure under
    the '27 header and a year-3 figure under '28 for any path with a hole in it. This is
    DEFENSIVE -- no such gap has been reproduced from the model, and the source comment
    in `shape_trajectory` describes the opposite direction (observable near, unobservable
    far) as the intent, which tail-padding handles correctly. Aligning by horizon costs
    nothing and removes the assumption.
    """
    gapped = _hand_payload([("Gapped", [0.0, 7.0, 5.0])])
    gapped["players"][0]["sgp"] = [p for p in gapped["players"][0]["sgp"] if p["horizon"] != 1]

    board = build_board(gapped, top="all", end=BASE + 3)
    row = board.rows[0]

    assert [c["horizon"] for c in row["by_year"]] == [2, 3], "fixture must have a hole at 1"
    assert row["year_cells"] == [None, pytest.approx(7.0), pytest.approx(5.0)], (
        "the year-2 figure belongs under 2028, not 2027"
    )
    assert len(row["year_cells"]) == len(board.year_columns)


def test_a_filtered_view_reports_both_denominators(payload: dict) -> None:
    """The rank is league-wide, so the count beside it has to say which pool it counts.

    `Board.scored` is counted AFTER the pool filter but its docstring calls it "the
    denominator the rank column is against". On /trajectory?pool=pitcher the page read
    "Showing 606 of 606 scored" while the # column ran to 1169 -- two numbers on screen
    contradicting each other, and a reader gauging staff depth takes the top pitcher's
    #12 as "12th of 606" when he is the best pitcher, 12th overall.
    """
    everyone = build_board(payload, top="all")
    pitchers = build_board(payload, top="all", pool="pitcher")

    assert pitchers.scored == len(pitchers.rows) == 2, "scored follows the view"
    assert pitchers.ranked == everyone.scored, "ranked is the league-wide denominator"
    assert pitchers.ranked > pitchers.scored
    assert max(r["rank_total"] for r in pitchers.rows) <= pitchers.ranked

    assert everyone.scored == everyone.ranked, "unfiltered, the two agree"


def test_the_derived_board_is_reused_across_requests(payload: dict, monkeypatch) -> None:
    """The payload is an immutable offline artifact, so deriving it once is safe.

    Measured on a live-shaped payload (1,169 players x 5 horizons x 2 scales, 651 KB):
    from_payload 10.5ms rebuilding 11,690 frozen dataclasses, totals + add_ranks 4.7ms.
    None of it depends on pool/top/mine, which only slice an already-ranked list -- and
    every control on the page is a full-page GET, so flipping pool, then end year, then
    scale, then top was four full rebuilds of the same thing.
    """
    from fantasy_baseball.web import trajectory_view

    trajectory_view.clear_board_cache()
    calls = {"n": 0}
    real = trajectory_view.from_payload

    def counted(pl):
        calls["n"] += 1
        return real(pl)

    monkeypatch.setattr(trajectory_view, "from_payload", counted)

    build_board(payload, end=BASE + 3)
    assert calls["n"] == 1
    build_board(payload, end=BASE + 3, pool="pitcher", top=25)
    build_board(payload, end=BASE + 3, scale="sgp")
    build_board(payload, end=BASE + 1)
    assert calls["n"] == 1, "filters and timeframes reuse the parsed sweep"


def test_a_new_push_invalidates_the_cache(payload: dict) -> None:
    """Vintage-keyed, so the next push is picked up rather than served stale."""
    from fantasy_baseball.web import trajectory_view

    trajectory_view.clear_board_cache()
    first = build_board(payload, top="all")

    # SAME player count and shape, so `generated_at` is the only part of the key that
    # differs -- otherwise `len(players)` does the invalidating and this proves nothing
    # about the mechanism it names.
    fresher = dict(payload, generated_at="2026-09-01T09:00:00-04:00")
    fresher["players"] = [
        dict(payload["players"][0], name="Renamed By The New Push"),
        *payload["players"][1:],
    ]
    second = build_board(fresher, top="all")

    assert len(second.rows) == len(first.rows), "the fixture must differ only in content"
    assert any(r["name"] == "Renamed By The New Push" for r in second.rows), (
        "a new generated_at must invalidate even when the shape is identical"
    )


def test_a_push_landing_mid_derivation_cannot_pin_the_old_board(payload: dict, monkeypatch) -> None:
    """A thread holding the previous vintage must not be able to write into the new one.

    Threaded WSGI, two requests straddling a push. T1 reads vintage A, misses, and is
    preempted inside the derivation -- it already holds A's players. T2 arrives with the
    freshly pushed payload B, swaps the vintage, derives B and stores it. T1 resumes and
    writes A's rows into the shared map. The vintage now says B while the rows are A's,
    nothing revalidates, and every later request is served the PRE-PUSH board while the
    page prints B's generated_at beside it.

    Simulated deterministically: `totals` is patched so the first derivation lands a
    complete B-vintage build before it returns, which is exactly the interleaving.
    """
    from fantasy_baseball.web import trajectory_view

    trajectory_view.clear_board_cache()
    newer = dict(payload, generated_at="2026-09-01T09:00:00-04:00")
    newer["players"] = payload["players"][:2]

    real_totals = trajectory_view.totals
    landed = {"done": False}

    def totals_with_a_push_landing(players, horizons, scale):
        rows = real_totals(players, horizons, scale)
        if not landed["done"]:
            landed["done"] = True
            build_board(newer, top="all")  # T2 completes while T1 is mid-derivation
        return rows

    monkeypatch.setattr(trajectory_view, "totals", totals_with_a_push_landing)
    build_board(payload, top="all")  # T1, holding the OLD payload
    monkeypatch.undo()

    after = build_board(newer, top="all")
    assert len(after.rows) == 2, "the post-push board must not be pinned to the pre-push rows"


def _spots_fixture():
    """Rosters for the `payload` fixture's six players.

    Two teams with DIFFERENT rosters, plus one rostered player who is not on the
    board at all ("Never Scored") -- without both, the filter and unscored
    assertions cannot fail. The ambiguity case needs a name collision, which this
    fixture deliberately does NOT have: it lives in
    `test_an_opponents_ambiguous_row_is_flagged_too`, which builds its own
    payload so the shared module-scoped fixture keeps its asserted arity.
    """

    return [
        _spot("Big Bat", "Theirs"),
        _spot("Small Bat", "Mine"),
        _spot("Under Water", "Mine"),
        _spot("Big Arm", "Mine", pool="pitcher"),
        _spot("Never Scored", "Mine"),
    ]


def test_selecting_a_team_narrows_the_board_to_that_roster(payload: dict) -> None:
    board = build_board(payload, spots=_spots_fixture(), my_team="Mine", team="Mine")
    assert {r["name"] for r in board.rows} == {"Small Bat", "Under Water", "Big Arm"}
    assert board.team == "Mine"


def test_a_teams_best_player_keeps_his_league_rank(payload: dict) -> None:
    """Ranking within the subset would make every team's best player a #1 and
    destroy the only comparison the board exists for."""
    everyone = build_board(payload, spots=_spots_fixture(), my_team="Mine")
    league = {r["name"]: r["rank_total"] for r in everyone.rows}
    mine = build_board(payload, spots=_spots_fixture(), my_team="Mine", team="Mine")

    assert [r["rank_total"] for r in mine.rows] == [league[r["name"]] for r in mine.rows]
    assert mine.rows[0]["rank_total"] != 1, "expected a league rank, not a within-team one"


def test_an_unknown_team_falls_back_to_the_whole_board(payload: dict) -> None:
    """The query string is user-editable and survives a team rename."""
    everyone = build_board(payload, spots=_spots_fixture(), my_team="Mine")
    junk = build_board(payload, spots=_spots_fixture(), my_team="Mine", team="Nobody FC")
    assert junk.team == "all"
    assert len(junk.rows) == len(everyone.rows)


def test_the_team_filter_composes_with_the_pool_filter(payload: dict) -> None:
    board = build_board(
        payload, spots=_spots_fixture(), my_team="Mine", team="Mine", pool="pitcher"
    )
    assert {r["name"] for r in board.rows} == {"Big Arm"}


def test_unscored_is_populated_only_for_a_selected_team(payload: dict) -> None:
    assert build_board(payload, spots=_spots_fixture(), my_team="Mine").unscored == []
    mine = build_board(payload, spots=_spots_fixture(), my_team="Mine", team="Mine")
    assert mine.unscored == ["Never Scored"]


def test_my_team_leads_the_dropdown(payload: dict) -> None:
    board = build_board(payload, spots=_spots_fixture(), my_team="Mine")
    assert board.teams == ("Mine", "Theirs")


def test_an_opponents_ambiguous_row_is_flagged_too(payload: dict) -> None:
    """`mine_ambiguous` only ever fired for my rows. Attributing an opponent's
    player on a guess is exactly as wrong.

    Reuses the `twin` construction from
    `test_a_colliding_name_is_flagged_rather_than_claimed` -- duplicate
    `players[0]` under a new id -- rather than sweeping a fresh panel, and
    rather than adding a collision to the module-scoped `payload`, whose arity
    is asserted by `test_top_all_shows_every_scored_row` (`scored == 6`).
    """

    first = payload["players"][0]
    twin = dict(payload)
    twin["players"] = [*payload["players"], {**first, "id": first["id"] + 10_000}]
    twin["generated_at"] = f"hand-{next(_HAND_SEQ)}"  # do not share the parse cache

    spots = [_spot(first["name"], "Theirs", first["pool"])]

    board = build_board(twin, top="all", spots=spots, my_team="Mine", team="Theirs")
    assert len(board.rows) == 2, "both rows match the only key available"
    assert all(r["owner_ambiguous"] for r in board.rows)
    assert not any(r["mine"] for r in board.rows), "these are an opponent's rows"

    # The all-teams view keeps today's behaviour: not mine, so not flagged.
    everyone = build_board(twin, top="all", spots=spots, my_team="Mine")
    assert not any(r["owner_ambiguous"] for r in everyone.rows)


def test_has_rosters_still_tracks_my_own_roster_not_the_read(payload: dict) -> None:
    """Two different facts. `has_rosters` gates the not-highlighted banner and
    must stay false when MY roster joined nothing, even though the read
    succeeded and the dropdown is perfectly usable."""

    others_only = [_spot("Big Bat", "Theirs")]
    board = build_board(payload, spots=others_only, my_team="Mine")
    assert board.has_rosters is False
    assert board.teams == ("Theirs",), "the dropdown still works for other teams"


def test_my_players_are_marked(payload: dict) -> None:
    """The first question a reader has on a keeper board is which of these are already
    his. Roster blobs carry no mlbam_id (#284), so the join is (normalized name,
    player_type) -- the same non-unique key the rest of the repo has to live with.
    """

    spots = [
        _spot("Big Bat", "Mine"),
        _spot("Big Arm", "Mine", "pitcher"),
    ]
    board = build_board(payload, top="all", spots=spots, my_team="Mine")
    marked = {r["name"] for r in board.rows if r["mine"]}
    assert marked == {"Big Bat", "Big Arm"}
    assert not any(r["owner_ambiguous"] for r in board.rows)

    unmarked = build_board(payload, top="all")
    assert not any(r["mine"] for r in unmarked.rows), "no roster read -- nothing claimed"


def test_an_empty_roster_read_is_not_a_successful_one(payload: dict) -> None:
    """ "You own none of these" and "we could not read your roster" must not look alike.

    `live_rosters` returns [] WITHOUT raising when the roster blobs are absent -- its own
    module docstring flags this: getting the asymmetry wrong "drops your own roster
    silently, which reads as 'you own nobody' rather than as an error". The route only
    caught exceptions, so a cold or half-written `cache:roster` gave `spots=[]`, which
    counted as a successful read: nothing highlighted AND no warning, so the page
    silently asserts the reader owns none of the top 50 keepers.
    """

    assert build_board(payload, spots=None, my_team="Mine").has_rosters is False, (
        "no read attempted"
    )
    assert build_board(payload, spots=[], my_team="Mine").has_rosters is False, (
        "an EMPTY read is not a successful one -- it cannot be told from a failed one"
    )
    mine_spot = [_spot("Big Bat", "Mine")]
    assert build_board(payload, spots=mine_spot, my_team="Mine").has_rosters is True

    # Spots present, but all on ANOTHER team: the read succeeded and the dropdown
    # works, but MY roster still joined nothing.
    others_spot = [_spot("Big Bat", "Theirs")]
    assert build_board(payload, spots=others_spot, my_team="Mine").has_rosters is False


def test_a_colliding_name_is_flagged_rather_than_claimed(payload: dict) -> None:
    """Two players can share (normalized name, player_type) -- the live board carries two
    hitters called Max Muncy. Marking both as mine on one roster hit would put a player
    the reader does not own on his keeper shortlist, so an ambiguous match says so.
    """

    twin = dict(payload)
    first = payload["players"][0]
    twin["players"] = [*payload["players"], {**first, "id": first["id"] + 10_000}]

    spots = [_spot(first["name"], "Mine", first["pool"])]
    board = build_board(twin, top="all", spots=spots, my_team="Mine")
    hits = [r for r in board.rows if r["mine"]]
    assert len(hits) == 2, "both rows match the only key available"
    assert all(r["owner_ambiguous"] for r in hits), "and both must say the match is unsure"


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
    """Reworded from a `version=99` check (2026-08-06, Hart's call) -- see
    `test_sweep.test_the_old_positional_payload_is_refused_by_shape_not_by_a_version`.

    The page-level guarantee is unchanged and is the reason this test exists: a payload
    this build cannot read must reach `season_routes`' error path, not render a board
    that silently means something else.
    """
    from fantasy_baseball.web import trajectory_view

    trajectory_view.clear_board_cache()
    stale = dict(payload, generated_at="positional-blob")
    stale["players"] = [
        dict(
            p,
            sgp=[
                [y["horizon"], y["mean"], y["p10"], y["p90"], y["n_effective"], 0] for y in p["sgp"]
            ],
        )
        for p in payload["players"]
    ]
    with pytest.raises(ValueError, match=re.escape("re-run scripts/push_trajectory_board.py")):
        build_board(stale)


def test_a_key_two_teams_roster_is_still_mine_and_still_flagged(payload: dict) -> None:
    """The regression that shipped between 2026-08-07's fix wave and the review after.

    Ownership was derived from whichever spot won a deterministic sort. So when an
    opponent's team name sorted first, my own rostered player rendered on the DEFAULT
    all-teams view as neither mine nor ambiguous -- no highlight, no (?) -- and the
    list that would have named him only appears under a team filter. The reader's
    page said, silently, "you do not own him".

    "Aardvarks" sorts before any real team here on purpose: under the old winner
    rule it took the key, and this test failed.
    """
    spots = [_spot("Big Bat", "Mine"), _spot("Big Bat", "Aardvarks")]
    board = build_board(payload, top="all", spots=spots, my_team="Mine")

    big = next(r for r in board.rows if r["name"] == "Big Bat")
    assert big["mine"], "an opponent rostering the same name does not take my player"
    assert big["owner_ambiguous"], "and the row must say the attribution is a guess"
    assert board.has_rosters, "my roster joined a row, whoever else also claims it"

    # And he shows under BOTH teams, because the board cannot tell which is which.
    for team in ("Mine", "Aardvarks"):
        rows = build_board(payload, top="all", spots=spots, my_team="Mine", team=team).rows
        assert any(r["name"] == "Big Bat" for r in rows), f"{team} rosters the name too"


def test_the_unscored_list_follows_the_pool_filter(payload: dict) -> None:
    """It renders under a table that may be showing one pool. Naming a pitcher
    beneath a hitters-only table reads as a hole in the hitter list."""
    spots = [
        _spot("Big Bat", "Mine"),
        _spot("Unpriced Bat", "Mine"),
        _spot("Unpriced Arm", "Mine", pool="pitcher"),
    ]
    kw = {"top": "all", "spots": spots, "my_team": "Mine", "team": "Mine"}

    assert build_board(payload, **kw).unscored == ["Unpriced Arm", "Unpriced Bat"]
    assert build_board(payload, pool="hitter", **kw).unscored == ["Unpriced Bat"]
    assert build_board(payload, pool="pitcher", **kw).unscored == ["Unpriced Arm"]


def _teams_fixture():
    """Rosters for the `payload` fixture, shaped so every assertion below can fail.

    THREE things are load-bearing and none may be dropped:
      * two teams with DIFFERENT rosters, or the ordering assertion is trivial;
      * "Mine" is deliberately the WEAKER of the two, so "my block is not promoted
        to the top" cannot pass by accident;
      * two teams with NO scored rows ("Empty FC", "Aardvark FC"), which is both
        the render-an-empty-team case and the only way to reach a total tie.
    """
    return [
        _spot("Big Bat", "Rivals"),
        _spot("Big Arm", "Rivals", pool="pitcher"),
        _spot("Small Bat", "Mine"),
        _spot("Small Arm", "Mine", pool="pitcher"),
        _spot("Never Scored", "Mine"),
        _spot("Ghost", "Empty FC"),
        _spot("Phantom", "Aardvark FC"),
    ]


def test_blocks_are_ordered_by_strength_and_mine_is_not_promoted(payload: dict) -> None:
    """The comparison IS the view. Sorting my own block to the top would destroy
    the ordering that justifies including it at all."""
    board = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine")

    names = [b.team for b in board.blocks]
    assert names[0] == "Rivals", "the strongest roster reads first"
    assert names.index("Mine") == 1, "mine sits where its strength puts it"
    assert [b.total for b in board.blocks] == sorted((b.total for b in board.blocks), reverse=True)
    assert next(b for b in board.blocks if b.is_mine).team == "Mine"
    assert not board.mine_missing


def test_teams_tied_on_total_are_ordered_by_name_not_by_roster_order(payload: dict) -> None:
    """Three teams total 0.0: "Aardvark FC" and "Empty FC" from `_teams_fixture`, plus
    "Zebra FC" -- added here as `my_team`, with a player of its own that is not on the
    board. That third team is load-bearing and cannot be dropped back to just the
    original two.

    `index_rosters` promotes `my_team` to the FRONT of `index.teams`, which is also
    `grouped`'s insertion order. Python's `sort` is stable, so with NO tie-break at all
    a three-way tie would simply keep that insertion order -- and "Zebra FC" being
    FIRST into `grouped` would put it first among the ties too, the opposite of where
    its name sorts. The original two-team version of this test used only "Aardvark FC"
    and "Empty FC", both of which -- coincidentally -- already sit in alphabetical
    order in `index.teams`' own insertion order, so a stable sort with no tie-break at
    all reproduced the "correct" answer by accident and the assertion could not fail.
    Only because "Zebra FC" is alphabetically LAST of the three, while being FIRST by
    insertion order, does this version actually distinguish "sorted by name" from
    "insertion order, stable-sorted."
    """
    spots = [*_teams_fixture(), _spot("Zebra Ghost", "Zebra FC")]
    forward = build_teams_board(payload, spots=spots, my_team="Zebra FC")
    reverse = build_teams_board(payload, spots=list(reversed(spots)), my_team="Zebra FC")

    empties = [b.team for b in forward.blocks if b.total == 0.0]
    assert empties == ["Aardvark FC", "Empty FC", "Zebra FC"], "ties break on name, ascending"
    assert [b.team for b in reverse.blocks] == [b.team for b in forward.blocks]


def test_a_team_with_nothing_scored_still_renders_with_its_unpriced_list(payload: dict) -> None:
    """#323's first named failure mode. Building the block list from the ROWS would
    drop this team and its unpriced list, leaving nothing on screen to say it exists."""
    board = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine")

    ghost = next(b for b in board.blocks if b.team == "Empty FC")
    assert ghost.rows == []
    assert ghost.scored == 0
    assert ghost.total == 0.0
    assert ghost.unscored == ["Ghost"]
    assert board.blocks[-1].total == 0.0, "and it sorts last"


def test_a_blocks_rows_carry_league_ranks(payload: dict) -> None:
    """Within-team ranks would make every team's best player read #1."""
    league = {r["name"]: r["rank_total"] for r in build_board(payload, top="all").rows}
    board = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine")

    for block in board.blocks:
        for row in block.rows:
            assert row["rank_total"] == league[row["name"]]
    assert any(b.rows and b.rows[0]["rank_total"] != 1 for b in board.blocks)


def test_per_team_slices_without_re_ranking(payload: dict) -> None:
    """N is a slice of an already-ranked list, so the first row must not move.

    "Mine" is the load-bearing block and cannot be dropped for "Rivals" alone. Rows
    reach `grouped` in `totals()` order -- every hitter, then every pitcher -- because
    `add_ranks` ranks off a temporary sorted view and leaves its input untouched.
    "Rivals" is appended [Big Bat(#1), Big Arm(#2)], which is ALREADY rank order, so
    deleting the per-block sort left this test green while `per_team` sliced the wrong
    players, `total` summed the wrong ones, and the block ordering the whole view exists
    to compare went with them. "Mine" is appended [Small Bat(#5), Small Arm(#4)] -- the
    opposite of rank order -- so at per_team=1 the surviving row is the sort's answer.
    """
    one = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine", per_team=1)
    two = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine", per_team=2)

    rivals_one = next(b for b in one.blocks if b.team == "Rivals")
    rivals_two = next(b for b in two.blocks if b.team == "Rivals")
    assert len(rivals_one.rows) == 1
    assert len(rivals_two.rows) == 2
    assert rivals_one.rows[0]["name"] == rivals_two.rows[0]["name"]
    assert rivals_one.scored == rivals_two.scored, "scored is the team's set, not the slice"

    mine_two = next(b for b in two.blocks if b.team == "Mine")
    assert [r["name"] for r in mine_two.rows] == ["Small Arm", "Small Bat"], (
        "the fixture must append these OUT of rank order, or the slice below cannot fail"
    )
    mine_one = next(b for b in one.blocks if b.team == "Mine")
    assert [r["name"] for r in mine_one.rows] == ["Small Arm"], (
        "per_team=1 keeps the better-ranked row, not the first one appended"
    )
    assert mine_one.total == mine_two.rows[0]["total"], "and the block total follows it"


def test_scored_follows_the_pool_filter_and_unscored_does_too(payload: dict) -> None:
    """A block that says "5 of 24" under a hitters-only table must mean 24 hitters,
    or the visible rows cannot add up to the number printed beside them."""
    both = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine")
    hitters = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine", pool="hitter")

    pitchers = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine", pool="pitcher")

    assert next(b for b in both.blocks if b.team == "Rivals").scored == 2
    assert next(b for b in hitters.blocks if b.team == "Rivals").scored == 1
    assert next(b for b in pitchers.blocks if b.team == "Rivals").scored == 1
    assert next(b for b in both.blocks if b.team == "Mine").unscored == ["Never Scored"]
    assert next(b for b in hitters.blocks if b.team == "Mine").unscored == ["Never Scored"]
    # THE HALF THAT CAN FAIL. "Never Scored" is a hitter, so `both` and `hitter` name him
    # either way -- dropping the `pool` argument from `unscored_for` left both green. Only
    # the pitcher view expects a DIFFERENT list, and an unpriced hitter under a
    # pitchers-only block reads as a hole in the pitching staff.
    assert next(b for b in pitchers.blocks if b.team == "Mine").unscored == []


def test_a_name_two_teams_roster_appears_in_both_blocks_flagged(payload: dict) -> None:
    """Membership, not a winner. Attributing the row to one team would take the
    other owner's player off his own block with nothing on screen to say so."""
    spots = [*_teams_fixture(), _spot("Big Bat", "Mine")]
    board = build_teams_board(payload, spots=spots, my_team="Mine")

    holders = [b.team for b in board.blocks if any(r["name"] == "Big Bat" for r in b.rows)]
    assert sorted(holders) == ["Mine", "Rivals"]
    for block in board.blocks:
        for row in block.rows:
            if row["name"] == "Big Bat":
                assert row["owner_ambiguous"], "the board cannot tell which Big Bat"


def test_mine_missing_when_my_team_names_no_block(payload: dict) -> None:
    """Ten unhighlighted blocks read as "you own none of these" -- a claim the page
    cannot support when the truth is that it never found the reader's roster."""
    board = build_teams_board(payload, spots=_teams_fixture(), my_team="Renamed FC")
    assert board.mine_missing
    assert not any(b.is_mine for b in board.blocks)

    assert not build_teams_board(payload, spots=_teams_fixture(), my_team="Mine").mine_missing
    assert build_teams_board(payload, spots=_teams_fixture(), my_team=None).mine_missing


def test_a_teams_block_carries_the_flag_threshold_and_its_flagged_rows(payload: dict) -> None:
    """The (!) rule is a tuned constant behind an open issue (#310), so the template
    renders the threshold from `meta` rather than restating it as prose -- exactly as the
    league board does. `TeamsBoard.meta` did not carry it, which meant the teams view
    could not have rendered the flag even if it wanted to: it was dropped structurally,
    and every block total then summed extrapolated rows with nothing on screen to say so.
    """
    from fantasy_baseball.trajectory.comps import MIN_LOCAL_SUPPORT

    spots = [*_teams_fixture(), _spot("Thin Support", "Mine")]
    board = build_teams_board(payload, spots=spots, my_team="Mine")

    assert board.meta["min_local_support"] == MIN_LOCAL_SUPPORT
    assert board.meta["min_local_support"] == build_board(payload).meta["min_local_support"], (
        "both views flag on the same rule, so both must read the same constant"
    )

    mine = next(b for b in board.blocks if b.team == "Mine")
    flagged = [r for r in mine.rows if r["extrapolated"]]
    assert [r["name"] for r in flagged] == ["Thin Support"], (
        "the fixture must put an extrapolated row in a block, or the flag is unreachable"
    )
    assert mine.total == pytest.approx(sum(r["total"] for r in mine.rows)), (
        "and it is summed into the total that orders the page like any other row"
    )


def test_per_team_and_end_year_clamp_junk_from_the_query_string(payload: dict) -> None:
    """These arrive from a URL a reader can edit."""
    board = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine", per_team="junk")
    assert board.per_team == 5
    assert build_teams_board(payload, spots=_teams_fixture(), per_team=0).per_team == 1
    assert build_teams_board(payload, spots=_teams_fixture(), per_team=999).per_team == 50
    assert build_teams_board(payload, spots=_teams_fixture(), end="nonsense").end_year == 2027


def _chart(payload: dict) -> dict:
    """The chart blob that PAIRS with `payload`: its stamp, keyed `(id, pool)`.

    A second cache key (#344), not a section of the board -- only this view reads career
    history and comps, and inline they more than doubled the blob every other view
    fetches. Tests that want the "no chart stored" path simply pass no chart.
    """
    return to_chart_payload(
        {
            (p["id"], p["pool"]): {
                # The subject's OWN base-season age is never in here: the push builds
                # history from `complete = live[~live["partial_season"]]`, which excludes
                # the season in progress by construction. These players are 27, so the
                # career stops at 26 -- a blob the writer could actually have written.
                "history": [[24, 14.0], [25, 16.0], [26, 20.0]],
                "comps": [
                    {
                        # The join key into `careers` below. Ids, never names (#284).
                        "id": 100 + i,
                        "name": f"Comp {i}",
                        "season": 2010 + i,
                        "rmse": 0.5 * i,
                        "path": [10.0 - i, 9.0 - i, 8.0 - i, 7.0 - i, 6.0 - i],
                    }
                    for i in range(1, 8)
                ],
            }
            for p in payload["players"]
        },
        # Deduped at the top level and keyed by pool, exactly as the push writes it: a
        # two-way comp has one arc per pool and a bare id would collapse them.
        #
        # Ages 22-30 SPAN the fixture players' age of 27 on purpose: a comp is selected
        # on an exact age match, so a stored arc that stopped before the subject's age
        # could not have been a comp at all, and a test asserting the match age is
        # drawable needs a fixture that could really have been written.
        #
        # The first value is `float(i)`, so an arc identifies which comp it belongs to
        # -- that is what makes the id join assertable rather than merely non-empty.
        careers={
            chart_key(100 + i, pool): [[22 + j, float(i + j)] for j in range(9)]
            for i in range(1, 8)
            for pool in ("hitter", "pitcher")
        },
        generated_at=str(payload["generated_at"]),
    )


def _view(payload: dict, **kwargs: Any) -> PlayerView:
    """`build_player_view` handed the chart blob paired with `payload`. The normal case;
    the un-paired ones are spelled out at their own tests."""
    return build_player_view(payload, chart=_chart(payload), **kwargs)


def _payload_with_a_twin(payload: dict, id_offset: int) -> tuple[dict, dict]:
    """The `payload` fixture plus a second player under `players[0]`'s NAME.

    Returns the payload and the original row, so a caller can assert against
    `first["id"]` and `first["id"] + id_offset`. The SIGN of the offset is the whole
    parameter: a positive one appends the larger id (the order `hits` arrives in
    anyway, which a candidate-ordering assertion cannot distinguish from a real sort),
    a negative one appends the smaller and only an actual `sorted()` produces it.
    """
    twin = dict(payload)
    first = twin["players"][0]
    twin["players"] = [*twin["players"], {**first, "id": first["id"] + id_offset}]
    twin["generated_at"] = f"hand-{next(_HAND_SEQ)}"  # do not share the parse cache
    return twin, first


def test_a_player_is_found_by_name_with_his_career_and_projection(payload: dict) -> None:
    view = _view(payload, player="Big Bat")
    assert view.found
    assert view.name == "Big Bat"
    assert [pt[0] for pt in view.history] == [24, 25, 26], "career ascends by age"
    assert len(view.projection) == 3, "the fixture sweeps three horizons"
    assert all(p["p10"] <= p["mean"] <= p["p90"] for p in view.projection)


def test_an_unknown_name_is_not_found_and_lists_nobody(payload: dict) -> None:
    view = _view(payload, player="Nobody At All")
    assert not view.found
    assert view.candidates == []
    # `found=False` and `candidates=[]` are both dataclass defaults -- a view that
    # somehow carried a stray history/projection alongside them would still pass those
    # two alone. This is the load-bearing half: nothing else on the empty view leaked.
    assert view.projection == []
    assert view.history == []
    assert view.name == "Nobody At All"


def test_an_ambiguous_name_lists_candidates_and_renders_no_chart(payload: dict) -> None:
    """Two players can share a normalized name -- the live board carries two hitters
    called Max Muncy. Guessing puts one man's career under another's name."""
    twin, first = _payload_with_a_twin(payload, 10_000)
    view = _view(twin, player=first["name"])
    assert not view.found, "an ambiguous name renders no chart"
    assert len(view.candidates) == 2
    assert {c["id"] for c in view.candidates} == {first["id"], first["id"] + 10_000}


def _two_way(payload: dict) -> dict:
    """A payload where one name is carried by a hitter row AND a pitcher row.

    The live board's shape for Shohei Ohtani: the two rows share an mlbam_id AND an age,
    differing only by slot and pool -- so the id is not the discriminator and only the
    pool is.
    """
    twin = dict(payload)
    hitter = twin["players"][0]
    twin["players"] = [
        *twin["players"],
        {**hitter, "pool": "pitcher", "slot": "SP", "floor": 3.0},
    ]
    twin["generated_at"] = f"hand-{next(_HAND_SEQ)}"  # do not share the parse cache
    return twin


def test_a_candidate_list_names_the_pool_it_offers_as_the_discriminator(payload: dict) -> None:
    """Two rows differing ONLY by pool render two identical lines without it."""
    view = _view(_two_way(payload), player="Big Bat")
    assert not view.found
    assert {c["pool"] for c in view.candidates} == {"hitter", "pitcher"}


def test_a_two_way_name_is_resolved_by_the_pool_selector(payload: dict) -> None:
    """`ppool`, NOT the board's `pool` filter: that one is the hitter/pitcher pill the
    player view merely passes through, and overloading it couples a board filter to name
    resolution."""
    twin = _two_way(payload)
    pitcher = _view(twin, player="Big Bat", ppool="pitcher")
    assert pitcher.found and pitcher.slot == "SP"
    hitter = _view(twin, player="Big Bat", ppool="hitter")
    assert hitter.found and hitter.slot == "OF"


def test_same_pool_namesakes_are_resolved_by_the_id_selector(payload: dict) -> None:
    """The live board carries two hitters called Max Muncy: same pool, different ids,
    so the pool selector cannot separate them and only `pid` can."""
    twin, first = _payload_with_a_twin(payload, 10_000)
    view = _view(twin, player=first["name"], pid=str(first["id"] + 10_000))
    assert view.found
    assert view.pid == str(first["id"] + 10_000)


def test_a_resolved_view_carries_its_own_narrowing_forward(payload: dict) -> None:
    """Every control link is built from `filter_state`, which reads these off the view.
    A resolved player whose view reported no narrowing would fall back to the candidate
    list the moment a scale pill was clicked."""
    twin = _two_way(payload)
    view = _view(twin, player="Big Bat", ppool="pitcher")
    assert (view.pid, view.ppool) == (str(twin["players"][0]["id"]), "pitcher")


def test_narrowing_that_matches_nothing_falls_back_to_the_full_candidate_list(
    payload: dict,
) -> None:
    """The search form is a GET that re-submits every key in `filter_state`, so a new
    name arrives carrying the PREVIOUS player's `pid`. Honouring it strictly would show
    "no player named X" for a player who is on the board -- a second dead end in place of
    the one this fixes."""
    twin, first = _payload_with_a_twin(payload, 10_000)
    view = _view(twin, player=first["name"], pid="999999", ppool="pitcher")
    assert len(view.candidates) == 2, "a stale narrowing is ignored, not obeyed"


def test_comps_are_sliced_to_n_and_n_is_clamped(payload: dict) -> None:
    assert len(_view(payload, player="Big Bat").comps) == 5, "default"
    assert len(_view(payload, player="Big Bat", n=3).comps) == 3
    assert len(_view(payload, player="Big Bat", n=999).comps) == 7, "what exists"
    assert len(_view(payload, player="Big Bat", n="junk").comps) == 5
    assert len(_view(payload, player="Big Bat", n=0).comps) == 1


def test_a_board_with_no_chart_data_still_renders_the_projection(payload: dict) -> None:
    """The shape prod holds: a board, and no `cache:trajectory_chart_data` at all. #332
    took /trajectory down by refusing a blob it could largely read; this renders what it
    has and lets the page say what is missing.

    NOT a mismatch. Nothing arrived, so there is nothing to disagree with the board, and
    the page must give the "predates / not pushed yet" explanation rather than the
    out-of-step one.
    """
    view = build_player_view(payload, player="Big Bat")
    assert view.found
    assert view.projection, "the fit is in every payload"
    assert view.history == []
    assert view.comps == []
    assert view.chart_vintage_mismatch is False


def test_chart_data_stamped_for_another_board_is_refused_not_drawn(payload: dict) -> None:
    """The failure the split creates, and the reason for the stamp.

    Two keys can be refreshed independently, so a board from noon can sit beside extras
    from Tuesday. Drawn together that is a stale career line under a fresh projection --
    silent, and both halves look plausible. The extras are dropped and the view says
    which of the two things went wrong.
    """
    stale = _chart(payload)
    stale["generated_at"] = "2020-01-01T00:00:00-05:00"

    view = build_player_view(payload, player="Big Bat", chart=stale)
    assert view.found and view.projection, "the board's own fit is unaffected"
    assert view.history == [], "a career line from another build is not drawn"
    assert view.comps == []
    assert view.chart_vintage_mismatch is True


def test_a_chart_blob_with_no_stamp_is_refused_rather_than_paired(payload: dict) -> None:
    """`None == None` would read as a match and pair two blobs on no evidence at all.

    Reachable from either side: a hand-built payload or an older push carries no
    `generated_at`, and the guard must refuse the pair rather than infer one.
    """
    unstamped = _chart(payload)
    unstamped.pop("generated_at")
    assert build_player_view(payload, player="Big Bat", chart=unstamped).chart_vintage_mismatch

    board = dict(payload)
    board.pop("generated_at")
    assert build_player_view(board, player="Big Bat", chart=_chart(payload)).chart_vintage_mismatch


@pytest.mark.parametrize("stamp", [None, ""], ids=["both-absent", "both-empty"])
def test_two_blobs_with_the_SAME_missing_stamp_are_still_refused(payload: dict, stamp) -> None:
    """What `in (None, "")` buys over a plain `!=`.

    The one-sided cases are refused by inequality alone, so they cannot catch a guard
    reduced to `str(board_at) != str(chart_at)`. These can: two blobs that agree on
    having NO vintage compare equal, and pairing them is pairing on no evidence -- the
    exact reasoning the vintage check exists for.
    """
    board, chart = dict(payload), _chart(payload)
    if stamp is None:
        board.pop("generated_at")
        chart.pop("generated_at")
    else:
        board["generated_at"] = chart["generated_at"] = stamp

    view = build_player_view(board, player="Big Bat", chart=chart)
    assert view.chart_vintage_mismatch is True
    assert view.history == [] and view.comps == []


def test_a_chart_blob_of_a_foreign_shape_is_refused_rather_than_raised(payload: dict) -> None:
    """The board written to the chart key -- one push produced both, so the stamps agree
    and the vintage check waves it through.

    Two levels, because `read_cache` hands back whatever was stored: a top-level list
    (the board is not a list, but any JSON array under this key lands here) and a
    mapping whose `players` is the board's LIST of rows. Both reach an attribute lookup
    that a list does not have; the route catches only `(ValueError, KeyError)`, so an
    unguarded `AttributeError` is a Flask 500 on a page whose projection renders fine.
    """
    board_shaped = {"generated_at": payload["generated_at"], "players": payload["players"]}
    view = build_player_view(payload, player="Big Bat", chart=board_shaped)
    assert view.chart_vintage_mismatch is True, "an unreadable blob is out of step, not absent"
    assert view.history == [] and view.comps == []
    assert view.projection, "and the board's own fit still renders"

    top_level_list = build_player_view(payload, player="Big Bat", chart=payload["players"])
    assert top_level_list.chart_vintage_mismatch is True
    assert top_level_list.projection


def test_the_chart_lookup_keeps_the_pool_so_a_two_way_player_keeps_his_own_career(
    payload: dict,
) -> None:
    """The reader half of `chart_key`. On a bare id the hitter row would draw the
    pitcher's career -- the live board's Ohtani, whose two rows share id and age."""
    twin = _two_way(payload)
    chart = _chart(twin)
    hitter_key = chart_key(twin["players"][0]["id"], "hitter")
    chart["players"][hitter_key] = {"history": [[24, 99.0]], "comps": []}

    hitter = build_player_view(twin, chart=chart, player="Big Bat", ppool="hitter", scale="sgp")
    pitcher = build_player_view(twin, chart=chart, player="Big Bat", ppool="pitcher", scale="sgp")
    assert hitter.history == [[24, 99.0]], "the hitter row read the hitter's entry"
    assert pitcher.history == [[24, 14.0], [25, 16.0], [26, 20.0]], "and not the other way"


def test_the_var_axis_nets_every_series_against_the_QUERY_players_floor(payload: dict) -> None:
    """Career, projection and comps alike. Netting each comp against its own slot would
    put lines on one axis that are not comparable -- the mixed-scale defect of #331."""
    var = _view(payload, player="Under Water", scale="var")
    sgp = _view(payload, player="Under Water", scale="sgp")
    floor = var.floor
    assert floor > 0, "fixture must net against a real floor"

    assert var.history[0][1] == pytest.approx(sgp.history[0][1] - floor)
    assert var.projection[0]["mean"] == pytest.approx(sgp.projection[0]["mean"] - floor)
    assert var.comps[0]["path"][0]["value"] == pytest.approx(
        sgp.comps[0]["path"][0]["value"] - floor
    )
    assert [c["name"] for c in var.comps] == [c["name"] for c in sgp.comps], "same comps"


def test_a_comp_path_is_truncated_to_the_projected_horizons(payload: dict) -> None:
    """The fixture's comp paths are stored 5 long; the fixture sweeps 3 horizons.

    Deleting the `[: len(row["sgp"])]` slice leaves every other test in this module
    green -- this is the one assertion that can catch it. The current push script
    always produces paths that already agree in length (`closest_paths` raises on a
    mismatch), so the slice defends a hand-built or future blob, not today's pipeline.
    """
    view = _view(payload, player="Big Bat")
    assert len(view.comps[0]["path"]) == 3, "a comp draws only the projected years"


def test_the_projection_and_comp_ages_are_derived_from_the_players_own_age(
    payload: dict,
) -> None:
    """`view.history[*][0]` is a pass-through of a fixture literal and cannot catch an
    off-by-one in the derived ages. `projection[*]["age"]` and `comps[*]["path"][*]["age"]`
    are both COMPUTED from `row["age"]` plus a horizon -- that arithmetic is unpinned
    anywhere else.
    """
    view = _view(payload, player="Big Bat")
    age = view.age
    assert [p["age"] for p in view.projection] == [age + 1, age + 2, age + 3]
    assert [pt["age"] for pt in view.comps[0]["path"]] == [age + 1, age + 2, age + 3]


def test_the_floor_field_is_the_applied_offset_not_the_raw_slot_floor(payload: dict) -> None:
    """`PlayerView.floor`'s own docstring calls it "what every series was netted
    against". Under scale="sgp" nothing was netted, so it must read 0.0 there -- not
    the slot floor, which a template would otherwise print as the netting rule for a
    chart where every line is untouched raw SGP.
    """
    var = _view(payload, player="Under Water", scale="var")
    sgp = _view(payload, player="Under Water", scale="sgp")
    assert var.floor == pytest.approx(6.0), "the slot floor, actually applied"
    assert sgp.floor == 0.0, "nothing was netted on the SGP scale"


def test_extrapolated_is_read_from_the_row_not_hardcoded(payload: dict) -> None:
    """`Thin Support` is the fixture's purpose-built (!) row -- see its own comment in
    the `payload` fixture. A hardcoded `extrapolated=False` on the found branch would
    still pass every other test in this module.
    """
    flagged = _view(payload, player="Thin Support")
    assert flagged.extrapolated is True

    calm = _view(payload, player="Big Bat")
    assert calm.extrapolated is False


def test_ambiguous_candidates_are_ordered_by_id(payload: dict) -> None:
    """The ambiguity test elsewhere compares a SET, so deleting the `sorted(...)` call
    passes it. Built with the smaller id appended LAST -- the opposite of the order
    `hits` would otherwise arrive in -- so only an actual sort produces this list.
    """
    twin, first = _payload_with_a_twin(payload, -10_000)
    view = _view(twin, player=first["name"])
    assert [c["id"] for c in view.candidates] == [first["id"] - 10_000, first["id"]]


def test_every_comp_carries_the_career_belonging_to_HIS_OWN_id(payload: dict) -> None:
    """A comp drawn as five forward points says nothing about the comp. The card needs
    his whole arc -- and it needs HIS.

    Each fixture arc starts at `float(i)` for comp id `100 + i`, so this asserts the
    join actually went through `chart_key(c["id"], pool)` rather than landing on a
    neighbour. That is the guard that matters now: the two lists this used to compare
    for parity were merged into one, so alignment is structural and comparing the merged
    list to itself asserts nothing -- what can still go wrong is the LOOKUP, which is the
    #284 defect class (a chart joined on a name, or on the wrong index).
    """
    view = _view(payload, player="Big Bat", n=3)
    floor = next(p["floor"] for p in payload["players"] if p["name"] == "Big Bat")

    assert len(view.comps) == 3
    assert [c["name"] for c in view.comps] == ["Comp 1", "Comp 2", "Comp 3"]
    for i, entry in enumerate(view.comps, start=1):
        assert entry["career"], "every fixture comp has an arc"
        ages = [pt[0] for pt in entry["career"]]
        assert ages == sorted(ages)
        assert entry["career"][0][1] == pytest.approx(float(i) - floor), (
            f"{entry['name']} must carry id {100 + i}'s arc, not another comp's"
        )


def test_every_comp_was_observed_at_the_age_he_matched_at(payload: dict) -> None:
    """`closest_paths` selects on `prepared.age == float(age)` -- an EXACT match -- so a
    comp necessarily has a season at the subject's own age, which is where every card
    draws its match rule (`data.age`, shipped once).

    Asserted against the stored ARC, not against a number the view synthesized: the
    previous version of this test compared `c["path"][0]["age"]` to `view.age + 1`, and
    both sides came from `sp.age` inside `build_player_view`, so it held by construction
    and could not fail whatever `closest_paths` returned. This one fails if a card is
    handed an arc that does not cover the age its rule is drawn at.
    """
    view = _view(payload, player="Big Bat")
    assert view.comps
    for c in view.comps:
        assert any(pt[0] == view.age for pt in c["career"]), (
            f"{c['name']} matched at {view.age} but his stored arc does not cover it"
        )


def test_a_comp_career_is_netted_against_the_QUERY_players_floor(payload: dict) -> None:
    """Same rule the comp PATHS already follow: the card asks what this arc would be
    worth in the subject's slot, so per-comp floors would put non-comparable lines on
    one axis."""
    row = next(p for p in payload["players"] if p["name"] == "Big Bat")
    var = _view(payload, player="Big Bat", scale="var")
    sgp = _view(payload, player="Big Bat", scale="sgp")

    assert var.comps[0]["career"][0][1] == pytest.approx(
        sgp.comps[0]["career"][0][1] - row["floor"]
    )


def test_a_blob_with_no_careers_yields_empty_arcs_rather_than_raising(payload: dict) -> None:
    """Every currently-deployed blob predates this feature. The page must render."""
    blob = _chart(payload)
    del blob["careers"]
    view = build_player_view(payload, player="Big Bat", chart=blob)

    assert view.comps, "the comps themselves still render"
    assert len(view.comps) == DEFAULT_COMPS, "and are sliced to `n` as usual"
    assert all(c["path"] for c in view.comps), "with their forward paths intact"
    assert all(c["career"] == [] for c in view.comps)


def test_a_careers_map_of_the_wrong_shape_is_refused_not_raised(payload: dict) -> None:
    """Refused rather than raised, for the reason #332 stands as: refusing a page over
    auxiliary data it could largely render is how /trajectory goes down."""
    blob = _chart(payload)
    blob["careers"] = [["not", "a", "mapping"]]
    view = build_player_view(payload, player="Big Bat", chart=blob)

    assert all(c["career"] == [] for c in view.comps)


def test_a_comp_stored_without_an_id_gets_an_empty_arc(payload: dict) -> None:
    """An older push wrote comps with no `id`. That is a missing join key, not a crash."""
    blob = _chart(payload)
    for block in blob["players"].values():
        for comp in block["comps"]:
            comp.pop("id", None)
    view = build_player_view(payload, player="Big Bat", chart=blob)

    assert all(c["career"] == [] for c in view.comps)


def test_a_two_way_comps_two_careers_do_not_collide(payload: dict) -> None:
    """One id, two pools, two different arcs. Keying on the bare id hands the hitter
    card the pitching career -- the same collapse `chart_key` exists to stop."""
    blob = _chart(payload)
    # Through `chart_key`, never a hand-written "101:hitter": it is imported at the top
    # of this file and the `_chart` fixture above builds its keys with it, so a literal
    # here would be a second spelling of the very join this test exists to police -- and
    # would keep writing the old shape if that format ever changed.
    first_comp_id = 101
    blob["careers"][chart_key(first_comp_id, "hitter")] = [[22, 1.0], [23, 2.0]]
    blob["careers"][chart_key(first_comp_id, "pitcher")] = [[22, 90.0], [23, 91.0]]

    hitter = build_player_view(payload, player="Big Bat", chart=blob)
    pitcher = build_player_view(payload, player="Big Arm", chart=blob)

    # Floors off the payload, never hardcoded: a literal here goes stale the moment a
    # fixture row moves.
    h_floor = next(p["floor"] for p in payload["players"] if p["name"] == "Big Bat")
    assert hitter.comps[0]["career"] == [[22, 1.0 - h_floor], [23, 2.0 - h_floor]]
    assert pitcher.comps[0]["career"][0][1] > 50


def _chart_including_base_age(payload: dict) -> dict:
    """`_chart` with the subject's own base-season age added to his career history.

    The panel produces this only after the season ends and the panel is rebuilt --
    `_live_seasons` un-flags the finished year, it enters `complete`, and it lands in
    `history` while `base_season` still names it.

    Built by MUTATING `_chart` rather than calling `to_chart_payload` again: the tuple
    key shape and the board-paired stamp are that helper's contract, and a second
    spelling of them here would keep passing against a shape the writer had stopped
    producing. Same pattern the neighbouring blob tests use.
    """
    blob = _chart(payload)
    for key in blob["players"]:
        blob["players"][key] = {"history": [[25, 16.0], [26, 20.0], [27, 21.0]], "comps": []}
    return blob


def test_the_paced_season_is_the_boards_now_netted_against_the_same_floor(
    payload: dict,
) -> None:
    """The gap at the base season was the most useful point on the chart. `now` is
    already in the board payload -- this only draws it."""
    row = next(p for p in payload["players"] if p["name"] == "Big Bat")
    var = _view(payload, player="Big Bat", scale="var")
    sgp = _view(payload, player="Big Bat", scale="sgp")

    assert var.paced == [row["age"], pytest.approx(row["now"] - row["floor"])]
    assert sgp.paced == [row["age"], pytest.approx(row["now"])]


def test_the_paced_season_is_ordered_after_every_realized_one(payload: dict) -> None:
    """The chart concatenates history + paced into one line with no re-sort, so the
    paced age must be strictly the largest."""
    view = _view(payload, player="Big Bat")
    assert view.paced is not None
    assert view.history, "the fixture stores a career"
    assert max(pt[0] for pt in view.history) < view.paced[0]


def test_a_base_season_already_realized_gets_no_paced_point(payload: dict) -> None:
    """The offseason case. A panel rebuilt after the season ends un-flags it, so it
    enters `history` -- and appending `now` beside it would draw two points at one age,
    one of them labelled a pace, on a finished year."""
    view = build_player_view(payload, player="Big Bat", chart=_chart_including_base_age(payload))
    assert view.age == 27, "the fixture's players are 27, which this blob's history covers"
    assert view.paced is None
    assert [pt[0] for pt in view.history] == [25, 26, 27], "history is left alone"


def test_the_paced_point_survives_a_chart_blob_that_does_not(payload: dict) -> None:
    """`now` comes from the BOARD, so it is never stale against the projection beside
    it. A refused chart blob costs the career line and the comps, not the anchor."""
    stale = _chart(payload)
    stale["generated_at"] = "some other build"
    view = build_player_view(payload, player="Big Bat", chart=stale)

    assert view.history == [] and view.comps == []
    assert view.chart_vintage_mismatch
    assert view.paced is not None, "board data, not chart data"


def test_an_unfound_player_has_no_paced_point(payload: dict) -> None:
    view = _view(payload, player="Nobody At All")
    assert view.paced is None
    assert view.paced_label == ""


@pytest.mark.parametrize(
    ("stored", "expected"),
    [(True, "2026 pace"), (False, "2026"), (None, "2026 pace")],
)
def test_the_paced_label_follows_the_boards_own_partial_flag(
    payload: dict, stored: object, expected: str
) -> None:
    """Built server-side like `axis_label`: the chart and the table read one string and
    cannot disagree about whether the season is finished. `None` is an old blob, which
    was written mid-season."""
    blob = {**payload}
    if stored is not None:
        blob["base_season_partial"] = stored
    blob["generated_at"] = f"hand-{next(_HAND_SEQ)}"  # do not share the parse cache
    view = build_player_view(blob, player="Big Bat", chart=None)
    assert view.paced_label == expected


def test_history_is_sorted_by_age_even_if_the_payload_is_not(payload: dict) -> None:
    """The push script's `groupby` happens to emit ascending; nothing in the payload
    schema guarantees it. An unsorted blob must still render a left-to-right career,
    or the chart zigzags -- pinning the PROPERTY, not a fixture literal that is already
    sorted going in.
    """
    scrambled = _chart(payload)
    key = chart_key(payload["players"][0]["id"], payload["players"][0]["pool"])
    scrambled["players"][key] = {"history": [[26, 20.0], [24, 14.0], [25, 16.0]], "comps": []}

    view = build_player_view(payload, player="Big Bat", chart=scrambled)
    ages = [pt[0] for pt in view.history]
    assert ages == sorted(ages), "career must render left-to-right by age"
    assert ages == [24, 25, 26]


# --------------------------------------------------------------------------
# #350: suggest-as-you-type search over the board
# --------------------------------------------------------------------------


def _named_payload(rows: list[tuple[int, str, str, int, str, list[float]]]) -> dict:
    """A payload with per-row control of id/name/pool/age/slot.

    `_hand_payload` above numbers ids sequentially and pins every row to one hitter
    slot, which cannot express the three shapes this search has to get right: two
    players sharing a name, one player appearing in both pools, and an accented name.
    """
    return {
        "base_season": BASE,
        "max_horizon": 3,
        "generated_at": f"named-{next(_HAND_SEQ)}",
        "players": [
            {
                "id": pid,
                "name": name,
                "pool": pool,
                "age": age,
                "slot": slot,
                "floor": 0.0,
                "now": 10.0,
                "prior": 10.0,
                "support": 0.9,
                "extrapolated": 0,
                "sgp": [
                    {
                        "horizon": h,
                        "mean": m,
                        "p10": m - 5,
                        "p90": m + 5,
                        "n_effective": 50.0,
                        "band_fell_back": 0,
                    }
                    for h, m in enumerate(means, start=1)
                ],
            }
            for pid, name, pool, age, slot, means in rows
        ],
    }


def test_search_matches_a_substring_not_just_a_prefix(payload: dict) -> None:
    """The headline complaint: `Witt` must find `Bobby Witt Jr.`. Here `bat` must find
    both Bats, neither of which starts with it.
    """
    names = [h["name"] for h in find_players(payload, "bat")]
    assert "Big Bat" in names and "Small Bat" in names, names


def test_search_ignores_case_and_accents() -> None:
    """Through `normalize_name`, the SAME function `build_player_view` resolves with.
    A suggestion the resolver would then fail on is worse than no suggestion.
    """
    blob = _named_payload([(1, "Yoan Moncada", "hitter", 27, "3B", [5.0, 5.0, 5.0])])
    for query in ("moncada", "MONCADA", "Moncada"):
        assert [h["name"] for h in find_players(blob, query)] == ["Yoan Moncada"], query


def test_exact_beats_prefix_beats_substring() -> None:
    """Ordering is stated in the issue so it is not invented per implementation.

    `Bat` is exactly one row's name, the prefix of another and inside a third; a
    ranking that only sorted by board rank would bury the exact match.
    """
    blob = _named_payload(
        [
            # Deliberately WORST by board rank, so a rank-only sort would put it last.
            (1, "Bat", "hitter", 27, "OF", [1.0, 1.0, 1.0]),
            (2, "Batting Glove", "hitter", 27, "OF", [9.0, 9.0, 9.0]),
            (3, "Big Bat", "hitter", 27, "OF", [8.0, 8.0, 8.0]),
        ]
    )
    assert [h["name"] for h in find_players(blob, "Bat")] == [
        "Bat",
        "Batting Glove",
        "Big Bat",
    ]


def test_ties_within_a_tier_break_by_board_rank() -> None:
    """Two equally-good matches are offered better-player-first."""
    blob = _named_payload(
        [
            (1, "Bat Weak", "hitter", 27, "OF", [1.0, 1.0, 1.0]),
            (2, "Bat Strong", "hitter", 27, "OF", [9.0, 9.0, 9.0]),
        ]
    )
    assert [h["name"] for h in find_players(blob, "bat")] == ["Bat Strong", "Bat Weak"]


def test_search_needs_two_characters(payload: dict) -> None:
    """Mirrors /api/players/find. One character over 1,169 rows is noise, not a
    suggestion, and the cap would decide the list instead of the query.
    """
    assert find_players(payload, "b") == []
    assert find_players(payload, "") == []
    assert find_players(payload, "  ") == []
    assert find_players(payload, "ba") != []


def test_search_is_capped() -> None:
    blob = _named_payload(
        [(i, f"Batter {i:03d}", "hitter", 27, "OF", [5.0, 5.0, 5.0]) for i in range(40)]
    )
    assert len(find_players(blob, "batter")) == 25
    assert len(find_players(blob, "batter", cap=3)) == 3


def test_every_suggestion_carries_the_fields_a_link_needs(payload: dict) -> None:
    """`id` and `pool` are required, not decorative: they are what `board_url` needs to
    produce a link that resolves straight to the player instead of the candidate page.
    """
    for hit in find_players(payload, "ba"):
        assert set(hit) == {"name", "id", "pool", "age", "slot"}, hit
        assert hit["id"] is not None and hit["pool"] in ("hitter", "pitcher")


def test_two_players_sharing_a_name_are_both_offered_and_distinguishable() -> None:
    """The live board has two Max Muncys. Collapsing them would hide one player."""
    blob = _named_payload(
        [
            (1, "Max Muncy", "hitter", 35, "1B", [7.0, 7.0, 7.0]),
            (2, "Max Muncy", "hitter", 22, "SS", [6.0, 6.0, 6.0]),
        ]
    )
    hits = find_players(blob, "muncy")
    assert len(hits) == 2, hits
    assert {h["id"] for h in hits} == {1, 2}
    assert {h["age"] for h in hits} == {35, 22}


def test_a_two_way_player_is_offered_once_per_pool() -> None:
    """Ohtani's two rows share an id and an age and differ only by pool and slot, so a
    list keyed on name or id alone offers one row where the board has two.
    """
    blob = _named_payload(
        [
            (660271, "Shohei Ohtani", "hitter", 31, "DH", [9.0, 9.0, 9.0]),
            (660271, "Shohei Ohtani", "pitcher", 31, "SP", [4.0, 4.0, 4.0]),
        ]
    )
    hits = find_players(blob, "ohtani")
    assert len(hits) == 2, hits
    assert {h["pool"] for h in hits} == {"hitter", "pitcher"}


def test_an_exact_name_still_resolves_rather_than_suggesting(payload: dict) -> None:
    """The fallback must not fire when the name resolves. A working search that started
    returning a one-item candidate list instead of the player would be a regression.
    """
    view = build_player_view(payload, player="Big Bat", chart=None)
    assert view.found is True
    assert view.candidates == []


def test_a_partial_name_falls_back_to_candidates_instead_of_a_dead_end(
    payload: dict,
) -> None:
    """The server-side half of #350, which works with JS off: `bat` matched nothing
    exactly, so offer what it does match rather than refusing.
    """
    view = build_player_view(payload, player="bat", chart=None)
    assert view.found is False
    names = [c["name"] for c in view.candidates]
    assert "Big Bat" in names and "Small Bat" in names, names
    for candidate in view.candidates:
        assert candidate["id"] is not None and candidate["pool"]


def test_a_name_absent_from_the_board_still_reports_absent(payload: dict) -> None:
    """A typo and a real exclusion must not read the same. Substring found nothing, so
    this stays the refusal -- with no candidates to imply otherwise.
    """
    view = build_player_view(payload, player="Nobody Here", chart=None)
    assert view.found is False
    assert view.candidates == []
