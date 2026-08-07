from __future__ import annotations

import itertools
import math
import re

import pytest

from fantasy_baseball.data.rosters import RosterSpot
from fantasy_baseball.trajectory.board import BoardRow
from fantasy_baseball.trajectory.sweep import (
    RANK_MOVE,
    sweep_pool,
    to_payload,
)
from fantasy_baseball.web.trajectory_view import DEFAULT_TOP, build_board, build_teams_board
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
    """Two teams with nothing scored both total 0.0. Left to dict order the page
    would reorder between reads -- the arbitrary-ordering defect `index_rosters`
    was fixed for in 06bf2646."""
    spots = _teams_fixture()
    forward = build_teams_board(payload, spots=spots, my_team="Mine")
    reverse = build_teams_board(payload, spots=list(reversed(spots)), my_team="Mine")

    empties = [b.team for b in forward.blocks if b.total == 0.0]
    assert empties == ["Aardvark FC", "Empty FC"], "ties break on name, ascending"
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
    """N is a slice of an already-ranked list, so the first row must not move."""
    one = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine", per_team=1)
    two = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine", per_team=2)

    rivals_one = next(b for b in one.blocks if b.team == "Rivals")
    rivals_two = next(b for b in two.blocks if b.team == "Rivals")
    assert len(rivals_one.rows) == 1
    assert len(rivals_two.rows) == 2
    assert rivals_one.rows[0]["name"] == rivals_two.rows[0]["name"]
    assert rivals_one.scored == rivals_two.scored, "scored is the team's set, not the slice"


def test_scored_follows_the_pool_filter_and_unscored_does_too(payload: dict) -> None:
    """A block that says "5 of 24" under a hitters-only table must mean 24 hitters,
    or the visible rows cannot add up to the number printed beside them."""
    both = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine")
    hitters = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine", pool="hitter")

    assert next(b for b in both.blocks if b.team == "Rivals").scored == 2
    assert next(b for b in hitters.blocks if b.team == "Rivals").scored == 1
    assert next(b for b in both.blocks if b.team == "Mine").unscored == ["Never Scored"]
    assert next(b for b in hitters.blocks if b.team == "Mine").unscored == ["Never Scored"]


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


def test_per_team_and_end_year_clamp_junk_from_the_query_string(payload: dict) -> None:
    """These arrive from a URL a reader can edit."""
    board = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine", per_team="junk")
    assert board.per_team == 5
    assert build_teams_board(payload, spots=_teams_fixture(), per_team=0).per_team == 1
    assert build_teams_board(payload, spots=_teams_fixture(), per_team=999).per_team == 50
    assert build_teams_board(payload, spots=_teams_fixture(), end="nonsense").end_year == 2027
