from __future__ import annotations

import itertools
import math

import pytest

from fantasy_baseball.trajectory.board import BoardRow
from fantasy_baseball.trajectory.sweep import RANK_MOVE, sweep_pool, to_payload
from fantasy_baseball.web.trajectory_view import DEFAULT_TOP, build_board
from tests._trajectory_panel import synthetic_panel

BASE = 2026
_HAND_SEQ = itertools.count()


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
    assert var_bands != sgp_bands, "the two scales are separate fits with separate bands"


def _hand_payload(players: list[tuple[str, list[float]]]) -> dict:
    """A payload with exact per-year means, so rank arithmetic can be pinned directly.

    `_pack` is [horizon, mean, p10, p90, n_eff, band_fell_back]; the band and support
    are irrelevant here and set wide/high so nothing else flags.
    """
    return {
        "version": 1,
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
                "floor": 4.0,
                "now": 10.0,
                "prior": 10.0,
                "support": 0.9,
                "extrapolated": 0,
                "var": [[h, m, m - 5, m + 5, 50.0, 0] for h, m in enumerate(means, start=1)],
                "sgp": [[h, m, m - 5, m + 5, 50.0, 0] for h, m in enumerate(means, start=1)],
            }
            for i, (name, means) in enumerate(players, start=1)
        ],
    }


def test_no_arrow_when_both_rankings_rest_on_zeros() -> None:
    """VAR clamps at zero, so every below-replacement player totals 0.0 and nexts 0.0.

    `add_ranks` still gives them distinct consecutive ranks, broken by name -- and the
    zero-set for `next` (year 1 only) is strictly larger than the zero-set for `total`
    (all years), so the two blocks start at different offsets and the difference is
    systematically non-zero on identical inputs. Simulated on a live-shaped 1,169-row
    pool, 432 of 469 all-zero rows cleared the arrow threshold, worst move -97 -- drawn
    beside a row reading 0.0 in every column.
    """
    board = build_board(
        _hand_payload(
            [
                ("Real One", [9.0, 9.0, 9.0]),
                # total > 0 but next == 0: sits in the `next` zero-block, not `total`'s.
                # Named to sort AFTER the all-zero rows, so the two zero-blocks
                # interleave and the ranks genuinely diverge -- which is what happens on
                # the live board, where the tie-break is alphabetical over 1,169 rows.
                *[(f"Zeta Bloom {i}", [0.0, 4.0, 4.0]) for i in range(6)],
                # zero on both: rank differs only by the offset above.
                *[(f"Alpha Sub {i}", [0.0, 0.0, 0.0]) for i in range(6)],
            ]
        ),
        top="all",
        end=BASE + 3,
    )
    rows = {r["name"]: r for r in board.rows}

    subs = [r for n, r in rows.items() if n.startswith("Alpha Sub")]
    assert subs, "fixture must contain all-zero rows"
    assert all(r["total"] == 0.0 and r["next"] == 0.0 for r in subs)
    assert any(abs(r["rank_next"] - r["rank_total"]) >= RANK_MOVE for r in subs), (
        "fixture must actually produce a raw rank gap, or this proves nothing"
    )
    assert all(r["rank_move"] == 0 for r in subs), (
        "a row that is 0.0 in every column has no hold-vs-start signal to draw"
    )
    assert rows["Real One"]["rank_move"] == 0


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
    gapped_payload["players"][0]["var"] = [
        p for p in gapped_payload["players"][0]["var"] if p[0] != 1
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
    gapped["players"][0]["var"] = [p for p in gapped["players"][0]["var"] if p[0] != 1]

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


def test_my_players_are_marked(payload: dict) -> None:
    """The first question a reader has on a keeper board is which of these are already
    his. Roster blobs carry no mlbam_id (#284), so the join is (normalized name,
    player_type) -- the same non-unique key the rest of the repo has to live with.
    """
    board = build_board(payload, top="all", mine={("big bat", "hitter"), ("big arm", "pitcher")})
    marked = {r["name"] for r in board.rows if r["mine"]}
    assert marked == {"Big Bat", "Big Arm"}
    assert not any(r["mine_ambiguous"] for r in board.rows)

    unmarked = build_board(payload, top="all")
    assert not any(r["mine"] for r in unmarked.rows), "no roster read -- nothing claimed"


def test_an_empty_roster_read_is_not_a_successful_one(payload: dict) -> None:
    """ "You own none of these" and "we could not read your roster" must not look alike.

    `live_rosters` returns [] WITHOUT raising when the roster blobs are absent -- its own
    module docstring flags this: getting the asymmetry wrong "drops your own roster
    silently, which reads as 'you own nobody' rather than as an error". The route only
    caught exceptions, so a cold or half-written `cache:roster` gave `mine=set()`, which
    counted as a successful read: nothing highlighted AND no warning, so the page
    silently asserts the reader owns none of the top 50 keepers.
    """
    assert build_board(payload, mine=None).has_rosters is False, "no read attempted"
    assert build_board(payload, mine=set()).has_rosters is False, (
        "an EMPTY read is not a successful one -- it cannot be told from a failed one"
    )
    assert build_board(payload, mine={("big bat", "hitter")}).has_rosters is True


def test_a_colliding_name_is_flagged_rather_than_claimed(payload: dict) -> None:
    """Two players can share (normalized name, player_type) -- the live board carries two
    hitters called Max Muncy. Marking both as mine on one roster hit would put a player
    the reader does not own on his keeper shortlist, so an ambiguous match says so.
    """
    twin = dict(payload)
    first = payload["players"][0]
    twin["players"] = [*payload["players"], {**first, "id": first["id"] + 10_000}]

    board = build_board(twin, top="all", mine={(first["name"].lower(), first["pool"])})
    hits = [r for r in board.rows if r["mine"]]
    assert len(hits) == 2, "both rows match the only key available"
    assert all(r["mine_ambiguous"] for r in hits), "and both must say the match is unsure"


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
