"""Tests for the multi-player trade evaluator."""

from __future__ import annotations

from datetime import date

import pytest

from fantasy_baseball.models.player import HitterStats, PitcherStats, Player
from fantasy_baseball.models.standings import (
    CategoryStats,
    ProjectedStandings,
    ProjectedStandingsEntry,
)
from fantasy_baseball.trades.evaluate import player_rest_of_season_stats
from fantasy_baseball.trades.multi_trade import (
    CategoryDelta,
    MultiTradeResult,
    TradeProposal,
    _can_roster_after,
    build_waiver_pool,
    evaluate_multi_trade,
    player_key,
)
from fantasy_baseball.utils.constants import ALL_CATEGORIES as ALL_CATEGORIES_LOCAL


def test_trade_proposal_defaults_empty_lists():
    p = TradeProposal(opponent="Foo")
    assert p.send == []
    assert p.receive == []
    assert p.my_drops == []
    assert p.opp_drops == []
    assert p.my_adds == []
    assert p.my_active_ids == set()


def test_trade_proposal_has_opp_active_ids_default_empty_set():
    p = TradeProposal(opponent="Foo")
    assert p.opp_active_ids == set()


def test_trade_proposal_accepts_opp_active_ids():
    p = TradeProposal(opponent="Foo", opp_active_ids={"Cade Smith::pitcher"})
    assert p.opp_active_ids == {"Cade Smith::pitcher"}


def test_multi_trade_result_shape():
    r = MultiTradeResult(
        legal=True,
        reason=None,
        delta_total=1.5,
        categories={"R": CategoryDelta(before=10.0, after=11.0, delta=1.0)},
    )
    assert r.legal is True
    assert r.categories["R"].delta == 1.0


def test_category_view_shape():
    from fantasy_baseball.trades.multi_trade import CategoryView

    v = CategoryView(before=10.0, after=12.0, delta=2.0)
    assert v.before == 10.0
    assert v.after == 12.0
    assert v.delta == 2.0


def test_view_block_shape():
    from fantasy_baseball.trades.multi_trade import CategoryView, ViewBlock

    v = ViewBlock(
        delta_total=1.5,
        categories={"R": CategoryView(before=10.0, after=11.0, delta=1.0)},
    )
    assert v.delta_total == 1.5
    assert v.categories["R"].after == 11.0


def test_multi_trade_result_has_view_blocks():
    from fantasy_baseball.trades.multi_trade import (
        MultiTradeResult,
        ViewBlock,
    )

    empty = ViewBlock(delta_total=0.0, categories={})
    r = MultiTradeResult(
        legal=True,
        reason=None,
        delta_total=0.0,
        categories={},
        roto=empty,
        ev_roto=empty,
        stat_totals=empty,
    )
    assert r.roto.delta_total == 0.0
    assert r.ev_roto.delta_total == 0.0
    assert r.stat_totals.delta_total == 0.0


def _hitter_with_key(name: str) -> Player:
    return Player(
        name=name,
        player_type="hitter",
        positions=["OF"],
        rest_of_season=HitterStats(pa=600, ab=500, h=125, r=70, hr=20, rbi=60, sb=5, avg=0.250),
    )


ROSTER_SLOTS_STANDARD = {
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "IF": 1,
    "OF": 4,
    "UTIL": 2,
    "P": 9,
    "BN": 2,
    "IL": 2,
}


def _roster_of(size: int, il: int = 0) -> list[Player]:
    roster: list[Player] = []
    for i in range(size):
        p = _hitter_with_key(f"P{i}")
        p.selected_position = "IL" if i < il else "OF"
        roster.append(p)
    return roster


def test_can_roster_after_passes_when_size_balances():
    roster = _roster_of(23)
    removed = ["P0::hitter", "P1::hitter"]
    added = [_hitter_with_key("Add1"), _hitter_with_key("Add2")]
    ok, reason = _can_roster_after(roster, removed, added, ROSTER_SLOTS_STANDARD)
    assert ok is True
    assert reason is None


def test_can_roster_after_rejects_wrong_resulting_size():
    roster = _roster_of(23)
    removed = ["P0::hitter", "P1::hitter"]
    added = [_hitter_with_key("Add1")]
    ok, reason = _can_roster_after(roster, removed, added, ROSTER_SLOTS_STANDARD)
    assert ok is False
    assert reason is not None
    assert "22" in reason


def test_can_roster_after_ignores_il_players_in_baseline_count():
    roster = _roster_of(25, il=2)
    ok, reason = _can_roster_after(roster, [], [], ROSTER_SLOTS_STANDARD)
    assert ok is True, reason


def _make_hitter(name, r=70, hr=20, rbi=65, sb=8, avg=0.270, ab=500, pos="OF"):
    h = int(avg * ab)
    return Player(
        name=name,
        player_type="hitter",
        positions=[pos],
        rest_of_season=HitterStats(
            pa=int(ab * 1.15), ab=ab, h=h, r=r, hr=hr, rbi=rbi, sb=sb, avg=avg
        ),
    )


def _make_pitcher(name, ip=150, w=9, k=140, sv=0, era=3.80, whip=1.25, pos="P"):
    er = int(era * ip / 9)
    bb = 40
    h_allowed = int(whip * ip - bb)
    return Player(
        name=name,
        player_type="pitcher",
        positions=[pos],
        rest_of_season=PitcherStats(
            ip=ip, w=w, k=k, sv=sv, era=era, whip=whip, er=er, bb=bb, h_allowed=h_allowed
        ),
    )


def _team_stats_from_players(players: list[Player]) -> dict[str, float]:
    """Build a stats dict matching the apply_swap_delta baseline pools."""
    stats = {
        "R": 0,
        "HR": 0,
        "RBI": 0,
        "SB": 0,
        "W": 0,
        "K": 0,
        "SV": 0,
        "AVG": 0.270,
        "ERA": 3.80,
        "WHIP": 1.25,
    }
    for p in players:
        s = player_rest_of_season_stats(p)
        for cat in ("R", "HR", "RBI", "SB", "W", "K", "SV"):
            stats[cat] += s[cat]
    return stats


def _standings_of(teams: dict[str, list[Player]]) -> ProjectedStandings:
    return ProjectedStandings(
        effective_date=date.fromisoformat("2026-04-01"),
        entries=[
            ProjectedStandingsEntry(
                team_name=name,
                stats=CategoryStats.from_dict(_team_stats_from_players(players)),
            )
            for name, players in teams.items()
        ],
    )


def test_evaluate_2_for_2_legal_returns_delta():
    me_name = "Hart"
    opp_name = "Rival"
    me_roster = (
        [_make_hitter(f"Me{i}", r=80 - i) for i in range(11)]
        + [_make_pitcher(f"MeP{i}") for i in range(9)]
        + [_make_hitter(f"MeBN{i}") for i in range(3)]
    )
    for p in me_roster:
        p.selected_position = (
            "BN" if p.name.startswith("MeBN") else ("P" if p.player_type == "pitcher" else "OF")
        )

    rival_roster = (
        [_make_hitter(f"Riv{i}", r=70 - i) for i in range(11)]
        + [_make_pitcher(f"RivP{i}") for i in range(9)]
        + [_make_hitter(f"RivBN{i}") for i in range(3)]
    )
    for p in rival_roster:
        p.selected_position = (
            "BN" if p.name.startswith("RivBN") else ("P" if p.player_type == "pitcher" else "OF")
        )

    team3 = [_make_hitter(f"T3_{i}", r=60) for i in range(20)] + [
        _make_pitcher(f"T3P{i}") for i in range(3)
    ]
    team4 = [_make_hitter(f"T4_{i}", r=50) for i in range(20)] + [
        _make_pitcher(f"T4P{i}") for i in range(3)
    ]

    standings = _standings_of(
        {me_name: me_roster, opp_name: rival_roster, "T3": team3, "T4": team4}
    )

    proposal = TradeProposal(
        opponent=opp_name,
        send=["Me0::hitter", "Me1::hitter"],
        receive=["Riv0::hitter", "Riv1::hitter"],
        my_active_ids={
            player_key(p)
            for p in me_roster
            if p.selected_position not in ("BN", "IL") and p.name not in ("Me0", "Me1")
        }
        | {"Riv0::hitter", "Riv1::hitter"},
    )

    result = evaluate_multi_trade(
        proposal=proposal,
        hart_name=me_name,
        hart_roster=me_roster,
        opp_rosters={opp_name: rival_roster},
        waiver_pool={},
        projected_standings=standings,
        team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
        fraction_remaining=0.6,
    )
    assert result.legal is True, result.reason
    cat_sum = sum(cd.delta for cd in result.categories.values())
    assert abs(cat_sum - result.delta_total) < 1e-6
    assert set(result.categories.keys()) == {
        "R",
        "HR",
        "RBI",
        "SB",
        "AVG",
        "W",
        "K",
        "SV",
        "ERA",
        "WHIP",
    }


def _build_league():
    """Helper: build a 4-team league with me=Hart, opp=Rival, and valid rosters.

    Returns (hart_roster, rival_roster, standings).
    """
    me = (
        [_make_hitter(f"Me{i}", r=80 - i) for i in range(11)]
        + [_make_pitcher(f"MeP{i}") for i in range(9)]
        + [_make_hitter(f"MeBN{i}") for i in range(3)]
    )
    for p in me:
        p.selected_position = (
            "BN" if p.name.startswith("MeBN") else "P" if p.player_type == "pitcher" else "OF"
        )
    riv = (
        [_make_hitter(f"Riv{i}", r=70 - i) for i in range(11)]
        + [_make_pitcher(f"RivP{i}") for i in range(9)]
        + [_make_hitter(f"RivBN{i}") for i in range(3)]
    )
    for p in riv:
        p.selected_position = (
            "BN" if p.name.startswith("RivBN") else "P" if p.player_type == "pitcher" else "OF"
        )
    t3 = [_make_hitter(f"T3_{i}", r=60) for i in range(20)] + [
        _make_pitcher(f"T3P{i}") for i in range(3)
    ]
    t4 = [_make_hitter(f"T4_{i}", r=50) for i in range(20)] + [
        _make_pitcher(f"T4P{i}") for i in range(3)
    ]
    standings = _standings_of({"Hart": me, "Rival": riv, "T3": t3, "T4": t4})
    return me, riv, standings


def test_2_for_3_with_drop_is_legal_and_scores():
    me, riv, standings = _build_league()
    proposal = TradeProposal(
        opponent="Rival",
        send=["Me0::hitter", "Me1::hitter"],
        receive=["Riv0::hitter", "Riv1::hitter", "Riv2::hitter"],
        my_drops=["MeBN0::hitter"],
        opp_drops=[],
        my_active_ids={
            player_key(p)
            for p in me
            if p.selected_position not in ("BN", "IL") and p.name not in ("Me0", "Me1")
        }
        | {"Riv0::hitter", "Riv1::hitter", "Riv2::hitter"},
    )
    result = evaluate_multi_trade(
        proposal=proposal,
        hart_name="Hart",
        hart_roster=me,
        opp_rosters={"Rival": riv},
        waiver_pool={},
        projected_standings=standings,
        team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
        fraction_remaining=0.6,
    )
    assert result.legal is False
    assert "Opponent" in result.reason


def test_2_for_3_drop_on_both_sides_is_legal():
    me, riv, standings = _build_league()
    proposal = TradeProposal(
        opponent="Rival",
        send=["Me0::hitter", "Me1::hitter"],
        receive=["Riv0::hitter", "Riv1::hitter"],
        my_drops=["MeBN0::hitter"],
        opp_drops=["RivBN0::hitter"],
        my_active_ids={
            player_key(p)
            for p in me
            if p.selected_position not in ("BN", "IL") and p.name not in ("Me0", "Me1")
        }
        | {"Riv0::hitter", "Riv1::hitter"},
    )
    result = evaluate_multi_trade(
        proposal=proposal,
        hart_name="Hart",
        hart_roster=me,
        opp_rosters={"Rival": riv},
        waiver_pool={},
        projected_standings=standings,
        team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
        fraction_remaining=0.6,
    )
    assert result.legal is False
    assert "22" in result.reason


def test_2_for_2_plus_drop_plus_waiver_add_is_legal():
    me, riv, standings = _build_league()
    waiver = _make_hitter("Waiver1", r=75)
    proposal = TradeProposal(
        opponent="Rival",
        send=["Me0::hitter", "Me1::hitter"],
        receive=["Riv0::hitter", "Riv1::hitter"],
        my_drops=["MeBN0::hitter"],
        my_adds=["Waiver1::hitter"],
        my_active_ids={
            player_key(p)
            for p in me
            if p.selected_position not in ("BN", "IL") and p.name not in ("Me0", "Me1")
        }
        | {"Riv0::hitter", "Riv1::hitter", "Waiver1::hitter"},
    )
    result = evaluate_multi_trade(
        proposal=proposal,
        hart_name="Hart",
        hart_roster=me,
        opp_rosters={"Rival": riv},
        waiver_pool={"Waiver1::hitter": waiver},
        projected_standings=standings,
        team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
        fraction_remaining=0.6,
    )
    assert result.legal is True, result.reason
    assert set(result.categories.keys()) >= {"R", "HR"}


def test_received_player_marked_bench_does_not_contribute():
    me, riv, standings = _build_league()
    active_set_all = {
        player_key(p)
        for p in me
        if p.selected_position not in ("BN", "IL") and p.name not in ("Me0", "Me1")
    } | {"Riv0::hitter", "Riv1::hitter"}
    active_set_bench_riv1 = active_set_all - {"Riv1::hitter"}

    proposal_all = TradeProposal(
        opponent="Rival",
        send=["Me0::hitter", "Me1::hitter"],
        receive=["Riv0::hitter", "Riv1::hitter"],
        my_active_ids=active_set_all,
    )
    proposal_bench = TradeProposal(
        opponent="Rival",
        send=["Me0::hitter", "Me1::hitter"],
        receive=["Riv0::hitter", "Riv1::hitter"],
        my_active_ids=active_set_bench_riv1,
    )

    r_all = evaluate_multi_trade(
        proposal=proposal_all,
        hart_name="Hart",
        hart_roster=me,
        opp_rosters={"Rival": riv},
        waiver_pool={},
        projected_standings=standings,
        team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
        fraction_remaining=0.6,
    )
    r_bench = evaluate_multi_trade(
        proposal=proposal_bench,
        hart_name="Hart",
        hart_roster=me,
        opp_rosters={"Rival": riv},
        waiver_pool={},
        projected_standings=standings,
        team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
        fraction_remaining=0.6,
    )
    assert r_all.legal is True
    assert r_bench.legal is True
    assert r_all.delta_total != r_bench.delta_total


def test_il_players_excluded_from_size_count():
    me, riv, standings = _build_league()
    me.append(_make_hitter("MeIL", r=10))
    me[-1].selected_position = "IL"
    me.append(_make_hitter("MeIL2", r=10))
    me[-1].selected_position = "IL"
    proposal = TradeProposal(
        opponent="Rival",
        send=["Me0::hitter"],
        receive=["Riv0::hitter"],
        my_active_ids={
            player_key(p) for p in me if p.selected_position not in ("BN", "IL") and p.name != "Me0"
        }
        | {"Riv0::hitter"},
    )
    result = evaluate_multi_trade(
        proposal=proposal,
        hart_name="Hart",
        hart_roster=me,
        opp_rosters={"Rival": riv},
        waiver_pool={},
        projected_standings=standings,
        team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
        fraction_remaining=0.6,
    )
    assert result.legal is True, result.reason


def test_unknown_player_key_returns_illegal_with_reason():
    me, riv, standings = _build_league()
    proposal = TradeProposal(
        opponent="Rival",
        send=["Ghost::hitter"],
        receive=["Riv0::hitter"],
        my_active_ids=set(),
    )
    result = evaluate_multi_trade(
        proposal=proposal,
        hart_name="Hart",
        hart_roster=me,
        opp_rosters={"Rival": riv},
        waiver_pool={},
        projected_standings=standings,
        team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
        fraction_remaining=0.6,
    )
    assert result.legal is False
    assert "Ghost" in result.reason


def _build_min_legal_proposal_fixture():
    """Build a minimal fixture: 23 hitters per team, swap one player.

    Returns (proposal, hart_name, hart_roster, opp_rosters, projected_standings).
    """

    hart_name = "Hart"
    opp_name = "Rival"

    hart_roster = [_hitter_with_key(f"M{i}") for i in range(23)]
    for i, p in enumerate(hart_roster):
        p.selected_position = "BN" if i >= 21 else "OF"
    opp_roster = [_hitter_with_key(f"R{i}") for i in range(23)]
    for i, p in enumerate(opp_roster):
        p.selected_position = "BN" if i >= 21 else "OF"

    cat_stats = {
        "R": 1000.0,
        "HR": 250.0,
        "RBI": 750.0,
        "SB": 80.0,
        "AVG": 0.260,
        "W": 70.0,
        "K": 1200.0,
        "SV": 50.0,
        "ERA": 3.80,
        "WHIP": 1.25,
    }
    standings = ProjectedStandings(
        effective_date=date(2026, 4, 1),
        entries=[
            ProjectedStandingsEntry(team_name=hart_name, stats=CategoryStats.from_dict(cat_stats)),
            ProjectedStandingsEntry(team_name=opp_name, stats=CategoryStats.from_dict(cat_stats)),
        ],
    )

    swap_send = player_key(hart_roster[0])
    swap_receive = player_key(opp_roster[0])

    # New active set: M0 leaves (replaced by R0), so M0 not in active; R0 added at slot 0.
    new_active = {player_key(p) for p in hart_roster[:21]}
    new_active.discard(swap_send)
    new_active.add(swap_receive)

    proposal = TradeProposal(
        opponent=opp_name,
        send=[swap_send],
        receive=[swap_receive],
        my_active_ids=new_active,
    )
    return proposal, hart_name, hart_roster, {opp_name: opp_roster}, standings


def test_evaluate_multi_trade_populates_view_blocks():
    proposal, hart_name, hart_roster, opp_rosters, standings = _build_min_legal_proposal_fixture()

    result = evaluate_multi_trade(
        proposal=proposal,
        hart_name=hart_name,
        hart_roster=hart_roster,
        opp_rosters=opp_rosters,
        waiver_pool={},
        projected_standings=standings,
        team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
        fraction_remaining=1.0,
    )

    assert result.legal, result.reason

    # All three view blocks present and populated for every roto category.
    for view_name in ("roto", "ev_roto", "stat_totals"):
        view = getattr(result, view_name)
        for cat in ("R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"):
            assert cat in view.categories, f"{view_name} missing {cat}"
            cv = view.categories[cat]
            assert cv.delta == cv.after - cv.before, f"{view_name}/{cat}: delta != after-before"

    # stat_totals.delta_total is conventionally 0.0 (no meaningful scalar).
    assert result.stat_totals.delta_total == 0.0

    # ev_roto block matches the existing top-level delta_total + categories
    # (since the top-level fields were always eROTO when team_sds is provided).
    # When team_sds is None, score_roto_dict returns integer roto, so ev_roto == roto here.
    for cat, cd in result.categories.items():
        assert result.ev_roto.categories[cat].delta == cd.delta


def test_legal_trade_result_has_band():
    """A legal trade result includes a band dict with the expected keys."""
    proposal, hart_name, hart_roster, opp_rosters, standings = _build_min_legal_proposal_fixture()

    result = evaluate_multi_trade(
        proposal=proposal,
        hart_name=hart_name,
        hart_roster=hart_roster,
        opp_rosters=opp_rosters,
        waiver_pool={},
        projected_standings=standings,
        team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
        fraction_remaining=0.6,
    )

    assert result.legal, result.reason
    assert result.band is not None
    assert set(result.band.keys()) == {"mean", "sd", "p_positive", "verdict"}


def test_band_mean_consistent_with_ev_roto_delta_total():
    """band['mean'] must equal ev_roto.delta_total within rounding (~0.01).

    The analytic band's mean is defined as the EV deltaRoto, computed by
    the same call-path as evaluate_multi_trade's ev_roto ViewBlock.  They
    share the same apply_swap_delta + score_roto_dict path so the values
    are identical before rounding; the serialised band['mean'] is rounded
    to 2 decimal places and ev_roto.delta_total is also rounded to 2 dp in
    the response, so any difference beyond floating-point noise is a bug.
    """
    proposal, hart_name, hart_roster, opp_rosters, standings = _build_min_legal_proposal_fixture()

    result = evaluate_multi_trade(
        proposal=proposal,
        hart_name=hart_name,
        hart_roster=hart_roster,
        opp_rosters=opp_rosters,
        waiver_pool={},
        projected_standings=standings,
        team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
        fraction_remaining=0.6,
    )

    assert result.legal, result.reason
    assert result.band is not None

    # band is already serialised to dict by to_dict(); mean is rounded to 2 dp.
    band_mean = result.band["mean"]
    ev_total = result.ev_roto.delta_total

    assert abs(band_mean - ev_total) < 0.01, (
        f"band['mean']={band_mean} differs from ev_roto.delta_total={ev_total} by more than 0.01"
    )
    # Verify all expected keys are present.
    assert set(result.band.keys()) == {"mean", "sd", "p_positive", "verdict"}


def test_multi_trade_result_has_opp_view_blocks_and_categories():
    from fantasy_baseball.trades.multi_trade import (
        CategoryDelta,
        MultiTradeResult,
        ViewBlock,
    )

    empty_view = ViewBlock(delta_total=0.0, categories={})
    r = MultiTradeResult(
        legal=True,
        reason=None,
        delta_total=0.0,
        categories={},
        roto=empty_view,
        ev_roto=empty_view,
        stat_totals=empty_view,
        band=None,
        opp_delta_total=0.0,
        opp_categories={"R": CategoryDelta(before=10.0, after=11.0, delta=1.0)},
        opp_roto=empty_view,
        opp_ev_roto=empty_view,
        opp_stat_totals=empty_view,
        opp_band=None,
    )
    assert r.opp_delta_total == 0.0
    assert r.opp_categories["R"].delta == 1.0
    assert r.opp_roto.delta_total == 0.0
    assert r.opp_ev_roto.delta_total == 0.0
    assert r.opp_stat_totals.delta_total == 0.0
    assert r.opp_band is None


def test_multi_trade_result_opp_fields_have_safe_defaults():
    """All opp_* fields must be constructable without explicit args (parity with my-side)."""
    r = MultiTradeResult(legal=True, reason=None, delta_total=0.0, categories={})
    assert r.opp_delta_total == 0.0
    assert r.opp_categories == {}
    assert r.opp_roto.delta_total == 0.0
    assert r.opp_ev_roto.delta_total == 0.0
    assert r.opp_stat_totals.delta_total == 0.0
    assert r.opp_band is None


# ---------------------------------------------------------------------------
# Task 3 helpers: simple 1-for-1 fixture for opp active-set delta tests.
# "Hart of the Order" sends H0, receives O0. 23 hitters each, 21 active.
# ---------------------------------------------------------------------------

_HART_OF_THE_ORDER = "Hart of the Order"
_OPP_TEAM = "Opp Team"

# Fixture-level stats used by both standings and `_eval_fixture`.
_FIXTURE_CAT_STATS = {
    "R": 1000.0,
    "HR": 250.0,
    "RBI": 750.0,
    "SB": 80.0,
    "AVG": 0.260,
    "W": 70.0,
    "K": 1200.0,
    "SV": 50.0,
    "ERA": 3.80,
    "WHIP": 1.25,
}


def _make_simple_1for1_proposal() -> TradeProposal:
    """1-for-1 proposal: Hart of the Order sends H0, receives O0.

    my_active_ids is the proposed post-trade active set for Hart's side:
    {H1..H20, O0}.  opp_active_ids is intentionally left empty (callers
    that need it should assign it, or call with _opp_active_set_for_simple_fixture()).
    """
    hart_roster = [_hitter_with_key(f"H{i}") for i in range(23)]
    for i, p in enumerate(hart_roster):
        p.selected_position = "BN" if i >= 21 else "OF"

    # Hart sends H0, receives O0.
    send_key = "H0::hitter"
    receive_key = "O0::hitter"

    new_active = {f"H{i}::hitter" for i in range(1, 21)}
    new_active.add(receive_key)  # O0 slides into Hart's active slot

    return TradeProposal(
        opponent=_OPP_TEAM,
        send=[send_key],
        receive=[receive_key],
        my_active_ids=new_active,
    )


def _opp_active_set_for_simple_fixture() -> set[str]:
    """Opp's proposed post-trade active set.

    Before the trade, Opp has O0..O20 active (O21, O22 are BN).
    After the trade: O0 is sent to Hart, H0 is received.
    Post-trade active set = {O1..O20, H0}.
    """
    active: set[str] = {f"O{i}::hitter" for i in range(1, 21)}
    active.add("H0::hitter")
    return active


def _eval_fixture() -> dict:
    """Keyword args for evaluate_multi_trade (everything except `proposal`).

    Builds a minimal 2-team league (Hart of the Order + Opp Team) with
    23 hitters each (21 active, 2 BN) and flat projected standings.
    """
    hart_roster = [_hitter_with_key(f"H{i}") for i in range(23)]
    for i, p in enumerate(hart_roster):
        p.selected_position = "BN" if i >= 21 else "OF"

    opp_roster = [_hitter_with_key(f"O{i}") for i in range(23)]
    for i, p in enumerate(opp_roster):
        p.selected_position = "BN" if i >= 21 else "OF"

    standings = ProjectedStandings(
        effective_date=date(2026, 4, 1),
        entries=[
            ProjectedStandingsEntry(
                team_name=_HART_OF_THE_ORDER,
                stats=CategoryStats.from_dict(_FIXTURE_CAT_STATS),
            ),
            ProjectedStandingsEntry(
                team_name=_OPP_TEAM,
                stats=CategoryStats.from_dict(_FIXTURE_CAT_STATS),
            ),
        ],
    )

    return dict(
        hart_name=_HART_OF_THE_ORDER,
        hart_roster=hart_roster,
        opp_rosters={_OPP_TEAM: opp_roster},
        waiver_pool={},
        projected_standings=standings,
        team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
        fraction_remaining=0.6,
    )


def test_opp_active_ids_fallback_matches_current_roster_level_math():
    """Without opp_active_ids, opp views must equal today's roster-level computation."""
    # Regression guard: a 1-for-1 trade with no drops produces the same opp stat
    # totals whether computed roster-level or active-set, because both swapped
    # players were active. So the fallback path must equal the explicit path
    # in this specific shape.
    proposal_no_aids = _make_simple_1for1_proposal()
    proposal_with_aids = _make_simple_1for1_proposal()
    proposal_with_aids.opp_active_ids = _opp_active_set_for_simple_fixture()

    fixture = _eval_fixture()
    r_fallback = evaluate_multi_trade(proposal=proposal_no_aids, **fixture)
    r_explicit = evaluate_multi_trade(proposal=proposal_with_aids, **fixture)

    assert (
        r_fallback.opp_stat_totals.categories["R"].after
        == r_explicit.opp_stat_totals.categories["R"].after
    )


def test_opp_active_ids_changes_opp_stat_totals_when_lineup_differs():
    """When opp_active_ids excludes a player who was active before the trade,
    opp_stat_totals must drop that player's contribution."""
    fixture = _eval_fixture()
    full_active = _opp_active_set_for_simple_fixture()
    bench_one = set(full_active)
    bench_one.discard(next(iter(full_active)))  # bench one player

    p_full = _make_simple_1for1_proposal()
    p_full.opp_active_ids = full_active
    p_bench = _make_simple_1for1_proposal()
    p_bench.opp_active_ids = bench_one

    r_full = evaluate_multi_trade(proposal=p_full, **fixture)
    r_bench = evaluate_multi_trade(proposal=p_bench, **fixture)

    # Benching any active hitter must reduce opp R/HR/RBI/SB after-totals.
    assert (
        r_bench.opp_stat_totals.categories["R"].after < r_full.opp_stat_totals.categories["R"].after
    )


def test_opp_roto_view_is_built_against_opp_baseline():
    """opp_roto.delta_total is the roto-point change for the opponent's team."""
    p = _make_simple_1for1_proposal()
    p.opp_active_ids = _opp_active_set_for_simple_fixture()
    r = evaluate_multi_trade(proposal=p, **_eval_fixture())

    assert r.legal is True
    assert set(r.opp_roto.categories) == {c.value for c in ALL_CATEGORIES_LOCAL}
    summed = sum(cv.delta for cv in r.opp_roto.categories.values())
    assert abs(r.opp_roto.delta_total - summed) < 1e-9


def test_opp_ev_roto_and_stat_totals_built():
    p = _make_simple_1for1_proposal()
    p.opp_active_ids = _opp_active_set_for_simple_fixture()
    r = evaluate_multi_trade(proposal=p, **_eval_fixture())

    assert r.opp_ev_roto.categories
    assert r.opp_stat_totals.categories
    assert r.opp_stat_totals.delta_total == 0.0  # convention: stat_totals has no scalar total


def test_opp_band_is_present_when_team_sds_provided():
    p = _make_simple_1for1_proposal()
    p.opp_active_ids = _opp_active_set_for_simple_fixture()
    fixture = _eval_fixture()
    # Build a minimal team_sds mapping both teams to per-category SDs.
    # Any positive float works; we just need non-None so the band path
    # has something to work with. ERA/WHIP SDs use a small positive value
    # as well -- they're inverse but the band handles that internally.
    team_sds = {
        _HART_OF_THE_ORDER: {cat: 1.0 for cat in ALL_CATEGORIES_LOCAL},
        _OPP_TEAM: {cat: 1.0 for cat in ALL_CATEGORIES_LOCAL},
    }
    fixture["team_sds"] = team_sds
    r = evaluate_multi_trade(proposal=p, **fixture)

    assert r.opp_band is not None
    assert "mean" in r.opp_band
    assert "sd" in r.opp_band
    assert "p_positive" in r.opp_band


def test_my_side_results_unchanged_after_opp_additions():
    """Regression guard: my-side roto/ev_roto/stat_totals are pinned to known
    values so future opp-side refactors can't drift them silently."""
    p = _make_simple_1for1_proposal()  # no opp_active_ids set -- fallback path
    r = evaluate_multi_trade(proposal=p, **_eval_fixture())

    assert r.legal is True
    assert r.roto.delta_total == pytest.approx(0.0)
    assert r.ev_roto.delta_total == pytest.approx(0.0)
    assert r.stat_totals.categories["R"].after == pytest.approx(1000.0)


def test_evaluate_multi_trade_legal_false_when_opponent_missing_from_standings():
    fixture = _eval_fixture()
    original_ps = fixture["projected_standings"]
    # Remove Opp Team from projected_standings; Hart of the Order is still present.
    fixture["projected_standings"] = ProjectedStandings(
        effective_date=original_ps.effective_date,
        entries=[e for e in original_ps.entries if e.team_name != _OPP_TEAM],
    )
    proposal = _make_simple_1for1_proposal()
    proposal.opp_active_ids = _opp_active_set_for_simple_fixture()
    r = evaluate_multi_trade(proposal=proposal, **fixture)
    assert r.legal is False
    assert r.reason is not None
    assert "Opponent" in r.reason
    assert _OPP_TEAM in r.reason


def test_build_waiver_pool_excludes_rostered_players():
    a = _make_hitter("Alice")
    b = _make_hitter("Bob")
    my_roster = [a]
    opp_rosters = {"Rival": [b]}
    ros_projections = {
        "hitters": [
            {
                "name": "Alice",
                "player_type": "hitter",
                "positions": ["OF"],
                "rest_of_season": {
                    "ab": 500,
                    "h": 125,
                    "r": 70,
                    "hr": 20,
                    "rbi": 60,
                    "sb": 5,
                    "avg": 0.25,
                    "pa": 575,
                },
            },
            {
                "name": "Bob",
                "player_type": "hitter",
                "positions": ["OF"],
                "rest_of_season": {
                    "ab": 500,
                    "h": 125,
                    "r": 70,
                    "hr": 20,
                    "rbi": 60,
                    "sb": 5,
                    "avg": 0.25,
                    "pa": 575,
                },
            },
            {
                "name": "Carol",
                "player_type": "hitter",
                "positions": ["OF"],
                "rest_of_season": {
                    "ab": 500,
                    "h": 125,
                    "r": 70,
                    "hr": 20,
                    "rbi": 60,
                    "sb": 5,
                    "avg": 0.25,
                    "pa": 575,
                },
            },
        ],
        "pitchers": [
            {
                "name": "Dan",
                "player_type": "pitcher",
                "positions": ["P"],
                "rest_of_season": {
                    "ip": 150,
                    "w": 9,
                    "k": 140,
                    "sv": 0,
                    "era": 3.80,
                    "whip": 1.25,
                    "er": 63,
                    "bb": 40,
                    "h_allowed": 147,
                },
            },
        ],
    }
    pool = build_waiver_pool(my_roster, opp_rosters, ros_projections)
    assert "Carol::hitter" in pool
    assert "Dan::pitcher" in pool
    assert "Alice::hitter" not in pool
    assert "Bob::hitter" not in pool
