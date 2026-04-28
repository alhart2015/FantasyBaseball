"""Tests for the multi-player trade evaluator."""

from __future__ import annotations

from datetime import date

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


def test_trade_proposal_defaults_empty_lists():
    p = TradeProposal(opponent="Foo")
    assert p.send == []
    assert p.receive == []
    assert p.my_drops == []
    assert p.opp_drops == []
    assert p.my_adds == []
    assert p.my_active_ids == set()


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
    )
    assert result.legal is False
    assert "Ghost" in result.reason


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
