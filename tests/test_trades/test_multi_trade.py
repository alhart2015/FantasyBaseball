"""Tests for the multi-player trade evaluator."""

from __future__ import annotations

from fantasy_baseball.models.player import HitterStats, PitcherStats, Player
from fantasy_baseball.trades.evaluate import player_rest_of_season_stats
from fantasy_baseball.trades.multi_trade import (
    CategoryDelta,
    MultiTradeResult,
    TradeProposal,
    _can_roster_after,
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
        name=name, player_type="hitter", positions=[pos],
        rest_of_season=HitterStats(pa=int(ab * 1.15), ab=ab, h=h,
                                    r=r, hr=hr, rbi=rbi, sb=sb, avg=avg),
    )


def _make_pitcher(name, ip=150, w=9, k=140, sv=0, era=3.80, whip=1.25, pos="P"):
    er = int(era * ip / 9)
    bb = 40
    h_allowed = int(whip * ip - bb)
    return Player(
        name=name, player_type="pitcher", positions=[pos],
        rest_of_season=PitcherStats(ip=ip, w=w, k=k, sv=sv, era=era, whip=whip,
                                     er=er, bb=bb, h_allowed=h_allowed),
    )


def _team_stats_from_players(players: list[Player]) -> dict[str, float]:
    """Build a stats dict matching the apply_swap_delta baseline pools."""
    stats = {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "W": 0, "K": 0, "SV": 0,
             "AVG": 0.270, "ERA": 3.80, "WHIP": 1.25}
    for p in players:
        s = player_rest_of_season_stats(p)
        for cat in ("R", "HR", "RBI", "SB", "W", "K", "SV"):
            stats[cat] += s[cat]
    return stats


def _standings_of(teams: dict[str, list[Player]]) -> list[dict]:
    return [{"name": name, "stats": _team_stats_from_players(players)}
            for name, players in teams.items()]


def test_evaluate_2_for_2_legal_returns_delta():
    me_name = "Hart"
    opp_name = "Rival"
    me_roster = [_make_hitter(f"Me{i}", r=80-i) for i in range(11)] + \
                [_make_pitcher(f"MeP{i}") for i in range(9)] + \
                [_make_hitter(f"MeBN{i}") for i in range(3)]
    for p in me_roster:
        p.selected_position = "BN" if p.name.startswith("MeBN") else (
            "P" if p.player_type == "pitcher" else "OF"
        )

    rival_roster = [_make_hitter(f"Riv{i}", r=70-i) for i in range(11)] + \
                   [_make_pitcher(f"RivP{i}") for i in range(9)] + \
                   [_make_hitter(f"RivBN{i}") for i in range(3)]
    for p in rival_roster:
        p.selected_position = "BN" if p.name.startswith("RivBN") else (
            "P" if p.player_type == "pitcher" else "OF"
        )

    team3 = [_make_hitter(f"T3_{i}", r=60) for i in range(20)] + \
            [_make_pitcher(f"T3P{i}") for i in range(3)]
    team4 = [_make_hitter(f"T4_{i}", r=50) for i in range(20)] + \
            [_make_pitcher(f"T4P{i}") for i in range(3)]

    standings = _standings_of({me_name: me_roster, opp_name: rival_roster,
                                "T3": team3, "T4": team4})

    proposal = TradeProposal(
        opponent=opp_name,
        send=["Me0::hitter", "Me1::hitter"],
        receive=["Riv0::hitter", "Riv1::hitter"],
        my_active_ids={player_key(p) for p in me_roster
                       if p.selected_position not in ("BN", "IL")
                       and p.name not in ("Me0", "Me1")}
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
        "R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"
    }
