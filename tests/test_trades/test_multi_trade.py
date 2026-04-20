"""Tests for the multi-player trade evaluator."""

from __future__ import annotations

from fantasy_baseball.models.player import HitterStats, Player
from fantasy_baseball.trades.multi_trade import (
    CategoryDelta,
    MultiTradeResult,
    TradeProposal,
    _can_roster_after,
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
