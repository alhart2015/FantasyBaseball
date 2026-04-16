"""Tests for refresh_steps.py — pure helpers extracted from
run_full_refresh that are specific to the refresh orchestration
(not general enough to push into a domain module)."""
import pytest

from fantasy_baseball.models.player import (
    HitterStats,
    PitcherStats,
    Player,
    PlayerType,
)
from fantasy_baseball.models.positions import Position
from fantasy_baseball.web.refresh_steps import merge_matched_and_raw_roster


def _player(name, player_type=PlayerType.HITTER, positions=None,
            selected_position=None, wsgp=0.0, ros=None):
    """Build a Player fixture.

    Accepts either ``HitterStats`` / ``PitcherStats`` for ``ros`` (the
    real stat-bag types in this codebase — there is no ``RosterStats``).
    Positions default to ``["OF"]`` for hitters and ``["SP"]`` for
    pitchers; ``Player`` accepts bare strings because ``Position`` is a
    ``StrEnum``.
    """
    positions = positions or (["OF"] if player_type == PlayerType.HITTER else ["SP"])
    selected_position = selected_position or positions[0]
    p = Player(
        name=name,
        positions=positions,
        player_type=player_type,
        selected_position=selected_position,
        yahoo_id=f"{name}::{player_type.value}",
    )
    p.rest_of_season = ros
    p.wsgp = wsgp
    return p


class TestMergeMatchedAndRawRoster:
    def test_matched_players_get_preseason_attached(self):
        soto = _player("Soto", ros=HitterStats(r=80))
        soto_pre = _player("Soto", ros=HitterStats(r=100, hr=35))
        result = merge_matched_and_raw_roster(
            matched=[soto],
            roster_raw=[{
                "name": "Soto", "positions": ["OF"],
                "selected_position": "OF", "player_id": "1", "status": "",
            }],
            preseason_lookup={"soto": soto_pre},
        )
        assert len(result) == 1
        assert result[0].preseason is soto_pre.rest_of_season

    def test_matched_player_without_preseason_entry(self):
        soto = _player("Soto", ros=HitterStats(r=80))
        result = merge_matched_and_raw_roster(
            matched=[soto],
            roster_raw=[{
                "name": "Soto", "positions": ["OF"],
                "selected_position": "OF", "player_id": "1", "status": "",
            }],
            preseason_lookup={},  # no preseason match
        )
        assert len(result) == 1
        # No preseason attached (attribute not set or stays as default)
        assert result[0].preseason is None

    def test_unmatched_raw_player_added_as_hitter(self):
        # Raw player not in matched list — should be added with
        # player_type inferred from positions (OF → HITTER).
        result = merge_matched_and_raw_roster(
            matched=[],
            roster_raw=[{
                "name": "Newbie", "positions": ["OF"],
                "selected_position": "OF", "player_id": "99", "status": "",
            }],
            preseason_lookup={},
        )
        assert len(result) == 1
        assert result[0].name == "Newbie"
        assert result[0].player_type == PlayerType.HITTER

    def test_unmatched_raw_player_added_as_pitcher(self):
        # SP positions → PITCHER
        result = merge_matched_and_raw_roster(
            matched=[],
            roster_raw=[{
                "name": "RookiePitcher", "positions": ["SP"],
                "selected_position": "P", "player_id": "100", "status": "",
            }],
            preseason_lookup={},
        )
        assert len(result) == 1
        assert result[0].player_type == PlayerType.PITCHER

    def test_matched_player_skipped_in_raw_iteration(self):
        # When a player is in BOTH matched and raw, only one entry should
        # appear in the result (the matched one).
        soto = _player("Soto")
        result = merge_matched_and_raw_roster(
            matched=[soto],
            roster_raw=[
                {"name": "Soto", "positions": ["OF"],
                 "selected_position": "OF", "player_id": "1", "status": ""},
                {"name": "Newbie", "positions": ["OF"],
                 "selected_position": "BN", "player_id": "99", "status": ""},
            ],
            preseason_lookup={},
        )
        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"Soto", "Newbie"}
