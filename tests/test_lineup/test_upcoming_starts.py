"""Unit tests for the rotation anchor + projection logic."""

from fantasy_baseball.lineup.upcoming_starts import (
    GameSlot,
    StartEntry,
)


def test_game_slot_fields():
    slot = GameSlot(
        date="2026-05-05",
        game_number=1,
        opponent="LAD",
        indicator="@",
        announced_starter="Bryan Woo",
    )
    assert slot.date == "2026-05-05"
    assert slot.game_number == 1
    assert slot.opponent == "LAD"
    assert slot.indicator == "@"
    assert slot.announced_starter == "Bryan Woo"


def test_start_entry_announced_default_false():
    entry = StartEntry(
        date="2026-05-05",
        day="Mon",
        opponent="LAD",
        indicator="@",
    )
    assert entry.announced is False


def test_start_entry_with_detail():
    entry = StartEntry(
        date="2026-05-05",
        day="Mon",
        opponent="LAD",
        indicator="@",
        announced=True,
        matchup_quality="Tough",
        detail={"ops": 0.789, "ops_rank": 4, "k_pct": 22.1, "k_rank": 18},
    )
    assert entry.announced is True
    assert entry.matchup_quality == "Tough"
    assert entry.detail["ops_rank"] == 4
