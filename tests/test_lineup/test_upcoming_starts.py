"""Unit tests for the rotation anchor + projection logic."""

from fantasy_baseball.lineup.upcoming_starts import (
    GameSlot,
    StartEntry,
    build_team_game_index,
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


def _pp(date_, away, home, awp="", hwp="", num=1):
    return {
        "date": date_,
        "game_number": num,
        "away_team": away,
        "home_team": home,
        "away_pitcher": awp or "TBD",
        "home_pitcher": hwp or "TBD",
    }


class TestBuildTeamGameIndex:
    def test_filters_to_target_team(self):
        pps = [
            _pp("2026-05-05", "SEA", "LAD", awp="Woo"),
            _pp("2026-05-05", "NYY", "BOS", awp="Cole"),
            _pp("2026-05-06", "TEX", "SEA", hwp="Castillo"),
        ]
        slots = build_team_game_index(pps, "SEA")
        assert len(slots) == 2
        assert slots[0].opponent == "LAD"
        assert slots[0].indicator == "@"
        assert slots[0].announced_starter == "Woo"
        assert slots[1].opponent == "TEX"
        assert slots[1].indicator == "vs"
        assert slots[1].announced_starter == "Castillo"

    def test_chronological_ordering(self):
        pps = [
            _pp("2026-05-07", "SEA", "TEX"),
            _pp("2026-05-05", "SEA", "LAD"),
            _pp("2026-05-06", "SEA", "TEX"),
        ]
        slots = build_team_game_index(pps, "SEA")
        assert [s.date for s in slots] == ["2026-05-05", "2026-05-06", "2026-05-07"]

    def test_doubleheader_sorts_by_game_number(self):
        pps = [
            _pp("2026-05-05", "SEA", "LAD", num=2, awp="Gilbert"),
            _pp("2026-05-05", "SEA", "LAD", num=1, awp="Woo"),
        ]
        slots = build_team_game_index(pps, "SEA")
        assert [s.game_number for s in slots] == [1, 2]
        assert slots[0].announced_starter == "Woo"
        assert slots[1].announced_starter == "Gilbert"

    def test_tbd_announced_starter_becomes_empty(self):
        pps = [_pp("2026-05-05", "SEA", "LAD", awp="TBD")]
        slots = build_team_game_index(pps, "SEA")
        assert slots[0].announced_starter == ""

    def test_empty_when_team_not_in_schedule(self):
        pps = [_pp("2026-05-05", "NYY", "BOS")]
        assert build_team_game_index(pps, "SEA") == []
