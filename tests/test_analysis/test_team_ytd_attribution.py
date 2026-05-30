"""Tests for compute_team_ytd_ab -- derives team YTD AB from
Team.ownership_periods intersected with per-game logs.

Same attribution model as SPoE (windowed by ownership) but sums actual
per-game AB rather than scaled preseason projections.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from fantasy_baseball.analysis.team_ytd_attribution import (
    _load_per_game_hitter_ab,
    compute_team_ytd_ab,
)
from fantasy_baseball.models.league import League
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.roster import Roster, RosterEntry
from fantasy_baseball.models.team import Team


def _entry(name: str, slot: Position) -> RosterEntry:
    """Minimal RosterEntry for a hitter in the given slot."""
    return RosterEntry(
        name=name,
        positions=[Position.OF],  # eligible positions don't matter for AB attribution
        selected_position=slot,
        status="",
        yahoo_id="",
    )


def test_attributes_active_slot_games_only():
    """A player active in week 1 then benched in week 2: only week-1 ABs count."""
    # Fixture: two snapshots a week apart. Player active in snapshot 1, bench in snapshot 2.
    snap1 = Roster(
        effective_date=date(2026, 4, 1),
        entries=[_entry("Test Hitter", Position.OF)],
    )
    snap2 = Roster(
        effective_date=date(2026, 4, 8),
        entries=[_entry("Test Hitter", Position.BN)],
    )
    team = Team(name="My Team", team_key="t.1", rosters=[snap1, snap2])
    league = League(season_year=2026, teams=[team])

    # Per-game logs: 4 games in week 1 (4 AB each = 16 AB),
    # 3 games in week 2 (3 AB each = 9 AB)
    game_logs = {
        "001": {
            "name": "Test Hitter",
            "type": "hitter",
            "games": [
                {"date": "2026-04-02", "ab": 4, "h": 1, "pa": 4},
                {"date": "2026-04-03", "ab": 4, "h": 2, "pa": 4},
                {"date": "2026-04-05", "ab": 4, "h": 0, "pa": 4},
                {"date": "2026-04-07", "ab": 4, "h": 1, "pa": 4},
                {"date": "2026-04-09", "ab": 3, "h": 1, "pa": 3},
                {"date": "2026-04-10", "ab": 3, "h": 0, "pa": 3},
                {"date": "2026-04-12", "ab": 3, "h": 1, "pa": 3},
            ],
        },
    }

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs=game_logs,
    )

    assert ab_by_team["My Team"] == 16  # only week-1 active games


def test_attributes_games_only_within_ownership_window():
    """A player traded mid-week: only games in the team's window count."""
    # Team A owns the player from 2026-04-01 to 2026-04-08
    snap_a1 = Roster(
        effective_date=date(2026, 4, 1),
        entries=[_entry("Traded Player", Position.OF)],
    )
    snap_a2 = Roster(  # Traded Player gone after this date
        effective_date=date(2026, 4, 8),
        entries=[],
    )
    team_a = Team(name="Team A", team_key="a.1", rosters=[snap_a1, snap_a2])
    # Team B owns the player from 2026-04-08 onward
    snap_b1 = Roster(effective_date=date(2026, 4, 1), entries=[])
    snap_b2 = Roster(
        effective_date=date(2026, 4, 8),
        entries=[_entry("Traded Player", Position.OF)],
    )
    team_b = Team(name="Team B", team_key="b.1", rosters=[snap_b1, snap_b2])
    league = League(season_year=2026, teams=[team_a, team_b])

    game_logs = {
        "002": {
            "name": "Traded Player",
            "type": "hitter",
            "games": [
                {"date": "2026-04-02", "ab": 4, "h": 1, "pa": 4},
                {"date": "2026-04-07", "ab": 4, "h": 2, "pa": 4},
                {"date": "2026-04-10", "ab": 4, "h": 1, "pa": 4},
                {"date": "2026-04-12", "ab": 4, "h": 0, "pa": 4},
            ],
        },
    }

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs=game_logs,
    )

    assert ab_by_team["Team A"] == 8  # 2 games while owned by A
    assert ab_by_team["Team B"] == 8  # 2 games while owned by B


def test_skips_pitcher_logs():
    """Pitcher entries in game_logs are ignored (no AB attribution)."""
    snap = Roster(
        effective_date=date(2026, 4, 1),
        entries=[_entry("Some Pitcher", Position.P)],
    )
    team = Team(name="Pitchers Only", team_key="p.1", rosters=[snap])
    league = League(season_year=2026, teams=[team])

    game_logs = {
        "003": {
            "name": "Some Pitcher",
            "type": "pitcher",
            "games": [
                {"date": "2026-04-02", "ip": 6, "k": 8, "w": 1},
            ],
        },
    }

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs=game_logs,
    )

    assert ab_by_team["Pitchers Only"] == 0


def test_skips_il_slots():
    """Players in IL slots accrue no AB even if they have games (rare but possible)."""
    snap = Roster(
        effective_date=date(2026, 4, 1),
        entries=[_entry("IL Hitter", Position.IL)],
    )
    team = Team(name="IL Team", team_key="i.1", rosters=[snap])
    league = League(season_year=2026, teams=[team])

    game_logs = {
        "004": {
            "name": "IL Hitter",
            "type": "hitter",
            "games": [
                {"date": "2026-04-02", "ab": 4, "h": 1, "pa": 4},
            ],
        },
    }

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs=game_logs,
    )

    assert ab_by_team["IL Team"] == 0


def test_skips_bench_slot():
    """Players on BN slot accrue no AB even with games."""
    snap = Roster(
        effective_date=date(2026, 4, 1),
        entries=[_entry("Bench Hitter", Position.BN)],
    )
    team = Team(name="Bench Team", team_key="b.1", rosters=[snap])
    league = League(season_year=2026, teams=[team])

    game_logs = {
        "005": {
            "name": "Bench Hitter",
            "type": "hitter",
            "games": [
                {"date": "2026-04-02", "ab": 4, "h": 1, "pa": 4},
            ],
        },
    }

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs=game_logs,
    )

    assert ab_by_team["Bench Team"] == 0


def test_unknown_player_contributes_zero():
    """A player on the roster but not in game_logs contributes 0 (no crash)."""
    snap = Roster(
        effective_date=date(2026, 4, 1),
        entries=[_entry("Unknown Hitter", Position.OF)],
    )
    team = Team(name="Unknown Team", team_key="u.1", rosters=[snap])
    league = League(season_year=2026, teams=[team])

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs={},
    )

    assert ab_by_team["Unknown Team"] == 0


def test_every_league_team_appears_in_output():
    """Even a team with no AB contributions appears in the output dict
    (consumer iterates standings entries and may need a zero default)."""
    team_empty = Team(name="Empty", team_key="e.1", rosters=[])
    team_with_data = Team(
        name="With Data",
        team_key="w.1",
        rosters=[Roster(effective_date=date(2026, 4, 1), entries=[_entry("X", Position.OF)])],
    )
    league = League(season_year=2026, teams=[team_empty, team_with_data])

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs={},
    )

    assert "Empty" in ab_by_team
    assert "With Data" in ab_by_team
    assert ab_by_team["Empty"] == 0
    assert ab_by_team["With Data"] == 0


def test_period_end_is_exclusive():
    """A game on the period_end date should NOT be counted (half-open window).

    Player owned 2026-04-01 to 2026-04-08; the 2026-04-08 game falls on the
    period_end and must be excluded.
    """
    snap_a1 = Roster(
        effective_date=date(2026, 4, 1),
        entries=[_entry("Boundary Player", Position.OF)],
    )
    snap_a2 = Roster(
        effective_date=date(2026, 4, 8),
        entries=[],
    )
    team_a = Team(name="Team A", team_key="a.1", rosters=[snap_a1, snap_a2])
    snap_b1 = Roster(effective_date=date(2026, 4, 1), entries=[])
    snap_b2 = Roster(
        effective_date=date(2026, 4, 8),
        entries=[_entry("Boundary Player", Position.OF)],
    )
    team_b = Team(name="Team B", team_key="b.1", rosters=[snap_b1, snap_b2])
    league = League(season_year=2026, teams=[team_a, team_b])

    game_logs = {
        "006": {
            "name": "Boundary Player",
            "type": "hitter",
            "games": [
                # On Team A's period_start (inclusive) -- counts for A
                {"date": "2026-04-01", "ab": 5, "h": 1, "pa": 5},
                # On Team A's period_end / Team B's period_start (exclusive A, inclusive B)
                {"date": "2026-04-08", "ab": 5, "h": 1, "pa": 5},
            ],
        },
    }

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs=game_logs,
    )

    assert ab_by_team["Team A"] == 5  # only 2026-04-01
    assert ab_by_team["Team B"] == 5  # only 2026-04-08


# -----------------------------------------------------------------------------
# Fix #9: pitcher-slot entry must not claim a same-named hitter's ABs
# -----------------------------------------------------------------------------


def test_pitcher_slot_does_not_attribute_hitter_games():
    """A roster entry in a pitcher slot must not pick up the hitter game log
    of a same-normalized-name player (Shohei Ohtani case).

    Ohtani is rostered as a hitter AND a pitcher. The hitter entry already
    accrues his ABs; if the pitcher entry's lookup against the hitter dict
    also fires, his ABs are double-counted.
    """
    snap = Roster(
        effective_date=date(2026, 4, 1),
        entries=[_entry("Shohei Ohtani", Position.P)],
    )
    team = Team(name="Two-Way", team_key="x.1", rosters=[snap])
    league = League(season_year=2026, teams=[team])

    game_logs = {
        "010": {
            "name": "Shohei Ohtani",
            "type": "hitter",
            "games": [
                {"date": "2026-04-02", "ab": 4, "h": 2, "pa": 4},
                {"date": "2026-04-03", "ab": 4, "h": 1, "pa": 4},
            ],
        },
    }

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs=game_logs,
    )

    assert ab_by_team["Two-Way"] == 0


def test_pitcher_slot_filter_covers_sp_and_rp_slots():
    """SP and RP slots must also skip the hitter-game lookup (same rationale)."""
    snap = Roster(
        effective_date=date(2026, 4, 1),
        entries=[
            _entry("Pitcher A", Position.SP),
            _entry("Pitcher B", Position.RP),
        ],
    )
    team = Team(name="Two-Way", team_key="x.1", rosters=[snap])
    league = League(season_year=2026, teams=[team])

    game_logs = {
        "011": {
            "name": "Pitcher A",
            "type": "hitter",  # corrupted/colliding row
            "games": [{"date": "2026-04-02", "ab": 4, "h": 2, "pa": 4}],
        },
        "012": {
            "name": "Pitcher B",
            "type": "hitter",
            "games": [{"date": "2026-04-03", "ab": 4, "h": 2, "pa": 4}],
        },
    }

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs=game_logs,
    )

    assert ab_by_team["Two-Way"] == 0


# -----------------------------------------------------------------------------
# Fix #10: same-day games in the last ownership window must be counted
# -----------------------------------------------------------------------------


def test_last_window_includes_todays_game():
    """A game dated TODAY in the last (current) ownership window is counted.

    Yahoo's stats.avg includes today's completed games. The half-open
    [start, end) clip on the last window would drop them. Verify the
    function counts a game whose date equals ``today``.
    """
    snap = Roster(
        effective_date=date(2026, 4, 1),
        entries=[_entry("Today Hitter", Position.OF)],
    )
    team = Team(name="Today Team", team_key="t.1", rosters=[snap])
    league = League(season_year=2026, teams=[team])

    today = date(2026, 4, 14)
    game_logs = {
        "020": {
            "name": "Today Hitter",
            "type": "hitter",
            "games": [
                {"date": "2026-04-13", "ab": 4, "h": 1, "pa": 4},
                {"date": today.isoformat(), "ab": 5, "h": 2, "pa": 5},
            ],
        },
    }

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=today,
        game_logs=game_logs,
    )

    assert ab_by_team["Today Team"] == 9  # yesterday (4) + today (5)


def test_non_last_window_remains_exclusive_on_end_boundary():
    """The closed-right relaxation must only apply to the last window.

    A game on the period_end date of a NON-last (mid-season) window still
    belongs to the NEXT window's owner -- preventing double-count.
    """
    snap_a1 = Roster(
        effective_date=date(2026, 4, 1),
        entries=[_entry("Boundary", Position.OF)],
    )
    snap_a2 = Roster(effective_date=date(2026, 4, 8), entries=[])
    team_a = Team(name="A", team_key="a.1", rosters=[snap_a1, snap_a2])

    snap_b1 = Roster(effective_date=date(2026, 4, 1), entries=[])
    snap_b2 = Roster(
        effective_date=date(2026, 4, 8),
        entries=[_entry("Boundary", Position.OF)],
    )
    team_b = Team(name="B", team_key="b.1", rosters=[snap_b1, snap_b2])
    league = League(season_year=2026, teams=[team_a, team_b])

    game_logs = {
        "021": {
            "name": "Boundary",
            "type": "hitter",
            "games": [
                {"date": "2026-04-08", "ab": 5, "h": 1, "pa": 5},
            ],
        },
    }

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs=game_logs,
    )

    assert ab_by_team["A"] == 0  # 4/8 is A's period_end (exclusive)
    assert ab_by_team["B"] == 5  # 4/8 is B's period_start (inclusive)


# -----------------------------------------------------------------------------
# Fix #12: defensive parsing -- malformed entries must not crash
# -----------------------------------------------------------------------------


def test_none_entry_does_not_crash():
    """A top-level entry that is ``None`` (corrupted row) is skipped."""
    snap = Roster(
        effective_date=date(2026, 4, 1),
        entries=[_entry("Hitter", Position.OF)],
    )
    team = Team(name="Team", team_key="t.1", rosters=[snap])
    league = League(season_year=2026, teams=[team])

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs={"x": None},
    )

    assert ab_by_team["Team"] == 0


def test_null_games_list_does_not_crash():
    """An entry with ``games: None`` (JSON null) is treated as empty."""
    snap = Roster(
        effective_date=date(2026, 4, 1),
        entries=[_entry("Null Games", Position.OF)],
    )
    team = Team(name="Team", team_key="t.1", rosters=[snap])
    league = League(season_year=2026, teams=[team])

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs={"x": {"name": "Null Games", "type": "hitter", "games": None}},
    )

    assert ab_by_team["Team"] == 0


def test_list_entry_does_not_crash():
    """A top-level entry that is a list instead of a dict is skipped."""
    snap = Roster(
        effective_date=date(2026, 4, 1),
        entries=[_entry("Hitter", Position.OF)],
    )
    team = Team(name="Team", team_key="t.1", rosters=[snap])
    league = League(season_year=2026, teams=[team])

    ab_by_team = compute_team_ytd_ab(
        league,
        season_start=date(2026, 3, 27),
        season_end=date(2026, 9, 28),
        today=date(2026, 4, 14),
        game_logs={"x": ["this", "is", "not", "an", "entry"]},
    )

    assert ab_by_team["Team"] == 0


# -----------------------------------------------------------------------------
# Fix #14: missing game-logs file -- warn and return {}
# -----------------------------------------------------------------------------


def test_missing_game_logs_file_logs_warning(caplog, tmp_path: Path):
    """Pointing _load_per_game_hitter_ab at a non-existent path logs a
    warning and returns an empty dict instead of silently degrading."""
    missing = tmp_path / "does_not_exist.json"
    with caplog.at_level(logging.WARNING, logger="fantasy_baseball.analysis.team_ytd_attribution"):
        result = _load_per_game_hitter_ab(game_logs=None, path=missing)
    assert result == {}
    assert any("not found" in rec.message.lower() for rec in caplog.records)


def test_corrupt_json_on_disk_logs_warning(caplog, tmp_path: Path):
    """A partial-write / corrupt JSON file is caught: warn and return {}."""
    bad = tmp_path / "corrupt.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="fantasy_baseball.analysis.team_ytd_attribution"):
        result = _load_per_game_hitter_ab(game_logs=None, path=bad)
    assert result == {}
    assert any("failed to load" in rec.message.lower() for rec in caplog.records)


# -----------------------------------------------------------------------------
# Fix #8: normalized-name collision detection
# -----------------------------------------------------------------------------


def test_normalized_name_collision_logs_warning(caplog):
    """Two distinct source names that normalize to the same key trigger a
    warning so the operator can see when attribution may double-count.

    ASCII "Jose Ramirez" and the accented form (e + combining acute,
    U+0301) both normalize to "jose ramirez" after NFKD + accent strip
    + lowercase. The accented form is built via chr() to keep this
    source file ASCII-only (per CLAUDE.md).
    """
    from fantasy_baseball.utils.name_utils import normalize_name

    accented = "Jos" + "e" + chr(0x301) + " Ramirez"

    game_logs = {
        "100": {
            "name": "Jose Ramirez",
            "type": "hitter",
            "games": [{"date": "2026-04-02", "ab": 4, "h": 1, "pa": 4}],
        },
        "101": {
            "name": accented,
            "type": "hitter",
            "games": [{"date": "2026-04-03", "ab": 3, "h": 0, "pa": 3}],
        },
    }
    with caplog.at_level(logging.WARNING, logger="fantasy_baseball.analysis.team_ytd_attribution"):
        out = _load_per_game_hitter_ab(game_logs=game_logs)

    # Sanity: both source names actually collide under normalize_name.
    assert normalize_name("Jose Ramirez") == normalize_name(accented)

    # Warning fired.
    assert any("collision" in rec.message.lower() for rec in caplog.records)

    # Merge still happens (no MLB-id to disambiguate) -- both games end up
    # in the same normalized-name bucket.
    merged = out[normalize_name("Jose Ramirez")]
    assert len(merged) == 2
