"""Tests for compute_team_ytd_ab -- derives team YTD AB from
Team.ownership_periods intersected with per-game logs.

Same attribution model as SPoE (windowed by ownership) but sums actual
per-game AB rather than scaled preseason projections.
"""

from __future__ import annotations

from datetime import date

from fantasy_baseball.analysis.team_ytd_attribution import compute_team_ytd_ab
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
