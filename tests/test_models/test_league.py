import sqlite3
from datetime import date

import pytest


def _team(name: str, team_key: str = ""):
    from fantasy_baseball.models.team import Team
    return Team(name=name, team_key=team_key, rosters=[])


def _snap(d: date, *teams: tuple[str, int]):
    from fantasy_baseball.models.standings import (
        CategoryStats, StandingsEntry, StandingsSnapshot,
    )
    entries = [
        StandingsEntry(name, f"k-{name}", rank, CategoryStats(r=100 + rank))
        for name, rank in teams
    ]
    return StandingsSnapshot(effective_date=d, entries=entries)


class TestLeagueBasics:
    def test_construction(self):
        from fantasy_baseball.models.league import League
        league = League(season_year=2026, teams=[], standings=[])
        assert league.season_year == 2026
        assert league.teams == []
        assert league.standings == []


class TestTeamLookups:
    def _league(self):
        from fantasy_baseball.models.league import League
        return League(
            season_year=2026,
            teams=[_team("Hart of the Order", "k-hart"), _team("Rivals", "k-riv")],
            standings=[],
        )

    def test_team_by_name(self):
        league = self._league()
        t = league.team_by_name("Rivals")
        assert t.name == "Rivals"
        assert t.team_key == "k-riv"

    def test_team_by_name_unknown_raises(self):
        league = self._league()
        with pytest.raises(KeyError, match="Unknown team"):
            league.team_by_name("Nobody")

    def test_team_by_key(self):
        league = self._league()
        t = league.team_by_key("k-hart")
        assert t.name == "Hart of the Order"

    def test_team_by_key_unknown_raises(self):
        league = self._league()
        with pytest.raises(KeyError, match="Unknown team_key"):
            league.team_by_key("k-nobody")


class TestStandingsLookups:
    def _league(self):
        from fantasy_baseball.models.league import League
        return League(
            season_year=2026,
            teams=[],
            standings=[
                _snap(date(2026, 3, 31), ("T1", 1)),
                _snap(date(2026, 4, 7), ("T1", 2)),
                _snap(date(2026, 4, 14), ("T1", 3)),
            ],
        )

    def test_latest_standings_returns_max_date(self):
        league = self._league()
        snap = league.latest_standings()
        assert snap.effective_date == date(2026, 4, 14)

    def test_latest_standings_raises_on_empty(self):
        from fantasy_baseball.models.league import League
        league = League(season_year=2026, teams=[], standings=[])
        with pytest.raises(ValueError, match="no standings"):
            league.latest_standings()

    def test_standings_as_of_exact_match(self):
        league = self._league()
        snap = league.standings_as_of(date(2026, 4, 7))
        assert snap is not None
        assert snap.effective_date == date(2026, 4, 7)

    def test_standings_as_of_between_snapshots(self):
        league = self._league()
        snap = league.standings_as_of(date(2026, 4, 10))
        assert snap is not None
        assert snap.effective_date == date(2026, 4, 7)

    def test_standings_as_of_before_first_returns_none(self):
        league = self._league()
        assert league.standings_as_of(date(2026, 1, 1)) is None

    def test_standings_as_of_after_last_returns_latest(self):
        league = self._league()
        snap = league.standings_as_of(date(2026, 6, 1))
        assert snap is not None
        assert snap.effective_date == date(2026, 4, 14)


@pytest.fixture
def conn_with_data():
    """SQLite connection pre-populated with fixture data for from_db tests."""
    from fantasy_baseball.data.db import create_tables
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    create_tables(c)

    # Two teams, three snapshots (two for team A, one for team B)
    rosters = [
        # snapshot_date, week_num, team, slot, player_name, positions, status, yahoo_id
        ("2026-04-07", 2, "Hart of the Order", "C",   "Ivan Herrera",      "C, Util",  None,    "11"),
        ("2026-04-07", 2, "Hart of the Order", "OF",  "Juan Soto",         "OF, Util", None,    "12"),
        ("2026-04-07", 2, "Hart of the Order", "BN",  "Marcus Semien",     "2B, Util", "DTD",   "13"),
        ("2026-04-14", 3, "Hart of the Order", "C",   "Ivan Herrera",      "C, Util",  None,    "11"),
        ("2026-04-14", 3, "Hart of the Order", "OF",  "Juan Soto",         "OF, Util", None,    "12"),
        ("2026-04-14", 3, "Hart of the Order", "2B",  "Marcus Semien",     "2B, Util", None,    "13"),
        ("2026-04-14", 3, "Rivals",            "C",   "William Contreras", "C, Util",  None,    "99"),
    ]
    c.executemany(
        "INSERT INTO weekly_rosters "
        "(snapshot_date, week_num, team, slot, player_name, positions, status, yahoo_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rosters,
    )

    standings = [
        # year, snapshot_date, team, team_key, rank, r, hr, rbi, sb, avg, w, k, sv, era, whip
        (2026, "2026-04-07", "Hart of the Order", "k-hart", 3,
         100, 40, 110, 15, 0.270, 55, 750, 30, 3.90, 1.18),
        (2026, "2026-04-07", "Rivals",            "k-riv",  5,
         95,  35, 100, 12, 0.265, 50, 700, 28, 4.10, 1.22),
        (2026, "2026-04-14", "Hart of the Order", "k-hart", 2,
         120, 45, 130, 20, 0.275, 60, 820, 33, 3.85, 1.15),
        (2026, "2026-04-14", "Rivals",            "k-riv",  4,
         110, 38, 115, 16, 0.268, 56, 760, 31, 3.95, 1.19),
    ]
    c.executemany(
        "INSERT INTO standings "
        "(year, snapshot_date, team, team_key, rank, r, hr, rbi, sb, avg, w, k, sv, era, whip) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        standings,
    )
    c.commit()
    yield c
    c.close()


class TestLeagueFromDb:
    def test_loads_teams_and_standings(self, conn_with_data):
        from fantasy_baseball.models.league import League
        league = League.from_db(conn_with_data, season_year=2026)
        assert {t.name for t in league.teams} == {"Hart of the Order", "Rivals"}
        assert len(league.standings) == 2

    def test_season_year_stored(self, conn_with_data):
        from fantasy_baseball.models.league import League
        league = League.from_db(conn_with_data, season_year=2026)
        assert league.season_year == 2026

    def test_team_has_sorted_rosters(self, conn_with_data):
        from fantasy_baseball.models.league import League
        league = League.from_db(conn_with_data, season_year=2026)
        team = league.team_by_name("Hart of the Order")
        dates = [r.effective_date for r in team.rosters]
        # Two snapshots for Hart
        assert date(2026, 4, 7) in dates
        assert date(2026, 4, 14) in dates

    def test_roster_entries_have_parsed_positions(self, conn_with_data):
        from fantasy_baseball.models.league import League
        from fantasy_baseball.models.positions import Position
        league = League.from_db(conn_with_data, season_year=2026)
        team = league.team_by_name("Hart of the Order")
        apr7 = team.roster_as_of(date(2026, 4, 7))
        assert apr7 is not None
        by_name = {e.name: e for e in apr7.entries}
        herrera = by_name["Ivan Herrera"]
        assert herrera.positions == [Position.C, Position.UTIL]
        assert herrera.selected_position is Position.C
        assert herrera.yahoo_id == "11"

    def test_status_nulls_become_empty_string(self, conn_with_data):
        from fantasy_baseball.models.league import League
        league = League.from_db(conn_with_data, season_year=2026)
        team = league.team_by_name("Hart of the Order")
        apr7 = team.roster_as_of(date(2026, 4, 7))
        assert apr7 is not None
        by_name = {e.name: e for e in apr7.entries}
        assert by_name["Ivan Herrera"].status == ""
        assert by_name["Marcus Semien"].status == "DTD"

    def test_standings_entries_populated(self, conn_with_data):
        from fantasy_baseball.models.league import League
        league = League.from_db(conn_with_data, season_year=2026)
        snap = league.standings_as_of(date(2026, 4, 14))
        assert snap is not None
        lookup = snap.by_team()
        hart = lookup["Hart of the Order"]
        assert hart.team_key == "k-hart"
        assert hart.rank == 2
        assert hart.stats.r == 120
        assert hart.stats.era == pytest.approx(3.85)

    def test_team_appears_in_weekly_rosters_but_not_standings(self, conn_with_data):
        """A team with roster rows but no standings row still loads."""
        conn_with_data.execute(
            "INSERT INTO weekly_rosters "
            "(snapshot_date, week_num, team, slot, player_name, positions) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("2026-04-14", 3, "Ghost Team", "OF", "Nobody", "OF, Util"),
        )
        conn_with_data.commit()

        from fantasy_baseball.models.league import League
        league = League.from_db(conn_with_data, season_year=2026)
        ghost = league.team_by_name("Ghost Team")
        assert len(ghost.rosters) == 1
        assert ghost.team_key == ""  # no standings row to provide one

    def test_team_appears_in_standings_but_not_weekly_rosters(self, conn_with_data):
        """A team with standings rows but no roster rows still loads."""
        conn_with_data.execute(
            "INSERT INTO standings "
            "(year, snapshot_date, team, team_key, rank, r, hr, rbi, sb, avg, w, k, sv, era, whip) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (2026, "2026-04-14", "Standings Only", "k-so", 6,
             80, 30, 90, 10, 0.260, 45, 650, 22, 4.20, 1.25),
        )
        conn_with_data.commit()

        from fantasy_baseball.models.league import League
        league = League.from_db(conn_with_data, season_year=2026)
        so = league.team_by_name("Standings Only")
        assert so.rosters == []
        assert so.team_key == "k-so"

    def test_unknown_position_token_raises(self, conn_with_data):
        """Unknown position tokens in weekly_rosters surface as errors."""
        conn_with_data.execute(
            "INSERT INTO weekly_rosters "
            "(snapshot_date, week_num, team, slot, player_name, positions) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("2026-04-14", 3, "Hart of the Order", "QB", "Bad Data", "QB, Util"),
        )
        conn_with_data.commit()

        from fantasy_baseball.models.league import League
        with pytest.raises(ValueError, match="Unknown position"):
            League.from_db(conn_with_data, season_year=2026)

    def test_team_key_backfills_from_standings(self, conn_with_data):
        """team_key on Team comes from the standings table."""
        from fantasy_baseball.models.league import League
        league = League.from_db(conn_with_data, season_year=2026)
        assert league.team_by_name("Hart of the Order").team_key == "k-hart"
        assert league.team_by_name("Rivals").team_key == "k-riv"
