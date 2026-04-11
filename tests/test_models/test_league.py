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
