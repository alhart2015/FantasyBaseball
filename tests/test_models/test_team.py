from datetime import date

import pytest


def _entry(name: str, slot: str = "OF"):
    from fantasy_baseball.models.positions import Position
    from fantasy_baseball.models.roster import RosterEntry
    p = Position.parse(slot)
    return RosterEntry(name=name, positions=[p], selected_position=p)


def _roster(d: date, *names: str):
    from fantasy_baseball.models.roster import Roster
    return Roster(effective_date=d, entries=[_entry(n) for n in names])


class TestTeamBasics:
    def test_team_construction(self):
        from fantasy_baseball.models.team import Team
        team = Team(name="Hart of the Order", team_key="k1", rosters=[])
        assert team.name == "Hart of the Order"
        assert team.team_key == "k1"
        assert team.rosters == []


class TestLatestRoster:
    def test_latest_roster_returns_max_effective_date(self):
        from fantasy_baseball.models.team import Team
        r1 = _roster(date(2026, 4, 7), "Alpha")
        r2 = _roster(date(2026, 4, 14), "Beta")
        team = Team("T", "k", [r1, r2])
        assert team.latest_roster() is r2

    def test_latest_roster_works_when_unsorted(self):
        """Team does not assume rosters are pre-sorted."""
        from fantasy_baseball.models.team import Team
        r1 = _roster(date(2026, 4, 7), "Alpha")
        r2 = _roster(date(2026, 4, 14), "Beta")
        team = Team("T", "k", [r2, r1])  # out of order on purpose
        assert team.latest_roster() is r2

    def test_latest_roster_raises_on_empty(self):
        from fantasy_baseball.models.team import Team
        team = Team("T", "k", [])
        with pytest.raises(ValueError, match="no rosters"):
            team.latest_roster()


class TestRosterAsOf:
    def _team(self):
        from fantasy_baseball.models.team import Team
        return Team("T", "k", [
            _roster(date(2026, 3, 31), "Opening Day Roster"),
            _roster(date(2026, 4, 7), "Week 2"),
            _roster(date(2026, 4, 14), "Week 3"),
        ])

    def test_exact_match(self):
        team = self._team()
        roster = team.roster_as_of(date(2026, 4, 7))
        assert roster is not None
        assert roster.effective_date == date(2026, 4, 7)
        assert roster.names() == {"Week 2"}

    def test_date_between_snapshots_returns_latest_prior(self):
        team = self._team()
        roster = team.roster_as_of(date(2026, 4, 10))  # between 04-07 and 04-14
        assert roster is not None
        assert roster.effective_date == date(2026, 4, 7)

    def test_date_after_latest_returns_latest(self):
        team = self._team()
        roster = team.roster_as_of(date(2026, 5, 1))
        assert roster is not None
        assert roster.effective_date == date(2026, 4, 14)

    def test_date_before_first_returns_none(self):
        team = self._team()
        assert team.roster_as_of(date(2026, 1, 1)) is None

    def test_empty_team_returns_none(self):
        from fantasy_baseball.models.team import Team
        team = Team("T", "k", [])
        assert team.roster_as_of(date(2026, 4, 11)) is None


class TestOwnershipPeriods:
    def _team_with_history(self):
        """Team that swapped Alpha -> Beta at the 04-14 lock."""
        from fantasy_baseball.models.team import Team
        return Team("T", "k", [
            _roster(date(2026, 3, 31), "Alpha", "Constant"),
            _roster(date(2026, 4, 7), "Alpha", "Constant"),
            _roster(date(2026, 4, 14), "Beta", "Constant"),
        ])

    def test_empty_team_returns_empty(self):
        from fantasy_baseball.models.team import Team
        team = Team("T", "k", [])
        periods = team.ownership_periods(
            season_start=date(2026, 3, 31),
            season_end=date(2026, 10, 1),
            today=date(2026, 4, 11),
        )
        assert periods == []

    def test_ownership_periods_walks_history(self):
        team = self._team_with_history()
        periods = team.ownership_periods(
            season_start=date(2026, 3, 31),
            season_end=date(2026, 10, 1),
            today=date(2026, 4, 20),
        )
        # Each entry yields (name, start, end)
        by_name: dict[str, list[tuple]] = {}
        for entry, start, end in periods:
            by_name.setdefault(entry.name, []).append((start, end))

        # Alpha owned from 03-31 to 04-14 (across two snapshots that contain them)
        assert by_name["Alpha"] == [
            (date(2026, 3, 31), date(2026, 4, 7)),
            (date(2026, 4, 7), date(2026, 4, 14)),
        ]
        # Beta owned from 04-14 through today (20th)
        assert by_name["Beta"] == [(date(2026, 4, 14), date(2026, 4, 20))]
        # Constant owned the whole way
        assert by_name["Constant"] == [
            (date(2026, 3, 31), date(2026, 4, 7)),
            (date(2026, 4, 7), date(2026, 4, 14)),
            (date(2026, 4, 14), date(2026, 4, 20)),
        ]

    def test_ownership_periods_clips_to_today(self):
        """Future-dated snapshots don't contribute days past `today`."""
        from fantasy_baseball.models.team import Team
        team = Team("T", "k", [
            _roster(date(2026, 4, 7), "Alpha"),
            _roster(date(2026, 4, 14), "Beta"),  # future relative to today
        ])
        periods = team.ownership_periods(
            season_start=date(2026, 3, 31),
            season_end=date(2026, 10, 1),
            today=date(2026, 4, 11),  # before the 04-14 snapshot
        )
        names = [e.name for e, _, _ in periods]
        # Beta's future-dated snapshot contributes nothing yet
        assert "Alpha" in names
        assert "Beta" not in names

    def test_ownership_periods_clips_to_season_window(self):
        """Preseason snapshots don't contribute days before season_start."""
        from fantasy_baseball.models.team import Team
        team = Team("T", "k", [
            _roster(date(2026, 3, 15), "Preseason"),  # before season
            _roster(date(2026, 4, 7), "In Season"),
        ])
        periods = team.ownership_periods(
            season_start=date(2026, 3, 31),
            season_end=date(2026, 10, 1),
            today=date(2026, 4, 20),
        )
        by_name = {
            e.name: (start, end) for e, start, end in periods
        }
        # Preseason snapshot's window gets clipped to [season_start, 04-07)
        assert by_name["Preseason"] == (date(2026, 3, 31), date(2026, 4, 7))
        assert by_name["In Season"] == (date(2026, 4, 7), date(2026, 4, 20))
