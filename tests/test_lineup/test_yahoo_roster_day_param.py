import datetime
from unittest.mock import MagicMock


class TestFetchRosterDayParam:
    def test_fetch_roster_without_day_calls_roster_no_args(self):
        from fantasy_baseball.lineup.yahoo_roster import fetch_roster

        team = MagicMock()
        team.roster.return_value = []
        league = MagicMock()
        league.to_team.return_value = team

        fetch_roster(league, "431.l.17492.t.3")

        team.roster.assert_called_once_with()

    def test_fetch_roster_with_day_passes_day_kwarg(self):
        from fantasy_baseball.lineup.yahoo_roster import fetch_roster

        team = MagicMock()
        team.roster.return_value = []
        league = MagicMock()
        league.to_team.return_value = team

        target = datetime.date(2026, 4, 14)
        fetch_roster(league, "431.l.17492.t.3", day=target)

        team.roster.assert_called_once_with(day=target)

    def test_fetch_roster_returns_parse_roster_output(self):
        from fantasy_baseball.lineup.yahoo_roster import fetch_roster

        team = MagicMock()
        team.roster.return_value = [
            {
                "name": "Ivan Herrera",
                "eligible_positions": ["C", "Util"],
                "selected_position": "C",
                "player_id": "12345",
                "status": "",
            }
        ]
        league = MagicMock()
        league.to_team.return_value = team

        result = fetch_roster(league, "k", day=datetime.date(2026, 4, 14))

        assert result == [
            {
                "name": "Ivan Herrera",
                "positions": ["C", "Util"],
                "selected_position": "C",
                "player_id": "12345",
                "status": "",
            }
        ]
