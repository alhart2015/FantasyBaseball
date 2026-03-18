import datetime
from unittest.mock import MagicMock
from fantasy_baseball.lineup.yahoo_roster import fetch_scoring_period


class TestFetchScoringPeriod:
    def test_returns_date_strings_from_yahoo(self):
        mock_league = MagicMock()
        mock_league.current_week.return_value = 12
        mock_league.week_date_range.return_value = (
            datetime.date(2026, 6, 15),
            datetime.date(2026, 6, 21),
        )

        start, end = fetch_scoring_period(mock_league)

        assert start == "2026-06-15"
        assert end == "2026-06-21"
        mock_league.current_week.assert_called_once()
        mock_league.week_date_range.assert_called_once_with(12)

    def test_falls_back_to_current_week_on_error(self):
        mock_league = MagicMock()
        mock_league.current_week.side_effect = Exception("Yahoo down")

        start, end = fetch_scoring_period(mock_league)

        # Should return Mon-Sun of current week as fallback
        today = datetime.date.today()
        monday = today - datetime.timedelta(days=today.weekday())
        sunday = monday + datetime.timedelta(days=6)
        assert start == monday.isoformat()
        assert end == sunday.isoformat()
