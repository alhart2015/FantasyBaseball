"""Tests for the local-time helpers."""

from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo


class TestLocalNow:
    def test_returns_timezone_aware_datetime(self):
        from fantasy_baseball.utils.time_utils import LOCAL_TZ, local_now

        result = local_now()
        assert isinstance(result, datetime)
        assert result.tzinfo is not None
        assert result.tzinfo == LOCAL_TZ

    def test_matches_eastern_wall_clock(self):
        """Given a specific UTC moment, local_now() should return the
        corresponding wall-clock time in ``America/New_York``.

        The canonical failure case: 2026-04-12 00:45 UTC is actually
        2026-04-11 20:45 EDT, one calendar day earlier.
        """
        from fantasy_baseball.utils import time_utils

        fixed_utc = datetime(2026, 4, 12, 0, 45, tzinfo=ZoneInfo("UTC"))

        # Patch the datetime.now used inside time_utils so we can pin
        # the result without touching the system clock.
        class _FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return fixed_utc.replace(tzinfo=None)
                return fixed_utc.astimezone(tz)

        with patch.object(time_utils, "datetime", _FakeDatetime):
            result = time_utils.local_now()

        # 2026-04-12 00:45 UTC → 2026-04-11 20:45 EDT (DST active in April)
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 11
        assert result.hour == 20
        assert result.minute == 45


class TestLocalToday:
    def test_returns_date_object(self):
        from fantasy_baseball.utils.time_utils import local_today

        result = local_today()
        assert isinstance(result, date)

    def test_returns_local_calendar_day_not_utc(self):
        """2026-04-12 00:45 UTC is 2026-04-11 in Eastern time — the
        local_today() result must reflect the Eastern calendar day,
        not the UTC one."""
        from fantasy_baseball.utils import time_utils

        fixed_utc = datetime(2026, 4, 12, 0, 45, tzinfo=ZoneInfo("UTC"))

        class _FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return fixed_utc.replace(tzinfo=None)
                return fixed_utc.astimezone(tz)

        with patch.object(time_utils, "datetime", _FakeDatetime):
            result = time_utils.local_today()

        assert result == date(2026, 4, 11)

    def test_handles_dst_transition(self):
        """America/New_York is UTC-5 in winter (EST) and UTC-4 in
        summer (EDT). The ZoneInfo handling should pick the right
        offset based on the wall-clock date."""
        from fantasy_baseball.utils import time_utils

        # Mid-January — EST (UTC-5)
        fixed_winter = datetime(2026, 1, 15, 4, 0, tzinfo=ZoneInfo("UTC"))

        class _WinterFake(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return fixed_winter.replace(tzinfo=None)
                return fixed_winter.astimezone(tz)

        with patch.object(time_utils, "datetime", _WinterFake):
            assert time_utils.local_today() == date(2026, 1, 14)
            # 04:00 UTC = 23:00 EST the previous day

        # Mid-July — EDT (UTC-4)
        fixed_summer = datetime(2026, 7, 15, 3, 0, tzinfo=ZoneInfo("UTC"))

        class _SummerFake(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return fixed_summer.replace(tzinfo=None)
                return fixed_summer.astimezone(tz)

        with patch.object(time_utils, "datetime", _SummerFake):
            assert time_utils.local_today() == date(2026, 7, 14)
            # 03:00 UTC = 23:00 EDT the previous day


class TestComputeEffectiveDate:
    def test_sunday_end_date_returns_following_tuesday(self):
        # Yahoo scoring period ends on a Sunday; effective date is the
        # following Tuesday (lineup lock day).
        from fantasy_baseball.utils.time_utils import compute_effective_date
        assert compute_effective_date("2026-04-19") == date(2026, 4, 21)

    def test_accepts_iso_string(self):
        from fantasy_baseball.utils.time_utils import compute_effective_date
        assert compute_effective_date("2026-05-03") == date(2026, 5, 5)

    def test_tuesday_input_returns_following_tuesday(self):
        # next_tuesday is strict — a Tuesday input still moves forward.
        from fantasy_baseball.utils.time_utils import compute_effective_date
        assert compute_effective_date("2026-04-21") == date(2026, 4, 28)


class TestNextTuesday:
    def test_sunday_advances_two_days(self):
        """The canonical production case: Yahoo returns a Mon–Sun
        scoring week ending on Sunday. next_tuesday(Sunday) must
        produce the Tuesday two days later."""
        from fantasy_baseball.utils.time_utils import next_tuesday
        assert next_tuesday(date(2026, 4, 12)) == date(2026, 4, 14)

    def test_monday_advances_one_day(self):
        """If the scoring week ever returns a Monday end_date, the
        formula end+1 would correctly produce Tuesday."""
        from fantasy_baseball.utils.time_utils import next_tuesday
        assert next_tuesday(date(2026, 4, 13)) == date(2026, 4, 14)

    def test_tuesday_advances_a_full_week(self):
        """'next Tuesday' strictly after ref. Passing a Tuesday must
        return the Tuesday one week later, never the same day. This
        prevents zero-day effective_date windows at the edge of a
        scoring period."""
        from fantasy_baseball.utils.time_utils import next_tuesday
        assert next_tuesday(date(2026, 4, 14)) == date(2026, 4, 21)

    def test_wednesday_advances_six_days(self):
        from fantasy_baseball.utils.time_utils import next_tuesday
        assert next_tuesday(date(2026, 4, 15)) == date(2026, 4, 21)

    def test_thursday_friday_saturday(self):
        from fantasy_baseball.utils.time_utils import next_tuesday
        assert next_tuesday(date(2026, 4, 16)) == date(2026, 4, 21)  # Thu
        assert next_tuesday(date(2026, 4, 17)) == date(2026, 4, 21)  # Fri
        assert next_tuesday(date(2026, 4, 18)) == date(2026, 4, 21)  # Sat

    def test_crosses_month_boundary(self):
        """Sun Apr 26 → Tue Apr 28."""
        from fantasy_baseball.utils.time_utils import next_tuesday
        assert next_tuesday(date(2026, 4, 26)) == date(2026, 4, 28)

    def test_crosses_year_boundary(self):
        """Sun Dec 27 2026 → Tue Dec 29 2026.
        Sun Dec 28 2025 → Tue Dec 30 2025 (within same year)
        Sun Dec 27 2020 → Tue Dec 29 2020 → check a Dec end-of-year."""
        from fantasy_baseball.utils.time_utils import next_tuesday
        # Tue Dec 29 → next Tue Jan 5
        assert next_tuesday(date(2026, 12, 29)) == date(2027, 1, 5)
