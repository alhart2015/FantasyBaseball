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
