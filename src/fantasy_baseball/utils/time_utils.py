"""Local-time helpers for the refresh pipeline and display layer.

The project runs on Render (UTC server) but the user is in Eastern
time. Using ``datetime.now()`` or ``date.today()`` directly produces
UTC dates, which drift from the user's wall clock by up to 5 hours
and can flip the calendar day. That's the wrong mental model for
a fantasy-league tool where "today" means "the day on the user's
wall clock".

These helpers return the current time / date in the user's local
timezone so every consumer gets consistent answers. The timezone is
hardcoded to ``America/New_York`` for now; parameterize via config
when there's ever a multi-user deployment.

Usage::

    from fantasy_baseball.utils.time_utils import local_now, local_today

    stamp = local_now().strftime("%Y-%m-%d %H:%M")
    today = local_today()
    year = local_today().year
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo


# The user is in Eastern time. Using a named zone (not a fixed offset)
# so DST transitions are handled automatically.
LOCAL_TZ = ZoneInfo("America/New_York")


def local_now() -> datetime:
    """Return the current datetime in the user's local timezone.

    Always returns a timezone-aware datetime. Use ``.strftime(...)``
    for display or ``.date()`` for the calendar date.
    """
    return datetime.now(LOCAL_TZ)


def local_today() -> date:
    """Return today's date in the user's local timezone.

    Equivalent to ``local_now().date()``. Prefer this over
    ``date.today()`` in any code that writes user-visible timestamps,
    computes day deltas for SPoE / ownership walks, or derives the
    current season year near a year boundary.
    """
    return local_now().date()
