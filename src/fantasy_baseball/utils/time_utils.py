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

from datetime import date, datetime, timedelta
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


def next_tuesday(ref: date) -> date:
    """Return the next Tuesday strictly after ``ref``.

    The user's Yahoo league locks lineups on Tuesday morning.
    ``fetch_scoring_period`` returns Yahoo's Mon–Sun scoring week,
    so ``end + 1`` lands on Monday — one day before the actual lock.
    This helper computes the Tuesday that comes *after* ``ref``:

        Sun  → next Tue (2 days later)
        Mon  → next Tue (1 day later)
        Tue  → following Tue (7 days later — never returns ``ref``)
        Wed  → next Tue (6 days later)

    If your league ever changes its lock day (or a config option is
    added), update the weekday target here. For now it's hardcoded
    to 1 (Tuesday) because the user's league is the only consumer.
    """
    TUESDAY = 1  # Monday = 0, Tuesday = 1, ..., Sunday = 6
    days_ahead = (TUESDAY - ref.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return ref + timedelta(days=days_ahead)


def compute_effective_date(end_date: str) -> date:
    """Return the next lineup-lock Tuesday strictly after ``end_date``.

    Yahoo's scoring period ends on a Sunday (``end_date``). The user's
    league locks lineups on Tuesday morning, so the effective date for
    fetching post-lock rosters is the next Tuesday strictly after that
    Sunday — ``end_date + 1`` would land on Monday, one day too early.
    """
    return next_tuesday(date.fromisoformat(end_date))


def compute_fraction_remaining(
    season_start: date, season_end: date, today: date
) -> float:
    """Return the fraction of the regular season still ahead of ``today``.

    Used for SD scaling on projected standings (``sqrt`` damps variance
    as the season progresses) and for ROS Monte Carlo weighting.

    Returns 0.0 if the season has not started (season_end == season_start)
    or if ``today`` is on/after ``season_end``. Lower bound only — does
    not clamp the upper bound, matching existing behavior.
    """
    total_days = (season_end - season_start).days
    if total_days <= 0:
        return 0.0
    remaining_days = max(0, (season_end - today).days)
    return remaining_days / total_days
