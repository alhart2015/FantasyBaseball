"""End-to-end orchestration for the hot-streaks pipeline.

Wraps the DB-refresh sequence (fetch logs/statcast, upsert projection
rates, recompute windows/thresholds/labels), the refit-or-load model
decision, the Yahoo fetch, and ``build_report`` into a single function
called by both the Sunday CLI and the dashboard refresh pipeline.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import duckdb

logger = logging.getLogger("streaks.pipeline")


_DEFAULT_MAX_FIT_AGE_DAYS = 14


def _should_refit(conn: duckdb.DuckDBPyConnection, *, max_age_days: int, force: bool) -> bool:
    """Return True iff models should be refit rather than loaded.

    True when ``force`` is set, when ``model_fits`` is empty, or when
    the most recent ``fit_timestamp`` is older than ``max_age_days``.
    """
    if force:
        return True
    row = conn.execute("SELECT MAX(fit_timestamp) FROM model_fits").fetchone()
    if row is None or row[0] is None:
        return True
    most_recent: datetime = row[0]
    if most_recent.tzinfo is None:
        most_recent = most_recent.replace(tzinfo=UTC)
    cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
    return bool(most_recent < cutoff)
