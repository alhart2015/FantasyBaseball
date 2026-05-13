"""Shared fixtures for streaks tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fantasy_baseball.streaks.data.schema import get_connection
from tests.test_streaks.test_predictors import _seed_pipeline


@pytest.fixture
def seeded_pipeline_conn_no_fits(tmp_path):
    """Connection with windows/thresholds/labels seeded for 2023+2024 but no model_fits."""
    db = tmp_path / "s.duckdb"
    conn = get_connection(db)
    _seed_pipeline(conn, season=2023)
    _seed_pipeline(conn, season=2024)
    yield conn
    conn.close()


def _stamp_fits(conn, *, age_days: int) -> None:
    """Backdate every row in model_fits to (now - age_days)."""
    ts = datetime.now(UTC) - timedelta(days=age_days)
    conn.execute("UPDATE model_fits SET fit_timestamp = ?", [ts])


@pytest.fixture
def seeded_pipeline_conn_with_recent_fits(seeded_pipeline_conn_no_fits):
    """Connection with model_fits aged 1 day (fresh)."""
    from fantasy_baseball.streaks.inference import refit_models_for_report

    refit_models_for_report(
        seeded_pipeline_conn_no_fits, season_set_train="2023-2024", window_days=14
    )
    _stamp_fits(seeded_pipeline_conn_no_fits, age_days=1)
    return seeded_pipeline_conn_no_fits


@pytest.fixture
def seeded_pipeline_conn_with_old_fits(seeded_pipeline_conn_no_fits):
    """Connection with model_fits aged 30 days (stale)."""
    from fantasy_baseball.streaks.inference import refit_models_for_report

    refit_models_for_report(
        seeded_pipeline_conn_no_fits, season_set_train="2023-2024", window_days=14
    )
    _stamp_fits(seeded_pipeline_conn_no_fits, age_days=30)
    return seeded_pipeline_conn_no_fits
