"""Shared fixtures for streaks tests."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from fantasy_baseball.streaks.data.load_model_fits import upsert_model_fits
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.models import ModelFit
from tests.test_streaks.test_predictors import _seed_pipeline


def _seed_two_seasons(conn) -> None:
    """Seed 2023 + 2024 so ``refit_models_for_report`` has a 2-season train set.

    The shared definition of "a two-season train seed" -- both the session
    seeded+fitted fixture and ``test_inference``'s fit tests use it, so the
    fixture's "byte-identical to a fresh fit" claim can't quietly diverge.
    """
    _seed_pipeline(conn, season=2023)
    _seed_pipeline(conn, season=2024)


@pytest.fixture
def seeded_pipeline_conn_no_fits(tmp_path):
    """Connection with windows/thresholds/labels seeded for 2023+2024 but no model_fits."""
    db = tmp_path / "s.duckdb"
    conn = get_connection(db)
    _seed_two_seasons(conn)
    yield conn
    conn.close()


def _synthetic_model_fit(*, age_days: int) -> ModelFit:
    """A single minimal ``model_fits`` row aged ``age_days``, for ``_should_refit``.

    ``_should_refit`` only reads ``MAX(fit_timestamp)`` -- it never inspects the
    coefficients -- so the staleness tests need one timestamped row to *exist*,
    not a real fit. Building it directly avoids the ~5s ``refit_models_for_report``
    (8-model GroupKFold grid search) plus the two-season ``_seed_pipeline`` that
    these tests used to pay for nothing. The Phase-B pipeline fields default to
    NULL, which is fine: nothing here reconstructs the pipeline.
    """
    return ModelFit(
        model_id="r_hot_2023-2024",
        category="r",
        direction="above",
        season_set="2023-2024",
        window_days=14,
        cold_method="empirical",
        chosen_C=1.0,
        cv_auc_mean=0.5,
        cv_auc_std=0.0,
        val_auc=0.5,
        n_train_rows=0,
        n_val_rows=0,
        fit_timestamp=datetime.now(UTC) - timedelta(days=age_days),
    )


def _conn_with_fit_aged(*, age_days: int) -> duckdb.DuckDBPyConnection:
    """In-memory connection holding one ``model_fits`` row aged ``age_days``."""
    conn = get_connection(":memory:")
    upsert_model_fits(conn, [_synthetic_model_fit(age_days=age_days)])
    return conn


@pytest.fixture
def conn_with_recent_fit():
    """Connection whose lone model_fits row is aged 1 day (fresh).

    Standalone (no pipeline seed, no refit): the staleness decision reads only
    the fit timestamp. See :func:`_synthetic_model_fit`.
    """
    conn = _conn_with_fit_aged(age_days=1)
    yield conn
    conn.close()


@pytest.fixture
def conn_with_old_fit():
    """Connection whose lone model_fits row is aged 30 days (stale).

    Standalone (no pipeline seed, no refit). See :func:`_synthetic_model_fit`.
    """
    conn = _conn_with_fit_aged(age_days=30)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def _seeded_fitted_db_path(tmp_path_factory):
    """Build the 2023+2024 seed + one real refit ONCE for the whole session.

    The expensive thing cached here is the 8-model GroupKFold refit (~1.2s)
    plus the seed; both are paid a single time, then handed to every plumbing
    test as an independent file copy (see :func:`seeded_fitted_conn`). Tests
    that exercise *scoring/report wiring* rather than the fit math reconstruct
    an identical models dict from these rows via ``load_models_from_fits`` --
    the round-trip test proves the reconstruction predicts byte-identically to
    a fresh fit, so the plumbing tests lose no fidelity by skipping their own
    refit.
    """
    from fantasy_baseball.streaks.inference import refit_models_for_report

    path = tmp_path_factory.mktemp("streaks_seed") / "seeded_fitted.duckdb"
    conn = get_connection(path)
    _seed_two_seasons(conn)
    refit_models_for_report(conn, season_set_train="2023-2024", window_days=14)
    conn.close()  # release the file lock so copies are clean
    return path


@pytest.fixture
def seeded_fitted_conn(_seeded_fitted_db_path, tmp_path):
    """A fresh, independently-mutable copy of the session seeded+fitted DB.

    Copying the single DuckDB file is milliseconds, so each test still gets an
    isolated database it can DELETE from without disturbing siblings -- without
    re-paying the seed or the fit.
    """
    dst = tmp_path / "seeded_fitted_copy.duckdb"
    shutil.copy(_seeded_fitted_db_path, dst)
    conn = get_connection(dst)
    yield conn
    conn.close()
