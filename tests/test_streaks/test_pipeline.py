"""Tests for streaks/pipeline.py — refit-or-load decision."""

from __future__ import annotations

from fantasy_baseball.streaks.pipeline import _should_refit


def test_should_refit_true_when_no_fits(seeded_pipeline_conn_no_fits) -> None:
    assert _should_refit(seeded_pipeline_conn_no_fits, max_age_days=14, force=False) is True


def test_should_refit_false_when_recent_fits(seeded_pipeline_conn_with_recent_fits) -> None:
    assert (
        _should_refit(seeded_pipeline_conn_with_recent_fits, max_age_days=14, force=False) is False
    )


def test_should_refit_true_when_stale_fits(seeded_pipeline_conn_with_old_fits) -> None:
    assert _should_refit(seeded_pipeline_conn_with_old_fits, max_age_days=14, force=False) is True


def test_should_refit_true_when_forced_even_if_recent(
    seeded_pipeline_conn_with_recent_fits,
) -> None:
    assert _should_refit(seeded_pipeline_conn_with_recent_fits, max_age_days=14, force=True) is True
