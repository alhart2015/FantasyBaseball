"""Tests for streaks/pipeline.py — refit-or-load decision."""

from __future__ import annotations

from pathlib import Path

from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.pipeline import _refresh_streaks_db, _should_refit


def test_should_refit_true_when_no_fits() -> None:
    # Empty model_fits -> refit. No pipeline seed needed: _should_refit reads
    # only MAX(fit_timestamp), so a bare connection exercises the decision.
    conn = get_connection(":memory:")
    assert _should_refit(conn, max_age_days=14, force=False) is True
    conn.close()


def test_should_refit_false_when_recent_fits(conn_with_recent_fit) -> None:
    assert _should_refit(conn_with_recent_fit, max_age_days=14, force=False) is False


def test_should_refit_true_when_stale_fits(conn_with_old_fit) -> None:
    assert _should_refit(conn_with_old_fit, max_age_days=14, force=False) is True


def test_should_refit_true_when_forced_even_if_recent(
    conn_with_recent_fit,
) -> None:
    assert _should_refit(conn_with_recent_fit, max_age_days=14, force=True) is True


def test_refresh_streaks_db_passes_incremental_to_fetch_season(
    seeded_pipeline_conn_no_fits, monkeypatch, tmp_path
) -> None:
    """The in-flight pipeline must fetch incrementally; otherwise top players
    freeze at the date of their first ingestion (see fetch_history bug fix)."""
    captured: dict[str, object] = {}

    def _capture(*, season, conn, **kw):
        captured["season"] = season
        captured.update(kw)
        return {"season": season, "stub": True}

    monkeypatch.setattr("fantasy_baseball.streaks.pipeline.fetch_season", _capture)

    # Minimal projection CSV so load_projection_rates_for_seasons doesn't blow up.
    projections_root: Path = tmp_path / "projections"
    season_dir = projections_root / "2024"
    season_dir.mkdir(parents=True)
    (season_dir / "steamer-hitters.csv").write_text(
        "Name,MLBAMID,PA,HR,R,RBI,SB,AVG\nA,2,600,30,90,108,24,0.275\n"
    )

    _refresh_streaks_db(
        seeded_pipeline_conn_no_fits,
        season=2024,
        season_set_train="2023-2024",
        projections_root=projections_root,
        skip_fetch=False,
    )

    assert captured.get("incremental") is True


def test_compute_streak_report_end_to_end(
    seeded_pipeline_conn_no_fits, monkeypatch, tmp_path
) -> None:
    """End-to-end with seeded DB + stubbed Yahoo fetch.

    Verifies the returned Report has the expected shape (one roster
    row + one FA row) and that models were refit (model_fits populated).
    """
    from fantasy_baseball.streaks.pipeline import compute_streak_report
    from fantasy_baseball.streaks.reports.sunday import YahooHitter

    # Project two hitters via a synthetic CSV so the name->mlbam map
    # resolves both stubbed Yahoo names. MLBAMIDs 2 and 4 are chosen
    # because the seeded fixture uses pid%2==0 to flag high-rate
    # players, and the CSV's HR/PA (30/600=0.05) matches the fixture's
    # high-rate hr_per_pa exactly — so the projection-rate upsert in
    # `_refresh_streaks_db` doesn't perturb the sparse `hr above`
    # training distribution enough to leave a CV fold single-class.
    projections_root = tmp_path / "projections"
    season_dir = projections_root / "2024"
    season_dir.mkdir(parents=True)
    (season_dir / "steamer-hitters.csv").write_text(
        "Name,MLBAMID,PA,HR,R,RBI,SB,AVG\n"
        "Roster Guy,2,600,30,90,108,24,0.275\n"
        "FA Guy,4,600,30,90,108,24,0.275\n"
    )

    def _fake_fetch_yahoo(league, *, team_name):
        return (
            [YahooHitter(name="Roster Guy", positions=("OF",), yahoo_id="2", status="")],
            [YahooHitter(name="FA Guy", positions=("OF",), yahoo_id="4", status="")],
        )

    monkeypatch.setattr("fantasy_baseball.streaks.pipeline._fetch_yahoo_hitters", _fake_fetch_yahoo)

    # Stub fetch_season — seeded DB already has data.
    monkeypatch.setattr(
        "fantasy_baseball.streaks.pipeline.fetch_season",
        lambda *, season, conn, **kw: {"season": season, "stub": True},
    )

    fake_league = object()
    # Anchor "today" just past the fixture's last window so the 4-day
    # staleness guard doesn't force every row neutral (which would strip
    # probabilities and make the assertion below vacuous).
    from datetime import timedelta

    last_window_end = seeded_pipeline_conn_no_fits.execute(
        "SELECT MAX(window_end) FROM hitter_windows"
    ).fetchone()[0]
    report = compute_streak_report(
        seeded_pipeline_conn_no_fits,
        league=fake_league,
        team_name="Hart of the Order",
        league_id=5652,
        projections_root=projections_root,
        scoring_season=2024,
        season_set_train="2023-2024",
        force_refit=False,
        today=last_window_end + timedelta(days=1),
    )

    assert report.team_name == "Hart of the Order"
    assert report.league_id == 5652
    assert len(report.roster_rows) == 1
    # FA row may or may not appear depending on composite=0 filter; assert
    # the overall report shape rather than the FA count.

    # Models must have been refit (no prior fits).
    n_fits = seeded_pipeline_conn_no_fits.execute("SELECT COUNT(*) FROM model_fits").fetchone()[0]
    assert n_fits > 0

    # At least one score must carry a live probability. A report where every
    # probability is None (models silently missing) shipped to production
    # once -- labels still light up, so only an end-to-end assertion on the
    # actual number catches it.
    all_scores = [
        score for row in (*report.roster_rows, *report.fa_rows) for score in row.scores.values()
    ]
    assert any(s.probability is not None for s in all_scores), (
        "no score carried a probability -- models fit but scoring never used them"
    )


def test_compute_streak_report_raises_when_zero_models_fit(
    seeded_pipeline_conn_no_fits, monkeypatch, tmp_path
) -> None:
    """Zero fitted models means every probability would be None -- the exact
    silent degradation that shipped once. The pipeline must raise (so the
    refresh keeps the previous good cache) instead of scoring model-less."""
    import pytest

    from fantasy_baseball.streaks import pipeline as pl

    monkeypatch.setattr(
        "fantasy_baseball.streaks.pipeline.fetch_season",
        lambda *, season, conn, **kw: {"season": season, "stub": True},
    )
    monkeypatch.setattr(
        "fantasy_baseball.streaks.pipeline.refit_models_for_report",
        lambda conn, **kw: {},
    )

    def _fail_if_reached(league, *, team_name):
        raise AssertionError("must raise before the Yahoo fetch")

    monkeypatch.setattr("fantasy_baseball.streaks.pipeline._fetch_yahoo_hitters", _fail_if_reached)

    projections_root = tmp_path / "projections"
    (projections_root / "2024").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="0 models"):
        pl.compute_streak_report(
            seeded_pipeline_conn_no_fits,
            league=object(),
            team_name="Hart of the Order",
            league_id=5652,
            projections_root=projections_root,
            scoring_season=2024,
            season_set_train="2023-2024",
            force_refit=True,
        )
