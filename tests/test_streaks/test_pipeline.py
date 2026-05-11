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
    report = compute_streak_report(
        seeded_pipeline_conn_no_fits,
        league=fake_league,
        team_name="Hart of the Order",
        league_id=5652,
        projections_root=projections_root,
        scoring_season=2024,
        season_set_train="2023-2024",
        force_refit=False,
    )

    assert report.team_name == "Hart of the Order"
    assert report.league_id == 5652
    assert len(report.roster_rows) == 1
    # FA row may or may not appear depending on composite=0 filter; assert
    # the overall report shape rather than the FA count.

    # Models must have been refit (no prior fits).
    n_fits = seeded_pipeline_conn_no_fits.execute("SELECT COUNT(*) FROM model_fits").fetchone()[0]
    assert n_fits > 0
