"""Tests for the streak report orchestrator (the /streaks page's payload)."""

from __future__ import annotations

from datetime import date, timedelta

from fantasy_baseball.streaks.inference import (
    Driver,
    PlayerCategoryScore,
)
from fantasy_baseball.streaks.reports.sunday import (
    ReportRow,
    YahooHitter,
    _composite,
    _max_probability,
    _row_from_scores,
    resolve_hitters,
)


def _score(
    *,
    player_id: int,
    category,
    label="neutral",
    probability: float | None = None,
    drivers: tuple[Driver, ...] = (),
    window_end: date | None = date(2026, 5, 10),
) -> PlayerCategoryScore:
    return PlayerCategoryScore(
        player_id=player_id,
        category=category,
        label=label,
        probability=probability,
        drivers=drivers,
        window_end=window_end,
    )


def _row(
    *,
    name: str,
    positions: tuple[str, ...],
    player_id: int,
    scores: dict,
) -> ReportRow:
    score_list = list(scores.values())
    return ReportRow(
        name=name,
        positions=positions,
        player_id=player_id,
        composite=_composite(score_list),
        scores=scores,
        max_probability=_max_probability(score_list),
    )


def test_composite_counts_hot_minus_cold() -> None:
    scores = [
        _score(player_id=1, category="hr", label="hot", probability=0.71),
        _score(player_id=1, category="r", label="hot", probability=0.65),
        _score(player_id=1, category="rbi", label="cold", probability=0.55),
        _score(player_id=1, category="sb", label="neutral"),
        _score(player_id=1, category="avg", label="neutral"),
    ]
    assert _composite(scores) == 1


def test_resolve_hitters_drops_pitchers() -> None:
    hitters = [
        YahooHitter(name="Aaron Judge", positions=("OF",), yahoo_id="1", status=""),
        YahooHitter(name="Tarik Skubal", positions=("SP",), yahoo_id="2", status=""),
    ]
    name_to_mlbam = {"aaron judge": 592450, "tarik skubal": 669373}
    resolved, unresolved = resolve_hitters(hitters, name_to_mlbam)
    assert len(resolved) == 1
    assert resolved[0][1] == 592450
    assert unresolved == []


def test_resolve_hitters_collects_unresolved_names() -> None:
    hitters = [
        YahooHitter(name="Aaron Judge", positions=("OF",), yahoo_id="1", status=""),
        YahooHitter(name="Unknown Prospect", positions=("OF",), yahoo_id="2", status=""),
    ]
    name_to_mlbam = {"aaron judge": 592450}
    resolved, unresolved = resolve_hitters(hitters, name_to_mlbam)
    assert [h.name for h, _ in resolved] == ["Aaron Judge"]
    assert unresolved == ["Unknown Prospect"]


def _hart_scores(player_id: int) -> dict:
    return {
        "hr": _score(player_id=player_id, category="hr", label="hot", probability=0.71),
        "r": _score(player_id=player_id, category="r", label="hot", probability=0.68),
        "rbi": _score(
            player_id=player_id,
            category="rbi",
            label="hot",
            probability=0.74,
            drivers=(
                Driver(feature="xwoba_avg", z_score=1.8),
                Driver(feature="ev_avg", z_score=1.2),
            ),
        ),
        "sb": _score(player_id=player_id, category="sb", label="neutral"),
        "avg": _score(player_id=player_id, category="avg", label="hot", probability=0.62),
    }


def _vlad_scores(player_id: int) -> dict:
    return {
        "hr": _score(player_id=player_id, category="hr", label="neutral"),
        "r": _score(player_id=player_id, category="r", label="neutral"),
        "rbi": _score(
            player_id=player_id,
            category="rbi",
            label="cold",
            probability=0.78,
            drivers=(Driver(feature="k_pct", z_score=1.6), Driver(feature="babip", z_score=-1.3)),
        ),
        "sb": _score(player_id=player_id, category="sb", label="neutral"),
        "avg": _score(player_id=player_id, category="avg", label="cold", probability=0.71),
    }


def test_build_report_end_to_end_against_seeded_db(seeded_fitted_conn) -> None:
    """build_report wires inference + Yahoo data + rendering together.

    Uses the shared seeded+fitted DB and synthetic Yahoo hitters with
    mlbam_ids that match the fixture player_ids. Asserts that the resulting
    Report contains the expected sections. This test exercises report
    *wiring*, not the fit math, so it reconstructs models from the shared fit
    via ``load_models_from_fits`` rather than refitting (the real end-to-end
    refit path stays covered by ``test_compute_streak_report_end_to_end``).
    """
    from fantasy_baseball.streaks.inference import load_models_from_fits
    from fantasy_baseball.streaks.reports.sunday import build_report

    conn = seeded_fitted_conn
    models = load_models_from_fits(conn)

    # Hitter player_ids in the fixture are 1..16. Yahoo names are
    # P1..P16 (matching the fixture's HitterGame.name). Build a
    # synthetic name->mlbam map.
    name_to_mlbam = {f"p{i}": i for i in range(1, 17)}
    roster = [
        YahooHitter(name="P2", positions=("OF",), yahoo_id="2", status=""),
        YahooHitter(name="P4", positions=("1B",), yahoo_id="4", status=""),
        # Pitcher -- should be filtered.
        YahooHitter(name="Tarik Skubal", positions=("SP",), yahoo_id="99", status=""),
    ]
    fas = [
        YahooHitter(name="P6", positions=("2B",), yahoo_id="6", status=""),
        YahooHitter(name="P8", positions=("3B",), yahoo_id="8", status=""),
    ]
    # Score the latest window in the fixture.
    latest_end = conn.execute(
        "SELECT MAX(window_end) FROM hitter_windows WHERE window_days = 14"
    ).fetchone()[0]
    today = latest_end if isinstance(latest_end, date) else date.fromisoformat(str(latest_end))

    report = build_report(
        conn,
        league_config_team_name="Hart of the Order",
        league_config_league_id=5652,
        models=models,
        roster_hitters=roster,
        fa_hitters=fas,
        name_to_mlbam=name_to_mlbam,
        today=today,
        season_set_train="2023-2024",
        scoring_season=2024,
        window_days=14,
        top_n_fas=10,
    )
    # Roster rows only contain hitters -- no pitcher snuck in.
    assert all("Tarik" not in r.name for r in report.roster_rows)
    assert len(report.roster_rows) == 2
    # FA rows zero composite are dropped -- assert <= len(fas).
    assert len(report.fa_rows) <= len(fas)
    # window_end reflects the seeded data.
    assert report.window_end is not None
    # The whole report round-trips through the /streaks page's cache payload.
    from fantasy_baseball.streaks.dashboard import deserialize_report, serialize_report

    assert deserialize_report(serialize_report(report)) == report


def test_build_report_end_to_end_keeps_and_annotates_inactive_roster(
    seeded_fitted_conn,
) -> None:
    """Stale roster players are kept, annotated with days-since, and neutralized.

    Same setup as ``test_build_report_end_to_end_against_seeded_db`` but with
    ``today`` pushed 30 days past the latest seeded window_end so every seeded
    player is inactive (>4 days stale). The report must retain the roster rows
    (belt-and-suspenders: inactive players are annotated, not dropped), stamp
    ``days_since_last_game`` on each, and force the composite to neutral.
    """
    from fantasy_baseball.streaks.inference import load_models_from_fits
    from fantasy_baseball.streaks.reports.sunday import build_report

    conn = seeded_fitted_conn
    models = load_models_from_fits(conn)

    name_to_mlbam = {f"p{i}": i for i in range(1, 17)}
    roster = [
        YahooHitter(name="P2", positions=("OF",), yahoo_id="2", status=""),
        YahooHitter(name="P4", positions=("1B",), yahoo_id="4", status=""),
        # Pitcher -- should be filtered.
        YahooHitter(name="Tarik Skubal", positions=("SP",), yahoo_id="99", status=""),
    ]
    fas = [
        YahooHitter(name="P6", positions=("2B",), yahoo_id="6", status=""),
        YahooHitter(name="P8", positions=("3B",), yahoo_id="8", status=""),
    ]
    latest_end = conn.execute(
        "SELECT MAX(window_end) FROM hitter_windows WHERE window_days = 14"
    ).fetchone()[0]
    latest_end = latest_end if isinstance(latest_end, date) else date.fromisoformat(str(latest_end))
    # Push today 30 days past the latest window so every player is stale.
    today = latest_end + timedelta(days=30)

    report = build_report(
        conn,
        league_config_team_name="Hart of the Order",
        league_config_league_id=5652,
        models=models,
        roster_hitters=roster,
        fa_hitters=fas,
        name_to_mlbam=name_to_mlbam,
        today=today,
        season_set_train="2023-2024",
        scoring_season=2024,
        window_days=14,
        top_n_fas=10,
    )
    # Inactive roster players are kept (not dropped), same count as the active test.
    assert len(report.roster_rows) == 2
    # Each row is annotated with the exact staleness (all share the latest window_end).
    assert all(r.days_since_last_game == 30 for r in report.roster_rows)
    # Each row is forced neutral.
    assert all(r.composite == 0 for r in report.roster_rows)


def _neutral_scores(window_end: date) -> list[PlayerCategoryScore]:
    return [
        _score(player_id=1, category=cat, window_end=window_end)
        for cat in ("hr", "r", "rbi", "sb", "avg")
    ]


def test_row_from_scores_marks_inactive_when_window_is_stale() -> None:
    window_end = date(2026, 6, 1)
    today = window_end + timedelta(days=10)  # 10 > 4
    row = _row_from_scores(
        name="Oneil Cruz",
        positions=("SS",),
        player_id=1,
        scores=_neutral_scores(window_end),
        today=today,
    )
    assert row.days_since_last_game == 10


def test_row_from_scores_active_when_window_is_recent() -> None:
    window_end = date(2026, 6, 1)
    today = window_end + timedelta(days=2)  # 2 <= 4
    row = _row_from_scores(
        name="Active Guy",
        positions=("OF",),
        player_id=1,
        scores=_neutral_scores(window_end),
        today=today,
    )
    assert row.days_since_last_game is None
