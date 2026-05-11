"""Tests for the Phase 5 Sunday report renderer + orchestrator."""

from __future__ import annotations

from datetime import date

from fantasy_baseball.streaks.inference import (
    Driver,
    PlayerCategoryScore,
)
from fantasy_baseball.streaks.reports.sunday import (
    DriverLine,
    Report,
    ReportRow,
    YahooHitter,
    _composite,
    _format_cell,
    _max_probability,
    _signed,
    render_markdown,
    render_terminal,
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


def test_signed_uses_true_minus_for_negative() -> None:
    assert _signed(3) == "+3"
    assert _signed(0) == "0"
    assert _signed(-2) == "−2"  # true Unicode minus


def test_format_cell_neutral_renders_dash() -> None:
    s = _score(player_id=1, category="hr", label="neutral")
    assert _format_cell(s, sparse=True) == "—"


def test_format_cell_hot_with_probability() -> None:
    s = _score(player_id=1, category="r", label="hot", probability=0.715)
    assert _format_cell(s, sparse=False) == "hot 0.71"


def test_format_cell_sparse_cold_without_model_shows_dash_for_probability() -> None:
    """HR/SB cold has no model in Phase 4; cell should render ``cold —``."""
    s = _score(player_id=1, category="hr", label="cold", probability=None)
    assert _format_cell(s, sparse=True) == "cold —"


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


def _build_sample_report() -> Report:
    judge = _row(
        name="Aaron Judge",
        positions=("OF",),
        player_id=592450,
        scores=_hart_scores(592450),
    )
    vlad = _row(
        name="Vlad Jr",
        positions=("1B",),
        player_id=665489,
        scores=_vlad_scores(665489),
    )
    polanco_scores = {
        "hr": _score(player_id=1, category="hr", label="neutral"),
        "r": _score(player_id=1, category="r", label="hot", probability=0.72),
        "rbi": _score(player_id=1, category="rbi", label="hot", probability=0.69),
        "sb": _score(player_id=1, category="sb", label="neutral"),
        "avg": _score(player_id=1, category="avg", label="hot", probability=0.65),
    }
    polanco = _row(
        name="Jorge Polanco",
        positions=("2B",),
        player_id=1,
        scores=polanco_scores,
    )
    driver_lines = (
        DriverLine(
            player_name="Aaron Judge",
            category="rbi",
            label="hot",
            probability=0.74,
            drivers=(
                Driver(feature="xwoba_avg", z_score=1.8),
                Driver(feature="ev_avg", z_score=1.2),
            ),
        ),
        DriverLine(
            player_name="Vlad Jr",
            category="rbi",
            label="cold",
            probability=0.78,
            drivers=(Driver(feature="k_pct", z_score=1.6), Driver(feature="babip", z_score=-1.3)),
        ),
    )
    return Report(
        report_date=date(2026, 5, 11),
        window_end=date(2026, 5, 10),
        team_name="Hart of the Order",
        league_id=5652,
        season_set_train="2023-2025",
        roster_rows=(judge, vlad),
        fa_rows=(polanco,),
        driver_lines=driver_lines,
        skipped=(),
    )


def test_render_markdown_contains_expected_sections() -> None:
    md = render_markdown(_build_sample_report())
    assert "# Streaks — Sunday Report — 2026-05-11" in md
    assert "## Your Roster — Hart of the Order (League 5652)" in md
    assert "## Top 10 Free Agent Signals" in md
    assert "## Drivers (top peripheral signal per active prediction)" in md
    # Roster columns present.
    assert "HR" in md and "R" in md and "RBI" in md and "SB" in md and "AVG" in md
    # Hot probability rendered correctly.
    assert "hot 0.71" in md
    assert "cold 0.78" in md
    # Driver lines.
    assert "xwoba_avg +1.8σ" in md
    assert "k_pct +1.6σ" in md
    assert "babip −1.3σ" in md


def test_render_markdown_snapshot_matches_golden() -> None:
    md = render_markdown(_build_sample_report())
    golden = """# Streaks — Sunday Report — 2026-05-11
*Models refit on 2023-2025; data through 2026-05-10*

## Your Roster — Hart of the Order (League 5652)
Sorted by composite (#hot − #cold), tiebreak max continuation probability.

| Player | Pos | Comp | HR | R | RBI | SB | AVG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Aaron Judge | OF | +4 | hot 0.71 | hot 0.68 | hot 0.74 | — | hot 0.62 |
| Vlad Jr | 1B | −2 | — | — | cold 0.78 | — | cold 0.71 |

## Top 10 Free Agent Signals
Top 10 available hitters by |composite|, non-neutral cats only.

| Player | Pos | Comp | Active Streaks |
| --- | --- | --- | --- |
| Jorge Polanco | 2B | +3 | hot 0.72 R, hot 0.69 RBI, hot 0.65 AVG |

## Drivers (top peripheral signal per active prediction)

**Aaron Judge — RBI hot 0.74**  →  xwoba_avg +1.8σ, ev_avg +1.2σ
**Vlad Jr — RBI cold 0.78**  →  k_pct +1.6σ, babip −1.3σ
"""
    assert md.strip() == golden.strip()


def test_render_terminal_contains_expected_content() -> None:
    term = render_terminal(_build_sample_report(), no_color=True)
    # Header.
    assert "Streaks — Sunday Report — 2026-05-11" in term
    # Roster + FA section headings.
    assert "Your Roster — Hart of the Order" in term
    assert "Top 10 Free Agent Signals" in term
    # Probabilities rendered.
    assert "hot 0.71" in term
    assert "cold 0.78" in term
    # Drivers.
    assert "Aaron Judge — RBI hot 0.74" in term


def test_render_terminal_no_color_strips_ansi() -> None:
    term = render_terminal(_build_sample_report(), no_color=True)
    assert "\033[" not in term


def test_render_terminal_with_color_inserts_ansi() -> None:
    term = render_terminal(_build_sample_report(), no_color=False)
    assert "\033[32m" in term  # green for hot
    assert "\033[31m" in term  # red for cold


def test_report_with_empty_roster_renders_placeholder() -> None:
    report = Report(
        report_date=date(2026, 5, 11),
        window_end=None,
        team_name="Hart of the Order",
        league_id=5652,
        season_set_train="2023-2025",
        roster_rows=(),
        fa_rows=(),
        driver_lines=(),
        skipped=(),
    )
    md = render_markdown(report)
    assert "_No roster hitters resolved._" in md
    assert "_No free-agent signals._" in md


def test_build_report_end_to_end_against_seeded_db() -> None:
    """build_report wires inference + Yahoo data + rendering together.

    Uses the same seeded DB as ``test_inference.py`` and synthetic Yahoo
    hitters with mlbam_ids that match the fixture player_ids. Asserts
    that the resulting Report contains the expected sections.
    """
    from fantasy_baseball.streaks.data.schema import get_connection
    from fantasy_baseball.streaks.inference import refit_models_for_report
    from fantasy_baseball.streaks.reports.sunday import build_report
    from tests.test_streaks.test_predictors import _seed_pipeline

    conn = get_connection(":memory:")
    _seed_pipeline(conn, season=2023)
    _seed_pipeline(conn, season=2024)
    models = refit_models_for_report(conn, season_set_train="2023-2024", window_days=14)

    # Hitter player_ids in the fixture are 1..16. Yahoo names are
    # P1..P16 (matching the fixture's HitterGame.name). Build a
    # synthetic name->mlbam map.
    name_to_mlbam = {f"p{i}": i for i in range(1, 17)}
    roster = [
        YahooHitter(name="P2", positions=("OF",), yahoo_id="2", status=""),
        YahooHitter(name="P4", positions=("1B",), yahoo_id="4", status=""),
        # Pitcher — should be filtered.
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
    # Roster rows only contain hitters — no pitcher snuck in.
    assert all("Tarik" not in r.name for r in report.roster_rows)
    assert len(report.roster_rows) == 2
    # FA rows zero composite are dropped — assert <= len(fas).
    assert len(report.fa_rows) <= len(fas)
    # window_end reflects the seeded data.
    assert report.window_end is not None
    # The whole report can be rendered without error.
    md = render_markdown(report)
    assert "Hart of the Order" in md


def test_report_with_skipped_players_renders_footer() -> None:
    report = Report(
        report_date=date(2026, 5, 11),
        window_end=date(2026, 5, 10),
        team_name="Hart of the Order",
        league_id=5652,
        season_set_train="2023-2025",
        roster_rows=(),
        fa_rows=(),
        driver_lines=(),
        skipped=("Prospect McProspect",),
    )
    md = render_markdown(report)
    assert "Skipped 1 player(s)" in md
    assert "Prospect McProspect" in md
