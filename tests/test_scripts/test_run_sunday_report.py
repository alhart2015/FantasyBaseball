"""End-to-end smoke test for ``scripts/streaks/run_sunday_report.py``.

Avoids the real Yahoo + Statcast pulls by mocking the fetch surfaces.
The DB is the same in-memory seeded fixture as ``test_inference.py``,
which exercises the inference + render code paths through the script
entry point.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "streaks"))


from tests.test_streaks.test_predictors import _seed_pipeline

# `run_sunday_report` module is loaded lazily inside the test so the
# sys.path injection above is in effect when the import runs.


def test_main_writes_report_against_seeded_db(tmp_path, monkeypatch) -> None:
    """End-to-end: invoke main() with mocked Yahoo + fetch_season.

    Asserts that the markdown file is written and contains the expected
    section headers.
    """
    import run_sunday_report  # type: ignore[import-not-found]

    from fantasy_baseball.streaks.data.schema import get_connection

    # Build a seeded DuckDB at a real on-disk path so the script can
    # open it via get_connection().
    db_path = tmp_path / "streaks.duckdb"
    conn = get_connection(db_path)
    try:
        _seed_pipeline(conn, season=2023)
        _seed_pipeline(conn, season=2024)
    finally:
        conn.close()

    # Use only one of the 2024-shaped projection CSVs to keep this test
    # fast; we synthesize the per-system CSV from the fixture player_ids.
    projections_root = tmp_path / "projections"
    season_dir = projections_root / "2024"
    season_dir.mkdir(parents=True)
    csv_text = "Name,MLBAMID,PA,HR,R,RBI,SB,AVG\n"
    for pid in range(1, 17):
        csv_text += f"P{pid},{pid},600,20,80,80,10,0.270\n"
    (season_dir / "steamer-hitters.csv").write_text(csv_text)

    league_yaml = tmp_path / "league.yaml"
    league_yaml.write_text(
        """
league:
  id: 5652
  num_teams: 10
  game_code: mlb
  team_name: "Hart of the Order"
  season_year: 2024
""".strip()
    )

    # Stub fetch_season — the seeded DB already has all the data we need.
    def _fake_fetch_season(*, season, conn, **kwargs):
        return {"season": season, "stub": True}

    # Stub yahoo auth + roster/FA fetch.
    fake_league = object()  # opaque sentinel — auth path never touches it.

    def _fake_get_yahoo_session(*_a, **_kw):
        return object()

    def _fake_get_league(*_a, **_kw):
        return fake_league

    def _fake_fetch_yahoo_data(league, *, team_name):
        from fantasy_baseball.streaks.reports.sunday import YahooHitter

        roster = [
            YahooHitter(name="P2", positions=("OF",), yahoo_id="2", status=""),
            YahooHitter(name="P4", positions=("1B",), yahoo_id="4", status=""),
        ]
        fas = [
            YahooHitter(name="P6", positions=("2B",), yahoo_id="6", status=""),
            YahooHitter(name="P8", positions=("3B",), yahoo_id="8", status=""),
        ]
        return roster, fas

    output_dir = tmp_path / "reports"
    with (
        patch("fantasy_baseball.streaks.pipeline.fetch_season", _fake_fetch_season),
        patch("fantasy_baseball.auth.yahoo_auth.get_yahoo_session", _fake_get_yahoo_session),
        patch("fantasy_baseball.auth.yahoo_auth.get_league", _fake_get_league),
        patch(
            "fantasy_baseball.streaks.pipeline._fetch_yahoo_hitters",
            _fake_fetch_yahoo_data,
        ),
        patch("fantasy_baseball.streaks.pipeline.local_today", lambda: date(2024, 6, 30)),
    ):
        exit_code = run_sunday_report.main(
            [
                "--db-path",
                str(db_path),
                "--league-config",
                str(league_yaml),
                "--projections-root",
                str(projections_root),
                "--output-dir",
                str(output_dir),
                "--season-set-train",
                "2023-2024",
                "--no-color",
                "--scoring-season",
                "2024",
            ]
        )
    assert exit_code == 0
    md_files = list(output_dir.glob("*.md"))
    assert len(md_files) == 1
    md_text = md_files[0].read_text(encoding="utf-8")
    assert "# Streaks — Sunday Report" in md_text
    assert "Hart of the Order" in md_text
    assert "## Your Roster" in md_text
    assert "## Top 10 Free Agent Signals" in md_text


def test_skip_refit_flag_errors_out() -> None:
    """The spec lists --skip-refit but it isn't wired yet — the CLI must
    fail loudly rather than silently doing nothing."""
    import run_sunday_report  # type: ignore[import-not-found]

    with pytest.raises(SystemExit) as exc_info:
        run_sunday_report.main(["--skip-refit", "--skip-fetch"])
    # The error message should mention --skip-refit.
    assert "skip-refit" in str(exc_info.value).lower() or exc_info.value.code != 0
