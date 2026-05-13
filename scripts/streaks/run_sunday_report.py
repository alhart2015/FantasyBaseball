"""CLI: produce a dated Phase 5 Sunday streaks report.

Thin wrapper around :func:`fantasy_baseball.streaks.pipeline.compute_streak_report`.
The shared pipeline handles the DB refresh, refit-or-load model decision,
Yahoo fetch, and report build. This script handles arg parsing, league
config loading, Yahoo authentication, and writing the markdown +
terminal output.

Usage::

    python scripts/streaks/run_sunday_report.py
    python scripts/streaks/run_sunday_report.py --skip-fetch
    python scripts/streaks/run_sunday_report.py --force-refit
    python scripts/streaks/run_sunday_report.py --output-dir /tmp/streaks
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection
from fantasy_baseball.streaks.pipeline import compute_streak_report
from fantasy_baseball.streaks.reports.sunday import render_markdown, render_terminal

logger = logging.getLogger("streaks.run_sunday_report")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Sunday streaks report.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="DuckDB path (default: data/streaks/streaks.duckdb).",
    )
    parser.add_argument(
        "--league-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "league.yaml",
        help="Path to league.yaml.",
    )
    parser.add_argument(
        "--projections-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "projections",
        help="Root of data/projections/ tree (subdirs per season).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "streaks" / "reports",
        help="Directory for the dated markdown report.",
    )
    parser.add_argument(
        "--season-set-train",
        default="2023-2025",
        help="Training season set for refit and label calibration.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip the game-log + Statcast pull (use cached DB data).",
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Refit models even if model_fits is recent. Default is reuse-when-recent (<=14 days).",
    )
    parser.add_argument(
        "--skip-refit",
        action="store_true",
        help="Deprecated. Raises an error instead -- use --force-refit to control refit cadence.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color in the terminal output (always disabled "
        "when output is not a TTY).",
    )
    parser.add_argument(
        "--scoring-season",
        type=int,
        help="Override the scoring season (defaults to config.season_year).",
    )
    args = parser.parse_args(argv)

    # Terminal renderer uses Unicode glyphs (true minus, sigma, em dash). On
    # Windows the default stdout encoding is cp1252, which raises on these.
    # reconfigure() is a no-op on POSIX where stdout is already utf-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config(args.league_config)
    scoring_season = args.scoring_season or config.season_year
    logger.info(
        "Sunday report for league %d (%s) — scoring season %d",
        config.league_id,
        config.team_name,
        scoring_season,
    )

    if args.skip_refit:
        raise SystemExit(
            "--skip-refit was removed in favor of model_fits reuse; "
            "use --force-refit to bypass reuse when needed."
        )

    conn = get_connection(args.db_path)
    try:
        from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session

        session = get_yahoo_session()
        league = get_league(session, config.league_id, config.game_code)

        report = compute_streak_report(
            conn,
            league=league,
            team_name=config.team_name,
            league_id=config.league_id,
            projections_root=args.projections_root,
            scoring_season=scoring_season,
            season_set_train=args.season_set_train,
            window_days=14,
            top_n_fas=10,  # CLI keeps the historical top-10; dashboard uses 50.
            force_refit=args.force_refit,
            skip_fetch=args.skip_fetch,
        )
    finally:
        conn.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{report.report_date.isoformat()}.md"
    out_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info("Wrote %s", out_path)

    use_color = not args.no_color and sys.stdout.isatty()
    print(render_terminal(report, no_color=not use_color))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
