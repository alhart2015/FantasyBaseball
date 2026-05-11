"""CLI: produce a dated Phase 5 Sunday streaks report.

End-to-end orchestrator:

1. Fetch 2026 game logs + Statcast (incremental skip if already loaded).
2. Load 2026 projection rates from CSVs into ``hitter_projection_rates``.
3. Rebuild ``hitter_windows`` from current games + statcast.
4. Recompute ``thresholds`` and re-apply ``hitter_streak_labels`` on the
   ``2023-2025`` calibration set.
5. Refit the 8 Phase 4 models on ``2023-2025``.
6. Pull the Yahoo roster + top FAs across hitter positions in the league
   configured in ``config/league.yaml``.
7. Score current 14d windows + attribute peripheral drivers.
8. Write a dated markdown file under ``data/streaks/reports/`` and
   pretty-print the same content to stdout.

Usage::

    python scripts/streaks/run_sunday_report.py
    python scripts/streaks/run_sunday_report.py --skip-fetch
    python scripts/streaks/run_sunday_report.py --skip-refit
    python scripts/streaks/run_sunday_report.py --output-dir /tmp/streaks
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.lineup.yahoo_roster import (
    fetch_free_agents,
    fetch_roster,
    fetch_teams,
    find_user_team_key,
)
from fantasy_baseball.streaks.data.fetch_history import fetch_season
from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates
from fantasy_baseball.streaks.data.projections import (
    load_projection_rates_for_seasons,
)
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection
from fantasy_baseball.streaks.inference import refit_models_for_report
from fantasy_baseball.streaks.labels import apply_labels
from fantasy_baseball.streaks.reports.sunday import (
    YahooHitter,
    build_name_to_mlbam_map,
    build_report,
    render_markdown,
    render_terminal,
)
from fantasy_baseball.streaks.thresholds import compute_thresholds
from fantasy_baseball.streaks.windows import compute_windows
from fantasy_baseball.utils.time_utils import local_today

logger = logging.getLogger("streaks.run_sunday_report")


# Hitter positions to scan for free agents. Pitcher streaks are out of
# scope for Phase 5.
_HITTER_FA_POSITIONS: tuple[str, ...] = ("C", "1B", "2B", "3B", "SS", "OF", "Util")


def _normalize_position(p: str) -> str:
    """Yahoo returns mixed-case position strings ("Util" vs "UTIL"); the
    streaks report only cares about hitter-vs-pitcher distinction, so
    upper-casing is enough.
    """
    return p.upper()


def _to_yahoo_hitter(entry: dict) -> YahooHitter:
    positions = tuple(_normalize_position(p) for p in entry.get("positions", []))
    return YahooHitter(
        name=entry["name"],
        positions=positions,
        yahoo_id=str(entry.get("player_id", "")),
        status=entry.get("status", "") or "",
    )


def _fetch_yahoo_data(league, *, team_name: str) -> tuple[list[YahooHitter], list[YahooHitter]]:
    """Fetch the user's roster + dedup'd FAs across hitter positions.

    Returns ``(roster_hitters, fa_hitters)``. The FA list is name-dedup'd
    across positions — Yahoo returns the same player under each of their
    eligible positions, so we'd see duplicates without this.
    """
    teams = fetch_teams(league)
    user_team_key = find_user_team_key(teams, team_name)
    roster_raw = fetch_roster(league, user_team_key)
    roster_hitters = [_to_yahoo_hitter(p) for p in roster_raw]

    # Fan out Yahoo FA fetches across positions in parallel — each call
    # is a synchronous HTTP round-trip and they're independent. Same
    # pattern as :class:`RefreshRun` in web/refresh_pipeline.py.
    def _fetch_one(pos: str) -> list[dict]:
        try:
            return fetch_free_agents(league, pos, count=50)
        except Exception:
            logger.exception("Free agent fetch failed at position %s; continuing", pos)
            return []

    seen: set[str] = set()
    fa_hitters: list[YahooHitter] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fa_raw in pool.map(_fetch_one, _HITTER_FA_POSITIONS):
            for fa in fa_raw:
                key = fa["name"].lower().strip()
                if key in seen:
                    continue
                seen.add(key)
                fa_hitters.append(_to_yahoo_hitter(fa))
    return roster_hitters, fa_hitters


def _refresh_streaks_db(
    conn,
    *,
    season: int,
    season_set_train: str,
    projections_root: Path,
    skip_fetch: bool,
) -> None:
    """Steps 1-4 of the pipeline: fetch -> load projections -> windows -> labels."""
    if not skip_fetch:
        logger.info("Fetching %d game logs + Statcast (incremental)...", season)
        summary = fetch_season(season=season, conn=conn)
        logger.info("fetch_season summary: %s", summary)
    else:
        logger.info("--skip-fetch set; using cached game logs + Statcast")

    logger.info("Loading %d projection rates from %s...", season, projections_root)
    rates = load_projection_rates_for_seasons(projections_root, [season])
    upsert_projection_rates(conn, rates)

    logger.info("Recomputing hitter_windows...")
    n_windows = compute_windows(conn)
    logger.info("  wrote %d window rows", n_windows)

    logger.info("Recomputing thresholds and labels on %s...", season_set_train)
    compute_thresholds(conn, season_set=season_set_train)
    n_labels = apply_labels(conn, season_set=season_set_train)
    logger.info("  wrote %d label rows", n_labels)


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
        "--skip-refit",
        action="store_true",
        help="Skip the in-process model refit (read latest model_fits row "
        "instead). Currently NOT implemented — kept as a flag for the "
        "design spec's flag list; toggling it just bypasses inference, "
        "which is not yet wired. Default behavior refits.",
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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config(args.league_config)
    scoring_season = args.scoring_season or config.season_year
    logger.info(
        "Sunday report for league %d (%s) — scoring season %d",
        config.league_id,
        config.team_name,
        scoring_season,
    )

    conn = get_connection(args.db_path)
    try:
        _refresh_streaks_db(
            conn,
            season=scoring_season,
            season_set_train=args.season_set_train,
            projections_root=args.projections_root,
            skip_fetch=args.skip_fetch,
        )

        if args.skip_refit:
            # The Phase 5 spec lists --skip-refit as a deferred optimization.
            # For now it's an error rather than a silent no-op — running the
            # report without a refit needs us to load fitted pipelines back
            # from disk, which we don't yet persist. Fail loudly.
            raise SystemExit(
                "--skip-refit is reserved for a future optimization; "
                "fitted pipelines are not currently persisted to disk."
            )

        logger.info("Refitting models on %s...", args.season_set_train)
        models = refit_models_for_report(
            conn,
            season_set_train=args.season_set_train,
            window_days=14,
        )
        logger.info("  fitted %d models", len(models))

        logger.info("Authenticating with Yahoo...")
        # Imported lazily so unit tests + --help don't trigger an oauth
        # file check.
        from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session

        session = get_yahoo_session()
        league = get_league(session, config.league_id, config.game_code)
        roster_hitters, fa_hitters = _fetch_yahoo_data(league, team_name=config.team_name)
        logger.info(
            "Yahoo fetch complete: %d roster, %d FAs (deduped)",
            len(roster_hitters),
            len(fa_hitters),
        )

        name_to_mlbam = build_name_to_mlbam_map(args.projections_root, season=scoring_season)
        if not name_to_mlbam:
            raise SystemExit(
                f"No name→mlbam mappings built — check that {args.projections_root}/"
                f"{scoring_season}/ contains hitter CSVs with Name + MLBAMID columns."
            )

        today = local_today()
        report = build_report(
            conn,
            league_config_team_name=config.team_name,
            league_config_league_id=config.league_id,
            models=models,
            roster_hitters=roster_hitters,
            fa_hitters=fa_hitters,
            name_to_mlbam=name_to_mlbam,
            today=today,
            season_set_train=args.season_set_train,
            scoring_season=scoring_season,
            window_days=14,
            top_n_fas=10,
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
