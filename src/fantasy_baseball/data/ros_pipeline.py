"""Rest-of-season projections pipeline: blend CSVs in memory → Redis.

Produces TWO Redis blobs from one CSV blend:
- ``cache:ros_projections`` — ROS-remaining counting stats (FanGraphs
  CSV values, untouched)
- ``cache:full_season_projections`` — same blend plus YTD actuals from
  ``game_log_totals:{hitters,pitchers}``, used by ``project_team_stats``
  for end-of-season standings projection.

Forward-looking decisions (transactions, trades, waivers, lineup
optimizer) read the ROS blob. Standings projection reads the
full-season blob.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.data.kv_store import get_kv
from fantasy_baseball.data.projections import (
    blend_projections,
    normalize_rest_of_season_to_full_season,
)
from fantasy_baseball.data.redis_store import get_game_log_totals
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.time_utils import local_today

log = logging.getLogger(__name__)

# A ROS snapshot older than this many days vs today means the daily
# FanGraphs fetch has likely stalled. Its counting stats still carry
# near-full-season magnitudes while YTD actuals have kept accumulating,
# so deriving full-season as ``YTD + ROS`` double-counts. Warn loudly.
ROS_SNAPSHOT_STALE_DAYS = 7


def _warn_if_ros_snapshot_stale(snapshot_date: str, progress_cb) -> None:
    """Emit a loud warning if the chosen ROS snapshot predates today by more
    than ``ROS_SNAPSHOT_STALE_DAYS``.

    Warn-and-proceed: the caller still writes both blobs. A non-ISO dir name
    is treated as non-datable and skipped (no false alarm).
    """
    try:
        snap = date.fromisoformat(snapshot_date)
    except ValueError:
        return
    days_stale = (local_today() - snap).days
    if days_stale > ROS_SNAPSHOT_STALE_DAYS:
        msg = (
            f"WARNING: ROS snapshot {snapshot_date} is {days_stale} days stale "
            f"(> {ROS_SNAPSHOT_STALE_DAYS}). full_season = YTD + ROS may "
            f"double-count: a stale snapshot still carries near-full-season "
            f"magnitudes while YTD has accumulated. Re-run the FanGraphs ROS "
            f"fetch to refresh the snapshot."
        )
        log.warning(msg)
        if progress_cb:
            progress_cb(msg)


def blend_and_cache_ros(
    projections_dir: Path,
    systems: list[str],
    weights: dict[str, float] | None,
    roster_names: set[str] | None,
    season_year: int,
    progress_cb=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend the latest ROS CSV snapshot in memory and write BOTH Redis blobs.

    Scans ``projections_dir/{season_year}/rest_of_season/{YYYY-MM-DD}/``
    for the most recent dated subdir, blends each system's ROS-only
    counting stats with no normalization, then derives a full-season
    view by adding season-to-date actuals from
    ``game_log_totals:{hitters,pitchers}``. Writes BOTH:

    - ``cache:ros_projections`` — ROS-only (FanGraphs CSV values
      untouched).
    - ``cache:full_season_projections`` — ROS + YTD actuals.

    Returns the ROS-only ``(hitters_df, pitchers_df)``; full-season is a
    derived view persisted to ``cache:full_season_projections`` and read via
    ``season_data.read_cache(CacheKey.FULL_SEASON_PROJECTIONS)``.

    Args:
        projections_dir: Root ``data/projections`` path. Year and
            ``rest_of_season/{date}`` subdirs live beneath it.
        systems: Projection system names (e.g. ``["steamer", "zips"]``).
        weights: Optional per-system blend weights; ``None`` → equal.
        roster_names: Optional rostered-player set for quality checks.
        season_year: Season year whose ROS snapshots to scan.
        progress_cb: Optional callback for per-system log lines.

    Raises:
        FileNotFoundError: if
            ``projections_dir/{season_year}/rest_of_season/`` is missing
            or contains no date subdirectories.
    """
    ros_root = projections_dir / str(season_year) / "rest_of_season"
    if not ros_root.is_dir():
        raise FileNotFoundError(f"ROS snapshot dir missing: {ros_root}")
    date_dirs = sorted(
        (p for p in ros_root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    if not date_dirs:
        raise FileNotFoundError(f"No ROS snapshot dirs under {ros_root}")
    latest = date_dirs[-1]
    snapshot_date = latest.name
    _warn_if_ros_snapshot_stale(snapshot_date, progress_cb)

    client = get_kv()
    # JSON round-trips coerce int keys to str;
    # normalize_rest_of_season_to_full_season looks up actuals with
    # int(mlbam_id), so we reverse the coercion at the boundary.
    # Writer invariant: keys are always numeric MLB AM IDs. See
    # ``data/mlb_game_logs.py`` — the writer builds each key as
    # ``str(player["mlbam_id"])`` where ``mlbam_id`` comes from the
    # MLB Stats API's ``person["id"]``, which is always a numeric ID.
    # We do NOT filter non-numeric keys here on purpose: a non-numeric
    # key would indicate a writer regression and we want the
    # ValueError to surface loudly rather than silently dropping data.
    hitter_totals = {int(k): v for k, v in get_game_log_totals(client, "hitters").items()}
    pitcher_totals = {int(k): v for k, v in get_game_log_totals(client, "pitchers").items()}

    # Blend in pure ROS-only mode — no normalizer. The blended output is
    # the FanGraphs ROS-remaining values, which is what
    # cache:ros_projections must hold.
    hitters_ros, pitchers_ros, _quality = blend_projections(
        latest,
        systems,
        weights,
        roster_names=roster_names,
        progress_cb=progress_cb,
        normalizer=None,
    )

    # Derive full-season by adding YTD actuals to the ROS-only blend.
    hitters_full = normalize_rest_of_season_to_full_season(
        hitters_ros,
        hitter_totals,
        PlayerType.HITTER,
    )
    pitchers_full = normalize_rest_of_season_to_full_season(
        pitchers_ros,
        pitcher_totals,
        PlayerType.PITCHER,
    )

    ros_payload = {
        "hitters": hitters_ros.to_dict(orient="records"),
        "pitchers": pitchers_ros.to_dict(orient="records"),
    }
    full_payload = {
        "hitters": hitters_full.to_dict(orient="records"),
        "pitchers": pitchers_full.to_dict(orient="records"),
    }
    # Single write path: write_cache routes to the KV (Upstash on Render,
    # sqlite locally) and wraps each payload in a provenance envelope.
    # Consumers read back via the envelope-aware read_cache/read_cache_dict.
    # Imported at call time to avoid a web-layer import at data-layer import
    # time.
    from fantasy_baseball.web.season_data import (
        reset_cache_job,
        set_cache_job,
        write_cache,
    )

    # Stamp the source snapshot date into the provenance envelope so a stale
    # blend is visible to consumers (see _warn_if_ros_snapshot_stale).
    snapshot_meta = {"_ros_snapshot_date": snapshot_date}
    job_token = set_cache_job("ros_fetch")
    try:
        write_cache(CacheKey.ROS_PROJECTIONS, ros_payload, extra_meta=snapshot_meta)
        write_cache(CacheKey.FULL_SEASON_PROJECTIONS, full_payload, extra_meta=snapshot_meta)
    finally:
        reset_cache_job(job_token)
    return hitters_ros, pitchers_ros
