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

from pathlib import Path

import pandas as pd

from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.data.kv_store import get_kv, is_remote
from fantasy_baseball.data.projections import (
    blend_projections,
    normalize_rest_of_season_to_full_season,
)
from fantasy_baseball.data.redis_store import (
    get_game_log_totals,
    set_full_season_projections,
    set_ros_projections,
)
from fantasy_baseball.models.player import PlayerType


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

    Returns the ROS-only ``(hitters_df, pitchers_df)`` — full-season is
    a derived view; callers that want it should read
    :func:`fantasy_baseball.data.redis_store.get_full_season_projections`.

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
    # ``write_cache`` writes to disk and (only on Render) through to
    # Upstash. Off Render, the local KV (sqlite) needs the explicit
    # ``set_*_projections`` writes so consumers reading via
    # ``get_{ros,full_season}_projections(kv)`` find the data; the
    # on-Render path would otherwise double-write the same blob to
    # Upstash. Imported at call time to avoid a web-layer import at
    # data-layer import time.
    from fantasy_baseball.web.season_data import write_cache

    write_cache(CacheKey.ROS_PROJECTIONS, ros_payload)
    write_cache(CacheKey.FULL_SEASON_PROJECTIONS, full_payload)
    if not is_remote():
        set_ros_projections(client, ros_payload)
        set_full_season_projections(client, full_payload)
    return hitters_ros, pitchers_ros
