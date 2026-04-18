"""Rest-of-season projections pipeline: blend CSVs in memory → Redis.

This replaces the SQLite-staging path (``load_rest_of_season_projections``
/ ``get_rest_of_season_projections`` in ``data.db``). The refresh and the
admin-triggered fetch both call :func:`blend_and_cache_ros` — it blends
the latest dated snapshot in memory using ``game_log_totals`` from Redis
for ROS → full-season normalization and writes the result straight to
``cache:ros_projections``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.data.projections import (
    blend_projections,
    normalize_rest_of_season_to_full_season,
)
from fantasy_baseball.data.redis_store import (
    get_default_client,
    get_game_log_totals,
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
    """Blend the latest ROS CSV snapshot in memory and cache to Redis.

    Scans ``projections_dir/{season_year}/rest_of_season/{YYYY-MM-DD}/``
    for the most recent dated subdir, normalizes each system's ROS
    counting stats to full-season by adding season-to-date actuals from
    ``game_log_totals:{hitters,pitchers}``, blends into weighted
    averages, and writes the result to ``cache:ros_projections``. No
    SQLite.

    Args:
        projections_dir: Root ``data/projections`` path. Year and
            ``rest_of_season/{date}`` subdirs live beneath it.
        systems: Projection system names (e.g. ``["steamer", "zips"]``).
        weights: Optional per-system blend weights; ``None`` → equal.
        roster_names: Optional rostered-player set for quality checks.
        season_year: Season year whose ROS snapshots to scan.
        progress_cb: Optional callback for per-system log lines.

    Returns:
        ``(hitters_df, pitchers_df)`` in the format produced by
        :func:`blend_projections`.

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

    client = get_default_client()
    # JSON round-trips coerce int keys to str;
    # normalize_rest_of_season_to_full_season looks up actuals with
    # int(mlbam_id), so we reverse the coercion at the boundary.
    # Writer invariant: keys are always numeric MLB AM IDs. See
    # ``data/mlb_game_logs.py`` — line 126 builds each key as
    # ``str(player["mlbam_id"])`` where ``mlbam_id`` comes from the
    # MLB Stats API's ``person["id"]`` (line 84), which is always a
    # numeric ID. We do NOT filter non-numeric keys here on purpose:
    # a non-numeric key would indicate a writer regression and we want
    # the ValueError to surface loudly rather than silently dropping data.
    hitter_totals = {
        int(k): v for k, v in get_game_log_totals(client, "hitters").items()
    }
    pitcher_totals = {
        int(k): v for k, v in get_game_log_totals(client, "pitchers").items()
    }

    def _normalizer(system_name, hitters_df, pitchers_df):
        if progress_cb:
            progress_cb(f"Normalizing {system_name} ROS → full-season")
        h = normalize_rest_of_season_to_full_season(
            hitters_df, hitter_totals, PlayerType.HITTER,
        )
        p = normalize_rest_of_season_to_full_season(
            pitchers_df, pitcher_totals, PlayerType.PITCHER,
        )
        return h, p

    hitters_df, pitchers_df, _quality = blend_projections(
        latest, systems, weights,
        roster_names=roster_names, progress_cb=progress_cb,
        normalizer=_normalizer,
    )

    # Imported at call time to avoid importing a web-layer module at
    # data-layer import time (circular-ish; narrower coupling here).
    # TODO(task-11): move write_cache into the data layer so this shim
    # (and the lazy import) can go away.
    from fantasy_baseball.web.season_data import write_cache
    write_cache(CacheKey.ROS_PROJECTIONS, {
        "hitters": hitters_df.to_dict(orient="records"),
        "pitchers": pitchers_df.to_dict(orient="records"),
    })
    return hitters_df, pitchers_df
