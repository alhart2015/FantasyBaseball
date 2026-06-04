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
# so deriving full-season as ``YTD + ROS`` double-counts. Refuse to
# overwrite the last-good Redis blob with a snapshot this stale.
ROS_SNAPSHOT_STALE_DAYS = 7


class StaleROSSnapshotError(RuntimeError):
    """Raised when the only available ROS snapshot is staler than
    ``ROS_SNAPSHOT_STALE_DAYS`` (or has an un-datable dir name).

    Blending it would overwrite the last-good ``cache:ros_projections`` with
    near-preseason magnitudes that the daily refresh then double-counts as
    ``YTD + ROS``. Aborting keeps the most recent good projections. This is the
    guard for the 2026-06-04 incident: a Cloudflare-403'd FanGraphs fetch left
    only the committed March snapshot on disk, and the old warn-and-proceed path
    overwrote fresh Redis with it.
    """


def _require_fresh_ros_snapshot(snapshot_date: str, progress_cb) -> None:
    """Abort the blend unless the chosen snapshot is within
    ``ROS_SNAPSHOT_STALE_DAYS`` of today.

    The snapshot dir name is normally ``YYYY-MM-DD`` but may carry a suffix
    (e.g. ``2026-06-04-manual``); the leading 10 chars are parsed as the ISO
    date. A name with no leading ISO date can't be dated -- treat it as
    un-verifiable and refuse, rather than silently overwrite good Redis.

    Raises:
        StaleROSSnapshotError: the snapshot is stale or has no leading ISO date.
    """
    try:
        snap = date.fromisoformat(snapshot_date[:10])
    except ValueError:
        msg = (
            f"Refusing to blend ROS snapshot '{snapshot_date}': name has no "
            f"leading ISO date, so freshness can't be verified. Keeping the "
            f"last-good cache:ros_projections."
        )
        log.warning(msg)
        if progress_cb:
            progress_cb(msg)
        raise StaleROSSnapshotError(msg) from None
    days_stale = (local_today() - snap).days
    if days_stale > ROS_SNAPSHOT_STALE_DAYS:
        msg = (
            f"Refusing to overwrite cache:ros_projections: latest ROS snapshot "
            f"{snapshot_date} is {days_stale} days stale (> {ROS_SNAPSHOT_STALE_DAYS}). "
            f"A failed FanGraphs fetch would otherwise regress fresh Redis to "
            f"near-preseason magnitudes that the refresh double-counts as YTD + "
            f"ROS. Keeping the most recent good ROS blob -- re-pull fresh ROS CSVs "
            f"and re-run."
        )
        log.warning(msg)
        if progress_cb:
            progress_cb(msg)
        raise StaleROSSnapshotError(msg)


def _numeric_keyed(totals: dict) -> dict[int, dict]:
    """Coerce a ``game_log_totals`` rollup (string mlbam_id keys) to int keys.

    ``normalize_rest_of_season_to_full_season`` matches on ``int(mlbam_id)``.
    Keys are numeric MLBAM ids by writer invariant (``mlb_game_logs`` writes
    ``str(person["id"])``); a non-numeric key cannot match a numeric id, so it
    is skipped with a loud warning -- surfacing a possible writer regression
    without crashing the whole job over one bad row. This is the single shared
    policy so the two call sites (ROS fetch + daily refresh) cannot diverge.
    """
    out: dict[int, dict] = {}
    bad: list[str] = []
    for k, v in (totals or {}).items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            bad.append(str(k))
    if bad:
        log.warning(
            "game_log_totals has %d non-numeric key(s), skipped from full-season "
            "derivation (possible writer regression): %s",
            len(bad),
            bad[:5],
        )
    return out


def derive_full_season(
    ros_hitters: pd.DataFrame,
    ros_pitchers: pd.DataFrame,
    hitter_totals: dict,
    pitcher_totals: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add YTD actuals to ROS-remaining projections -> full-season totals.

    ``hitter_totals`` / ``pitcher_totals`` are the raw ``game_log_totals``
    rollups (string mlbam_id keys); :func:`_numeric_keyed` coerces them once.
    Shared by the ROS-fetch job (which persists ``cache:full_season_projections``)
    and the daily refresh (which re-derives full-season from the SAME current
    game logs its team-YTD overlay uses, avoiding a stale-vintage mismatch), so
    the derivation cannot drift between the two.
    """
    full_hitters = normalize_rest_of_season_to_full_season(
        ros_hitters, _numeric_keyed(hitter_totals), PlayerType.HITTER
    )
    full_pitchers = normalize_rest_of_season_to_full_season(
        ros_pitchers, _numeric_keyed(pitcher_totals), PlayerType.PITCHER
    )
    return full_hitters, full_pitchers


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
        StaleROSSnapshotError: if the latest snapshot dir is staler than
            ``ROS_SNAPSHOT_STALE_DAYS`` (or un-datable). Nothing is written,
            so the last-good ``cache:ros_projections`` is preserved.
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
    # Refuse to overwrite the last-good Redis blob with a stale snapshot
    # (raises StaleROSSnapshotError). Must run BEFORE any KV read/write below
    # so a stale blend touches nothing -- "use the most recent ROS" mitigation.
    _require_fresh_ros_snapshot(snapshot_date, progress_cb)

    client = get_kv()
    hitter_totals = get_game_log_totals(client, "hitters")
    pitcher_totals = get_game_log_totals(client, "pitchers")

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

    # Derive full-season by adding YTD actuals to the ROS-only blend, via the
    # shared helper the daily refresh also uses (so the two can't drift).
    hitters_full, pitchers_full = derive_full_season(
        hitters_ros, pitchers_ros, hitter_totals, pitcher_totals
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
    # blend is visible to consumers (see _require_fresh_ros_snapshot).
    snapshot_meta = {"_ros_snapshot_date": snapshot_date}
    job_token = set_cache_job("ros_fetch")
    try:
        write_cache(CacheKey.ROS_PROJECTIONS, ros_payload, extra_meta=snapshot_meta)
        write_cache(CacheKey.FULL_SEASON_PROJECTIONS, full_payload, extra_meta=snapshot_meta)
    finally:
        reset_cache_job(job_token)
    return hitters_ros, pitchers_ros
