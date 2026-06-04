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

# A ROS snapshot older than this many days vs today means the FanGraphs fetch
# has stalled. Its counting stats still carry near-full-season magnitudes while
# YTD actuals keep accumulating, so deriving full-season as ``YTD + ROS``
# double-counts. The fetch is a DAILY job, so a healthy run always produces a
# same-day snapshot; anything older means a failed/skipped fetch. 1 day of grace
# absorbs timezone slack and a hand-staged snapshot blended the next morning.
# Refuse to overwrite the last-good Redis blob with a snapshot this stale, and
# warn on the READ side when a frozen blob ages past this (see refresh_pipeline).
ROS_SNAPSHOT_STALE_DAYS = 2


class StaleROSSnapshotError(RuntimeError):
    """Raised when the latest datable ROS snapshot is staler than
    ``ROS_SNAPSHOT_STALE_DAYS``.

    Blending it would overwrite the last-good ``cache:ros_projections`` with
    near-preseason magnitudes that the daily refresh then double-counts as
    ``YTD + ROS``. Aborting keeps the most recent good projections. This is the
    guard for the 2026-06-04 incident: a Cloudflare-403'd FanGraphs fetch left
    only the committed March snapshot on disk, and the old warn-and-proceed path
    overwrote fresh Redis with it.
    """


def parse_snapshot_date(dir_name: str) -> date | None:
    """Parse the date a ``rest_of_season`` subdir name (or a stamped
    ``_ros_snapshot_date``) encodes.

    The name is normally ``YYYY-MM-DD`` but may carry a suffix (e.g.
    ``2026-06-04-manual`` for a hand-staged snapshot); the leading 10 chars are
    parsed as the ISO date. Returns ``None`` when there is no leading ISO date.
    The single source of truth for "this name means date X", shared by snapshot
    selection, the write-side guard, and the read-side warning so they can't
    disagree.
    """
    try:
        return date.fromisoformat(dir_name[:10])
    except ValueError:
        return None


def ros_snapshot_days_stale(snap: date) -> int:
    """Days a ROS snapshot dated ``snap`` lags today (negative if in the future).

    The single source of the staleness arithmetic shared by the write-side guard
    (:func:`_require_fresh_ros_snapshot`) and the read-side warning
    (``refresh_pipeline._warn_if_ros_blob_stale``), so the two enforcement points
    can't drift. Both compare the result against :data:`ROS_SNAPSHOT_STALE_DAYS`.
    """
    return (local_today() - snap).days


def _require_fresh_ros_snapshot(snap: date, label: str, progress_cb) -> None:
    """Abort the blend unless ``snap`` is within ``ROS_SNAPSHOT_STALE_DAYS`` of
    today. ``label`` is the dir name, for the message only.

    Runs BEFORE any KV read/write so a stale blend touches nothing -- keeping the
    last-good ``cache:ros_projections`` ("use the most recent ROS").

    Raises:
        StaleROSSnapshotError: the snapshot is too stale to blend.
    """
    days_stale = ros_snapshot_days_stale(snap)
    if days_stale > ROS_SNAPSHOT_STALE_DAYS:
        msg = (
            f"Refusing to overwrite cache:ros_projections: latest ROS snapshot "
            f"{label} is {days_stale} days stale (> {ROS_SNAPSHOT_STALE_DAYS}). "
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
            or contains no date-named subdirectories.
        StaleROSSnapshotError: if the latest dated snapshot is staler than
            ``ROS_SNAPSHOT_STALE_DAYS``. Nothing is written, so the last-good
            ``cache:ros_projections`` is preserved.
    """
    ros_root = projections_dir / str(season_year) / "rest_of_season"
    if not ros_root.is_dir():
        raise FileNotFoundError(f"ROS snapshot dir missing: {ros_root}")
    # Pick the latest snapshot BY PARSED DATE, ignoring any dir whose name has no
    # leading ISO date (a stray/helper dir). A raw string sort would let an
    # undatable name like "manual-latest" sort after the dated dirs and shadow a
    # perfectly fresh snapshot, aborting every blend.
    dated = [
        (p, d)
        for p in ros_root.iterdir()
        if p.is_dir() and (d := parse_snapshot_date(p.name)) is not None
    ]
    if not dated:
        raise FileNotFoundError(f"No datable ROS snapshot dirs under {ros_root}")
    latest, snap = max(dated, key=lambda pd: pd[1])
    snapshot_date = latest.name
    # Refuse to overwrite the last-good Redis blob with a stale snapshot
    # (raises StaleROSSnapshotError). Must run BEFORE any KV read/write below
    # so a stale blend touches nothing -- "use the most recent ROS" mitigation.
    _require_fresh_ros_snapshot(snap, snapshot_date, progress_cb)

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
