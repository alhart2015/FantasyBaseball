"""Write the validated transcriptions into the two seams the pipeline reads.

The Yahoo-free refresh does not invent a new way to learn the league
state -- it reuses the stale-data path that already exists
(``FB_SKIP_YAHOO=1`` / ``RefreshRun(skip_yahoo=True)``). That path reads
exactly two things it cannot compute for itself:

- ``weekly_rosters_history`` -- the hash the Yahoo run writes in
  ``refresh_pipeline._write_snapshots_and_load_league`` and that
  ``League.from_redis`` reads back to build every team's roster.
- ``cache:standings`` -- the enveloped blob
  ``RefreshRun._reuse_cached_standings`` reads to recover the standings
  and the user's ``team_key``.

So the seeder writes those two (plus the canonical ``standings_history``
snapshot, which drives the trends charts and is the other half of
``League.from_redis``), through the SAME writer functions the Yahoo path
calls. No new key family is invented and no payload is hand-shaped:
``write_roster_snapshot``, ``write_standings_snapshot`` and
``write_cache_to`` own those shapes, and going through them is what keeps
a manual blob readable by every existing consumer.

Two safety properties this module is responsible for:

1. **It refuses to write to the Yahoo baseline store.** Manual mode is
   isolated by pointing ``FANTASY_LOCAL_KV_PATH`` at a whole separate
   SQLite file (``data/manual.db``); a seed that landed in
   ``data/local.db`` would overwrite real Yahoo history with hand-typed
   rows, and afterwards there would be no way to tell. The resolved
   absolute path is echoed BEFORE the first write, every run.
2. **Everything it writes is stamped.** The cache envelope carries
   ``_source``/``_manual``, every roster row carries ``source``, and the
   store gets a ``manual_seed_provenance`` breadcrumb. A future reader who
   finds this data must not be able to mistake it for Yahoo's.

This module acquires and shapes data only -- see the package docstring.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fantasy_baseball.data.cache_keys import (
    MANUAL_PROVENANCE_KEY,
    CacheKey,
    redis_key,
)

# ``_DEFAULT_LOCAL_DB`` is the single definition of the Yahoo baseline store's
# location. Re-deriving it here would let the two drift, and the refusal below
# is only worth having if it points at the same file ``kv_store`` does.
from fantasy_baseball.data.kv_store import _DEFAULT_LOCAL_DB, KVStore
from fantasy_baseball.data.redis_store import write_roster_snapshot, write_standings_snapshot
from fantasy_baseball.manual.transcripts import ManualRosterSnapshot
from fantasy_baseball.models.standings import Standings
from fantasy_baseball.web.season_data import (
    reset_cache_job,
    set_cache_job,
    unwrap_cache_envelope,
    write_cache_to,
)

log = logging.getLogger(__name__)

#: Stamped into every blob this module writes. Grep-able, and unmistakable.
MANUAL_SOURCE = "manual-transcription"

#: Job label stamped into the cache envelope's ``_meta._job`` -- the same field
#: the dashboard refresh and the ROS fetch use to record their writer.
MANUAL_JOB_LABEL = "manual-seed"

#: Store-level breadcrumb: one plain key describing the whole seed, so an
#: operator opening an unfamiliar .db can tell in a single read whether it is
#: the Yahoo baseline or a hand-transcribed store. Defined in ``data.cache_keys``
#: and re-exported here: ``data.rosters`` reads it, and the pipeline importing
#: ``manual`` for a key would point the dependency the wrong way.
PROVENANCE_KEY = MANUAL_PROVENANCE_KEY

#: Any store with this file name is presumed to be the Yahoo baseline.
BASELINE_DB_NAME = "local.db"

#: Marker key added to every seeded roster row.
ROW_SOURCE_FIELD = "source"


class ManualSeedRefused(RuntimeError):
    """The seeder declined to write because the target store is not isolated.

    Raised BEFORE any write, so a refused seed leaves the store exactly as
    it was.
    """


@dataclass(frozen=True)
class SeedStats:
    """What a successful :func:`seed_manual_kv` wrote."""

    teams: int
    players: int
    snapshot_date: str
    standings_date: str = ""
    kv_path: str = ""


def resolve_kv_path(client: KVStore) -> Path | None:
    """Return the absolute file backing ``client``, or None if it has none.

    Thin alias for ``kv_sync.store_path``. It used to read the PRIVATE
    ``SqliteKVStore._path`` through a ``getattr(..., None)`` that fails open --
    so a rename would have turned this, and the three refusals that gate on it,
    into silent no-ops. ``SqliteKVStore.path`` is public and pinned by a test
    now; this stays as the name ``manual/`` callers already use.
    """
    from fantasy_baseball.data.kv_sync import store_path

    return store_path(client)


def describe_kv_target(client: KVStore) -> str:
    """Human-readable description of where ``client`` would write."""
    path = resolve_kv_path(client)
    if path is not None:
        return str(path)
    return f"{type(client).__name__} (no local file -- remote or in-memory store)"


def assert_isolated_store(client: KVStore) -> Path:
    """Return the target path, or raise if it is not an isolated manual store.

    Refuses, in order:

    1. ``RENDER`` set to anything non-empty -- manual data must never be
       written from (or to) the production environment.
    2. a client with no local file -- an Upstash client would put
       hand-typed rows straight into production.
    3. any store named ``local.db`` -- the Yahoo baseline, or a copy of it
       carrying the same name.
    4. the exact baseline path ``kv_store`` resolves by default.
    """
    render = os.environ.get("RENDER", "")
    if render:
        raise ManualSeedRefused(
            f"RENDER={render!r} is set. The manual seeder writes hand-transcribed "
            "data and must never run against the production environment. Unset "
            "RENDER and point FANTASY_LOCAL_KV_PATH at data/manual.db."
        )

    path = resolve_kv_path(client)
    if path is None:
        raise ManualSeedRefused(
            f"Refusing to seed {describe_kv_target(client)}: the manual seeder only "
            "writes a local SQLite store (data/manual.db). Build the client with "
            "SqliteKVStore(path), or set FANTASY_LOCAL_KV_PATH before the first "
            "fantasy_baseball import."
        )

    if path.name.lower() == BASELINE_DB_NAME or path == _DEFAULT_LOCAL_DB.resolve():
        raise ManualSeedRefused(
            f"Refusing to seed {path}: that is the Yahoo baseline KV store. "
            "Hand-transcribed rosters and standings written there would overwrite "
            "real Yahoo history irreversibly. Run scripts/bootstrap_manual_kv.py "
            "first and seed the copy (data/manual.db) instead."
        )
    return path


def read_team_keys(client: KVStore) -> dict[str, str]:
    """Return ``{team_name: team_key}`` from the store's ``cache:standings``.

    Yahoo team keys are LOOKED UP here, never hand-typed into the YAML: the
    manual store is bootstrapped from the Yahoo baseline, so the last real
    standings blob still carries every ``team_key``.
    ``transcripts.load_manual_standings`` takes this mapping and attaches the
    keys to the transcribed rows.

    Returns ``{}`` -- never raises -- when the blob is missing, corrupt, or an
    unexpected shape. A missing key degrades to ``""``, which
    ``League.from_redis`` tolerates.
    """
    try:
        raw = client.get(redis_key(CacheKey.STANDINGS))
    except Exception as exc:  # pragma: no cover - backend-specific failure
        log.warning("read_team_keys: KV read failed: %s", exc)
        return {}
    if raw is None:
        log.warning(
            "read_team_keys: no cache:%s in this store -- every team_key will be "
            "empty. Bootstrap the manual store from data/local.db first.",
            CacheKey.STANDINGS,
        )
        return {}
    try:
        payload = unwrap_cache_envelope(json.loads(raw))
    except json.JSONDecodeError:
        log.warning("read_team_keys: cache:%s is corrupt JSON", CacheKey.STANDINGS)
        return {}
    if not isinstance(payload, dict):
        log.warning("read_team_keys: cache:%s is not a mapping", CacheKey.STANDINGS)
        return {}
    teams = payload.get("teams")
    if not isinstance(teams, list):
        log.warning("read_team_keys: cache:%s has no 'teams' list", CacheKey.STANDINGS)
        return {}

    out: dict[str, str] = {}
    for row in teams:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        team_key = row.get("team_key")
        if isinstance(name, str) and name and isinstance(team_key, str) and team_key:
            out[name] = team_key
    return out


def _stamp_rows(rows: list[dict[str, str]], team: str) -> list[dict[str, str]]:
    """Copy roster rows, adding the provenance marker.

    The extra key rides along inside ``weekly_rosters_history``. Every reader
    (``League.from_redis``, ``_reuse_cached_roster_raw``,
    ``compute_team_ytd_ab``) picks named fields out of these dicts, so an
    additional one is inert -- but it makes a manual day in the hash
    self-identifying next to the Yahoo days around it.
    """
    if not rows:
        log.warning("seed: team %r transcribed with zero players", team)
    return [{**row, ROW_SOURCE_FIELD: MANUAL_SOURCE} for row in rows]


def seed_manual_kv(
    client: KVStore,
    standings: Standings,
    rosters: ManualRosterSnapshot,
    *,
    echo: Callable[[str], None] = print,
) -> SeedStats:
    """Write the transcriptions into ``client`` and return what was written.

    Three writes, all through the writers the Yahoo pipeline itself uses:

    1. :func:`redis_store.write_roster_snapshot` once per team, keyed by
       ``rosters.snapshot_date`` -- this is ``weekly_rosters_history``. It
       replaces that team's rows for that date rather than appending, so
       re-seeding after fixing a transcription typo is safe.
    2. :func:`redis_store.write_standings_snapshot` -- ``standings_history``,
       keyed by ``standings.effective_date``.
    3. :func:`season_data.write_cache_to` for ``cache:standings`` -- the blob
       ``_reuse_cached_standings`` reads. ``write_cache_to`` takes an explicit
       client, so the seeder never depends on process-global ``get_kv()``
       resolution: the caller decides where this lands.

    ``client`` is validated by :func:`assert_isolated_store` first, and the
    resolved absolute path is echoed before anything is written.

    Raises:
        ManualSeedRefused: the target store is not an isolated manual store.
        ValueError: the roster transcription contains no teams.
    """
    path = assert_isolated_store(client)
    snapshot_date = rosters.snapshot_date.isoformat()
    standings_date = standings.effective_date.isoformat()

    echo(f"Seeding MANUAL (hand-transcribed, NOT Yahoo) data into: {path}")
    echo(
        f"  roster snapshot date: {snapshot_date} ({len(rosters.rows_by_team)} teams) | "
        f"standings effective date: {standings_date} ({len(standings.entries)} teams)"
    )

    if not rosters.rows_by_team:
        raise ValueError(
            "Nothing to seed: the roster transcription has no teams. "
            "Fill in data/manual/rosters.yaml before seeding."
        )
    if snapshot_date != standings_date:
        # Not fatal -- the pipeline reads the two independently -- but mixing
        # vintages silently is exactly the sort of thing that later reads as a
        # real roster change, so say it out loud.
        echo(
            "  WARNING: roster snapshot date and standings effective date differ; "
            "this run will mix two vintages."
        )

    players = 0
    token = set_cache_job(MANUAL_JOB_LABEL)
    try:
        for team, rows in rosters.rows_by_team.items():
            write_roster_snapshot(client, snapshot_date, team, _stamp_rows(rows, team))
            players += len(rows)

        write_standings_snapshot(client, standings)

        extra_meta = {
            "_source": MANUAL_SOURCE,
            "_manual": True,
            "_yahoo": False,
            "_manual_roster_snapshot": snapshot_date,
            "_manual_standings_date": standings_date,
            "_manual_kv_path": str(path),
        }
        write_cache_to(client, CacheKey.STANDINGS, standings.to_json(), extra_meta)
    finally:
        reset_cache_job(token)

    stats = SeedStats(
        teams=len(rosters.rows_by_team),
        players=players,
        snapshot_date=snapshot_date,
        standings_date=standings_date,
        kv_path=str(path),
    )
    client.set(
        PROVENANCE_KEY,
        json.dumps(
            {
                "source": MANUAL_SOURCE,
                "yahoo": False,
                "seeded_at": datetime.now(UTC).isoformat(),
                "kv_path": stats.kv_path,
                "roster_snapshot_date": stats.snapshot_date,
                "standings_effective_date": stats.standings_date,
                "teams": stats.teams,
                "players": stats.players,
                "note": (
                    "weekly_rosters_history, standings_history and cache:standings in "
                    "this store were seeded from hand-transcribed YAML under "
                    "data/manual/, NOT from the Yahoo API. Do not sync this store "
                    "anywhere."
                ),
            }
        ),
    )

    echo(f"  seeded {stats.teams} teams / {stats.players} players; stamped {PROVENANCE_KEY}")
    return stats
