#!/usr/bin/env python3
"""Publish the hand-transcribed manual store to production Upstash, so Render shows it.

While Yahoo access is gone, ``data/manual.db`` is the only current league state there
is. ``scripts/run_manual_refresh.py`` builds it locally; this copies it up so the
deployed dashboard serves the same standings, rosters, lineup and projections.

    python scripts/run_manual_refresh.py     # build the store, as before
    python scripts/publish_manual.py --dry-run
    python scripts/publish_manual.py         # send it

WHAT IT SENDS. Every key in the store whose value differs from prod, plus the four
history hashes field by field -- see ``kv_sync.publish_local_to_remote`` for the rules
(upsert only, provenance stamp first, ``cache:meta`` last, trajectory blobs and job logs
excluded). The stamp is what makes Render treat the data as manual: it shows the
"hand-entered" banner and refuses the Refresh button, which would otherwise run
stale-data mode over the transcription.

WHAT IT KEEPS. Before the first write, every prod value about to be replaced is saved
to ``data/backups/upstash-before-publish-<UTC stamp>.json.gz``. Prod has no history of
its own, and the first publish replaces the last Yahoo snapshot of every ``cache:*``
key.

NOT SENT: the trajectory board. It has its own push, ``scripts/push_trajectory_board.py``,
because it is fitted offline on data Render does not have.

Exit codes: 0 ok, 1 started then failed, 2 refused before touching anything.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RC_OK = 0
RC_FAILED = 1
RC_REFUSED = 2

DEFAULT_BACKUP_DIR = PROJECT_ROOT / "data" / "backups"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from fantasy_baseball.manual.environment import DEFAULT_MANUAL_KV_PATH

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--kv-path",
        type=Path,
        default=DEFAULT_MANUAL_KV_PATH,
        help="the manual store to publish (default: data/manual.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compare with prod and report what would change; write nothing",
    )
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    return parser.parse_args(argv)


def source_refusal(kv_path: Path) -> str | None:
    """Why ``kv_path`` must not be published, or None when it may be.

    Checked BEFORE the store is opened: ``SqliteKVStore`` creates a missing file, and an
    empty store carries no stamp -- so opening first would publish nothing while
    leaving a stray database behind.
    """
    if os.environ.get("RENDER"):
        return (
            f"RENDER is set ({os.environ['RENDER']!r}). This script runs on a laptop and "
            "reaches prod explicitly; unset RENDER and re-run."
        )
    if not kv_path.is_file():
        return (
            f"{kv_path} does not exist. Build it with scripts/bootstrap_manual_kv.py and "
            "scripts/run_manual_refresh.py first."
        )
    return None


def stamp_refusal(stamp_raw: str | None, kv_path: Path) -> str | None:
    """Why a store whose provenance stamp reads ``stamp_raw`` must not be published."""
    if stamp_raw is None:
        return (
            f"{kv_path} carries no manual provenance stamp, so it is not a manual store "
            "(most likely the Yahoo baseline). Refusing: publishing it would push Yahoo-era "
            "data to prod with nothing marking it."
        )
    try:
        stamp = json.loads(stamp_raw)
    except json.JSONDecodeError:
        return f"{kv_path}: the manual provenance stamp is not valid JSON."
    if not isinstance(stamp, dict) or not stamp.get("seeded"):
        return (
            f"{kv_path} is bootstrapped but not seeded -- it still holds only the Yahoo "
            "copy. Run scripts/run_manual_refresh.py first."
        )
    return None


def write_backup(stats, backup_dir: Path) -> Path:
    """Save every prod value the publish is about to replace. Returns the file."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = backup_dir / f"upstash-before-publish-{stamp}.json.gz"
    payload = {
        "saved_at": datetime.now(UTC).isoformat(),
        "strings": stats.previous,
        "hash_fields": stats.previous_hash_fields,
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def _describe(stamp_raw: str, meta_raw: str | None) -> list[str]:
    from fantasy_baseball.web.season_data import unwrap_cache_envelope

    stamp = json.loads(stamp_raw)
    meta = unwrap_cache_envelope(json.loads(meta_raw)) if meta_raw else {}
    return [
        f"  rosters as of   : {stamp.get('roster_snapshot_date', '?')}",
        f"  standings as of : {stamp.get('standings_effective_date', '?')}",
        f"  last refresh    : {meta.get('last_refresh', '?') if isinstance(meta, dict) else '?'}",
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    kv_path = args.kv_path.resolve()

    refusal = source_refusal(kv_path)
    if refusal:
        print(f"REFUSING: {refusal}")
        return RC_REFUSED

    from fantasy_baseball.data.cache_keys import MANUAL_PROVENANCE_KEY, CacheKey, redis_key
    from fantasy_baseball.data.kv_store import SqliteKVStore, build_explicit_upstash_kv
    from fantasy_baseball.data.kv_sync import publish_local_to_remote

    local = SqliteKVStore(kv_path)
    stamp_raw = local.get(MANUAL_PROVENANCE_KEY)
    refusal = stamp_refusal(stamp_raw, kv_path)
    if refusal:
        print(f"REFUSING: {refusal}")
        return RC_REFUSED
    assert stamp_raw is not None  # stamp_refusal returned None

    remote = build_explicit_upstash_kv()
    host = urlparse(os.environ.get("UPSTASH_REDIS_REST_URL", "")).netloc or "?"
    meta_key = redis_key(CacheKey.META)
    print("=" * 72)
    print("PUBLISH MANUAL STORE -> PRODUCTION")
    print(f"  from : {kv_path}")
    print(f"  to   : Upstash {host}")
    print(f"  mode : {'DRY RUN (writes nothing)' if args.dry_run else 'LIVE'}")
    for line in _describe(stamp_raw, local.get(meta_key)):
        print(line)
    print("=" * 72)

    backup: list[Path] = []

    def _save(stats) -> None:
        backup.append(write_backup(stats, args.backup_dir))
        print(f"  saved the {stats.strings_changed} prod values being replaced to {backup[0]}")

    try:
        stats = publish_local_to_remote(
            local=local, remote=remote, dry_run=args.dry_run, before_write=_save
        )
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        if backup:
            print(f"  prod may be part-way; the replaced values are in {backup[0]}")
        return RC_FAILED

    print(f"\n  {stats.summary()}")
    if args.dry_run:
        print("\n  --dry-run: nothing written.")
        return RC_OK

    # Read back the two keys that decide what Render shows. A publish that silently
    # wrote nothing leaves the page on stale data that still renders.
    if remote.get(MANUAL_PROVENANCE_KEY) != stamp_raw or remote.get(meta_key) != local.get(
        meta_key
    ):
        print("\nFAILED: prod does not read back the stamp and meta just published.")
        return RC_FAILED
    print("\n  verified: prod carries this store's provenance stamp and meta.")
    print("  The trajectory board is separate: python scripts/push_trajectory_board.py")
    return RC_OK


if __name__ == "__main__":
    sys.exit(main())
