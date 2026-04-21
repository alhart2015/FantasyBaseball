"""One-shot migration: rewrite standings_history Redis hash into canonical shape.

Legacy entries have ``{"team", "r", "hr", ...}`` (no 'name', lowercase,
no effective_date wrapper). Canonical shape is what ``Standings.to_json``
emits. Idempotent: entries already in canonical shape are skipped.

Run once after the refactor is merged:

    python scripts/migrate_standings_history.py

Writes directly to prod Upstash via ``build_explicit_upstash_kv`` — the
whole point of this migration is to repair the remote hash that the
Render refresh reads. ``get_kv()`` would silently target the local
SQLite backend when ``RENDER`` is unset, making the script a no-op
against prod. Requires Upstash creds via the usual kv_store env vars.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fantasy_baseball.data import redis_store  # noqa: E402
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv  # noqa: E402
from fantasy_baseball.models.standings import (  # noqa: E402
    CategoryStats,
    Standings,
    StandingsEntry,
)


def _from_legacy_json(payload: dict, *, snapshot_date: str) -> Standings:
    """Parse the legacy ``{"teams": [{"team", lowercase_stat_keys}]}`` shape."""
    if "teams" not in payload:
        raise ValueError(f"{snapshot_date}: payload missing 'teams' wrapper")
    rows = payload["teams"]
    entries: list[StandingsEntry] = []
    for row in rows:
        if "team" not in row:
            raise ValueError(f"{snapshot_date}: row missing 'team' field — not legacy shape either")
        stats = CategoryStats(
            r=float(row["r"]) if row.get("r") is not None else 0.0,
            hr=float(row["hr"]) if row.get("hr") is not None else 0.0,
            rbi=float(row["rbi"]) if row.get("rbi") is not None else 0.0,
            sb=float(row["sb"]) if row.get("sb") is not None else 0.0,
            avg=float(row["avg"]) if row.get("avg") is not None else 0.0,
            w=float(row["w"]) if row.get("w") is not None else 0.0,
            k=float(row["k"]) if row.get("k") is not None else 0.0,
            sv=float(row["sv"]) if row.get("sv") is not None else 0.0,
            era=float(row["era"]) if row.get("era") is not None else 99.0,
            whip=float(row["whip"]) if row.get("whip") is not None else 99.0,
        )
        entries.append(
            StandingsEntry(
                team_name=row["team"],
                team_key=row.get("team_key") or "",
                rank=int(row.get("rank") or 0),
                stats=stats,
                yahoo_points_for=None,
            )
        )
    return Standings(effective_date=date.fromisoformat(snapshot_date), entries=entries)


def rewrite_hash(client) -> dict[str, int]:
    """Walk the standings_history hash and rewrite legacy entries.

    Returns a stats dict: {"rewritten": N, "skipped": N, "errors": N}.
    """
    raw_map = client.hgetall(redis_store.STANDINGS_HISTORY_KEY)
    stats = {"rewritten": 0, "skipped": 0, "errors": 0}

    for snap_date, raw in raw_map.items():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[{snap_date}] corrupt JSON — skipping")
            stats["errors"] += 1
            continue

        if not isinstance(payload, dict):
            print(f"[{snap_date}] not a dict payload — skipping")
            stats["errors"] += 1
            continue

        try:
            Standings.from_json(payload)
            print(f"[{snap_date}] already canonical — skip")
            stats["skipped"] += 1
            continue
        except (ValueError, KeyError, TypeError):
            # Partially-canonical rows (missing team_key/rank) raise KeyError;
            # malformed types raise TypeError. Fall through to the legacy parser
            # in all cases so a single bad row does not halt the migration.
            pass

        # Intermediate shape: rows already canonical ({"name", "stats": {...}})
        # but outer effective_date wrapper missing. Inject from hash key and
        # retry canonical parse before falling through to the flat-row parser.
        s: Standings | None = None
        if "effective_date" not in payload and "teams" in payload:
            try:
                s = Standings.from_json({"effective_date": snap_date, **payload})
            except (ValueError, KeyError, TypeError):
                s = None

        if s is None:
            try:
                s = _from_legacy_json(payload, snapshot_date=snap_date)
            except (ValueError, KeyError, TypeError) as e:
                print(f"[{snap_date}] legacy parse failed: {e} — SKIPPING (fix manually)")
                stats["errors"] += 1
                continue

        client.hset(
            redis_store.STANDINGS_HISTORY_KEY,
            snap_date,
            json.dumps(s.to_json()),
        )
        print(f"[{snap_date}] rewritten")
        stats["rewritten"] += 1

    return stats


def main() -> int:
    try:
        client = build_explicit_upstash_kv()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    print("Migrating standings_history on prod Upstash...")
    stats = rewrite_hash(client)
    print()
    print(f"Rewritten: {stats['rewritten']}")
    print(f"Skipped (already canonical): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    return 0 if stats["errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
