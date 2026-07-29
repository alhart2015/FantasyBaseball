"""Player positions, keyed by normalized name, for netting SGP against the right
replacement level.

Prefers the live `cache:positions` blob and falls back to the committed
`data/player_positions.json`. Two shapes have to be reconciled: the cache is
already keyed by normalized name, the file by display name, and the file spells
the utility slot `"Util"` where `sgp.replacement` expects `"UTIL"`. An unmatched
slot is silently skipped by `calculate_var`, so a case mismatch would quietly
charge a hitter the wrong floor rather than fail.
"""

from __future__ import annotations

import json
from pathlib import Path

from fantasy_baseball.utils.name_utils import normalize_name

_LOCAL = Path(__file__).resolve().parents[3] / "data" / "player_positions.json"


def _clean(slots: object) -> list[str]:
    if not isinstance(slots, list):
        return []
    return [str(slot).strip().upper() for slot in slots if str(slot).strip()]


def load_positions(local_path: Path | None = None) -> dict[str, list[str]]:
    """Map normalized player name -> uppercase eligible slots.

    The live blob wins where both have a player: Yahoo eligibility moves during a
    season and the committed file is a point-in-time cache.
    """
    merged: dict[str, list[str]] = {}
    path = local_path if local_path is not None else _LOCAL
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for name, slots in raw.items():
                cleaned = _clean(slots)
                if cleaned:
                    merged[normalize_name(str(name))] = cleaned
    merged.update(_load_cached())
    return merged


def _load_cached() -> dict[str, list[str]]:
    """Positions from the KV blob, or nothing if it is unreachable.

    Import-local and failure-tolerant: this module is used by an offline analysis
    script that must still run with no network and no credentials.
    """
    try:
        from fantasy_baseball.data.kv_store import get_kv

        blob = get_kv().get("cache:positions")
    except Exception:
        return {}
    if not blob:
        return {}
    payload = json.loads(blob) if isinstance(blob, str) else blob
    if isinstance(payload, dict):
        payload = payload.get("_data", payload)
    if not isinstance(payload, dict):
        return {}
    out: dict[str, list[str]] = {}
    for name, slots in payload.items():
        cleaned = _clean(slots)
        if cleaned:
            out[normalize_name(str(name))] = cleaned
    return out
