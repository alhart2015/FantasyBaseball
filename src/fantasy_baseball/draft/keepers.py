"""Keeper resolution shared between the legacy CLI and the dashboard.

Keepers in ``config/league.yaml`` are listed by display name (e.g.
``"Juan Soto"``). Resolving that to the right board row is non-trivial:
accents are normalized at write time, namesakes share the same
normalized form, and the player_id format depends on the projection
loader. The CLI in ``scripts/run_draft.py`` and the dashboard in
``web/app.py`` need the same matching rules — keep the logic here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fantasy_baseball.utils.name_utils import normalize_name


def index_by_normalized_name(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group board rows by normalized name for O(1) keeper lookup.

    Each group is the full list of rows that share a normalized name —
    callers tie-break by VAR (or whatever criterion fits). Falls back to
    re-normalizing ``name`` when the row doesn't carry ``name_normalized``.
    """
    by_norm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row.get("name_normalized") or normalize_name(row.get("name", ""))
        by_norm.setdefault(key, []).append(row)
    return by_norm


def find_keeper_match(
    name: str,
    by_norm: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Return the highest-VAR row matching ``name``, or ``None`` if absent.

    When two players share a normalized name (a real-world namesake
    collision in this league: "Jose Ramirez" the 3B and "Jose Ramirez"
    the reliever), VAR picks the keeper-worthy one — namesakes filtered
    by the projection loader for low playing time get near-zero VAR.
    """
    candidates = by_norm.get(normalize_name(name), [])
    if not candidates:
        return None
    # Use ``is None`` instead of ``r.get("var") or 0.0`` — the falsy form
    # treats var=0.0 the same as missing, which is fine in this max() but
    # bites in any sort/index context (see recs_integration._var_key).
    return max(candidates, key=lambda r: r["var"] if r.get("var") is not None else 0.0)
