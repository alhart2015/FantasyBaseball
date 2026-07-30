"""Player positions, keyed by normalized name, for netting SGP against the right
replacement level.

Composes three things the repo already owns rather than re-deriving them:
`web.season_data.read_cache_dict` for the live blob (via `CacheKey`, so a key typo
fails loudly) and `models.positions.Position.parse` for the slot tokens -- which
absorbs both the `"Util"` vs `"UTIL"` Yahoo casing and suffixed slots like
`"OF2"`. An unmatched slot is silently skipped by `calculate_var`, so normalizing
here is what stops a hitter being charged the wrong floor.

Only the merge is new: the live blob wins where both know a player, because Yahoo
eligibility moves during a season and the committed file is a point-in-time cache.
The file still contributes several hundred names the blob lacks, so both are read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.models.positions import Position
from fantasy_baseball.utils.name_utils import normalize_name

POSITIONS_JSON = Path(__file__).resolve().parents[3] / "data" / "player_positions.json"


def _slots(raw: object) -> list[str]:
    """Canonical slot names, dropping anything `Position` does not recognize."""
    if not isinstance(raw, list):
        return []
    out = []
    for token in raw:
        try:
            out.append(Position.parse(str(token)).value)
        except ValueError:
            continue
    return out


def _by_normalized_name(source: Mapping[str, object]) -> dict[str, list[str]]:
    return {
        normalize_name(str(name)): slots for name, raw in source.items() if (slots := _slots(raw))
    }


def _local_positions(path: Path) -> Mapping[str, object]:
    """The committed JSON, read as UTF-8.

    Not `data.yahoo_players.load_positions_cache`: that opens without an explicit
    encoding, so on Windows a name outside cp1252 raises UnicodeDecodeError and
    takes the whole ranking down. The committed file is ASCII today only because
    `scripts/fetch_positions_mlb.py` dumps with `ensure_ascii=True`, which is not a
    guarantee this module should rely on. A non-dict payload is ignored rather than
    raising AttributeError.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _live_positions() -> Mapping[str, object]:
    """The live blob, or empty if it is unreachable.

    `read_cache_dict` is imported lazily: it lives in the season-dashboard read
    path, and importing that at module scope would make this package unusable from
    an offline analysis script with no credentials, which is the main caller. The
    committed file already covers most players, so failure here is survivable.
    """
    try:
        from fantasy_baseball.web.season_data import read_cache_dict

        return read_cache_dict(CacheKey.POSITIONS) or {}
    except Exception:
        return {}


def load_positions(local_path: Path | None = None) -> dict[str, list[str]]:
    """Map normalized player name -> canonical eligible slots."""
    path = local_path if local_path is not None else POSITIONS_JSON
    merged: dict[str, list[str]] = {}
    if path.is_file():
        merged.update(_by_normalized_name(_local_positions(path)))
    merged.update(_by_normalized_name(_live_positions()))
    return merged
