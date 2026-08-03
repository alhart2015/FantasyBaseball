"""Who owns whom, from the live roster blobs.

Two cache keys together cover the league, and they are NOT the same shape:

    cache:roster        a BARE LIST -- your own team, with no team name on it
    cache:opp_rosters   a dict {team_name: [player, ...]} -- the other nine

Getting that asymmetry wrong drops your own roster silently, which reads as "you own
nobody" rather than as an error.

**Joins fall back to (normalized name, player_type).** Roster blobs carry a Yahoo
`player_id` and a `player_type` but no `mlbam_id` (#284), while every board in this repo
is MLBAM-keyed. Two different players sharing a normalized name and type therefore
collapse onto one owner. That residual is irreducible here -- the fix is populating
`mlbam_id` at Yahoo ingest, not more matching logic -- so callers are given
`unmatched_names` and should say what they could not place rather than let a failed join
read as "nobody owns him".

`parse_rosters` is separated from `live_rosters` because the second crosses the RENDER
gate to reach prod Upstash, which `build_explicit_upstash_kv` refuses under pytest. The
shape handling is the part worth testing, so it is testable without a network.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.name_utils import normalize_name


@dataclass(frozen=True)
class RosterSpot:
    """One player on one fantasy roster."""

    name: str
    #: Accent-stripped, lowercased. The join key, paired with `player_type`.
    normalized: str
    player_type: str
    team: str
    #: Yahoo's own id. Carried because it IS unique where the name is not, so a caller
    #: that later gets mlbam_id at ingest can upgrade the join without re-reading this.
    yahoo_id: str
    #: "", "IL10", "DTD", ... -- an injured player is still owned.
    status: str
    selected_position: str


def parse_rosters(
    roster_blob: Any, opp_blob: Any, my_team: str
) -> list[RosterSpot]:
    """Flatten both blobs into one list. Pure -- no network, no environment."""
    spots: list[RosterSpot] = []
    # (payload, team-if-the-payload-is-a-bare-list)
    for payload, mine in ((opp_blob, None), (roster_blob, my_team)):
        if payload is None:
            continue
        data = payload.get("_data", payload) if isinstance(payload, dict) else payload
        # opp_rosters is {team: [players]}; roster is a BARE LIST -- your own team.
        groups = data.items() if isinstance(data, dict) else [(mine, data)]
        for team, players in groups:
            if not isinstance(players, list) or team is None:
                continue
            for p in players:
                if not isinstance(p, dict) or not p.get("name"):
                    continue
                spots.append(
                    RosterSpot(
                        name=str(p["name"]),
                        normalized=normalize_name(str(p["name"])),
                        player_type=str(p.get("player_type", "")),
                        team=str(team),
                        yahoo_id=str(p.get("player_id", "")),
                        status=str(p.get("status", "") or ""),
                        selected_position=str(p.get("selected_position", "") or ""),
                    )
                )
    return spots


def owner_map(spots: list[RosterSpot]) -> dict[tuple[str, str], str]:
    """(normalized name, player_type) -> owning team.

    Collisions collapse, by construction -- see the module docstring. Later spots win,
    which is arbitrary; that is the #284 residual, not a decision worth encoding.
    """
    return {(s.normalized, s.player_type): s.team for s in spots}


def live_rosters(my_team: str, *, project_root: Path | None = None) -> list[RosterSpot]:
    """Read both roster blobs from PROD Upstash and flatten them.

    Crosses the RENDER gate deliberately: `get_kv()` off Render returns the local SQLite
    mirror, which is only as fresh as the last sync, and roster membership is exactly the
    kind of live state that goes stale silently.
    """
    os.environ["RENDER"] = "true"
    from dotenv import load_dotenv

    root = project_root or Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env")

    from .cache_keys import CacheKey, redis_key
    from .kv_store import build_explicit_upstash_kv

    kv = build_explicit_upstash_kv()

    def read(key: CacheKey) -> Any:
        raw = kv.get(redis_key(key))
        if raw is None:
            return None
        return json.loads(raw) if isinstance(raw, str) else raw

    return parse_rosters(read(CacheKey.ROSTER), read(CacheKey.OPP_ROSTERS), my_team)
