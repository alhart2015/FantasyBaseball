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
`mlbam_id` at Yahoo ingest, not more matching logic. Callers join on that key and
should NAME what they could not place rather than let a failed join read as "nobody owns
him".

`parse_rosters` is separated from `live_rosters` because the second reaches prod Upstash,
which `build_explicit_upstash_kv` refuses to do under pytest. The shape handling is the
part worth testing, so it is testable without a network.

**Prod Upstash is the wrong store in manual mode.** The Yahoo-free pipeline isolates
itself in a whole separate KV file, and its rosters are hand-transcribed and current;
prod is Yahoo-vintage and, while Yahoo auth is down, up to a month stale. Serving prod
rosters into a page otherwise built from manual data mixes two vintages that both look
plausible. So `live_rosters` reads the MANUAL store's own roster snapshot instead --
same-vintage data, which is what the caller actually wanted -- and refuses only when
that store has no snapshot to give (see `ManualStoreRefused`).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from ..utils.name_utils import normalize_name

log = logging.getLogger(__name__)


class ManualStoreRefused(RuntimeError):
    """The manual KV store is active and has no roster snapshot to serve.

    Raised INSTEAD OF falling back to prod Upstash, which is the whole point: a
    month-stale Yahoo roster spliced into an otherwise-manual page looks plausible
    and is wrong. "No roster data" is the honest answer; "you own nobody" is not.

    The state is reachable: `scripts/bootstrap_manual_kv.py` creates the store and
    stamps it manual before any seed has run, so a dashboard opened between those
    two steps lands here. The message says to seed.

    A `RuntimeError` subclass, so the season dashboard's trajectory route catches it
    with the handler it already has for a failed roster read and renders the board
    unmarked. `scripts/trajectory_board.py` also catches it, but RE-RAISES under
    `--by-team`, where a team view with no rosters would render nothing at all -- a
    loud stop beats an empty table that reads as "every team is empty".
    """


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


def parse_rosters(roster_blob: Any, opp_blob: Any, my_team: str) -> list[RosterSpot]:
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
                    )
                )
    return spots


def owner_map(spots: list[RosterSpot]) -> dict[tuple[str, str], str]:
    """(normalized name, player_type) -> owning team.

    Collisions collapse, by construction -- see the module docstring. Later spots win,
    which is arbitrary; that is the #284 residual, not a decision worth encoding.
    """
    return {(s.normalized, s.player_type): s.team for s in spots}


def manual_store_active() -> bool:
    """True when this process's KV store holds hand-transcribed (manual) data.

    Reads the store-level breadcrumb `manual.seed` stamps on a seeded manual store,
    rather than sniffing `FANTASY_LOCAL_KV_PATH`. That precision is the whole point:
    every pytest run and every ad-hoc `FANTASY_LOCAL_KV_PATH` also redirects the
    store, and none of those are manual runs -- keying on the env var would refuse
    reads that are perfectly legitimate. One breadcrumb read, from a local SQLite
    file the caller's process has already opened.

    Never consults a remote store: on Render `get_kv()` IS production Upstash, which
    is the store `live_rosters` wants, and a manual run refuses to start with
    `RENDER` set at all -- so the answer there is False without a round trip.
    """
    from .kv_store import get_kv, is_remote

    if is_remote():
        return False

    from ..manual.seed import PROVENANCE_KEY

    try:
        return get_kv().get(PROVENANCE_KEY) is not None
    except (OSError, sqlite3.Error):
        # Fail OPEN on a STORAGE fault, deliberately. An unreadable store is not
        # evidence of manual mode, and refusing on it would take the ordinary Yahoo
        # caller down for an unrelated fault. In manual mode the store is a local
        # SQLite file the page has already read successfully to get this far.
        #
        # Narrow on purpose. `except Exception` would swallow a TypeError or
        # AttributeError out of the probe itself -- a DEFECT in the detection logic,
        # answering "not manual" for a store that is manual, which is exactly how
        # month-stale prod rosters end up spliced into a manual page. That failure
        # has to be loud. Off Render `get_kv()` is a local SQLite store, so these
        # two are the fault modes it actually has.
        log.warning("live_rosters: manual-store check failed; assuming Yahoo mode")
        return False


def live_rosters(my_team: str) -> list[RosterSpot]:
    """Read both roster blobs from PROD Upstash and flatten them.

    Delegates to `manual_rosters` when this process is running against the manual KV
    store. The Upstash reach below is right for the Yahoo caller and wrong for that
    one: it would splice month-stale prod rosters into a page whose other half is
    fresh manual data, with both halves looking plausible. The manual store already
    holds the same answer at the page's own vintage; `ManualStoreRefused` comes back
    only when it does not.

    `build_explicit_upstash_kv` rather than `get_kv()`, deliberately: off Render `get_kv`
    returns the local SQLite mirror, which is only as fresh as the last sync, and roster
    membership is exactly the kind of live state that goes stale silently -- a trade since
    the last sync would put a player on the wrong team with nothing looking wrong.

    It does NOT set `RENDER`. Several scripts do that before calling this constructor,
    which is cargo: the env gate lives in `get_kv` alone and this path never consults it,
    so the assignment cannot change what is returned. What it CAN do is steer `get_kv` --
    a process-wide singleton -- for whatever runs next, which for a library function
    reachable from the web app is a side effect with no upside. `.env` loading is likewise
    already handled inside `_build_upstash_kv`.
    """
    if manual_store_active():
        return manual_rosters(my_team)

    from .cache_keys import CacheKey, redis_key
    from .kv_store import build_explicit_upstash_kv

    kv = build_explicit_upstash_kv()

    def decode(raw: Any) -> Any:
        if raw is None:
            return None
        return json.loads(raw) if isinstance(raw, str) else raw

    # One round trip. The two blobs are independent and `mget` is on the KVStore protocol
    # for both backends, so fetching them separately just bought an extra network wait.
    roster_raw, opp_raw = kv.mget(redis_key(CacheKey.ROSTER), redis_key(CacheKey.OPP_ROSTERS))
    return parse_rosters(decode(roster_raw), decode(opp_raw), my_team)


def manual_rosters(my_team: str) -> list[RosterSpot]:
    """Ownership from the MANUAL store's own latest roster snapshot.

    The manual pipeline seeds `weekly_rosters_history` from `data/manual/rosters.yaml`
    for every team in the league, so the answer `live_rosters` wants is already in the
    store this process is pointed at -- at the same vintage as everything else on the
    page. Reading it is strictly better than refusing: the By-team keeper view, which
    needs ownership for all ten teams, works in manual mode instead of falling through
    to an empty block list and silently rendering the league board.

    `my_team` is accepted and unused, deliberately: every row in the hash carries its
    own `team`, including yours, so unlike the Upstash pair there is no bare list that
    needs a name attached. Keeping the parameter lets this stand in for `live_rosters`
    at the same call signature.

    `player_type` is derived from the eligibility string -- see
    `player_type_for_positions`. Roster rows carry no type field, and a two-way player
    is two rows, so deriving it is what keeps `name::player_type` intact.

    Raises:
        ManualStoreRefused: the store holds no roster snapshot (bootstrapped but not
            yet seeded). Falling back to prod Upstash here would be the exact vintage
            mix this whole path exists to prevent.
    """
    from ..models.positions import player_type_for_positions
    from .kv_store import get_kv
    from .redis_store import get_latest_weekly_rosters

    del my_team  # every row names its own team; see the docstring

    rows = get_latest_weekly_rosters(get_kv())
    if not rows:
        message = (
            "live_rosters: this process is running against the hand-transcribed manual "
            "KV store and that store holds no roster snapshot. Refusing to fall back to "
            "production Upstash -- those rosters are Yahoo-vintage and can be a month "
            "stale, and mixing them into a page built from manual data would look "
            "plausible and be wrong. Run scripts/run_manual_refresh.py to seed it."
        )
        log.error(message)
        raise ManualStoreRefused(message)

    spots: list[RosterSpot] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("player_name")
        team = row.get("team")
        if not name or not team:
            continue
        spots.append(
            RosterSpot(
                name=str(name),
                normalized=normalize_name(str(name)),
                player_type=player_type_for_positions(str(row.get("positions", ""))),
                team=str(team),
                yahoo_id=str(row.get("yahoo_id", "") or ""),
                status=str(row.get("status", "") or ""),
            )
        )
    log.info("live_rosters: served %d spots from the manual store", len(spots))
    return spots
