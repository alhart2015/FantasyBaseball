"""Fetch and store prospect pedigree (#433): MLB Pipeline rankings and draft picks.

Hitters with little MLB history are where our projections trail FanGraphs most. Their
minor-league lines are in the store (#435); this adds what scouts thought of them.

* Rankings: MLB Pipeline's Top 100 and each club's Top 30, 2011 on (the first years are
  a Top 50 and Top 10s), from MLB's content API. One row per player per list, in list
  order. Only the rank, the list and the player id are kept: the rest of an entry
  (grades, ETA, position, bio) is the player's page as it reads *today*, which would
  leak later seasons into a backtest. A season's lists are its preseason ones for
  finished seasons, but the current season's are edited as players graduate, so they
  aren't the preseason lists. (Checked against MLB debuts: each 2011-2025 list still
  holds 56-120 hitters who debuted that season; 2026's holds 28, and 2 who debuted
  earlier against 30-80 in other years.)
* Draft: every pick of the June (Rule 4) draft, from the MLB Stats API. One row per pick,
  so a player drafted twice (out of high school, then college) has two rows.

Layout (under the store root, read through :func:`store.connect`)::

    prospect_rankings/YYYY.parquet   season, list, rank, list_size, player_id
    draft/YYYY.parquet               one row per pick
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from fantasy_baseball.pitch_data.store import _write_parquet

logger = logging.getLogger(__name__)

_CONTENT_URL = "https://dapi.cms.mlbinfra.com/v2/content/en-us"
_DRAFT_URL = "https://statsapi.mlb.com/api/v1/draft"
FIRST_RANKINGS_SEASON = 2011
# Club slugs in MLB Pipeline's list names (sel-pr-<season>-<slug>); old names alias these.
TEAM_LISTS = (
    "angels", "astros", "athletics", "bluejays", "braves", "brewers", "cardinals", "cubs",
    "dbacks", "dodgers", "giants", "guardians", "mariners", "marlins", "mets", "nationals",
    "orioles", "padres", "phillies", "pirates", "rangers", "rays", "redsox", "reds",
    "rockies", "royals", "tigers", "twins", "whitesox", "yankees",
)  # fmt: skip
TOP100 = "top100"
# A few seasons' lists go by another name; tried when the usual one isn't there.
_LIST_ALIASES = {"dbacks": ("diamondbacks",), "bluejays": ("blue-jays",)}


def rankings_path(root: Path, season: int) -> Path:
    return root / "prospect_rankings" / f"{season}.parquet"


def draft_path(root: Path, year: int) -> Path:
    return root / "draft" / f"{year}.parquet"


def _get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    import requests

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    out: dict[str, Any] = resp.json()
    return out


def _list_items(season: int, name: str) -> list[dict[str, Any]]:
    import requests

    for slug in (name, *_LIST_ALIASES.get(name, ())):
        try:
            data = _get_json(f"{_CONTENT_URL}/sel-pr-{season}-{slug}", {"$limit": 100})
            break
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 404:
                raise
    else:
        raise RuntimeError(f"no {season} list for {name}")
    items: list[dict[str, Any]] = data.get("items", [])
    if len(items) >= 100 and data.get("pagination", {}).get("nextUrl"):
        raise RuntimeError(f"sel-pr-{season}-{name}: more than 100 entries")
    return items


def rankings_rows(season: int, name: str, items: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per ranked player, rank = list order (1 = best). Entries without a
    player id (none seen so far) are dropped with a warning, keeping the others' ranks."""
    ids = [item.get("fields", {}).get("playerId") for item in items]
    missing = sum(i is None for i in ids)
    if missing:
        logger.warning("sel-pr-%s-%s: %s entries without a player id", season, name, missing)
    return pd.DataFrame(
        {
            "season": season,
            "list": name,
            "rank": [r for r, i in enumerate(ids, start=1) if i is not None],
            "list_size": len(items),
            "player_id": [int(i) for i in ids if i is not None],
        }
    )


def fetch_rankings_season(root: Path, season: int, *, refresh: bool = False) -> int:
    """Write the season's Top 100 and club Top 30s; returns the row count. A season on
    disk is kept unless ``refresh``."""
    path = rankings_path(root, season)
    if path.exists() and not refresh:
        return len(pd.read_parquet(path))
    frames = [rankings_rows(season, TOP100, _list_items(season, TOP100))]
    for team in TEAM_LISTS:
        frames.append(rankings_rows(season, team, _list_items(season, team)))
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise RuntimeError(f"{season}: no prospect lists")
    _write_parquet(df, path)
    return len(df)


_DRAFT_DTYPES = {
    "year": "int64",
    "player_id": "int64",
    "round": "string",
    "pick_number": "Int64",
    "pick_value": "Int64",
    "signing_bonus": "Int64",
    "pre_draft_rank": "Int64",
    "school": "string",
    "birth_date": "string",
    "team_id": "Int64",
}


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def draft_rows(year: int, data: dict[str, Any]) -> pd.DataFrame:
    """One row per June-draft pick with a player id."""
    rows = []
    for rnd in data.get("drafts", {}).get("rounds", []):
        for pick in rnd.get("picks", []):
            person = pick.get("person") or {}
            if "id" not in person or pick.get("isPass"):
                continue
            if pick.get("draftType", {}).get("code") != "JR":
                continue
            rows.append(
                {
                    "year": year,
                    "player_id": int(person["id"]),
                    "round": str(pick.get("pickRound")),
                    "pick_number": _int(pick.get("pickNumber")),
                    "pick_value": _int(pick.get("pickValue")),
                    "signing_bonus": _int(pick.get("signingBonus")),
                    "pre_draft_rank": _int(pick.get("rank")),
                    "school": (pick.get("school") or {}).get("name"),
                    "birth_date": person.get("birthDate"),
                    "team_id": _int((pick.get("team") or {}).get("id")),
                }
            )
    # Fixed dtypes, so a year where a column is all blank still stacks with the others.
    return pd.DataFrame(rows, columns=list(_DRAFT_DTYPES)).astype(_DRAFT_DTYPES)


def fetch_draft_year(root: Path, year: int, *, refresh: bool = False) -> int:
    """Write the year's June draft; returns the pick count. A year on disk is kept
    unless ``refresh``."""
    path = draft_path(root, year)
    if path.exists() and not refresh:
        return len(pd.read_parquet(path))
    df = draft_rows(year, _get_json(f"{_DRAFT_URL}/{year}"))
    if df.empty:
        raise RuntimeError(f"{year}: no draft picks")
    _write_parquet(df, path)
    return len(df)
