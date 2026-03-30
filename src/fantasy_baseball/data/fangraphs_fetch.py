"""Fetch ROS projections from FanGraphs via their __NEXT_DATA__ JSON endpoint."""

import json
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Maps config system names to FanGraphs type= query parameter values
ROS_TYPE_CODES: dict[str, str] = {
    "steamer": "steamerr",
    "zips": "rzips",
    "atc": "ratcdc",
    "the-bat-x": "rthebatx",
    "oopsy": "roopsydc",
}

_COLUMN_RENAME: dict[str, str] = {
    "PlayerName": "Name",
    "playerid": "PlayerId",
    "xMLBAMID": "MLBAMID",
}

_NEXT_DATA_RE = re.compile(
    r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>',
    re.DOTALL,
)

_FG_BASE_URL = (
    "https://www.fangraphs.com/projections"
    "?type={type_code}&stats={stats_type}&pos=all&team=0&players=0&lg=all&pageitems=2000"
)


def _fetch_fangraphs_data(type_code: str, stats_type: str) -> list[dict]:
    """Fetch player projection rows from FanGraphs.

    Sends a GET request to the FanGraphs projections page, extracts the
    ``__NEXT_DATA__`` JSON blob embedded in the HTML, and returns the list of
    player dicts found under ``props.pageProps.data.data`` (or similar paths).

    Returns an empty list on any failure (network error, missing data, etc.).
    """
    url = _FG_BASE_URL.format(type_code=type_code, stats_type=stats_type)
    headers = {"User-Agent": _USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception:
        return []

    match = _NEXT_DATA_RE.search(resp.text)
    if not match:
        return []

    try:
        blob = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []

    # FanGraphs embeds projection rows at props.pageProps.data.data
    try:
        queries = blob["props"]["pageProps"]["data"]["data"]
    except (KeyError, TypeError):
        return []

    if not isinstance(queries, list):
        return []

    return queries


def _to_csv(players: list[dict], filepath: Path) -> None:
    """Convert a list of player dicts to a CSV file, renaming standard columns."""
    df = pd.DataFrame(players)
    rename = {k: v for k, v in _COLUMN_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def fetch_ros_projections(
    output_dir,
    systems: list[str],
    season_year: int,
    progress_cb=None,
) -> dict[str, str]:
    """Fetch ROS projections for each requested system and save as CSVs.

    Files are saved as::

        output_dir/{season_year}/ros/{today}/{system}-hitters.csv
        output_dir/{season_year}/ros/{today}/{system}-pitchers.csv

    A 2-second sleep is inserted between each HTTP request to be polite to
    FanGraphs servers.

    Parameters
    ----------
    output_dir:
        Root projections directory (e.g. ``data/projections``).
    systems:
        List of system names from config (e.g. ``["steamer", "zips"]``).
    season_year:
        The season year, used as part of the output path.
    progress_cb:
        Optional callable invoked with ``(system, stats_type, status)`` strings
        after each fetch completes.

    Returns
    -------
    dict mapping system name -> ``"ok"`` or ``"error: <reason>"``.
    """
    output_dir = Path(output_dir)
    today = date.today().isoformat()
    snapshot_dir = output_dir / str(season_year) / "ros" / today

    results: dict[str, str] = {}
    first_request = True

    for system in systems:
        type_code = ROS_TYPE_CODES.get(system)
        if type_code is None:
            results[system] = "error: unknown system"
            continue

        system_ok = True
        for stats_type in ("bat", "pit"):
            if not first_request:
                time.sleep(2)
            first_request = False

            players = _fetch_fangraphs_data(type_code, stats_type)

            if not players:
                results[system] = f"error: no data returned for {stats_type}"
                system_ok = False
                if progress_cb:
                    progress_cb(system, stats_type, "error")
                break

            suffix = "hitters" if stats_type == "bat" else "pitchers"
            csv_path = snapshot_dir / f"{system}-{suffix}.csv"
            try:
                _to_csv(players, csv_path)
            except Exception as exc:
                results[system] = f"error: {exc}"
                system_ok = False
                if progress_cb:
                    progress_cb(system, stats_type, "error")
                break

            if progress_cb:
                progress_cb(system, stats_type, "ok")

        if system_ok:
            results[system] = "ok"

    return results
