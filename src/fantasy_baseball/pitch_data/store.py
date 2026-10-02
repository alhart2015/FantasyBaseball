"""Fetch and store raw pitch-level data: every Statcast pitch, sprint speed, batting order.

Everything lands as parquet under one root (``data/pitch_data/`` by default) and is
kept raw -- all Savant columns, no renames, no filtering -- so later steps of #399
decide what to use. Query it through :func:`connect`, which exposes DuckDB views.

Layout::

    pitches/season=YYYY/<start>_<end>.parquet   one file per weekly chunk
    lineups/YYYY.parquet                         one row per player per lineup slot per
                                                 game: batting order + box-score batting line
    sprint_speed/YYYY.parquet                    Savant sprint-speed leaderboard
    schedule/YYYY.parquet                        first/last scheduled regular-season date

Resumability: a file is final -- never fetched again -- only if it was *written* more
than ``SETTLE_DAYS`` after the end of the dates it covers. A file written earlier (a
mid-season or next-day run) is re-fetched on every run until a settled copy replaces it,
so late Savant corrections and the rest of a season are never frozen out. A lineup file
missing a current column (e.g. one written before the batting line was stored) is not
final either.

Pitch chunks are fixed weeks counted from opening day and capped at the last scheduled
regular-season date, never at today, so a chunk's file name does not change while its
week is still being played. Writing a chunk also deletes any other file in that season
whose dates overlap it (e.g. after the schedule moves the season's last date).

Completeness: a weekly pitch chunk is only written when every completed regular-season
game the MLB schedule lists for those dates appears in it. Savant can return a short or
empty CSV without raising, and a silently short chunk would otherwise be marked final.

The pitch window is the regular season (first to last regular-season date). Rows Savant
returns inside that window for other game types (e.g. spring games around an overseas
opening series) are kept; filter on ``game_type = 'R'`` when reading.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from fantasy_baseball.analysis.game_logs import FULL_HITTER_FIELDS, full_hitter_line
from fantasy_baseball.data.mlb_game_logs import (
    _fetch_boxscore,
    _fetch_season_games,
    _game_context,
    _is_regular_final,
)
from fantasy_baseball.streaks.data.statcast import chunk_date_range

logger = logging.getLogger(__name__)

SETTLE_DAYS = 2
FETCH_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 30
_BOXSCORE_WORKERS = 8

Games = list[dict[str, Any]]


def is_settled(end: date, today: date) -> bool:
    """True once data ending on ``end`` can no longer change."""
    return end < today - timedelta(days=SETTLE_DAYS)


def is_final(path: Path, end: date, required_columns: Iterable[str] = ()) -> bool:
    """True if ``path`` exists, was written after data ending on ``end`` had settled, and
    has every ``required_columns`` column -- a file from before a column was added is
    not final, so it gets re-fetched."""
    if not path.exists():
        return False
    written = date.fromtimestamp(path.stat().st_mtime)
    if not is_settled(end, written):
        return False
    return set(required_columns) <= set(pq.read_schema(path).names)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file so an interrupted run never leaves a half file that looks final."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


# --- schedule -------------------------------------------------------------------


def season_window(games: Games) -> tuple[date, date]:
    """First and last regular-season date in a schedule."""
    dates = [date.fromisoformat(_game_context(g)[2]) for g in games if g.get("gameType") == "R"]
    if not dates:
        raise ValueError("schedule has no regular-season games")
    return min(dates), max(dates)


def was_played(game: dict[str, Any]) -> bool:
    """A regular-season game that was actually played.

    The schedule marks cancelled and postponed entries ``abstractGameState == "Final"``
    too; only ``codedGameState == "F"`` (Final, or Completed Early) has pitches.
    """
    return _is_regular_final(game) and game.get("status", {}).get("codedGameState") == "F"


def final_game_pks_by_date(games: Games) -> dict[date, set[int]]:
    """Played regular-season gamePks, keyed by official date.

    A suspended-and-resumed game is listed under both dates with one gamePk, and which
    date Savant files its pitches under is not pinned down, so such games are left out
    of the completeness check rather than risk a chunk that can never pass.
    """
    dates_by_pk: dict[int, set[date]] = {}
    for g in games:
        if was_played(g):
            game_pk, _, d = _game_context(g)
            dates_by_pk.setdefault(game_pk, set()).add(date.fromisoformat(d))
    out: dict[date, set[int]] = {}
    for game_pk, dates in dates_by_pk.items():
        if len(dates) == 1:
            out.setdefault(next(iter(dates)), set()).add(game_pk)
    return out


# --- pitches --------------------------------------------------------------------


def pitch_chunk_path(root: Path, season: int, start: date, end: date) -> Path:
    return root / "pitches" / f"season={season}" / f"{start.isoformat()}_{end.isoformat()}.parquet"


def _remove_overlapping_chunks(path: Path, start: date, end: date) -> None:
    """Delete other chunk files in ``path``'s season that cover any of start..end."""
    for other in path.parent.glob("*.parquet"):
        if other == path:
            continue
        o_start, o_end = (date.fromisoformat(x) for x in other.stem.split("_"))
        if o_start <= end and start <= o_end:
            logger.info("pitches: removing overlapping chunk %s", other.name)
            other.unlink()


def missing_game_pks(pitches: pd.DataFrame, expected: Iterable[int]) -> set[int]:
    """Scheduled final games with no pitch rows."""
    have = set(pitches["game_pk"].astype(int)) if "game_pk" in pitches.columns else set()
    return set(expected) - have


def _statcast_range(start: date, end: date) -> pd.DataFrame:
    from pybaseball import statcast

    df: pd.DataFrame = statcast(start.isoformat(), end.isoformat(), verbose=False)
    return df


def _fetch_with_retries(
    fetch: Callable[[date, date], pd.DataFrame], start: date, end: date
) -> pd.DataFrame | None:
    """Savant now and then answers with a malformed CSV that pybaseball cannot parse.

    Retry a few times; None means every attempt failed.
    """
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            return fetch(start, end)
        except Exception:
            logger.warning(
                "pitches %s..%s: fetch attempt %d/%d failed",
                start,
                end,
                attempt,
                FETCH_ATTEMPTS,
                exc_info=True,
            )
            if attempt < FETCH_ATTEMPTS:
                time.sleep(RETRY_SLEEP_SECONDS)
    return None


def fetch_pitches_season(
    root: Path,
    season: int,
    today: date,
    *,
    games: Games | None = None,
    fetch: Callable[[date, date], pd.DataFrame] = _statcast_range,
) -> dict[str, int]:
    """Fetch every weekly pitch chunk of ``season`` that is not already final on disk.

    Returns counts of chunks written / skipped / incomplete. An incomplete chunk (games
    missing, or every fetch attempt failed) is logged and left unwritten so the next run
    retries it.
    """
    games = games if games is not None else _fetch_season_games(season)
    start, end = season_window(games)
    expected_by_date = final_game_pks_by_date(games)
    summary = {"written": 0, "skipped": 0, "incomplete": 0, "rows": 0}

    for chunk_start, chunk_end in chunk_date_range(start, end):
        if chunk_start > today:
            break
        path = pitch_chunk_path(root, season, chunk_start, chunk_end)
        if is_final(path, chunk_end):
            summary["skipped"] += 1
            continue
        expected = {
            pk for d, pks in expected_by_date.items() if chunk_start <= d <= chunk_end for pk in pks
        }
        df = _fetch_with_retries(fetch, chunk_start, min(chunk_end, today))
        if df is None:
            summary["incomplete"] += 1
            continue
        missing = missing_game_pks(df, expected)
        if missing:
            logger.warning(
                "pitches %s..%s: %d of %d scheduled games missing (%s); not written",
                chunk_start,
                chunk_end,
                len(missing),
                len(expected),
                sorted(missing)[:5],
            )
            summary["incomplete"] += 1
            continue
        if df.empty:
            continue
        _write_parquet(df, path)
        _remove_overlapping_chunks(path, chunk_start, chunk_end)
        summary["written"] += 1
        summary["rows"] += len(df)
        logger.info("pitches %s..%s: %d rows", chunk_start, chunk_end, len(df))
    return summary


# --- lineups --------------------------------------------------------------------


# A lineup file without these is from before #402 and gets re-fetched.
LINEUP_STAT_COLUMNS = tuple(FULL_HITTER_FIELDS.values())


def lineup_path(root: Path, season: int) -> Path:
    return root / "lineups" / f"{season}.parquet"


def lineup_rows(boxscore: dict[str, Any], game_pk: int, game_date: str) -> list[dict[str, Any]]:
    """One row per player who holds a batting-order slot in this box score.

    ``battingOrder`` is a 3-digit string: hundreds digit = lineup spot (1-9), the rest =
    substitution index (0 = the starter, 1 = first player in that spot after him).
    The player's batting line for the game rides along (``FULL_HITTER_FIELDS`` columns);
    a missing field is 0, e.g. a pinch runner who never batted.
    """
    rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        team = boxscore.get("teams", {}).get(side, {})
        team_id = team.get("team", {}).get("id")
        for entry in team.get("players", {}).values():
            order = entry.get("battingOrder")
            person_id = entry.get("person", {}).get("id")
            if order is None or person_id is None:
                continue
            order_int = int(order)
            batting = entry.get("stats", {}).get("batting") or {}
            rows.append(
                {
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "team_id": team_id,
                    "is_home": side == "home",
                    "player_id": int(person_id),
                    "batting_order": order_int,
                    "lineup_spot": order_int // 100,
                    "sub_index": order_int % 100,
                    "position": entry.get("position", {}).get("abbreviation"),
                    **full_hitter_line(batting),
                }
            )
    return rows


def fetch_lineups_season(
    root: Path,
    season: int,
    *,
    games: Games | None = None,
    fetch_boxscore: Callable[[int], dict[str, Any]] = _fetch_boxscore,
) -> int | None:
    """Fetch batting order for every played regular-season game. Returns rows written.

    Skips (returns None) when the season's file is already final (see :func:`is_final`). Any box-score failure
    raises without writing, so the next run retries the season whole.
    """
    games = games if games is not None else _fetch_season_games(season)
    path = lineup_path(root, season)
    _, season_end = season_window(games)
    if is_final(path, season_end, LINEUP_STAT_COLUMNS):
        return None
    finals = [_game_context(g) for g in games if was_played(g)]
    # A suspended game is listed under both its start and resume dates with one gamePk;
    # fetch it once, dated to the later (completion) date.
    by_pk: dict[int, str] = {}
    for pk, _, d in finals:
        by_pk[pk] = max(d, by_pk.get(pk, d))

    def _one(item: tuple[int, str]) -> list[dict[str, Any]]:
        pk, d = item
        return lineup_rows(fetch_boxscore(pk), pk, d)

    with ThreadPoolExecutor(max_workers=_BOXSCORE_WORKERS) as pool:
        rows = [r for game_rows in pool.map(_one, by_pk.items()) for r in game_rows]
    df = pd.DataFrame(rows)
    _write_parquet(df, path)
    logger.info("lineups %s: %d games, %d rows", season, len(by_pk), len(df))
    return len(df)


# --- sprint speed ---------------------------------------------------------------


def sprint_speed_path(root: Path, season: int) -> Path:
    return root / "sprint_speed" / f"{season}.parquet"


def _savant_sprint_speed(season: int) -> pd.DataFrame:
    from pybaseball import statcast_sprint_speed

    # min_opp=1 keeps every runner; competitive_runs is in the table to filter on later.
    df: pd.DataFrame = statcast_sprint_speed(season, min_opp=1)
    return df


def fetch_sprint_speed_season(
    root: Path,
    season: int,
    *,
    games: Games | None = None,
    fetch: Callable[[int], pd.DataFrame] = _savant_sprint_speed,
) -> int | None:
    """Fetch the season's sprint-speed leaderboard. Returns rows, or None if already final."""
    path = sprint_speed_path(root, season)
    if path.exists():
        games = games if games is not None else _fetch_season_games(season)
        if is_final(path, season_window(games)[1]):
            return None
    df = fetch(season)
    if df.empty:
        raise ValueError(f"sprint speed {season}: Savant returned no rows")
    # The leaderboard has no year column; add one so the seasons can be stacked.
    df = df.assign(season=season)
    _write_parquet(df, path)
    logger.info("sprint speed %s: %d rows", season, len(df))
    return len(df)


# --- schedule bounds --------------------------------------------------------------


def schedule_path(root: Path, season: int) -> Path:
    return root / "schedule" / f"{season}.parquet"


# codedGameState for cancelled and postponed entries: listed on the schedule, never played
# on that date.
_NOT_PLAYED_ON_DATE = frozenset({"C", "D"})


def write_season_schedule(root: Path, season: int, games: Games) -> tuple[date, date]:
    """Store the first and last scheduled regular-season dates, played or still to come.

    Readers use this, not the last game on disk, to tell a finished season from one in
    progress. Cancelled and postponed entries are left out so a rained-out final day
    does not push the end past the last game actually played.
    """
    dates = [
        date.fromisoformat(_game_context(g)[2])
        for g in games
        if g.get("gameType") == "R"
        and g.get("status", {}).get("codedGameState") not in _NOT_PLAYED_ON_DATE
    ]
    if not dates:
        raise ValueError(f"schedule {season}: no regular-season games")
    first, last = min(dates), max(dates)
    df = pd.DataFrame({"season": [season], "first_date": [first], "last_date": [last]})
    _write_parquet(df, schedule_path(root, season))
    return first, last


# --- reading --------------------------------------------------------------------


def connect(root: Path) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with ``pitches``, ``lineups``, ``sprint_speed`` and ``schedule`` views.

    A view is only created when its files exist. ``pitches`` carries a ``season`` column
    from the directory name; columns Savant added in later years are NULL in earlier ones.
    """
    conn = duckdb.connect()
    patterns = {
        "pitches": "pitches/*/*.parquet",
        "lineups": "lineups/*.parquet",
        "sprint_speed": "sprint_speed/*.parquet",
        "schedule": "schedule/*.parquet",
    }
    for name, pattern in patterns.items():
        if not any(root.glob(pattern)):
            continue
        conn.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet("
            f"'{(root / pattern).as_posix()}', union_by_name = true, hive_partitioning = true)"
        )
    return conn
