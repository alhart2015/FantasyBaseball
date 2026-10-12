"""Fetch and store raw pitch-level data: every Statcast pitch, sprint speed, batting order.

Everything lands as parquet under one root (``data/pitch_data/`` by default) and is
kept raw -- all Savant columns, no renames, no filtering -- so later steps of #399
decide what to use. Query it through :func:`connect`, which exposes DuckDB views.

Layout::

    pitches/season=YYYY/<start>_<end>.parquet   one file per weekly chunk
    lineups/YYYY.parquet                         one row per player per lineup slot per
                                                 game: batting order + box-score batting line
    pitching/YYYY.parquet                        one row per pitcher per game: order of
                                                 appearance + box-score pitching line (#465)
    sprint_speed/YYYY.parquet                    Savant sprint-speed leaderboard
    schedule/YYYY.parquet                        first/last scheduled regular-season date
    games/YYYY.parquet                           each regular-season game: date, ballpark,
                                                 home and away team (played or not)
    milb_season/, milb_weekly/                   minor-league hitting lines (see milb.py)

Resumability: a file is final -- never fetched again -- only if it was *written* more
than ``SETTLE_DAYS`` after the end of the dates it covers. A file written earlier (a
mid-season or next-day run) is re-fetched on every run until a settled copy replaces it,
so late Savant corrections and the rest of a season are never frozen out. A lineup file
missing a current column (e.g. one written before the batting line was stored) is not
final either; the same goes for a pitching file.

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

from fantasy_baseball.analysis.game_logs import (
    FULL_HITTER_FIELDS,
    FULL_PITCHER_FIELDS,
    full_hitter_line,
    full_pitcher_line,
)
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
    return fetch_boxscore_season(
        root, season, ("lineups",), games=games, fetch_boxscore=fetch_boxscore
    )["lineups"]


# --- pitching lines ---------------------------------------------------------------


# A pitching file without these is from an older field list and gets re-fetched.
PITCHING_STAT_COLUMNS = (*FULL_PITCHER_FIELDS.values(), "resumed_date")
# Checked per team and game: the stored pitcher lines must add up to the box score's
# team pitching line. Not earned runs: by scoring rule 9.16(i) a reliever who enters
# after an error can be charged earned runs the team is not, so the pitchers' ER can add
# up to more than the team's (21 of 2,429 games in 2026). The pitchers' own ER are the
# official ones, the ones ERA is built from.
PITCHING_TEAM_CHECK = ("outs", "bf", "k", "bb", "h", "hr", "r", "pitches")
_PITCHER_FIELD = {col: field for field, col in FULL_PITCHER_FIELDS.items()}


def pitching_path(root: Path, season: int) -> Path:
    return root / "pitching" / f"{season}.parquet"


def _pitching_line(team: dict[str, Any], pitcher_id: int) -> dict[str, Any] | None:
    """His ``stats.pitching`` block in this team's box score; None when it is missing."""
    entry = team.get("players", {}).get(f"ID{pitcher_id}", {})
    line: dict[str, Any] | None = entry.get("stats", {}).get("pitching")
    return line


def pitching_rows(boxscore: dict[str, Any], game_pk: int, game_date: str) -> list[dict[str, Any]]:
    """One row per pitcher who appeared in this box score. ``appearance`` is his place in
    the team's ``pitchers`` list (1 = the team's first pitcher, 2 the next, ...); ``gs``
    is who was credited with the start (an opener is appearance 1 with gs 1, the bulk
    man after him gs 0). The pitching line rides along (``FULL_PITCHER_FIELDS`` columns;
    a missing field is 0). A listed pitcher with no pitching line gets no row
    (:func:`pitching_problems` reports him); the others keep their places.
    """
    rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        team = boxscore.get("teams", {}).get(side, {})
        team_id = team.get("team", {}).get("id")
        for i, person_id in enumerate(team.get("pitchers", []), start=1):
            line = _pitching_line(team, person_id)
            if line is None:
                continue
            rows.append(
                {
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "team_id": team_id,
                    "is_home": side == "home",
                    "player_id": int(person_id),
                    "appearance": i,
                    **full_pitcher_line(line),
                }
            )
    return rows


def pitching_problems(boxscore: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    """What looks wrong with one game's pitching ``rows`` (from :func:`pitching_rows`):
    ``side:no line for <id>`` for a listed pitcher with no pitching line, and
    ``side:<column>`` for each ``PITCHING_TEAM_CHECK`` column whose stored rows don't add
    up to the team's pitching line (a total the team line lacks is not checked)."""
    out = []
    for side in ("home", "away"):
        team = boxscore.get("teams", {}).get(side, {})
        out += [
            f"{side}:no line for {pid}"
            for pid in team.get("pitchers", [])
            if _pitching_line(team, pid) is None
        ]
        total = team.get("teamStats", {}).get("pitching", {})
        mine = [r for r in rows if r["is_home"] == (side == "home")]
        for col in PITCHING_TEAM_CHECK:
            expected = total.get(_PITCHER_FIELD[col])
            if expected is not None and sum(r[col] for r in mine) != int(expected):
                out.append(f"{side}:{col}")
    return out


def fetch_pitching_season(
    root: Path,
    season: int,
    *,
    games: Games | None = None,
    fetch_boxscore: Callable[[int], dict[str, Any]] = _fetch_boxscore,
) -> int | None:
    """Fetch every pitcher's line for every played regular-season game (see
    :func:`fetch_boxscore_season`). Returns rows written; None when the season's file is
    already final."""
    return fetch_boxscore_season(
        root, season, ("pitching",), games=games, fetch_boxscore=fetch_boxscore
    )["pitching"]


# --- one pass over a season's box scores --------------------------------------------


BOXSCORE_KINDS = ("lineups", "pitching")


def _played_dates(games: Games) -> dict[int, tuple[str, str | None]]:
    """game_pk -> (date, resumed) for every played game. ``date`` is its official date
    (the latest, if listed under more than one). A suspended game is listed twice with one
    gamePk and the same official date, the day it started; the day it was finished is the
    schedule's ``resumeGameDate`` -> ``resumed``. None for a game played in one go."""
    resumed = {g["gamePk"]: str(g["resumeGameDate"])[:10] for g in games if g.get("resumeGameDate")}
    out: dict[int, tuple[str, str | None]] = {}
    for pk, _, d in (_game_context(g) for g in games if was_played(g)):
        out[pk] = (max(d, out[pk][0]) if pk in out else d, resumed.get(pk))
    return out


def fetch_boxscore_season(
    root: Path,
    season: int,
    kinds: Iterable[str] = BOXSCORE_KINDS,
    *,
    games: Games | None = None,
    fetch_boxscore: Callable[[int], dict[str, Any]] = _fetch_boxscore,
) -> dict[str, int | None]:
    """Fetch every played regular-season box score once and write the ``kinds`` files
    (``lineups``, ``pitching``) that aren't already final (see :func:`is_final`).

    Returns rows written per kind: None when its file was already final, 0 when there
    were no rows (a season not started yet: nothing is written, so no empty file stands
    in for it). Any box-score failure raises without writing, so the next run retries
    the season whole.

    Every row is dated to the game's official date (``game_date``), for a suspended game
    the day it started. Pitching rows also carry ``resumed_date``, the day a suspended
    game was finished (None otherwise): a pitcher in such a game pitched on one of the
    two, and the box score doesn't say which.

    A game whose pitching rows look wrong (:func:`pitching_problems`) is kept, and logged:
    it is MLB's own box score, with nothing here to fix it from.
    """
    games = games if games is not None else _fetch_season_games(season)
    _, season_end = season_window(games)
    paths = {"lineups": lineup_path(root, season), "pitching": pitching_path(root, season)}
    columns = {"lineups": LINEUP_STAT_COLUMNS, "pitching": PITCHING_STAT_COLUMNS}
    result: dict[str, int | None] = {}
    todo = []
    for kind in kinds:
        if kind not in paths:
            raise ValueError(f"unknown box-score kind {kind!r}; expected {BOXSCORE_KINDS}")
        result[kind] = None
        if not is_final(paths[kind], season_end, columns[kind]):
            todo.append(kind)
    if not todo:
        return result
    dates = _played_dates(games)

    def _one(item: tuple[int, tuple[str, str | None]]) -> dict[str, Any]:
        pk, (game_date, resumed) = item
        box = fetch_boxscore(pk)
        out: dict[str, Any] = {}
        if "lineups" in todo:
            out["lineups"] = lineup_rows(box, pk, game_date)
        if "pitching" in todo:
            rows = [{**r, "resumed_date": resumed} for r in pitching_rows(box, pk, game_date)]
            out["pitching"] = rows
            out["problems"] = pitching_problems(box, rows)
        return out

    with ThreadPoolExecutor(max_workers=_BOXSCORE_WORKERS) as pool:
        per_game = dict(zip(dates, pool.map(_one, dates.items()), strict=True))
    if "pitching" in todo:
        bad = sorted(
            f"{pk} ({', '.join(g['problems'])})" for pk, g in per_game.items() if g["problems"]
        )
        if bad:
            logger.warning(
                "pitching %s: %d games whose pitcher lines look wrong: %s",
                season,
                len(bad),
                "; ".join(bad),
            )
    for kind in todo:
        df = pd.DataFrame([r for g in per_game.values() for r in g[kind]])
        result[kind] = len(df)
        if df.empty:
            logger.info("%s %s: no games played yet, nothing written", kind, season)
            continue
        _write_parquet(df, paths[kind])
        logger.info("%s %s: %d games, %d rows", kind, season, len(dates), len(df))
    return result


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


def games_path(root: Path, season: int) -> Path:
    return root / "games" / f"{season}.parquet"


def write_season_games(root: Path, season: int, games: Games) -> int:
    """Store one row per regular-season game on the schedule: ``game_pk``, ``game_date``,
    ``venue_id`` / ``venue_name`` (the ballpark -- not always the home team's usual one:
    the Tokyo, London and Seoul series, the 2025 A's in Sacramento), ``home_team_id``,
    ``away_team_id`` and ``played`` (``was_played``). Rewritten on every run, like the
    schedule bounds: it is one cheap call and future games can still move."""
    rows = [
        {
            "game_pk": g["gamePk"],
            "season": season,
            "game_date": date.fromisoformat(_game_context(g)[2]),
            "venue_id": g.get("venue", {}).get("id"),
            "venue_name": g.get("venue", {}).get("name"),
            "home_team_id": g["teams"]["home"]["team"]["id"],
            "away_team_id": g["teams"]["away"]["team"]["id"],
            "played": was_played(g),
        }
        for g in games
        if g.get("gameType") == "R"
    ]
    if not rows:
        raise ValueError(f"games {season}: no regular-season games")
    # A postponed game is listed again on its new date with the same gamePk: keep the
    # played entry (or the last one listed).
    df = (
        pd.DataFrame(rows)
        .sort_values(["played", "game_date"])
        .drop_duplicates("game_pk", keep="last")
        .sort_values(["game_date", "game_pk"])
        .reset_index(drop=True)
    )
    _write_parquet(df, games_path(root, season))
    return len(df)


# --- reading --------------------------------------------------------------------


def connect(root: Path) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with ``pitches``, ``lineups``, ``pitching``, ``sprint_speed``, ``schedule``,
    ``games``, ``milb_season``, ``milb_weekly``, ``prospect_rankings`` and ``draft`` views.

    A view is only created when its files exist. ``pitches`` carries a ``season`` column
    from the directory name; columns Savant added in later years are NULL in earlier ones.
    """
    conn = duckdb.connect()
    patterns = {
        "pitches": "pitches/*/*.parquet",
        "lineups": "lineups/*.parquet",
        "pitching": "pitching/*.parquet",
        "sprint_speed": "sprint_speed/*.parquet",
        "schedule": "schedule/*.parquet",
        "games": "games/*.parquet",
        "milb_season": "milb_season/*.parquet",
        "milb_weekly": "milb_weekly/*/*.parquet",
        "prospect_rankings": "prospect_rankings/*.parquet",
        "draft": "draft/*.parquet",
    }
    for name, pattern in patterns.items():
        if not any(root.glob(pattern)):
            continue
        conn.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet("
            f"'{(root / pattern).as_posix()}', union_by_name = true, hive_partitioning = true)"
        )
    return conn
