"""Fetch and store raw minor-league hitting lines from the MLB Stats API (#435).

Rookies are where our projections trail FanGraphs (#433), because we had no
minor-league history. This pulls it into the same store as the MLB data, kept raw: every
column of the API's splits (``json_normalize``'d, so names like
``stat.plateAppearances``), plus ``sport_id`` (the level), ``season`` and, for weekly
lines, the window dates. Grading the levels against each other is a later step.

Layout (under the store root, read through :func:`store.connect`)::

    milb_season/YYYY.parquet                one row per player x level: the season line
    milb_weekly/YYYY/<start>_<end>.parquet  the same, for one window of dates

Levels are ``MILB_LEVELS``, one ``sportId`` each; the API returns one line per player per
level per query, with ``team`` / ``league`` the player's last club there and
``numTeams`` how many he played for. Player ids are MLBAM, the same as the MLB data.

Weekly windows follow the hitter table's as-of grid (opening day plus multiples of the
table's step), so a table row's "this season so far" is exactly the windows that end
before its as-of date. The first window starts on January 1 and the last runs to
December 31, so every regular-season game falls in exactly one window. There was no
minor-league season in 2020.

Resumability follows the store: a file is final once it was written after the dates it
covers had settled (``store.is_final``). A season file counts as covering the MLB season
(the minor-league regular season ends by then). Once the season has settled and its file
and every window are on disk, the window lines must add up to the season line for every
player and level, or the fetch fails: the API can return a short page without raising.
When they don't add up and the season file is from an earlier run, the season lines are
fetched once more first, so a short season page can't get stuck as final. The
date-range query does drop the odd game (5 seasons, about 20 player-levels each, a few
PA; the games are in the player's game log). A player-level that still doesn't add up
gets its weekly lines rebuilt from his game log, which carries each game's date; those
rows have ``from_game_log`` set, count stats summed, ``team`` / ``league`` from his
last game in the window (he may have been traded since), and the rest of their columns
from his season line. Only if they still don't add up does it fail.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from fantasy_baseball.keepers.mlb_stats import _MLB_PEOPLE_URL, fetch_stats_splits
from fantasy_baseball.pitch_data.store import _write_parquet, is_final, is_settled

logger = logging.getLogger(__name__)

# sportId -> level. 15 (short-season A) ended with the 2021 reorganization; 13 and 14
# were renamed then (Advanced A -> High-A, Class A -> Single-A) but kept their tiers.
MILB_LEVELS = {11: "AAA", 12: "AA", 13: "A+", 14: "A", 15: "A-", 16: "Rk"}
NO_MILB_SEASONS = frozenset({2020})
_WORKERS = 4
PA = "stat.plateAppearances"

Window = tuple[date, date]
Fetch = Callable[[dict[str, str | int]], list[dict[str, Any]]]
# (player, season, sportId) -> that player's game-by-game splits at the level
GameLog = Callable[[int, int, int], list[dict[str, Any]]]
# Numeric stat fields that are not counts, so summing games would be wrong.
_NOT_COUNTS = frozenset({"stat.age"})


def fetch_game_log(player: int, season: int, sport_id: int) -> list[dict[str, Any]]:
    import requests

    params: dict[str, str | int] = {
        "stats": "gameLog",
        "group": "hitting",
        "season": season,
        "sportId": sport_id,
    }
    resp = requests.get(f"{_MLB_PEOPLE_URL}/{player}/stats", params=params, timeout=60)
    resp.raise_for_status()
    stats = resp.json().get("stats", [])
    splits: list[dict[str, Any]] = stats[0]["splits"] if stats else []
    return splits


def milb_season_path(root: Path, season: int) -> Path:
    return root / "milb_season" / f"{season}.parquet"


def milb_week_path(root: Path, season: int, window: Window) -> Path:
    start, end = window
    return root / "milb_weekly" / str(season) / f"{start.isoformat()}_{end.isoformat()}.parquet"


def week_windows(first: date, last: date, step_days: int) -> list[Window]:
    """Date windows that tile the year around the as-of grid ``first + k * step_days``
    (k = 0 .. while <= ``last``): January 1 to the day before ``first``, one window per
    step, and the last as-of date to December 31."""
    as_ofs = [
        first + timedelta(days=k * step_days) for k in range((last - first).days // step_days + 1)
    ]
    starts = [date(first.year, 1, 1), *as_ofs]
    ends = [d - timedelta(days=1) for d in as_ofs] + [date(first.year, 12, 31)]
    return list(zip(starts, ends, strict=True))


def _lines(season: int, fetch: Fetch, window: Window | None = None) -> pd.DataFrame:
    """Every level's hitting lines for the season, or for ``window`` within it."""
    frames = []
    for sport_id in MILB_LEVELS:
        params: dict[str, str | int] = {
            "stats": "season" if window is None else "byDateRange",
            "group": "hitting",
            "season": season,
            "sportId": sport_id,
            "playerPool": "all",
        }
        if window is not None:
            params["startDate"], params["endDate"] = (d.isoformat() for d in window)
        splits = fetch(params)
        if splits:
            frames.append(pd.json_normalize(splits).assign(sport_id=sport_id))
    empty = pd.DataFrame(
        {"player.id": pd.Series(dtype="int64"), "sport_id": pd.Series(dtype="int64"), PA: []}
    )
    df = pd.concat(frames, ignore_index=True) if frames else empty
    # The splits carry the season as a string; one int column stacks cleanly.
    df = df.assign(season=season)
    if window is not None:
        df = df.assign(window_start=window[0], window_end=window[1], from_game_log=False)
    return df


def weekly_mismatches(season_lines: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """Player x level rows whose window PA don't add up to the season line's PA."""
    keys = ["player.id", "sport_id"]
    whole = season_lines.groupby(keys)[PA].sum()
    parts = weekly.groupby(keys)[PA].sum() if len(weekly) else pd.Series(dtype=float)
    both = pd.concat({"season_pa": whole, "weekly_pa": parts}, axis=1).fillna(0)
    return both[both["season_pa"] != both["weekly_pa"]].reset_index()


def fetch_milb_season(
    root: Path,
    season: int,
    first: date,
    last: date,
    *,
    step_days: int,
    today: date,
    fetch: Fetch = fetch_stats_splits,
    game_log: GameLog = fetch_game_log,
) -> dict[str, int]:
    """Fetch what is missing or still changing for ``season``: the season lines and each
    weekly window that has started. ``first`` / ``last``: the MLB regular season's first
    and last dates. Returns counts of files written, skipped as final and failed."""
    result = {"written": 0, "final": 0, "failed": 0}
    if season in NO_MILB_SEASONS:
        return result
    season_path = milb_season_path(root, season)
    season_fetched = False
    if is_final(season_path, last):
        result["final"] += 1
    elif first <= today:
        _write_season_lines(season, fetch, season_path)
        result["written"] += 1
        season_fetched = True

    todo = []
    for window in week_windows(first, last, step_days):
        if window[0] > today:
            break
        if is_final(milb_week_path(root, season, window), window[1]):
            result["final"] += 1
        else:
            todo.append(window)

    def one(window: Window) -> bool:
        try:
            _write_parquet(_lines(season, fetch, window), milb_week_path(root, season, window))
            return True
        except Exception:
            logger.exception("milb %s %s..%s: failed", season, *window)
            return False

    with ThreadPoolExecutor(_WORKERS) as pool:
        for ok in pool.map(one, todo):
            result["written" if ok else "failed"] += 1
    logger.info("milb %s weekly: %s", season, result)

    # Once the season has settled and every window is on disk (final, or fetched just
    # now), the windows must add up to the season line.
    if is_settled(last, today) and not result["failed"]:
        windows = week_windows(first, last, step_days)
        if not season_fetched and len(
            weekly_mismatches(pd.read_parquet(season_path), _read_weekly(root, season, windows))
        ):
            # A short season page from an earlier run would otherwise stay final for good.
            logger.info(
                "milb %s: weekly lines don't add up; fetching the season lines again", season
            )
            _write_season_lines(season, fetch, season_path)
            result["written"] += 1
        check_season(root, season, windows, game_log)
    return result


def _write_season_lines(season: int, fetch: Fetch, path: Path) -> None:
    df = _lines(season, fetch)
    if df.empty:
        raise ValueError(f"milb {season}: the API returned no lines at any level")
    _write_parquet(df, path)
    logger.info("milb %s season: %d lines", season, len(df))


def _read_weekly(root: Path, season: int, windows: list[Window]) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(milb_week_path(root, season, w)) for w in windows], ignore_index=True
    )


def check_season(root: Path, season: int, windows: list[Window], game_log: GameLog) -> None:
    """Make the weekly lines add up to the season lines: rebuild any player-level that
    doesn't from his game log, then raise if any still doesn't."""
    season_lines = pd.read_parquet(milb_season_path(root, season))
    bad = weekly_mismatches(season_lines, _read_weekly(root, season, windows))
    if len(bad):
        logger.info("milb %s: %d player-levels don't add up; using game logs", season, len(bad))
        _rebuild_from_game_logs(root, season, windows, season_lines, bad, game_log)
        bad = weekly_mismatches(season_lines, _read_weekly(root, season, windows))
    if len(bad):
        raise ValueError(
            f"milb {season}: weekly PA don't add up to the season line for {len(bad)} "
            f"player-levels, e.g. {bad.head().to_dict('records')}"
        )


def _rebuild_from_game_logs(
    root: Path,
    season: int,
    windows: list[Window],
    season_lines: pd.DataFrame,
    bad: pd.DataFrame,
    game_log: GameLog,
) -> None:
    """Replace each ``bad`` player-level's weekly rows with sums of his game log.

    A player-level is only replaced when he has exactly one season line and his game
    log adds up to it. Otherwise his API rows stay as they are, so the check that
    follows still fails on him: deleting them would either lose real lines for good
    (the rewritten files are final) or, with no season line to compare against, make
    the check pass on nothing."""
    keys = ["player.id", "sport_id"]
    rebuilt: list[pd.DataFrame] = []
    fixed_keys: list[tuple[int, int]] = []
    for player, sport_id in bad[keys].itertuples(index=False):
        line = season_lines[
            (season_lines["player.id"] == player) & (season_lines["sport_id"] == sport_id)
        ]
        if len(line) != 1:
            logger.warning(
                "milb %s: player %s sport %s has %d season lines; not rebuilt",
                season,
                player,
                sport_id,
                len(line),
            )
            continue
        games = pd.json_normalize(game_log(int(player), season, int(sport_id)))
        if games.empty:
            continue
        counts = [
            c
            for c in games.columns
            if c.startswith("stat.")
            and c not in _NOT_COUNTS
            and pd.api.types.is_numeric_dtype(games[c])
        ]
        games = games.assign(_day=pd.to_datetime(games["date"]).dt.date).sort_values("_day")
        base = line.drop(columns=[c for c in line.columns if c.startswith("stat.")])
        # His club in a window is his club in its games, not his last club of the season.
        club = [c for c in games.columns if c.startswith(("team.", "league.")) and c in base]
        rows = []
        for start, end in windows:
            in_window = games[(games["_day"] >= start) & (games["_day"] <= end)]
            if in_window.empty:
                continue
            last_game = in_window.iloc[-1]
            rows.append(
                base.assign(
                    **{c: last_game[c] for c in club},
                    **{c: in_window[c].sum() for c in counts},
                    window_start=start,
                    window_end=end,
                    from_game_log=True,
                )
            )
        rebuilt_pa = sum(r[PA].sum() for r in rows if PA in r.columns)
        if rebuilt_pa != line[PA].iloc[0]:
            continue
        rebuilt.extend(rows)
        fixed_keys.append((player, sport_id))
    if not fixed_keys:
        return
    fixed = pd.DataFrame(fixed_keys, columns=keys).assign(_drop=True)
    new = pd.concat(rebuilt, ignore_index=True)
    for window in windows:
        path = milb_week_path(root, season, window)
        old = pd.read_parquet(path).merge(fixed, on=keys, how="left")
        keep = old["_drop"].isna()
        add = new[new["window_start"] == window[0]]
        if keep.all() and add.empty:
            continue  # nothing of a rebuilt player-level in this window
        _write_parquet(pd.concat([old[keep].drop(columns="_drop"), add], ignore_index=True), path)
