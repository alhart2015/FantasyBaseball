"""Rolling-window aggregator for the streaks project.

Generates one row per (player_id, calendar_date in [first_played, last_played],
window_days in {3, 7, 14}) populated into ``hitter_windows`` — the threshold
calibration and label assignment in :mod:`thresholds` and :mod:`labels`
build on top of these rows.

Implementation: load ``hitter_games`` into pandas, per-player reindex to the
calendar between first and last played (zero-fill off-days), apply pandas
rolling sums, then join per-window Statcast averages computed in DuckDB.
Pandas is the right tool here because per-player reindexing + rolling +
join is awkward in pure SQL; DuckDB handles the percentile work in
:mod:`thresholds`.

Idempotent: the upsert uses ``INSERT OR REPLACE`` keyed by
``(player_id, window_end, window_days)``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import fields

import duckdb
import pandas as pd

from fantasy_baseball.streaks.models import HitterWindow, PtBucket

logger = logging.getLogger(__name__)

WINDOW_DAYS: tuple[int, ...] = (3, 7, 14)
PT_BUCKETS: tuple[PtBucket, ...] = ("low", "mid", "high")

# Box-score components we sum across windows. PA stays the canonical PA
# count even though it's also derivable from the components — the loader's
# PA-identity check is upstream so we trust the stored value.
_SUM_COLS: tuple[str, ...] = (
    "pa",
    "ab",
    "h",
    "hr",
    "r",
    "rbi",
    "sb",
    "bb",
    "k",
    "b2",
    "b3",
    "sf",
    "hbp",
)


def _build_per_day_frames(conn: duckdb.DuckDBPyConnection) -> dict[int, pd.DataFrame]:
    """Build per-player dense-calendar daily frames once.

    Returns ``{player_id: per_day_frame}`` where each frame is indexed by
    every calendar day in ``[first_played, last_played]`` (zero-fill on
    off-days, doubleheaders collapsed to one row). Hoisted out of
    :func:`_compute_rolling_sums` so :func:`compute_windows` can build it
    once and reuse across the 3 window sizes.
    """
    games = conn.execute(f"SELECT player_id, date, {', '.join(_SUM_COLS)} FROM hitter_games").df()
    if games.empty:
        return {}
    games["date"] = pd.to_datetime(games["date"])
    frames: dict[int, pd.DataFrame] = {}
    for player_id, player_games in games.groupby("player_id", sort=False):
        first_played = player_games["date"].min()
        last_played = player_games["date"].max()
        idx = pd.date_range(first_played, last_played, freq="D")
        frames[int(player_id)] = (
            player_games.set_index("date")[list(_SUM_COLS)]
            .groupby(level=0)
            .sum()  # collapse doubleheaders into one daily row
            .reindex(idx, fill_value=0)
        )
    return frames


def _rolling_sums_from_per_day(
    per_day_frames: dict[int, pd.DataFrame], window_days: int
) -> pd.DataFrame:
    """Apply a rolling-sum pass over pre-built per-day frames."""
    if not per_day_frames:
        return pd.DataFrame(columns=["player_id", "window_end", "window_days", *_SUM_COLS])
    out_frames: list[pd.DataFrame] = []
    for player_id, per_day in per_day_frames.items():
        # fillna(0) makes the NOT NULL invariant on _SUM_COLS explicit at the
        # cast site rather than relying on upstream schema enforcement.
        rolling = per_day.rolling(window=window_days, min_periods=1).sum().fillna(0).astype(int)
        rolling = rolling.reset_index().rename(columns={"index": "window_end"})
        rolling.insert(0, "player_id", player_id)
        rolling["window_days"] = window_days
        out_frames.append(rolling)
    return pd.concat(out_frames, ignore_index=True)


def _compute_rolling_sums(conn: duckdb.DuckDBPyConnection, window_days: int) -> pd.DataFrame:
    """Return a DataFrame of rolling sums for every (player, calendar-date) pair.

    Columns: player_id, window_end (Timestamp), window_days, plus one column
    per name in ``_SUM_COLS``. No PA<5 filter applied here — caller filters.
    """
    return _rolling_sums_from_per_day(_build_per_day_frames(conn), window_days)


def _add_rate_stats(sums: pd.DataFrame) -> pd.DataFrame:
    """Add avg, babip, iso, k_pct, bb_pct columns to a rolling-sums frame.

    NaN where the denominator is zero (e.g. zero-PA windows from off-day
    rows that haven't been filtered yet).
    """
    out = sums.copy()
    ab = out["ab"].astype("float64")
    pa = out["pa"].astype("float64")
    babip_denom = (out["ab"] - out["k"] - out["hr"] + out["sf"]).astype("float64")
    iso_num = (out["b2"] + 2 * out["b3"] + 3 * out["hr"]).astype("float64")

    # ``.where(denom > 0)`` returns NaN where the denominator is zero, so
    # we never produce inf (numerators are finite ints). No need for the
    # deprecated ``mode.use_inf_as_na`` option context.
    out["avg"] = (out["h"] / ab).where(ab > 0)
    out["babip"] = ((out["h"] - out["hr"]) / babip_denom).where(babip_denom > 0)
    out["iso"] = (iso_num / ab).where(ab > 0)
    out["k_pct"] = (out["k"] / pa).where(pa > 0)
    out["bb_pct"] = (out["bb"] / pa).where(pa > 0)
    return out


_STATCAST_SUM_COLS: tuple[str, ...] = (
    "ls_sum",
    "ls_n",
    "barrel_sum",
    "barrel_n",
    "xwoba_sum",
    "xwoba_n",
)


def _build_daily_statcast(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Run the (player_id, date) statcast GROUP BY once and return a flat frame.

    Returned columns: player_id (int), date (Timestamp), ls_sum, ls_n,
    barrel_sum, barrel_n, xwoba_sum, xwoba_n. Empty frame if the source
    table has no rows.

    Hoisted out of :func:`_add_statcast_peripherals` so :func:`compute_windows`
    can run the SQL aggregate once and reuse the result across the 3
    window sizes.
    """
    daily = conn.execute(
        """
        SELECT
            player_id,
            date,
            SUM(launch_speed) FILTER (WHERE launch_speed IS NOT NULL) AS ls_sum,
            COUNT(launch_speed) FILTER (WHERE launch_speed IS NOT NULL) AS ls_n,
            SUM(CASE WHEN barrel THEN 1 ELSE 0 END) FILTER (WHERE barrel IS NOT NULL) AS barrel_sum,
            COUNT(*) FILTER (WHERE barrel IS NOT NULL) AS barrel_n,
            SUM(estimated_woba_using_speedangle) FILTER (WHERE estimated_woba_using_speedangle IS NOT NULL) AS xwoba_sum,
            COUNT(estimated_woba_using_speedangle) FILTER (WHERE estimated_woba_using_speedangle IS NOT NULL) AS xwoba_n
        FROM hitter_statcast_pa
        GROUP BY player_id, date
        """
    ).df()
    if daily.empty:
        return daily
    daily["date"] = pd.to_datetime(daily["date"])
    daily["player_id"] = daily["player_id"].astype(int)
    return daily


def _add_statcast_peripherals_from_daily(
    daily: pd.DataFrame, sums: pd.DataFrame, window_days: int
) -> pd.DataFrame:
    """Left-join per-window EV/barrel%/xwOBA averages onto ``sums``.

    ``sums`` must contain a single ``window_days`` value. ``daily`` is
    the pre-aggregated per-player statcast frame produced by
    :func:`_build_daily_statcast`. Players in ``sums`` with no rows in
    ``daily`` get NaN peripherals via the final left-merge.

    Implementation: per-player dense calendar + pandas rolling sum, same
    pattern :func:`_build_per_day_frames` uses for box-score rolls. The
    calendar spans both the player's statcast date range and the window
    of (player_id, window_end) keys we need so every requested window
    has a row.
    """
    if sums.empty:
        out = sums.copy()
        for col in ("ev_avg", "barrel_pct", "xwoba_avg"):
            out[col] = pd.NA
        return out

    wd = int(window_days)

    # Targets we need a peripheral row for: every (player_id, window_end)
    # in ``sums``. Used to bound the per-player calendar so the rolling
    # output always contains the window_end key for the merge.
    sums_end_by_pid = sums.groupby("player_id", sort=False)["window_end"].agg(["min", "max"])

    peripheral_blocks: list[pd.DataFrame] = []
    if not daily.empty:
        # Pre-build a {player_id: per-player daily frame} mapping.
        # Tight loop below indexes into this rather than re-grouping.
        daily_grouped = {int(pid): df for pid, df in daily.groupby("player_id", sort=False)}
        for pid_int, (end_min, end_max) in sums_end_by_pid.iterrows():
            pid_int = int(pid_int)
            daily_player = daily_grouped.get(pid_int)
            if daily_player is None:
                continue  # no statcast PAs for this player; left-merge -> NaN
            first_d = daily_player["date"].min()
            last_d = daily_player["date"].max()
            calendar_start = min(first_d, end_min - pd.Timedelta(days=wd - 1))
            calendar_end = max(last_d, end_max)
            idx = pd.date_range(calendar_start, calendar_end, freq="D")
            per_day = (
                daily_player.set_index("date")[list(_STATCAST_SUM_COLS)]
                .groupby(level=0)
                .sum()
                .reindex(idx, fill_value=0)
            )
            rolling = per_day.rolling(window=wd, min_periods=1).sum()
            rolling = rolling.reset_index().rename(columns={"index": "window_end"})
            rolling.insert(0, "player_id", pid_int)
            peripheral_blocks.append(rolling)

    if peripheral_blocks:
        roll_df = pd.concat(peripheral_blocks, ignore_index=True)
        ls_n = roll_df["ls_n"].astype("float64")
        barrel_n = roll_df["barrel_n"].astype("float64")
        xwoba_n = roll_df["xwoba_n"].astype("float64")
        peripherals = pd.DataFrame(
            {
                "player_id": roll_df["player_id"].astype(int),
                "window_end": roll_df["window_end"],
                "ev_avg": (roll_df["ls_sum"] / ls_n).where(ls_n > 0),
                "barrel_pct": (roll_df["barrel_sum"] / barrel_n).where(barrel_n > 0),
                "xwoba_avg": (roll_df["xwoba_sum"] / xwoba_n).where(xwoba_n > 0),
            }
        )
    else:
        peripherals = pd.DataFrame(
            columns=["player_id", "window_end", "ev_avg", "barrel_pct", "xwoba_avg"]
        )

    # Add window_days back so the merge key matches sums.
    peripherals["window_days"] = wd
    return sums.merge(peripherals, on=["player_id", "window_end", "window_days"], how="left")


def _add_statcast_peripherals(conn: duckdb.DuckDBPyConnection, sums: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper: build daily statcast then join.

    The optimized :func:`compute_windows` builds the daily aggregate once
    via :func:`_build_daily_statcast` and reuses across window sizes;
    direct callers (tests, ad-hoc tooling) can still use this single-call
    form. ``sums`` must contain a single ``window_days`` value.
    """
    if sums.empty:
        out = sums.copy()
        for col in ("ev_avg", "barrel_pct", "xwoba_avg"):
            out[col] = pd.NA
        return out
    unique_wd = sums["window_days"].astype(int).unique()
    if len(unique_wd) != 1:
        raise ValueError(
            "_add_statcast_peripherals expects sums to contain a single window_days "
            f"value; got {sorted(unique_wd.tolist())}"
        )
    daily = _build_daily_statcast(conn)
    return _add_statcast_peripherals_from_daily(daily, sums, int(unique_wd[0]))


def _assign_pt_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Assign 'low' (5-9) / 'mid' (10-19) / 'high' (>=20) based on PA.

    Caller must have already filtered PA<5 rows out — this helper does not
    re-filter (the schema requires pt_bucket NOT NULL).
    """
    out = df.copy()
    bins = [4, 9, 19, 10**9]
    out["pt_bucket"] = pd.cut(out["pa"], bins=bins, labels=list(PT_BUCKETS), right=True).astype(
        "string"
    )
    return out


# Derived from the dataclass per the models.py single-source-of-truth convention.
_HITTER_WINDOWS_COLS: tuple[str, ...] = tuple(f.name for f in fields(HitterWindow))


def compute_windows(conn: duckdb.DuckDBPyConnection) -> int:
    """Rebuild ``hitter_windows`` from ``hitter_games`` + ``hitter_statcast_pa``.

    Generates rows for every (player, calendar_date in
    [first_played, last_played], window_days in {3, 7, 14}) where the
    window's PA >= 5. Returns the total row count written.

    Idempotent: ``INSERT OR REPLACE`` keyed by (player_id, window_end, window_days).
    """
    overall_t0 = time.perf_counter()
    n_games_row = conn.execute("SELECT COUNT(*) FROM hitter_games").fetchone()
    n_games = int(n_games_row[0]) if n_games_row is not None else 0
    logger.info("compute_windows: starting; %d rows in hitter_games", n_games)

    # Build the per-player dense calendar once and reuse across window sizes —
    # the SQL read + groupby + reindex is the dominant cost in this function.
    t0 = time.perf_counter()
    per_day_frames = _build_per_day_frames(conn)
    total_per_day_rows = sum(len(f) for f in per_day_frames.values())
    logger.info(
        "compute_windows: built %d per-day frames (%d total rows) in %.2fs",
        len(per_day_frames),
        total_per_day_rows,
        time.perf_counter() - t0,
    )

    # Pre-compute the per-(player_id, date) statcast aggregate once and reuse
    # across window sizes — same idea as per_day_frames. Returned as a dict of
    # per-player dense-calendar frames so each window-size pass can roll over
    # them without re-running the GROUP BY.
    t0 = time.perf_counter()
    daily_statcast = _build_daily_statcast(conn)
    logger.info(
        "compute_windows: built daily statcast aggregate (%d rows) in %.2fs",
        len(daily_statcast),
        time.perf_counter() - t0,
    )

    all_rows: list[pd.DataFrame] = []
    for window_days in WINDOW_DAYS:
        t0 = time.perf_counter()
        sums = _rolling_sums_from_per_day(per_day_frames, window_days)
        n_pre = len(sums)
        sums = sums[sums["pa"] >= 5].copy()
        logger.info(
            "compute_windows: rolling sums for %dd done in %.2fs (%d -> %d rows after PA>=5)",
            window_days,
            time.perf_counter() - t0,
            n_pre,
            len(sums),
        )
        if sums.empty:
            continue

        t0 = time.perf_counter()
        sums = _add_rate_stats(sums)
        logger.info(
            "compute_windows: rate stats applied for %dd in %.2fs",
            window_days,
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        sums = _add_statcast_peripherals_from_daily(daily_statcast, sums, window_days)
        logger.info(
            "compute_windows: peripherals applied for %dd in %.2fs (%d rows)",
            window_days,
            time.perf_counter() - t0,
            len(sums),
        )

        t0 = time.perf_counter()
        sums = _assign_pt_bucket(sums)
        logger.info(
            "compute_windows: pt_bucket assigned for %dd in %.2fs",
            window_days,
            time.perf_counter() - t0,
        )
        all_rows.append(sums)

    if not all_rows:
        logger.info("compute_windows: no rows produced; returning 0")
        return 0

    t0 = time.perf_counter()
    out = pd.concat(all_rows, ignore_index=True)[list(_HITTER_WINDOWS_COLS)]
    logger.info(
        "compute_windows: concatenated %d frames -> %d rows in %.2fs",
        len(all_rows),
        len(out),
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    n_written = _bulk_replace_hitter_windows(conn, out)
    logger.info(
        "compute_windows: wrote %d rows to hitter_windows in %.2fs",
        n_written,
        time.perf_counter() - t0,
    )
    logger.info("compute_windows: total elapsed %.2fs", time.perf_counter() - overall_t0)
    return n_written


def _bulk_replace_hitter_windows(conn: duckdb.DuckDBPyConnection, out: pd.DataFrame) -> int:
    """Replace contents of ``hitter_windows`` with rows in ``out``.

    Uses DuckDB's pandas registration to bulk-INSERT in C rather than
    row-by-row ``executemany`` (which is O(N) Python-level binding and
    runs into the minutes range for ~1M rows). Functionally equivalent
    to ``INSERT OR REPLACE`` keyed by the PK because we DELETE the table
    contents first inside the same transaction.
    """
    # Coerce window_end to date (DuckDB DATE column) — pandas may have it
    # as a Timestamp; the DuckDB pandas-register path handles datetime64
    # correctly so we don't need to convert per-row.
    df = out  # noqa: F841 — referenced via DuckDB's pandas scan below
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute("DELETE FROM hitter_windows")
        conn.execute(
            f"INSERT INTO hitter_windows ({', '.join(_HITTER_WINDOWS_COLS)}) "
            f"SELECT {', '.join(_HITTER_WINDOWS_COLS)} FROM df"
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return len(out)
