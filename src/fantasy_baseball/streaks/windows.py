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


def _add_statcast_peripherals(conn: duckdb.DuckDBPyConnection, sums: pd.DataFrame) -> pd.DataFrame:
    """Left-join per-window EV/barrel%/xwOBA averages from hitter_statcast_pa.

    Computed in DuckDB SQL keyed on (player_id, date) so the aggregation
    runs in the database. We then build a dense per-player calendar of
    those daily aggregates, run a pandas rolling sum once per player, and
    merge the rolling result back onto ``sums`` keyed on
    (player_id, window_end, window_days).

    This is O(N_players x N_days) rather than O(N_windows x N_daily_rows),
    which is the prior implementation's bottleneck on the real corpus
    (~7 minutes per window size dropped to ~5 seconds).
    """
    if sums.empty:
        out = sums.copy()
        for col in ("ev_avg", "barrel_pct", "xwoba_avg"):
            out[col] = pd.NA
        return out

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
    daily["date"] = pd.to_datetime(daily["date"])

    sum_cols = ("ls_sum", "ls_n", "barrel_sum", "barrel_n", "xwoba_sum", "xwoba_n")

    peripheral_frames: list[pd.DataFrame] = []
    for window_days, window_group in sums.groupby("window_days", sort=False):
        wd = int(window_days)
        # Per-player roll: build a dense calendar that spans both the
        # statcast date range AND every window_end we need, so the
        # rolling-sum result has a row at each window_end (including
        # off-days for that player). For players in ``sums`` with no
        # statcast PAs we emit NaN peripherals via the left-merge below.
        sums_pids = window_group["player_id"].astype(int).unique()
        per_player_rolling: list[pd.DataFrame] = []
        if len(daily) > 0:
            # Group statcast daily rows by player.
            daily_grouped = daily.groupby("player_id", sort=False)
            # Need to know per-player min/max window_end so the calendar
            # extends to cover every requested window_end (a player with
            # statcast data only in early April might still need a row
            # for a window ending in May where launch_speed sums roll to 0).
            sums_end_by_pid = window_group.groupby("player_id", sort=False)["window_end"].agg(
                ["min", "max"]
            )
            for pid, daily_player in daily_grouped:
                pid_int = int(pid)
                if pid_int not in sums_pids:
                    # Player has statcast data but no windows in this batch;
                    # skip — nothing would merge through.
                    continue
                first_d = daily_player["date"].min()
                last_d = daily_player["date"].max()
                end_min = sums_end_by_pid.loc[pid_int, "min"]
                end_max = sums_end_by_pid.loc[pid_int, "max"]
                # Calendar must span [min(first_d, end_min - (wd-1)), max(last_d, end_max)]
                # so every window_end has a row and rolling sums see all
                # statcast days that could contribute to those windows.
                calendar_start = min(first_d, end_min - pd.Timedelta(days=wd - 1))
                calendar_end = max(last_d, end_max)
                idx = pd.date_range(calendar_start, calendar_end, freq="D")
                # Collapse multiple rows on the same date (defensive — the
                # SQL groups by (player, date) so this is usually a no-op),
                # then reindex to the dense calendar with zero-fill.
                per_day = (
                    daily_player.set_index("date")[list(sum_cols)]
                    .groupby(level=0)
                    .sum()
                    .reindex(idx, fill_value=0)
                )
                rolling = per_day.rolling(window=wd, min_periods=1).sum()
                rolling = rolling.reset_index().rename(columns={"index": "window_end"})
                rolling.insert(0, "player_id", pid_int)
                per_player_rolling.append(rolling)
        if per_player_rolling:
            roll_df = pd.concat(per_player_rolling, ignore_index=True)
            roll_df["window_days"] = wd
            # Compute averages from the rolling sums; NaN where denom == 0.
            ls_n = roll_df["ls_n"].astype("float64")
            barrel_n = roll_df["barrel_n"].astype("float64")
            xwoba_n = roll_df["xwoba_n"].astype("float64")
            peripheral = pd.DataFrame(
                {
                    "player_id": roll_df["player_id"].astype(int),
                    "window_end": roll_df["window_end"],
                    "window_days": roll_df["window_days"].astype(int),
                    "ev_avg": (roll_df["ls_sum"] / ls_n).where(ls_n > 0),
                    "barrel_pct": (roll_df["barrel_sum"] / barrel_n).where(barrel_n > 0),
                    "xwoba_avg": (roll_df["xwoba_sum"] / xwoba_n).where(xwoba_n > 0),
                }
            )
            peripheral_frames.append(peripheral)

    if peripheral_frames:
        peripherals = pd.concat(peripheral_frames, ignore_index=True)
    else:
        peripherals = pd.DataFrame(
            columns=["player_id", "window_end", "window_days", "ev_avg", "barrel_pct", "xwoba_avg"]
        )
    return sums.merge(peripherals, on=["player_id", "window_end", "window_days"], how="left")


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
    # Build the per-player dense calendar once and reuse across window sizes —
    # the SQL read + groupby + reindex is the dominant cost in this function.
    per_day_frames = _build_per_day_frames(conn)
    all_rows: list[pd.DataFrame] = []
    for window_days in WINDOW_DAYS:
        sums = _rolling_sums_from_per_day(per_day_frames, window_days)
        sums = sums[sums["pa"] >= 5].copy()
        if sums.empty:
            continue
        sums = _add_rate_stats(sums)
        sums = _add_statcast_peripherals(conn, sums)
        sums = _assign_pt_bucket(sums)
        all_rows.append(sums)

    if not all_rows:
        return 0

    out = pd.concat(all_rows, ignore_index=True)[list(_HITTER_WINDOWS_COLS)]
    placeholders = ", ".join(["?"] * len(_HITTER_WINDOWS_COLS))
    sql = (
        f"INSERT OR REPLACE INTO hitter_windows ({', '.join(_HITTER_WINDOWS_COLS)}) "
        f"VALUES ({placeholders})"
    )
    # Vectorized NaN-to-None for DuckDB binding (faster than per-row pd.isna scan
    # at ~500K rows x 17 cols).
    out_obj = out.astype(object).where(out.notna(), None)
    rows = list(out_obj.itertuples(index=False, name=None))
    conn.executemany(sql, rows)
    logger.info("Wrote %d rows to hitter_windows", len(rows))
    return len(rows)
