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

from fantasy_baseball.streaks.models import HitterWindow

logger = logging.getLogger(__name__)

WINDOW_DAYS: tuple[int, ...] = (3, 7, 14)

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


def _compute_rolling_sums(conn: duckdb.DuckDBPyConnection, window_days: int) -> pd.DataFrame:
    """Return a DataFrame of rolling sums for every (player, calendar-date) pair.

    Columns: player_id, window_end (Timestamp), window_days, plus one column
    per name in ``_SUM_COLS``. No PA<5 filter applied here — caller filters.
    """
    games = conn.execute(f"SELECT player_id, date, {', '.join(_SUM_COLS)} FROM hitter_games").df()
    if games.empty:
        return pd.DataFrame(columns=["player_id", "window_end", "window_days", *_SUM_COLS])

    games["date"] = pd.to_datetime(games["date"])

    out_frames: list[pd.DataFrame] = []
    for player_id, player_games in games.groupby("player_id", sort=False):
        first_played = player_games["date"].min()
        last_played = player_games["date"].max()
        # Reindex to a continuous daily calendar between first and last played.
        idx = pd.date_range(first_played, last_played, freq="D")
        per_day = (
            player_games.set_index("date")[list(_SUM_COLS)]
            .groupby(level=0)
            .sum()  # collapse doubleheaders into one daily row
            .reindex(idx, fill_value=0)
        )
        # fillna(0) makes the NOT NULL invariant on _SUM_COLS explicit at the
        # cast site rather than relying on upstream schema enforcement.
        rolling = per_day.rolling(window=window_days, min_periods=1).sum().fillna(0).astype(int)
        rolling = rolling.reset_index().rename(columns={"index": "window_end"})
        rolling.insert(0, "player_id", int(player_id))
        rolling["window_days"] = window_days
        out_frames.append(rolling)

    return pd.concat(out_frames, ignore_index=True)


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
    runs in the database; the result is a small per-(player, day) frame
    that we then sum-then-divide across the window in pandas.

    Perf note: the inner per-window mask loop is O(windows × daily_rows).
    On a 3-season corpus that's ~250K windows × ~120K daily rows. If this
    becomes the dominant cost in compute_windows, replace with cumulative
    sums on the dense calendar (mirroring _compute_rolling_sums) so each
    window is a vectorized cumulative[end] − cumulative[start − 1] subtract.
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

    out_rows: list[dict[str, float | int | pd.Timestamp]] = []
    for window_days, window_group in sums.groupby("window_days", sort=False):
        end_dates = window_group[["player_id", "window_end"]].drop_duplicates()
        for _, row in end_dates.iterrows():
            pid = int(row["player_id"])
            end = row["window_end"]
            start = end - pd.Timedelta(days=int(window_days) - 1)
            mask = (daily["player_id"] == pid) & (daily["date"] >= start) & (daily["date"] <= end)
            sub = daily.loc[mask]
            ls_n = int(sub["ls_n"].sum())
            barrel_n = int(sub["barrel_n"].sum())
            xwoba_n = int(sub["xwoba_n"].sum())
            out_rows.append(
                {
                    "player_id": pid,
                    "window_end": end,
                    "window_days": int(window_days),
                    "ev_avg": float(sub["ls_sum"].sum()) / ls_n if ls_n else float("nan"),
                    "barrel_pct": float(sub["barrel_sum"].sum()) / barrel_n
                    if barrel_n
                    else float("nan"),
                    "xwoba_avg": float(sub["xwoba_sum"].sum()) / xwoba_n
                    if xwoba_n
                    else float("nan"),
                }
            )
    peripherals = pd.DataFrame(out_rows)
    return sums.merge(peripherals, on=["player_id", "window_end", "window_days"], how="left")


def _assign_pt_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Assign 'low' (5-9) / 'mid' (10-19) / 'high' (>=20) based on PA.

    Caller must have already filtered PA<5 rows out — this helper does not
    re-filter (the schema requires pt_bucket NOT NULL).
    """
    out = df.copy()
    bins = [4, 9, 19, 10**9]
    labels = ["low", "mid", "high"]
    out["pt_bucket"] = pd.cut(out["pa"], bins=bins, labels=labels, right=True).astype("string")
    return out


# Derived from the dataclass per the streaks/models.py single-source-of-truth
# convention (matches the load.py pattern for HitterGame and HitterStatcastPA).
_HITTER_WINDOWS_COLS: tuple[str, ...] = tuple(f.name for f in fields(HitterWindow))


def compute_windows(conn: duckdb.DuckDBPyConnection) -> int:
    """Rebuild ``hitter_windows`` from ``hitter_games`` + ``hitter_statcast_pa``.

    Generates rows for every (player, calendar_date in
    [first_played, last_played], window_days in {3, 7, 14}) where the
    window's PA >= 5. Returns the total row count written.

    Idempotent: ``INSERT OR REPLACE`` keyed by (player_id, window_end, window_days).
    """
    all_rows: list[pd.DataFrame] = []
    for window_days in WINDOW_DAYS:
        sums = _compute_rolling_sums(conn, window_days=window_days)
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
    rows = [
        tuple(None if pd.isna(v) else v for v in r) for r in out.itertuples(index=False, name=None)
    ]
    conn.executemany(sql, rows)
    logger.info("Wrote %d rows to hitter_windows", len(rows))
    return len(rows)
