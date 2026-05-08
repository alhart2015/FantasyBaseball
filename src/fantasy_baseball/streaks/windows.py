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

import duckdb
import pandas as pd

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
        rolling = per_day.rolling(window=window_days, min_periods=1).sum().astype(int)
        rolling = rolling.reset_index().rename(columns={"index": "window_end"})
        rolling.insert(0, "player_id", int(player_id))
        rolling["window_days"] = window_days
        out_frames.append(rolling)

    return pd.concat(out_frames, ignore_index=True)
