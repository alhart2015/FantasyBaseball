"""Find a hitter's events strictly before a date, by index arithmetic.

Shared by the plate-appearance batcher (``sequence.py``) and the pitch store
(``pretrain.py``) so the leak-sensitive rule -- "only events before the as-of date" --
lives in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sort key that orders events by hitter, then day: hitter rank * DAY_SPAN + day.
DAY_SPAN = 1_000_000


def to_days(dates: pd.Series | np.ndarray) -> np.ndarray:
    """Dates as whole days since 1970."""
    days: np.ndarray = pd.to_datetime(dates).to_numpy().astype("datetime64[D]").astype(np.int64)
    return days


class HitterTimeline:
    """Index over events already sorted by hitter, then time.

    ``first[i]`` / ``last[i]`` bound hitter ``players[i]``'s events (half-open).
    """

    def __init__(self, player: np.ndarray, days: np.ndarray) -> None:
        if len(player) and np.any(np.diff(player.astype(np.int64)) < 0):
            raise ValueError("events must be sorted by hitter")
        self.players, self.first = np.unique(player, return_index=True)
        self.last = np.append(self.first[1:], len(player))
        self.key = np.searchsorted(self.players, player) * DAY_SPAN + days
        if len(self.key) and np.any(np.diff(self.key) < 0):
            raise ValueError("events must be sorted by time within each hitter")

    def rank(self, player_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(rank in ``players``, known?) for each id."""
        rank = np.minimum(np.searchsorted(self.players, player_ids), max(len(self.players) - 1, 0))
        known = (len(self.players) > 0) & (self.players[rank] == player_ids)
        return rank, known

    def before(
        self, player_ids: np.ndarray, as_of_days: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(start, end) of each hitter's events strictly before his as-of day (half-open;
        empty for an unknown hitter)."""
        rank, known = self.rank(player_ids)
        start = np.where(known, self.first[rank], 0)
        end = np.where(known, np.searchsorted(self.key, rank * DAY_SPAN + as_of_days, "left"), 0)
        return start, end
