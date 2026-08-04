"""The synthetic trajectory panel shared by every test that needs one.

The panel schema is an input contract to `prepare()` / `sweep_pool`: mlbam_id, season,
age, sgp, and `partial_season` on the production path. It had three byte-equivalent
copies -- test_sweep.py, test_trajectory_view.py, and inline 2,100 lines into
test_season_routes.py -- so adding a required column meant finding all three, and the
likely outcome was two getting updated while the third kept passing against a shape
production no longer produces.

Lives beside `_cache_helpers.py`, which exists for the same reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_panel(n: int = 160, seasons: range = range(2010, 2019)) -> pd.DataFrame:
    """A population several seasons deep, so a 3-horizon fit is not a pair of NaNs.

    Seeded, so every caller gets the same panel and a fixture that depends on a
    particular player's support or rank stays reproducible.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        level = float(rng.uniform(4.0, 22.0))
        for offset, season in enumerate(seasons):
            rows.append((i, season, 24 + offset, max(level + float(rng.normal(0, 2.0)), 0.0)))
    return pd.DataFrame(rows, columns=["mlbam_id", "season", "age", "sgp"])
