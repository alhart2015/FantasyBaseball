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

from fantasy_baseball.trajectory.shape import MAX_LAG

#: Seasons of LEAD-IN before the panel's nominal first year, so `build_history` can
#: anchor a row at that year instead of censoring it.
#:
#: `shape.MAX_LAG` deep, matching the fit's own window. Without it the shallowest row the
#: fit keeps is `MAX_LAG` seasons in -- age 28 on this panel -- and every fixture that
#: asks about a 24-to-27-year-old silently loses its cohort: `closest_careers` returns
#: nothing (no candidate shares those ages), `prepared.age` has no entry to index, and the
#: tests fail on empty lookups rather than on anything they assert.
#:
#: THE SEASON-TO-AGE MAPPING IS UNCHANGED -- 2010 is still age 24 -- BUT THE VALUES ARE
#: NOT, and the difference matters to whoever reads this next. Prepending draws shifts the
#: whole `default_rng(0)` stream, so every `sgp` moved: measured against the pre-change
#: generator, 0 of 1440 overlapping (mlbam_id, season) rows kept their value, mean
#: |diff| 6.6 SGP and max 24.0. An earlier version of this note claimed every fixture
#: "reads exactly what it did before" -- false, and exactly the sentence that would send
#: someone debugging a fixture failure past the real cause.
#:
#: What IS safe is that no fixture here asserts a specific SGP: they assert orderings,
#: band containment, effective sizes and rank relationships, which the draws being
#: exchangeable preserves. A future fixture that pins a literal value would not be.
_LEAD_IN = MAX_LAG


def synthetic_panel(n: int = 160, seasons: range = range(2010, 2019)) -> pd.DataFrame:
    """A population several seasons deep, so a 3-horizon fit is not a pair of NaNs.

    `seasons` names the years a caller cares about; `_LEAD_IN` earlier ones are prepended
    so those years survive the lag window. A caller passing its own range gets the same
    treatment and does not have to know the fit's depth.

    Seeded, so every caller gets the same panel and a fixture that depends on a
    particular player's support or rank stays reproducible.
    """
    # Drawn as part of each player's own sequence rather than from a second generator,
    # so the panel stays reproducible from one seed. This does NOT preserve the old
    # values -- see `_LEAD_IN`; it just keeps the construction single-seeded.
    full = range(seasons.start - _LEAD_IN, seasons.stop)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        level = float(rng.uniform(4.0, 22.0))
        for offset, season in enumerate(full):
            age = 24 - _LEAD_IN + offset
            rows.append((i, season, age, max(level + float(rng.normal(0, 2.0)), 0.0)))
    return pd.DataFrame(rows, columns=["mlbam_id", "season", "age", "sgp"])
