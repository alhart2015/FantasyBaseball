"""The draft board's hitter projection: ours blended with the board's FanGraphs blend (#459).

``ours`` (the hitter net) stays independent of FanGraphs. This is the separate step
after it, as for pitchers (``pitcher_ros/final.py``): per stat,
``w x ours + (1 - w) x FanGraphs`` on rates (R, HR, RBI, SB per PA; AVG), with
FanGraphs = the draft board's own blend (``data.projections.blend_projections`` of the
league's systems on file, at the league's weights), as published.

``WEIGHTS`` come from ``scripts/fit_board_hitter_blend.py``: picked by MSE on actuals
(``evaluate.squared_error``: R/HR/RBI/SB on totals over the PA each hitter got, AVG in
points weighted by AB) over hitters with 300+ actual PA on a 0.05 grid, each season's
held out of its own weights; ours = run 459-base (6 seeds), the held-out twin of
proj2027seq.

* R, HR, RBI, AVG: fit on 2022-2026. Held out, the blend beat the FanGraphs blend on
  all four (R 86.9 vs 94.0, RBI 134.1 vs 149.0, AVG 532 vs 553: 100 sure; HR 29.7 vs
  30.7, 98) and ours on all four (99-100).
* SB: fit on 2024-2026 only. 2023 brought the bigger bases and pitch clock and steals
  jumped; a net trained only on earlier seasons can't know that, and 2023 alone wants
  all FanGraphs (MSE rises with every step toward ours) while 2024, 2025 and 2026 each
  want 0.5+. Over all five seasons the held-out blend lost to FanGraphs (35.2 vs 34.0);
  on 2024-2026, held out, it beat FanGraphs (32.5 vs 34.2, 98 sure) and ours (89).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fantasy_baseball import blend_fit
from fantasy_baseball.hitter_ros.evaluate import SCALE, squared_error
from fantasy_baseball.hitter_ros.features import TARGETS

# Weight on ours per stat (1 = all ours, 0 = all FanGraphs).
WEIGHTS: dict[str, float] = {"r": 0.60, "hr": 0.45, "rbi": 0.70, "sb": 0.60, "avg": 0.50}


def error_weight(scored: pd.DataFrame) -> np.ndarray:
    """Each row's weight on its squared rate error in ``evaluate.squared_error``:
    ``pa^2`` for R/HR/RBI/SB (``(pa x err)^2``), ``ab x 1000^2`` for AVG (PA where a row
    has no AB)."""
    pa = scored["pa"].to_numpy(float)
    ab = scored["ab"].to_numpy(float) if "ab" in scored.columns else np.full(len(pa), np.nan)
    is_avg = (scored["stat"] == "avg").to_numpy()
    weight: np.ndarray = np.where(is_avg, np.where(np.isnan(ab), pa, ab) * SCALE["avg"] ** 2, pa**2)
    return weight


SCORING = blend_fit.Scoring(TARGETS, squared_error, error_weight)


def board_rates(ours: pd.DataFrame, board_fg: pd.DataFrame) -> pd.DataFrame:
    """The draft board's final hitter rates: ours with the board's FanGraphs blend (its
    rates, as published) at ``WEIGHTS``. A hitter only one side projects gets that
    side's rates."""
    return blend_fit.blend_rates(ours, board_fg, WEIGHTS, TARGETS)
