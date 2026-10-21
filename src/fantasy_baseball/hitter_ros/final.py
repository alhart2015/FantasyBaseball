"""The draft board's hitter projection: ours blended with the board's FanGraphs blend (#459).

``ours`` (the hitter net) stays independent of FanGraphs. This is the separate step
after it, as for pitchers (``pitcher_ros/final.py``): per stat,
``w x ours + (1 - w) x FanGraphs`` on rates (R, HR, RBI, SB per PA; AVG), with
FanGraphs = the draft board's own blend (``data.projections.blend_projections`` of the
league's systems on file, at the league's weights), as published.

``WEIGHTS`` come from ``scripts/fit_board_hitter_blend.py``: picked by MSE on actuals
(``evaluate.squared_error``: R/HR/RBI/SB on totals over the PA each hitter got, AVG in
points weighted by AB) on a 0.05 grid, each season held out of its own weights, over
every hitter the board would blend -- the board's FanGraphs blend (built and matched
exactly as the board's: ``draft.our_players``) projects 100+ PA, ~600-660 a season --
who batted. Ours = run 459-base (6 seeds), the held-out twin of proj2027seq.

* R, HR, RBI, AVG: fit on 2022-2026. Held out, the blend beat the FanGraphs blend on
  all four (R 47.1 vs 50.6, RBI 71.3 vs 78.8, AVG 912 vs 948: 100 sure; HR 15.2 vs
  15.7, 99) and ours on all four (100). It holds for part-timers (projected 100-400
  PA: R, RBI, SB 99-100 sure; HR 85, AVG 89) as for regulars.
* SB: fit on 2024-2026 only (``FIT_SEASONS``). 2023 brought the bigger bases and pitch
  clock and steals jumped; a net trained only on earlier seasons can't know that, and
  2023 alone wanted all FanGraphs while 2024, 2025 and 2026 each want 0.5+. Over all
  five seasons the held-out blend lost to FanGraphs on SB; on 2024-2026 it beats
  FanGraphs (100 sure) and ours (88). The window was chosen after seeing 2023, so this
  is a judgment (the user's call, 2026-10-21), not a held-out result.
"""

from __future__ import annotations

import pandas as pd

from fantasy_baseball import blend_fit
from fantasy_baseball.hitter_ros.evaluate import squared_error
from fantasy_baseball.hitter_ros.features import TARGETS

# The seasons a stat's weight is fit and scored on, when not every season (see the
# module doc; scripts/fit_board_hitter_blend.py reads this). SB: the seasons since the
# 2023 rule change -- a judgment made after seeing 2023 disagree, not a held-out
# result: no season scored this way was left unseen when the window was picked.
FIT_SEASONS: dict[str, tuple[int, ...]] = {"sb": (2024, 2025, 2026)}
# Weight on ours per stat (1 = all ours, 0 = all FanGraphs).
WEIGHTS: dict[str, float] = {"r": 0.60, "hr": 0.45, "rbi": 0.70, "sb": 0.60, "avg": 0.55}


# How hitters are scored (evaluate.squared_error): the fit's MSE.
SCORING = blend_fit.Scoring(TARGETS, squared_error)


def board_rates(ours: pd.DataFrame, board_fg: pd.DataFrame) -> pd.DataFrame:
    """The draft board's final hitter rates: ours with the board's FanGraphs blend (its
    rates, as published) at ``WEIGHTS``. A hitter only one side projects gets that
    side's rates."""
    return blend_fit.blend_rates(ours, board_fg, WEIGHTS, TARGETS)
