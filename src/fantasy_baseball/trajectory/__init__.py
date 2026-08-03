"""Career SGP trajectory from historical comparables (#303).

Answers: a player is on pace for N SGP in his age-A season -- what does the rest of
his career look like? The estimate is the forward path of every historical
player-season that started from the same place, with players who left the league
scored as the zero they are worth to a roster slot.

DELIBERATELY STANDALONE. This does not read from, write to, or alter keeper value
(`keepers/`, `scripts/keeper_forecast.py`, `scripts/keeper_value.py`). It is its own
metric to explore; if it eventually earns the keeper forecast's job, that is a later
and explicit decision.

What it produces is `E[future SGP | age, current SGP]` -- a forecast, NOT an aging
curve. It already contains regression to the mean: the drop in year one is mostly
a high season being partly luck, not decline. Do not compose it with another
regression step, and do not read its slope as aging.

Layers:
  `panel`   load and clean the 2000-2026 season panel, score each season in SGP
  `era`     restate every season in the run environment the SGP denominators price
  `comps`   match on (age, SGP) and average the forward paths
  `shape`   fit forward SGP on last year AND this year, kernel-weighted -- the DEFAULT
  `value`   restate any of them as value above the position-aware waiver floor

`shape` is the default matcher because it wins where the decision is. Level matching
under-predicts a star coming off a down year by **3.31 SGP a year** (n=239, out of
sample, the query player removed from the panel so neither estimator can match him to
himself); shape is unbiased there and beats it on every elite slice.

    scripts/backtest_trajectory.py --pool hitter        # +1, hitters
    elite big drop (<70% of prior)  n=239  RMSE 6.03 -> 4.78  bias -3.31 -> -0.15

Re-measure rather than trust: those figures come from that script and nowhere else, so
a regression shows up as a changed table instead of a stale docstring. **Hitters only.**
The pitcher pool has never been validated and shape is its default too -- #313.

`comps` is kept, not deprecated: it is the simpler estimator, it needs only one season,
and it remains the honest baseline any future matcher has to beat.
"""

from .comps import PathPoint, Trajectory, comp_trajectory
from .era import era_normalize, league_rates
from .panel import DEFAULT_PANEL_DIR, load_scored_panel
from .shape import Anchors, Prepared, prepare, shape_trajectory
from .value import best_floor, resolve_slots

__all__ = [
    "DEFAULT_PANEL_DIR",
    "Anchors",
    "PathPoint",
    "Prepared",
    "Trajectory",
    "best_floor",
    "comp_trajectory",
    "era_normalize",
    "league_rates",
    "load_scored_panel",
    "prepare",
    "resolve_slots",
    "shape_trajectory",
]
