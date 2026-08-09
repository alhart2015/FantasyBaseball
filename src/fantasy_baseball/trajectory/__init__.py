"""Career SGP trajectory from historical comparables (#303).

Answers: a player is on pace for N SGP in his age-A season -- what does the rest of
his career look like? The estimate is the forward path of every historical
player-season that started from the same place, with players who left the league
scored as the zero they are worth to a roster slot.

DELIBERATELY STANDALONE. This does not read from, write to, or alter keeper value
(`keepers/`, now ingest and normalization only). It is its own
metric to explore; if it eventually earns the keeper forecast's job, that is a later
and explicit decision.

What it produces is `E[future SGP | age, current SGP]` -- a forecast, NOT an aging
curve. It already contains regression to the mean: the drop in year one is mostly
a high season being partly luck, not decline. Do not compose it with another
regression step, and do not read its slope as aging.

Layers:
  `panel`   load and clean the 2000-2026 season panel, score each season in SGP
  `era`     restate every season in the run environment the SGP denominators price
  `model`   the shapes and panel semantics an estimator returns and relies on
  `shape`   fit forward SGP on last year AND this year, kernel-weighted -- the estimator
  `value`   restate any of them as value above the position-aware waiver floor

`shape` is the default matcher because it wins where the decision is. Level matching
under-predicts a star coming off a down year by **3.31 SGP a year** (n=239, out of
sample, the query player removed from the panel so neither estimator can match him to
himself); shape is unbiased there and beats it on every elite slice.

    elite big drop (<70% of prior)  n=239  RMSE 6.03 -> 4.78  bias -3.31 -> -0.15

Measured 2026-08-03 at 26d26149, the last commit that still carried both matchers and
`scripts/backtest_trajectory.py`. That harness was retired with them (#325), so these
figures can no longer be re-run -- check them out of git history rather than trusting
this paragraph.

**Both pools are validated.** #313 closed 2026-08-03 (PR #326) with the pitcher run:
shape beats level matching on every well-powered slice there too, but by fixing
CALIBRATION rather than by discriminating -- bias -1.69 -> +0.17 on the elite slice,
against a 53% win rate on the big-drop slice, which is a coin flip. It also beats
`track`, and on pitchers `track` is worse than plain level matching: shape ADAPTS the
weight on the prior season to the pool where track assumes it.

`comps` -- level matching and the `track` hard-band variant -- was RETIRED in #325 once
shape had beaten it out of sample in both pools. `model` holds the primitives the two
had shared. The evidence that licensed the removal is in PR #353 and issue #313; the
harness that produced it went with the matchers, so re-deriving those tables means
reading the git history rather than re-running a script.
"""

from .era import era_normalize, league_rates
from .model import PathPoint, Trajectory
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
    "era_normalize",
    "league_rates",
    "load_scored_panel",
    "prepare",
    "resolve_slots",
    "shape_trajectory",
]
