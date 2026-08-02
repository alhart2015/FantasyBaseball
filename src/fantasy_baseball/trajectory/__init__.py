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
"""

from .comps import PathPoint, Trajectory, comp_trajectory
from .era import era_normalize, league_rates
from .panel import DEFAULT_PANEL_DIR, load_scored_panel

__all__ = [
    "DEFAULT_PANEL_DIR",
    "PathPoint",
    "Trajectory",
    "comp_trajectory",
    "era_normalize",
    "league_rates",
    "load_scored_panel",
]
