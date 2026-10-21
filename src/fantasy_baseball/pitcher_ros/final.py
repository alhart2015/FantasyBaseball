"""The final pitcher projection: ours blended with FanGraphs (#499).

``ours`` (the pitcher net, ``model``) stays independent of FanGraphs: no FanGraphs
stats, shares or closer calls go into it, so it can be improved and judged on its own.
This is the separate step after it: per stat, ``w x ours + (1 - w) x FanGraphs``, where
FanGraphs is the plain average of its systems' rates. The weights differ before the
season and during it (``WEIGHTS``), because what FanGraphs knows that we don't differs:
before the season it is mostly the closer calls they take from news (a perfect "who
opens as closer" flag only tied FanGraphs' saves: #499), during it our net reads the
season so far better.

``WEIGHTS`` come from ``scripts/fit_pitcher_blend.py`` on run 035-saves-net, picked by
MSE over every scored pitcher on a 0.05 grid, with FanGraphs = the average of
``SYSTEMS[span]``:

* Preseason (2022-2026 pooled; Steamer + ZiPS, the systems on file every season):
  picked leaving each season out, the weights moved at most 0.10 from these, and the
  blend beat both ours and FanGraphs on K, W, SV and ERA over all pitchers
  (fantasy-relevant: K and SV beat both; ERA and WHIP tied ours).
* Mid-season (2026's 13 snapshots only, the one season with FanGraphs rest-of-season
  files; ATC + OOPSY + Steamer): **not validated**. They are fitted on the rows they
  are scored on. No honest held-out check exists yet: every snapshot's actuals run to
  the end of the same season, so any split of one season's snapshots shares games.
  In-sample, SV moved most (0.55); the rest are near all ours. Provisional: refit with
  a season held out once a second season of snapshots is stored.

* Board (``BOARD``, ``scripts/fit_board_blend.py``): ours **age-corrected**
  (``aging``, its fit leaving the season out) against the draft board's own FanGraphs
  blend, built and matched exactly as the board's (``draft.our_players.board_blend``:
  Steamer + ZiPS 2022-2025, all five 2026), as published, its K, W, SV per its systems'
  batters faced (TBF). 2022-2026, every pitcher the board would blend (projected 100+
  BF, ~700-750 a season) who pitched: weights picked leaving each season out moved at
  most 0.15 from these, and with them the blend beat the FanGraphs blend on every stat
  (K 118.1 vs 122.1, W, ERA, WHIP 100 sure; SV 9.31 vs 9.66, 95) and ours on K, SV,
  ERA, WHIP (100) with W 91. :func:`board_rates`.

The backtest scores only pitchers every system projects, so the weights were picked
on those. ``final_rates`` blends every pitcher anyone projects: FanGraphs is the
average of the systems that project him (``fg_average``), and a pitcher only one side
projects gets that side's rates.

Refit each season (the script prints the fitted weights next to these, and warns when
the run's FanGraphs systems differ from ``SYSTEMS``).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from fantasy_baseball import blend_fit
from fantasy_baseball.pitcher_ros.evaluate import squared_error
from fantasy_baseball.pitcher_ros.stats import TARGETS

PRESEASON = "preseason"
MIDSEASON = "midseason"
BOARD = "board"
# Weight on ours per stat (1 = all ours, 0 = all FanGraphs).
WEIGHTS: dict[str, dict[str, float]] = {
    PRESEASON: {"k": 0.50, "w": 0.70, "sv": 0.30, "era": 0.50, "whip": 0.75},
    MIDSEASON: {"k": 0.85, "w": 1.00, "sv": 0.55, "era": 0.90, "whip": 0.95},
    BOARD: {"k": 0.45, "w": 0.75, "sv": 0.30, "era": 0.50, "whip": 0.65},
}
# The FanGraphs systems each span's WEIGHTS were fitted with (their average).
SYSTEMS: dict[str, tuple[str, ...]] = {
    PRESEASON: ("steamer", "zips"),
    MIDSEASON: ("atc", "oopsy", "steamer"),
}
# How pitchers are scored (pitcher_ros.evaluate): the fit's MSE.
SCORING = blend_fit.Scoring(TARGETS, squared_error)


def blend_rates(ours: pd.DataFrame, fg: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    """Per stat, ``weights[stat] x ours + (1 - weights[stat]) x fg``. Both are rates
    indexed by ``player_id`` with every stat in ``TARGETS`` (K, W, SV per batter faced;
    ERA, WHIP). Every pitcher in either comes back: one only side projects (or projects
    a stat for) gets that side's rate."""
    return blend_fit.blend_rates(ours, fg, weights, TARGETS)


def fg_average(systems: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Per pitcher and stat, the plain average of the systems that project it (rates
    indexed by ``player_id``, every stat in ``TARGETS``). Unlike ``evaluate.blend``,
    which keeps only the pitchers every system covers, a pitcher one system misses
    still gets the others' average."""
    if not systems:
        raise ValueError("no FanGraphs systems")
    stacked = pd.concat([f[list(TARGETS)].astype(float) for f in systems.values()])
    return stacked.groupby(level=0).mean()


def final_rates(ours: pd.DataFrame, systems: Mapping[str, pd.DataFrame], span: str) -> pd.DataFrame:
    """The final projection: ``blend_rates`` of ours with ``fg_average(systems)`` at
    ``WEIGHTS[span]``. ``systems`` must be ``SYSTEMS[span]``, the ones the weights were
    fitted with: a different FanGraphs average needs its own fit."""
    if span not in SYSTEMS:  # BOARD's FanGraphs side is the board's blend: board_rates
        raise ValueError(f"span must be one of {sorted(SYSTEMS)}, not {span!r}")
    if set(systems) != set(SYSTEMS[span]):
        raise ValueError(
            f"{span} weights were fitted with FanGraphs = {list(SYSTEMS[span])}, "
            f"got {sorted(systems)}; refit (scripts/fit_pitcher_blend.py) first"
        )
    return blend_rates(ours, fg_average(systems), WEIGHTS[span])


def board_rates(ours: pd.DataFrame, board_fg: pd.DataFrame) -> pd.DataFrame:
    """The draft board's final pitcher rates: ``blend_rates`` of ours (age-corrected)
    with the board's FanGraphs blend (its rates, as published) at ``WEIGHTS[BOARD]``."""
    return blend_rates(ours, board_fg, WEIGHTS[BOARD])
