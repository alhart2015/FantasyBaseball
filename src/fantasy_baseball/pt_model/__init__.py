"""Playing-time projection: how much a player will play in season N (#290).

Two things in this repo are about playing time and they are not the same:

* `utils.playing_time` -- the VARIANCE SHAPE the Monte Carlo samples around a
  playing-time volume it was already handed (mean scale, CV, quantile ladder).
* this package -- the MODEL that produces that volume in the first place, from a
  player's prior-season history.

They should eventually meet (MC availability is a listed consumer of #290), which
is exactly why they get distinct names now instead of two modules called
`playing_time`.

Distinct too from the keeper composite's `pt` family, which RANKS observed playing
time. That family is the first consumer this model is meant to back.

* `panel` -- the historical per-player-per-season training panel (#291). Pure; the
  fetching and caching live in `keepers.mlb_stats` and `scripts/build_pt_panel.py`.
"""

from __future__ import annotations

from fantasy_baseball.pt_model.panel import (
    HITTER_PANEL_COLUMNS,
    build_hitter_panel,
)

__all__ = [
    "HITTER_PANEL_COLUMNS",
    "build_hitter_panel",
]
