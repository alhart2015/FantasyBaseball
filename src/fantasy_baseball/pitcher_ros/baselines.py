"""Two deliberately simple pitcher projections, scored next to FanGraphs as a floor.

* ``league_avg``: every pitcher gets the league's rates. Knows nothing about anyone.
* ``marcel``: Marcel-style, the hitter one (``hitter_ros/baselines.py``) with batters
  faced for PA. A weighted sum of the pitcher's own counts -- this season so far x6,
  last season x5, the two seasons before x3.5 each -- plus 1200 batters faced of
  league-average pitching, then divided out. ERA and WHIP regress per out, over the
  outs those 1200 batters make at the league's rate. No aging, no park, no Statcast,
  no role.

Both use only games before the projection date, plus league rates from the
``LEAGUE_SEASONS`` seasons before the projected one. A pitcher with no MLB games yet
gets league average from both.
"""

from __future__ import annotations

import pandas as pd

from fantasy_baseball.pitcher_ros.stats import (
    COUNTS,
    PER_BF,
    TARGETS,
    rates_from_counts,
    totals,
)

MARCEL_WEIGHTS = {"std": 6.0, "p1": 5.0, "older": 3.5}
REGRESS_BF = 1200.0
LEAGUE_SEASONS = 3


def league_rates(games: pd.DataFrame, season: int) -> pd.Series:
    """League rates over the ``LEAGUE_SEASONS`` seasons before ``season``, plus
    ``outs_per_bf``."""
    prior = games[games["season"].between(season - LEAGUE_SEASONS, season - 1)]
    if prior.empty:
        raise ValueError(f"no season before {season} to take league rates from")
    sums = prior[["bf", "outs", "k", "w", "sv", "er", "h", "bb"]].sum()
    rates: pd.Series = rates_from_counts(pd.DataFrame([sums])).iloc[0]
    rates["outs_per_bf"] = sums["outs"] / sums["bf"]
    return rates


def league_average(players: pd.Index, league: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({s: league[s] for s in TARGETS}, index=players)


def marcel(
    players: pd.Index, games: pd.DataFrame, season: int, before: pd.Timestamp, league: pd.Series
) -> pd.DataFrame:
    """Marcel for ``players`` in ``season``, from games before ``before`` (Jan 1 of
    ``season`` for a preseason projection, the snapshot date mid-season)."""
    known = games[games["game_date"] < before]
    windows = {
        "std": known[known["season"] == season],
        "p1": known[known["season"] == season - 1],
        "older": known[known["season"].between(season - 3, season - 2)],
    }
    weighted = pd.DataFrame(0.0, index=players, columns=list(COUNTS))
    for w, g in windows.items():
        weighted += MARCEL_WEIGHTS[w] * totals(g).reindex(players, fill_value=0)
    out = pd.DataFrame(
        {s: (weighted[s] + REGRESS_BF * league[s]) / (weighted["bf"] + REGRESS_BF) for s in PER_BF},
        index=players,
    )
    regress_outs = REGRESS_BF * league["outs_per_bf"]
    outs = weighted["outs"] + regress_outs
    out["era"] = 27 * (weighted["er"] + regress_outs * league["era"] / 27) / outs
    out["whip"] = 3 * (weighted["h"] + weighted["bb"] + regress_outs * league["whip"] / 3) / outs
    return out[list(TARGETS)]


def baseline_predictions(
    players: pd.Index, games: pd.DataFrame, season: int, before: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Both baselines for ``players``, rates indexed by ``player_id``."""
    league = league_rates(games, season)
    return {
        "marcel": marcel(players, games, season, before, league),
        "league_avg": league_average(players, league),
    }
