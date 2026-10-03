"""Two deliberately simple projections, scored next to the net as a floor (#410).

* ``league_avg``: every player gets the league's rates. Knows nothing about anyone.
* ``marcel``: Marcel-style. A weighted sum of the player's own counts -- this season so
  far x6, last season x5, the two seasons before x3.5 each -- plus 1200 PA of
  league-average performance, then divided out. No aging, no park, no Statcast.

Both use only what the training table's inputs use (counts before the as-of date) plus
league rates from the three complete seasons before the test season. If either one
scores close to FanGraphs or the net, the metric is rewarding caution more than skill.
"""

from __future__ import annotations

import pandas as pd

from fantasy_baseball.hitter_ros.features import COUNTS, TARGETS, rates_from_counts

MARCEL_WEIGHTS = {"std": 6.0, "p1": 5.0, "older": 3.5}
REGRESS_PA = 1200.0
LEAGUE_SEASONS = 3
ID_COLS = ["player_id", "season", "week", "as_of"]


def league_rates(table: pd.DataFrame, season: int) -> pd.Series:
    """League rates over the complete seasons in the ``LEAGUE_SEASONS`` before ``season``."""
    prior = table[
        table["season"].between(season - LEAGUE_SEASONS, season - 1)
        & table["season_complete"]
        & (table["week"] == 0)
    ]
    if prior.empty:
        raise ValueError(f"no complete season before {season} to take league rates from")
    totals = pd.DataFrame([{c: prior[f"ros_{c}"].sum() for c in COUNTS}])
    rates: pd.Series = rates_from_counts(totals).iloc[0]
    rates["ab_per_pa"] = totals["ab"].iloc[0] / totals["pa"].iloc[0]
    return rates


def league_average(rows: pd.DataFrame, league: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({s: league[s] for s in TARGETS}, index=rows.index)


def marcel(rows: pd.DataFrame, league: pd.Series) -> pd.DataFrame:
    def weighted(c: str) -> pd.Series:
        w = MARCEL_WEIGHTS
        older = rows[f"p3_{c}"] - rows[f"p1_{c}"]
        return w["std"] * rows[f"std_{c}"] + w["p1"] * rows[f"p1_{c}"] + w["older"] * older

    pa, ab = weighted("pa"), weighted("ab")
    regress_ab = REGRESS_PA * league["ab_per_pa"]
    out = pd.DataFrame(
        {
            s: (weighted(s) + REGRESS_PA * league[s]) / (pa + REGRESS_PA)
            for s in ("r", "hr", "rbi", "sb")
        },
        index=rows.index,
    )
    out["avg"] = (weighted("h") + regress_ab * league["avg"]) / (ab + regress_ab)
    return out[list(TARGETS)]


def baseline_predictions(table: pd.DataFrame, season: int) -> dict[str, pd.DataFrame]:
    """Both baselines for every row of ``season``, shaped like the net's predictions."""
    rows = table[table["season"] == season]
    league = league_rates(table, season)
    return {
        name: pd.concat([rows[ID_COLS], fn(rows, league)], axis=1)
        for name, fn in (("league_avg", league_average), ("marcel", marcel))
    }
