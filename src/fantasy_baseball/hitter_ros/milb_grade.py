"""Grade minor-league hitting lines by level: what a line is worth in MLB terms (#435).

Hitting .300 in AAA means more than .300 in Low-A, and .300 in the hitter-friendly
Pacific Coast League means less than .300 in the International League. Three steps:

1. **League-relative.** Each line's rates are divided by its league's rates that season
   (``relative_lines``), which takes out league and era environments. A league with
   fewer than ``MIN_LEAGUE_PA`` PA that season (stray PA filed under the wrong league)
   uses its level's pooled rates instead.
2. **Level factors** (``level_factors``): how much a league-relative rate shrinks on the
   way to MLB, measured on first promotions: players with at least ``MIN_PAIR_PA`` PA
   at a level and at the next level up (or MLB) in the same or the following season,
   who had never batted at the higher level before. Each lower-level season counts
   once (paired with the same season when it can be, else the next), so ``pairs`` is
   that many player-seasons. Veterans going back and forth between AAA and MLB are
   left out: they are not the rookies these factors are for, and they made AAA steals
   look more valuable than they are for prospects (SB x0.85 vs. x0.80 without them).
   A factor is the ratio of the PA-weighted sums of the two relative rates (weight: the
   harmonic mean of the two PA). AAA and AA are measured against MLB directly; the lower levels are
   chained (``CHAIN``), because few of their players reach MLB within a season.
   Factors only use pairs whose later season is inside ``seasons``, so a backtest can
   grade with seasons before the test season only.
3. **Translation** (``translate``): relative rate x factor = an MLB-league-relative rate.

Selection: only players who earned a promotion make pairs, and a promotion follows a
hot stretch, so part of each factor is that luck wearing off. That is wanted here: the
factor answers "what does this line usually turn into", which includes it.

The Mexican League is filed under AAA by the API but left out (``EXCLUDED_LEAGUES``):
mostly veterans, a different talent pool, and almost no players reach MLB from it to
grade it with. Pitchers' plate appearances are left out everywhere.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.features import rates_from_counts
from fantasy_baseball.pitch_data.milb import MILB_LEVELS

MLB = 1  # sportId of the major leagues
EXCLUDED_LEAGUES = {125: "Mexican League"}
MIN_LEAGUE_PA = 5000
MIN_PAIR_PA = 100
MIN_PAIRS = 30
# Measured straight against MLB; the rest step up to the level named here.
DIRECT = (11, 12)
CHAIN = {13: 12, 14: 13, 15: 14, 16: 14}
PROMOTION_LAGS = (0, 1)  # the next level in the same season, or the season after

# Count columns: name here -> the API's stat field / the box score's column.
COUNTS = {
    "pa": "plateAppearances",
    "ab": "atBats",
    "h": "hits",
    "hr": "homeRuns",
    "r": "runs",
    "rbi": "rbi",
    "sb": "stolenBases",
    "bb": "baseOnBalls",
    "k": "strikeOuts",
    "sf": "sacFlies",
}
PER_PA = ("hr", "r", "rbi", "sb", "bb", "k")
STATS = ("avg", *PER_PA, "babip")
KEYS = ["season", "sport_id", "league_id"]


def load_milb_lines(conn: duckdb.DuckDBPyConnection, by_window: bool = False) -> pd.DataFrame:
    """Season x player x level x league totals from the weekly lines (whose league is
    the window's club, so a traded player's PA land in the right league), with the
    player's season age at the level. Pitchers and ``EXCLUDED_LEAGUES`` left out.
    ``by_window``: one row per weekly window too (``window_end``), not season totals."""
    sums = ", ".join(f'sum("stat.{v}") AS {k}' for k, v in COUNTS.items())
    excluded = ", ".join(str(i) for i in EXCLUDED_LEAGUES)
    window = ", w.window_end" if by_window else ""
    return conn.execute(
        f"""
        WITH age AS (
            SELECT season, "player.id" AS player_id, sport_id, max("stat.age") AS age
            FROM milb_season GROUP BY ALL
        )
        SELECT w.season, w."player.id" AS player_id, w.sport_id, w."league.id" AS league_id,
               {sums}, any_value(age.age) AS age{window}
        FROM milb_weekly w
        LEFT JOIN age ON age.season = w.season AND age.player_id = w."player.id"
                     AND age.sport_id = w.sport_id
        WHERE coalesce(w."position.type", '') <> 'Pitcher'
          AND w."league.id" NOT IN ({excluded})
        GROUP BY w.season, w."player.id", w.sport_id, w."league.id"{window}
        HAVING sum("stat.plateAppearances") > 0
        """
    ).df()


def load_mlb_lines(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Season x player MLB box-score totals, shaped like ``load_milb_lines``. Players
    with at least half their PA as the pitcher are left out."""
    sums = ", ".join(f"sum({k}) AS {k}" for k in COUNTS)
    lines = conn.execute(
        f"""
        SELECT year(CAST(game_date AS DATE)) AS season, player_id, {sums},
               coalesce(sum(pa) FILTER (WHERE position = 'P'), 0) AS pa_as_pitcher
        FROM lineups GROUP BY ALL
        """
    ).df()
    hitters = (lines["pa"] > 0) & (lines["pa_as_pitcher"] < 0.5 * lines["pa"])
    return (
        lines[hitters]
        .drop(columns="pa_as_pitcher")
        .assign(sport_id=MLB, league_id=0, age=np.nan)
        .reset_index(drop=True)
    )


def rates(counts: pd.DataFrame) -> pd.DataFrame:
    """``STATS`` from the ``COUNTS`` columns (any other column is ignored); NaN where the
    denominator is 0. AVG and R/HR/RBI/SB per PA are ``features.rates_from_counts``."""
    c = counts[list(COUNTS)].astype(float)
    out = rates_from_counts(c)
    for s in ("bb", "k"):
        out[s] = c[s] / c["pa"].where(c["pa"] > 0)
    bip = c["ab"] - c["k"] - c["hr"] + c["sf"]
    out["babip"] = (c["h"] - c["hr"]) / bip.where(bip > 0)
    return out[list(STATS)]


def league_rates(lines: pd.DataFrame, keys: list[str] = KEYS) -> pd.DataFrame:
    """Rates per season x level x league (index ``keys``, which end in ``league_id``;
    add a column such as ``window_end`` to keep periods apart). A league under
    ``MIN_LEAGUE_PA`` in its period gets its level's pooled rates for the period."""
    counts = list(COUNTS)
    league = lines.groupby(keys)[counts].sum()
    level = lines.groupby(keys[:-1])[counts].sum()
    small = league["pa"] < MIN_LEAGUE_PA
    pooled = level.reindex(league.index.droplevel(keys[-1])).set_axis(league.index)
    return rates(league.where(~small, pooled))


def relative_lines(
    lines: pd.DataFrame, by_league: pd.DataFrame | None = None, keys: list[str] = KEYS
) -> pd.DataFrame:
    """``lines`` plus ``rel_<stat>``: each rate over its league's rate in the same
    period. ``by_league``: ``league_rates`` to divide by (default: from ``lines``)."""
    ref = league_rates(lines, keys) if by_league is None else by_league
    own = rates(lines)
    league = ref.reindex(pd.MultiIndex.from_frame(lines[keys])).set_axis(lines.index)
    return lines.assign(**{f"rel_{s}": own[s] / league[s] for s in STATS})


def player_levels(rel: pd.DataFrame) -> pd.DataFrame:
    """One row per season x player x level: count totals, PA-weighted relative rates
    (a player in two leagues at one level is graded against each), age, and
    ``age_vs_level`` = age minus the level-season's PA-weighted mean age."""
    keys = ["season", "player_id", "sport_id"]
    weighted = rel.assign(
        **{f"_w_{s}": rel[f"rel_{s}"] * rel["pa"] for s in STATS},
        **{f"_n_{s}": rel["pa"].where(rel[f"rel_{s}"].notna(), 0.0) for s in STATS},
    )
    sums = weighted.groupby(keys)[
        [*COUNTS, *(f"_w_{s}" for s in STATS), *(f"_n_{s}" for s in STATS)]
    ].sum()
    out = sums[list(COUNTS)].copy()
    for s in STATS:
        out[f"rel_{s}"] = sums[f"_w_{s}"] / sums[f"_n_{s}"].where(sums[f"_n_{s}"] > 0)
    out["age"] = rel.groupby(keys)["age"].max()
    out = out.reset_index()
    mean_age = level_mean_ages(out).reindex(pd.MultiIndex.from_frame(out[["season", "sport_id"]]))
    return out.assign(age_vs_level=out["age"].to_numpy() - mean_age.to_numpy())


def level_mean_ages(levels: pd.DataFrame) -> pd.Series:
    """PA-weighted mean age per (season, sport_id), over rows with an age (MLB rows
    have none)."""
    known = levels[levels["age"].notna()]
    group = [known["season"], known["sport_id"]]
    ages: pd.Series = (known["age"] * known["pa"]).groupby(group).sum() / known["pa"].groupby(
        group
    ).sum()
    return ages


def season_totals(window_lines: pd.DataFrame) -> pd.DataFrame:
    """``load_milb_lines(by_window=True)`` rows summed back to season x player x level x
    league totals: the same as ``load_milb_lines()`` without a second scan."""
    keys = ["season", "player_id", "sport_id", "league_id"]
    return (
        window_lines.groupby(keys)
        .agg(**{c: (c, "sum") for c in COUNTS}, age=("age", "max"))
        .reset_index()
    )


def _step(levels: pd.DataFrame, lo: int, hi: int, seasons: tuple[int, int]) -> pd.Series:
    """One level -> the next (or MLB): ``pairs`` and a factor per stat."""
    first, last = seasons
    a = levels[(levels["sport_id"] == lo) & (levels["pa"] >= MIN_PAIR_PA)]
    b = levels[(levels["sport_id"] == hi) & (levels["pa"] >= MIN_PAIR_PA)]
    pairs = []
    for lag in PROMOTION_LAGS:
        later = b[b["season"].between(first + lag, last)]
        later = later.assign(season=later["season"] - lag, _lag=lag)
        pairs.append(a.merge(later, on=["season", "player_id"], suffixes=("_lo", "_hi")))
    m = pd.concat(pairs, ignore_index=True)
    # First promotions only: no PA at the higher level in any earlier season.
    debut = levels[levels["sport_id"] == hi].groupby("player_id")["season"].min()
    m = m[~(m["player_id"].map(debut) < m["season"])]
    # Each lower-level season once, with the earliest higher-level season it pairs with.
    m = m.sort_values("_lag", kind="stable").drop_duplicates(["season", "player_id"])
    weight = 2.0 / (1.0 / m["pa_lo"] + 1.0 / m["pa_hi"])
    out = {"pairs": float(len(m))}
    for s in STATS:
        ok = m[f"rel_{s}_lo"].notna() & m[f"rel_{s}_hi"].notna()
        num = (weight[ok] * m.loc[ok, f"rel_{s}_hi"]).sum()
        den = (weight[ok] * m.loc[ok, f"rel_{s}_lo"]).sum()
        out[s] = num / den if den > 0 else np.nan
    return pd.Series(out)


def level_factors(levels: pd.DataFrame, seasons: tuple[int, int]) -> pd.DataFrame:
    """Factor to MLB per level (index: sportId) and stat, from pairs that fall inside
    ``seasons`` (inclusive, both ends of a pair). ``levels``: ``player_levels`` rows for
    the minor leagues and MLB together. ``pairs`` counts the pairs of the level's own
    step. A level with fewer than ``MIN_PAIRS`` pairs, or chained to one, is left out."""
    out: dict[int, pd.Series] = {}
    for level in DIRECT:
        step = _step(levels, level, MLB, seasons)
        if step["pairs"] >= MIN_PAIRS:
            out[level] = step
    for level, up in CHAIN.items():  # in order: each one's target is already done
        step = _step(levels, level, up, seasons)
        if step["pairs"] >= MIN_PAIRS and up in out:
            out[level] = pd.concat([step[["pairs"]], step[list(STATS)] * out[up][list(STATS)]])
    # Explicit columns, so a window without enough pairs at any level still has them.
    frame = pd.DataFrame(out, index=["pairs", *STATS], dtype=float).T
    frame.index.name = "sport_id"
    return frame


def translate(levels: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """``levels`` plus ``mlb_rel_<stat>`` = relative rate x the level's factor (NaN for a
    level without one, and for MLB rows)."""
    f = factors.reindex(levels["sport_id"]).set_axis(levels.index)
    return levels.assign(**{f"mlb_rel_{s}": levels[f"rel_{s}"] * f[s] for s in STATS})


def factor_table(factors: pd.DataFrame) -> pd.DataFrame:
    """``factors`` with level names for an index, for printing."""
    return factors.rename(index=MILB_LEVELS)
