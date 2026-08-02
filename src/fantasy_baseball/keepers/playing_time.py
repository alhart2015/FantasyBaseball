"""Project next season's playing time from a player's own multi-year history.

Replaces the one-year gap regression the volume term used to run on. That model saw
only the most recent season, so a durable star who lost half a year to injury was
forecast as a part-time player -- Juan Soto came out at 467 PA off five straight
650+ PA seasons, which ranked him below a fringe regular.

The curve, fit on 2010-2025 (n~3200 player-seasons, 12 target seasons):

    PA_next = intercept
            + b1*PA(-1) + b3*PA(-3)
            + b_age*age
            + b_ppg*PA_per_game(-1)
            + b_short*shortfall

**`ppg1` -- plate appearances per GAME played -- does the most work (t = 12.2), and it
is what separates an injury from a bad role.** Two players can both land on 360 PA: one
played 84 games batting second (4.3 PA/game, hurt), the other 140 games in a platoon
(2.6 PA/game, healthy but marginal). Season PA alone cannot tell them apart; PA-per-game
says which is which, and the first is far likelier to return to a full workload. Adding
it cut holdout RMSE from 159.3 to 155.9 overall and from 179.6 to 166.8 on
injury-shortened stars, both at t > 5, with no cost to durable regulars.

**`shortfall` stays, and only because it removes a bias.** It is
`max(0, best_of_the_two_prior_seasons - PA(-1))`: how far below his own established
norm a player fell. One-sided on purpose -- losing 250 PA to a hamstring says something
very different than gaining 250 PA off a breakout. On accuracy alone `ppg1` makes it
redundant, but dropping it leaves a SIGNIFICANT -26 PA under-forecast on the
injury-shortened slice (t = -2.6); with it, that falls to -13 (t = -1.3, noise).
Accuracy and calibration disagreed here and calibration won: a systematic error against
one identifiable class of player mis-ranks all of them together.

Deliberately NOT in the model, each tested and dropped: `pa2` (the second lag adds
nothing once `pa1` and `pa3` are present), `has2`/`has3` debut flags (`ppg1` already
distinguishes a real rookie role from a fragile veteran, and dropping them IMPROVED the
young-player slice), `gshort` (games lost -- absorbed by `ppg1` plus `shortfall`), and a
two-stage `games x PA-per-game` product, which beat this model on injured players but
was significantly WORSE on durable regulars (t = -4.5): multiplying two noisy estimates
loses to one where there is no disruption to explain.

Everything is 162-game normalized, so 2020 (60 scheduled games) is comparable.

Pure and I/O-free. `scripts/build_pt_panel.py` produces the panel this consumes.

**Hitters only.** The pitcher IP panel is not built, so pitcher volume still runs on
the one-year gap model in `persistence.py`. That asymmetry is real and is why the two
pools' volume terms are not directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Feature order is the coefficient order. A positional zip over these must not
# silently mis-pair, so the tuple is the single source of truth.
PA_FEATURES: tuple[str, ...] = ("pa1", "pa3", "age", "ppg1", "shortfall")

FULL_SEASON_GAMES = 162
# A real everyday hitter cannot exceed this; a two-game cameo can otherwise compute a
# nonsense per-game rate off a rounding artifact.
_MAX_PA_PER_GAME = 5.5


@dataclass(frozen=True)
class PlayingTimeCurve:
    """A fitted playing-time projection. `coefficients` is aligned to `PA_FEATURES`."""

    intercept: float
    coefficients: tuple[float, ...]
    n: int
    rmse: float

    def as_dict(self) -> dict[str, float]:
        return {
            "intercept": self.intercept,
            **dict(zip(PA_FEATURES, self.coefficients, strict=True)),
        }

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Projected next-season playing time, floored at zero.

        A player cannot take negative plate appearances; the linear form can produce
        one for an old player with almost no recent history, and a negative volume
        would flip the sign of every counting stat built on it.
        """
        missing = [c for c in PA_FEATURES if c not in features.columns]
        if missing:
            raise KeyError(f"playing-time features missing {missing}")
        out = pd.Series(self.intercept, index=features.index, dtype=float)
        for name, beta in zip(PA_FEATURES, self.coefficients, strict=True):
            out = out + beta * features[name].astype(float)
        return out.clip(lower=0.0)


def normalize_to_full_season(volume: pd.Series, scheduled_games: pd.Series) -> pd.Series:
    """Scale a season's volume to a 162-game schedule so 2020 is comparable."""
    return volume * FULL_SEASON_GAMES / scheduled_games


def plate_appearances_per_game(pa: pd.Series, games: pd.Series) -> pd.Series:
    """Role quality: plate appearances per game actually played.

    Deliberately NOT normalized to a full schedule -- it is already a per-game rate, so
    a 60-game 2020 and a 162-game season are directly comparable, and so is a season
    still in progress. A player with no games yields 0.0 rather than NaN: "did not play"
    is a real observation the curve should see.
    """
    rate = pa.divide(games.where(games > 0))
    return rate.fillna(0.0).clip(upper=_MAX_PA_PER_GAME)


def build_features(
    pa1: pd.Series,
    pa2: pd.Series,
    pa3: pd.Series,
    age: pd.Series,
    ppg1: pd.Series,
) -> pd.DataFrame:
    """Assemble the model matrix.

    `pa2` is an INPUT but not a feature: the second lag adds nothing on its own and is
    here only because `shortfall` measures against the better of the two prior seasons.

    A NaN lag means the player did not play that season, which is a real observation of
    zero -- not missing data.
    """
    lags = {
        name: series.fillna(0.0).astype(float)
        for name, series in (("pa1", pa1), ("pa2", pa2), ("pa3", pa3))
    }
    prior_best = pd.concat([lags["pa2"], lags["pa3"]], axis=1).max(axis=1)
    return pd.DataFrame(
        {
            "pa1": lags["pa1"],
            "pa3": lags["pa3"],
            "age": age.astype(float),
            "ppg1": ppg1.fillna(0.0).astype(float),
            # One-sided on purpose: a surplus above the prior norm is not the mirror
            # image of a shortfall below it. See the module docstring.
            "shortfall": (prior_best - lags["pa1"]).clip(lower=0.0),
        },
        index=pa1.index,
    )


def fit_curve(features: pd.DataFrame, target: pd.Series) -> PlayingTimeCurve:
    """Ordinary least squares of `target` on `PA_FEATURES` plus an intercept."""
    frame = features.reindex(columns=list(PA_FEATURES)).copy()
    frame["__y"] = target
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) <= len(PA_FEATURES) + 1:
        raise ValueError(
            f"need more than {len(PA_FEATURES) + 1} complete rows to fit, got {len(frame)}"
        )
    y = frame.pop("__y").to_numpy()
    design = np.column_stack([np.ones(len(frame)), frame.to_numpy()])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return PlayingTimeCurve(
        intercept=float(beta[0]),
        coefficients=tuple(float(b) for b in beta[1:]),
        n=len(frame),
        rmse=float(np.sqrt((resid**2).mean())),
    )


def lag_panel(panel: pd.DataFrame, *, min_recent: float = 0.0) -> pd.DataFrame:
    """Reshape a per-player-per-season panel into (features, target) rows.

    `panel` is `scripts/build_pt_panel.py`'s output: one row per (mlbam_id, season)
    across each player's career span, with unobserved seasons carried as NaN.

    In-progress seasons (`partial_season`) are dropped as TARGETS -- training on a
    two-thirds-complete year as though it were finished would teach the curve that
    everyone collapses. They remain available to the caller as features.
    """
    for col in ("mlbam_id", "season", "pa", "games", "age", "scheduled_games"):
        if col not in panel.columns:
            raise KeyError(f"panel missing {col!r}; got {sorted(panel.columns)}")
    frame = panel.loc[~panel.get("partial_season", False).astype(bool)].copy()
    frame["pa162"] = normalize_to_full_season(frame["pa"].fillna(0.0), frame["scheduled_games"])
    frame["ppg"] = plate_appearances_per_game(frame["pa"].fillna(0.0), frame["games"].fillna(0.0))

    wide = frame.pivot_table(index="mlbam_id", columns="season", values="pa162")
    ppg = frame.pivot_table(index="mlbam_id", columns="season", values="ppg")
    ages = frame.pivot_table(index="mlbam_id", columns="season", values="age")

    rows = []
    for season in sorted(wide.columns):
        lags = [season - 1, season - 2, season - 3]
        if not all(lag in wide.columns for lag in lags):
            continue
        built = build_features(
            wide[lags[0]], wide[lags[1]], wide[lags[2]], ages[season], ppg[lags[0]]
        )
        built["target"] = wide[season].fillna(0.0)
        built["season"] = season
        built["mlbam_id"] = wide.index
        # A player with no trace in any lag season had not debuted; he is not a
        # zero-playing-time observation, he is absent.
        built = built.loc[wide[lags].notna().any(axis=1)]
        rows.append(built.loc[built["age"].notna()])
    if not rows:
        raise ValueError("panel spans too few seasons to form a single lagged row")
    out = pd.concat(rows, ignore_index=True)
    return out.loc[out["pa1"] >= min_recent].reset_index(drop=True)
