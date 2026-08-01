"""Project next season's playing time from a player's own multi-year history.

Replaces the one-year gap regression the volume term used to run on. That model saw
only the most recent season, so a durable star who lost half a year to injury was
forecast as a part-time player -- Juan Soto came out at 467 PA off five straight
650+ PA seasons, which ranked him below a fringe regular.

The curve, fit on 2010-2025 (n~3200 player-seasons, 12 target seasons):

    PA_next = intercept
            + b1*PA(-1) + b2*PA(-2) + b3*PA(-3)
            + b_age*age
            + b_has2*has_2nd_season + b_has3*has_3rd_season
            + b_short*shortfall

**`shortfall` is the term that matters, and it is deliberately one-sided.** It is
`max(0, best_of_the_two_prior_seasons - PA(-1))`: how far below his own established
norm a player fell last year. Only the shortfall gets a coefficient, never a surplus,
because the two are not symmetric -- losing 250 PA to a hamstring says something very
different about next year than gaining 250 PA off a breakout. Measured: players with a
600+ PA history who lost 150+ PA won back about a quarter of it, and a plain one-year
model under-forecast them by 70-90 PA systematically. The hinge removes roughly 80% of
that bias (+70.3 -> +17.8 PA) while also being the best model on the full population.

`has2`/`has3` separate "did not play" from "did not exist". A rookie's second- and
third-prior seasons are structural zeros, not evidence of fragility, and without these
flags the curve reads a young everyday player as an injury risk.

Everything is 162-game normalized, so 2020 (60 scheduled games) is comparable.

Pure and I/O-free. `scripts/build_pt_panel.py` produces the panel this consumes; the
fit is driven from `scripts/keeper_persistence.py --pt-curve`.

**Hitters only.** The pitcher IP panel is not built yet, so pitcher volume still runs
on the one-year gap model in `persistence.py`. That asymmetry is real and is why the
two pools' volume terms are not directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Feature order is the coefficient order. A positional zip over these must not
# silently mis-pair, so the tuple is the single source of truth.
PA_FEATURES: tuple[str, ...] = ("pa1", "pa2", "pa3", "age", "has2", "has3", "shortfall")

# Seasons a player must be past his debut for a lag to mean "did not play" rather than
# "was not yet in the majors".
_LAG_SEASONS = (2, 3)
FULL_SEASON_GAMES = 162


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


def build_features(
    pa1: pd.Series,
    pa2: pd.Series,
    pa3: pd.Series,
    age: pd.Series,
    seasons_since_debut: pd.Series,
) -> pd.DataFrame:
    """Assemble the model matrix from three lags plus age and career length.

    A NaN lag means the player did not play that season, which is a real observation of
    zero -- not missing data. `has2`/`has3` carry the distinction between that and not
    having debuted yet, so the zero is not read as fragility for a rookie.
    """
    lags = {
        name: series.fillna(0.0).astype(float)
        for name, series in (("pa1", pa1), ("pa2", pa2), ("pa3", pa3))
    }
    ssd = seasons_since_debut.astype(float)
    prior_best = pd.concat([lags["pa2"], lags["pa3"]], axis=1).max(axis=1)
    return pd.DataFrame(
        {
            **lags,
            "age": age.astype(float),
            **{f"has{n}": (ssd >= n).astype(float) for n in _LAG_SEASONS},
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


def lag_panel(panel: pd.DataFrame, *, volume: str = "pa", min_recent: float = 0.0) -> pd.DataFrame:
    """Reshape a per-player-per-season panel into (features, target) rows.

    `panel` is `scripts/build_pt_panel.py`'s output: one row per (mlbam_id, season)
    across each player's career span, with unobserved seasons carried as NaN.

    In-progress seasons (`partial_season`) are dropped as TARGETS -- training on a
    two-thirds-complete year as though it were finished would teach the curve that
    everyone collapses. They are still available to the caller as features.
    """
    for col in ("mlbam_id", "season", volume, "age", "seasons_since_debut", "scheduled_games"):
        if col not in panel.columns:
            raise KeyError(f"panel missing {col!r}; got {sorted(panel.columns)}")
    frame = panel.loc[~panel.get("partial_season", False).astype(bool)].copy()
    frame[volume] = normalize_to_full_season(frame[volume].fillna(0.0), frame["scheduled_games"])

    wide = frame.pivot_table(index="mlbam_id", columns="season", values=volume)
    ages = frame.pivot_table(index="mlbam_id", columns="season", values="age")
    debut = frame.pivot_table(index="mlbam_id", columns="season", values="seasons_since_debut")

    rows = []
    for season in sorted(wide.columns):
        lags = [season - 1, season - 2, season - 3]
        if not all(lag in wide.columns for lag in lags):
            continue
        built = build_features(
            wide[lags[0]], wide[lags[1]], wide[lags[2]], ages[season], debut[season]
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
