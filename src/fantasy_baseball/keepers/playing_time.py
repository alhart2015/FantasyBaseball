"""Project next season's playing time from a player's own multi-year history.

Replaces the one-year gap regression the volume term used to run on. That model saw
only the most recent season, so a durable star who lost half a year to injury was
forecast as a part-time player -- Juan Soto came out at 467 PA off five straight
650+ PA seasons, which ranked him below a fringe regular.

One curve shape serves both pools, fit on 2010-2025:

    volume_next = intercept
                + b1*volume(-1) + b3*volume(-3)
                + b_age*age
                + b_role*role(-1)
                + b_short*shortfall
                [+ b_start*start_share(-1)]        pitchers only

`volume` is PA for hitters and IP for pitchers; `role` is the per-appearance rate --
PA per game played, or innings per appearance.

For pitchers, read `role` and `start_share` as ONE joint term, never separately. A
starter has both a high per-appearance rate and a start share near 1, so the pair is
near-collinear (VIF ~35) and the fit resolves it as a large positive role coefficient
minus a large negative start_share. The sum is stable and the prediction is fine, but
neither number means anything alone, and an off-diagonal pitcher -- an opener, a bulk
reliever, a swingman -- can swing tens of innings on which side of that cancellation
he lands.

**`role` is what separates an injury from a job.** Two hitters both land on 360 PA:
one played 84 games batting second (4.3 PA/game, hurt), the other 140 games in a
platoon (2.6 PA/game, healthy but marginal). Season volume alone cannot tell them
apart, and only the first is likely to return to a full workload. For pitchers the
same term carries something even more basic -- innings per appearance is what makes a
starter a starter -- which is why `start_share` joins it there: innings-per-appearance
alone cannot separate a swingman from a short starter.

Measured, holdout RMSE (leave-one-season-out) against a one-lag model, on the
EXIT-CORRECTED population (see `include_exits` in `lag_panel`):

    hitters    all 175.3 -> 164.6 (t=9.8)   injured 214.9 -> 184.8 (t=5.4)
    pitchers   all  49.9 ->  49.0 (t=5.4)   durable  60.0 ->  59.3 (t=2.9)

Role helps injured hitters most and durable PITCHERS most, which is the right shape:
for a pitcher, role is not a nuance, it is the whole starter/reliever distinction.

(An earlier revision quoted 159.3 -> 155.9 and 47.6 -> 46.7. Those were measured on the
survivor-only population, which excludes career endings and so understates the error
every model makes on it. Same comparison, honester denominator.)

**`shortfall` is HITTERS ONLY, and earns its place there by removing a bias rather
than by adding accuracy.** It is `max(0, best_of_the_two_prior_seasons - volume(-1))`:
how far below his own norm a player fell. One-sided on purpose -- losing 250 PA to a
hamstring says something very different than gaining 250 off a breakout. On the
exit-corrected fit it costs nothing in RMSE (164.63 with, 164.60 without) but takes the
injury-shortened-hitter bias from -24.8 PA (t=-1.99, significant) to -17.0 (t=-1.36,
noise). Accuracy and calibration disagreed and calibration won: a systematic error
against one identifiable class mis-ranks all of them together.

It was DROPPED for pitchers. On the survivor-only fit it removed a -15 IP bias there,
which is what justified it; the exit correction removes that bias by itself, leaving
the term with a NEGATIVE coefficient (-0.026 -- it would subtract from a pitcher who
fell short of his own norm, the opposite of the mechanism) and no measurable job:
RMSE 48.95 with against 48.94 without, injured bias +4.3 against +5.1, both inside
noise. A parameter with a nonsensical sign that buys nothing does not ship.

Tested and dropped, so they are not retried: the second volume lag (adds nothing once
lags 1 and 3 are in), debut flags (`role` already tells a real rookie job from a
fragile veteran, and dropping them IMPROVED the young-player slice), games-lost, an
age quadratic, and a two-stage `games x per-game-rate` product -- better on injured
players but significantly WORSE on durable ones (t=-4.5), because multiplying two
noisy estimates loses where there is no disruption to explain.

Volume is 162-game normalized, so 2020 (60 scheduled games) is comparable. `role` is
NOT normalized: it is already a per-appearance rate, which also makes it readable off
a season still in progress.

Pure and I/O-free. `scripts/build_pt_panel.py` produces the panels this consumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Feature order is coefficient order; a positional zip must not silently mis-pair, so
# these tuples are the single source of truth. `start_share` is pitcher-only.
HITTER_FEATURES: tuple[str, ...] = ("vol1", "vol3", "age", "role", "shortfall")
PITCHER_FEATURES: tuple[str, ...] = ("vol1", "vol3", "age", "role", "start_share")
FEATURES: dict[str, tuple[str, ...]] = {
    "hitter": HITTER_FEATURES,
    "pitcher": PITCHER_FEATURES,
}

FULL_SEASON_GAMES = 162
# Per-appearance ceilings. A handful of games can otherwise compute a nonsense rate off
# a rounding artifact -- 40 PA over 2 games, or a 9-inning complete game as a "rate".
_MAX_ROLE = {"hitter": 5.5, "pitcher": 9.0}
# Panel column names per pool: (volume, appearances).
_PANEL_COLUMNS = {"hitter": ("pa", "games"), "pitcher": ("ip", "games")}


@dataclass(frozen=True)
class PlayingTimeCurve:
    """A fitted playing-time projection. `coefficients` aligns to `FEATURES[kind]`."""

    kind: str
    intercept: float
    coefficients: tuple[float, ...]
    n: int
    rmse: float

    @property
    def features(self) -> tuple[str, ...]:
        return FEATURES[self.kind]

    def as_dict(self) -> dict[str, float]:
        return {
            "intercept": self.intercept,
            **dict(zip(self.features, self.coefficients, strict=True)),
        }

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Projected next-season volume; NaN where the linear form goes negative.

        A player cannot take negative plate appearances or throw negative innings, and
        a negative volume would flip the sign of every counting stat built on it. This
        returns NaN rather than clipping to 0 so the CALLER can fall back to another
        estimator: a hard 0 looks like a real forecast and silently prints a zeroed
        line, where NaN is a signal the curve could not score this player.
        """
        missing = [c for c in self.features if c not in features.columns]
        if missing:
            raise KeyError(f"{self.kind} playing-time features missing {missing}")
        out = pd.Series(self.intercept, index=features.index, dtype=float)
        for name, beta in zip(self.features, self.coefficients, strict=True):
            out = out + beta * features[name].astype(float)
        return out.where(out > 0.0)


def normalize_to_full_season(volume: pd.Series, scheduled_games: pd.Series) -> pd.Series:
    """Scale a season's volume to a 162-game schedule so 2020 is comparable."""
    return volume * FULL_SEASON_GAMES / scheduled_games


def per_appearance(volume: pd.Series, appearances: pd.Series, kind: str) -> pd.Series:
    """Role: volume per appearance -- PA per game, or innings per outing.

    Deliberately NOT schedule-normalized: it is already a per-appearance rate, so a
    60-game 2020, a full season and a season still in progress are all comparable. No
    appearances yields 0.0 rather than NaN, since "did not play" is a real observation
    the curve should see.
    """
    if kind not in _MAX_ROLE:
        raise ValueError(f"kind must be 'hitter' or 'pitcher', got {kind!r}")
    rate = volume.divide(appearances.where(appearances > 0))
    return rate.fillna(0.0).clip(upper=_MAX_ROLE[kind])


def carry_forward_role(
    volumes: Sequence[pd.Series],
    appearances: Sequence[pd.Series],
    kind: str,
    starts: Sequence[pd.Series] | None = None,
) -> tuple[pd.Series, pd.Series | None]:
    """Role (and, for pitchers, start share) from the most recent OBSERVED season.

    `volumes`/`appearances`/`starts` are ordered newest-first: the base season, then the
    one before it, and so on. A player with no row in the base season falls back to the
    next season that has one, because a batting-order slot or a rotation job is sticky
    -- far more so than a workload. Nothing observed anywhere yields 0.0.

    **Both terms are returned together, and that is the point.** For pitchers they are
    one near-collinear joint term (VIF ~35): the prediction is a large positive `role`
    minus a large negative `start_share`, so advancing one without the other is not a
    partial fix, it is a wrong answer. Carrying role forward while leaving start_share
    pinned to the base year gave a rehabbing starter his full starter credit with none
    of the offset -- about +29 IP, a 25-30% inflation. Returning a tuple makes the two
    impossible to separate at a call site.
    """
    if len(volumes) != len(appearances):
        raise ValueError("volumes and appearances must be the same length")
    if kind == "pitcher" and starts is None:
        raise ValueError("pitcher carry-forward requires starts")
    if starts is not None and len(starts) != len(volumes):
        raise ValueError("starts must be the same length as volumes")

    role: pd.Series | None = None
    share: pd.Series | None = None
    for step, (vol, app) in enumerate(zip(volumes, appearances, strict=True)):
        observed = vol.notna()
        step_role = per_appearance(vol, app, kind).where(observed)
        role = step_role if role is None else role.fillna(step_role)
        if starts is not None:
            apps = app.where(app > 0)
            # Sourced from the SAME season index as the role above, never the base year.
            step_share = (starts[step] / apps).where(observed)
            share = step_share if share is None else share.fillna(step_share)
    if role is None:
        raise ValueError("no seasons supplied")
    return role.fillna(0.0), None if share is None else share.fillna(0.0)


def build_features(
    vol1: pd.Series,
    vol2: pd.Series,
    vol3: pd.Series,
    age: pd.Series,
    role: pd.Series,
    kind: str = "hitter",
    start_share: pd.Series | None = None,
) -> pd.DataFrame:
    """Assemble the model matrix for `kind`.

    `vol2` is an INPUT but never a feature: the second lag adds nothing on its own and
    is here only because `shortfall` measures against the better of the two prior
    seasons. A NaN lag means the player did not play, which is a real observation of
    zero -- not missing data.
    """
    if kind not in FEATURES:
        raise ValueError(f"kind must be 'hitter' or 'pitcher', got {kind!r}")
    lags = {
        name: series.fillna(0.0).astype(float)
        for name, series in (("vol1", vol1), ("vol2", vol2), ("vol3", vol3))
    }
    prior_best = pd.concat([lags["vol2"], lags["vol3"]], axis=1).max(axis=1)
    out = pd.DataFrame(
        {
            "vol1": lags["vol1"],
            "vol3": lags["vol3"],
            "age": age.astype(float),
            "role": role.fillna(0.0).astype(float),
            # One-sided on purpose: a surplus above the prior norm is not the mirror
            # image of a shortfall below it. See the module docstring.
            "shortfall": (prior_best - lags["vol1"]).clip(lower=0.0),
        },
        index=vol1.index,
    )
    if kind == "pitcher":
        if start_share is None:
            raise ValueError("pitcher features require start_share")
        out["start_share"] = start_share.fillna(0.0).astype(float)
    return out.loc[:, list(FEATURES[kind])]


def fit_curve(features: pd.DataFrame, target: pd.Series, kind: str = "hitter") -> PlayingTimeCurve:
    """Ordinary least squares of `target` on `FEATURES[kind]` plus an intercept."""
    names = FEATURES[kind]
    frame = features.reindex(columns=list(names)).copy()
    frame["__y"] = target
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) <= len(names) + 1:
        raise ValueError(f"need more than {len(names) + 1} complete rows to fit, got {len(frame)}")
    y = frame.pop("__y").to_numpy()
    design = np.column_stack([np.ones(len(frame)), frame.to_numpy()])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return PlayingTimeCurve(
        kind=kind,
        intercept=float(beta[0]),
        coefficients=tuple(float(b) for b in beta[1:]),
        n=len(frame),
        rmse=float(np.sqrt((resid**2).mean())),
    )


def lag_panel(
    panel: pd.DataFrame,
    kind: str = "hitter",
    *,
    min_recent: float = 0.0,
    include_exits: bool = True,
) -> pd.DataFrame:
    """Reshape a per-player-per-season panel into (features, target) rows.

    `panel` is `scripts/build_pt_panel.py`'s output: one row per (mlbam_id, season)
    across each player's career span, with unobserved seasons carried as NaN.

    In-progress seasons (`partial_season`) are dropped as TARGETS -- training on a
    two-thirds-complete year as though it were finished would teach the curve that
    everyone collapses. They remain available to the caller as features.

    `include_exits` controls whether a career ENDING trains. The panel spans
    first-observed to last-observed season, so the year after a player's final one has
    no row, his age is NaN, and he is dropped -- meaning the curve learns
    `E[volume | he plays again]` and over-forecasts exactly the aging and marginal
    players a keeper decision has to price. With it on, that season's age is derived
    from the prior one and the exit trains as `target = 0`, matching the survivorship
    correction `build_volume_transition` already applies to the persistence share.

    Only ONE exit row per career is added. The guard is the age derivation itself, NOT
    `min_recent`: the season AFTER the exit has no lag row to derive an age from, so it
    drops on the `age.notna()` filter below. `min_recent` also excludes it at the
    300/50 floors the scripts pass, but it is 0.0 by default here and cannot be relied
    on. Long-retired players therefore contribute no tail of zeros either way.
    """
    if kind not in FEATURES:
        raise ValueError(f"kind must be 'hitter' or 'pitcher', got {kind!r}")
    volume, appearances = _PANEL_COLUMNS[kind]
    needed = ["mlbam_id", "season", volume, appearances, "age", "scheduled_games"]
    if kind == "pitcher":
        needed.append("starts")
    for col in needed:
        if col not in panel.columns:
            raise KeyError(f"panel missing {col!r}; got {sorted(panel.columns)}")

    frame = panel.loc[~panel.get("partial_season", False).astype(bool)].copy()
    vol = frame[volume].fillna(0.0)
    app = frame[appearances].fillna(0.0)
    frame["_vol"] = normalize_to_full_season(vol, frame["scheduled_games"])
    frame["_role"] = per_appearance(vol, app, kind)
    if kind == "pitcher":
        frame["_start"] = (frame["starts"].fillna(0.0) / app.where(app > 0)).fillna(0.0)

    wide = frame.pivot_table(index="mlbam_id", columns="season", values="_vol")
    role = frame.pivot_table(index="mlbam_id", columns="season", values="_role")
    ages = frame.pivot_table(index="mlbam_id", columns="season", values="age")
    starts = (
        frame.pivot_table(index="mlbam_id", columns="season", values="_start")
        if kind == "pitcher"
        else None
    )

    rows = []
    for season in sorted(wide.columns):
        lags = [season - 1, season - 2, season - 3]
        if not all(lag in wide.columns for lag in lags):
            continue
        age = ages[season]
        if include_exits:
            age = age.fillna(ages[lags[0]] + 1)
        built = build_features(
            wide[lags[0]],
            wide[lags[1]],
            wide[lags[2]],
            age,
            role[lags[0]],
            kind,
            None if starts is None else starts[lags[0]],
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
    return out.loc[out["vol1"] >= min_recent].reset_index(drop=True)
