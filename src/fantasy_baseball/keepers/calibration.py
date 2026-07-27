"""Assemble ZiPS-vs-actual year pairs and measure the fit sample.

The one non-negotiable methodological constraint (spec 6.1): the base must be
ZiPS_Y, built knowing only through Y-1, so it has NOT already absorbed year Y --
mirroring production, where ZiPS 2027 has never seen 2026. Using ZiPS_{Y+1} as
the base would fit how much surprise ZiPS already absorbed and drive the
coefficient to zero by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from fantasy_baseball.keepers.actuals import normalize_hitting, normalize_pitching
from fantasy_baseball.keepers.fold import shrink
from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season
from fantasy_baseball.keepers.vintages import load_vintage

# Year Y of each usable (Y, Y+1) pair. 2025 needs a complete 2026 season; 2021 has
# no ZiPS vintage on disk (data/projections starts at 2022).
PAIR_YEARS = (2022, 2023, 2024)

PT_COL = {"hitter": "pa", "pitcher": "ip"}


@dataclass(frozen=True)
class YearPair:
    """One (Y, Y+1) observation set, already aligned on mlbam_id."""

    year: int
    base: pd.DataFrame  # ZiPS_Y rates
    residual: pd.DataFrame  # actual_Y rates - ZiPS_Y rates
    target: pd.DataFrame  # actual_{Y+1} rates
    realized_pt: pd.Series  # actual_Y playing time (drives shrink and gate)
    target_pt: pd.Series  # actual_{Y+1} playing time


LAST_COMPLETE_SEASON = 2025


def build_pairs(
    player_type: str,
    cache_dir: Path,
    projections_root: Path,
    years: tuple[int, ...] = PAIR_YEARS,
) -> list[YearPair]:
    """Assemble (Y, Y+1) observation sets.

    Two properties are load-bearing and were both wrong in an earlier draft:

    * The frames carry the PLAYING TIME column alongside the rates. PT is the
      twelfth coefficient, and spec requirement 12 -- the systematic mean of the
      PT residual -- is the single hardest constraint on the estimator. Stripping
      PT here would make that requirement unaddressable.
    * Membership is `zips INTERSECT actual_Y` ONLY. Intersecting year Y+1 as well would
      precondition the sample on having survived, inflating the measured survival
      rate by 7-9 points AND removing non-survivors before any estimator sees
      them -- making spec requirement 5 unmeasurable. Absentees get a NaN target
      and 0.0 target playing time, which is the honest encoding of "did not play".
    """
    if player_type not in {"hitter", "pitcher"}:
        raise ValueError(f"player_type must be 'hitter' or 'pitcher', got {player_type!r}")
    group = "hitting" if player_type == "hitter" else "pitching"
    normalize = normalize_hitting if player_type == "hitter" else normalize_pitching
    pairs: list[YearPair] = []
    for year in years:
        if year + 1 > LAST_COMPLETE_SEASON:
            # fetch_or_cache never invalidates, so a mid-season pull would freeze
            # permanently. Fail loud rather than cache an in-progress season.
            raise ValueError(
                f"pair {year}->{year + 1} needs a complete {year + 1} season; "
                f"last complete is {LAST_COMPLETE_SEASON}"
            )
        zips_h, zips_p = load_vintage(year, projections_root)
        zips = zips_h if player_type == "hitter" else zips_p
        act_y = normalize(fetch_mlb_season(cache_dir, year, group))
        act_next = normalize(fetch_mlb_season(cache_dir, year + 1, group))
        ids = zips.index.intersection(act_y.index)
        cols = list(zips.columns)  # rates AND the playing-time column
        pt_col = PT_COL[player_type]
        target = act_next.reindex(ids)[cols].copy()
        # A player absent from the year-Y+1 leaderboard has NO Y+1 rate (NaN is the
        # only honest answer) but he does have a well-defined Y+1 MLB playing time
        # of zero. Leaving PT as NaN here would drop every non-survivor from the
        # playing-time fit and evaluation, which is precisely the survivorship
        # deletion spec 6.3 warns against. Rates stay NaN; only PT is filled.
        target[pt_col] = target[pt_col].fillna(0.0)
        pairs.append(
            YearPair(
                year=year,
                base=zips.loc[ids, cols],
                residual=act_y.loc[ids, cols] - zips.loc[ids, cols],
                target=target,
                realized_pt=act_y.loc[ids, pt_col],
                target_pt=target[pt_col],
            )
        )
    return pairs


def survivorship(pairs: list[YearPair], threshold: float) -> pd.DataFrame:
    """Per pair: how many cleared `threshold` in year Y, and how many again in Y+1.

    Fitting on survivors alone measures persistence GIVEN continued play, which
    biases the playing-time coefficient upward. Spec 6.3 requires this measured on
    the actual fit sample, not on the wider MLB population.
    """
    rows = []
    for pair in pairs:
        in_year = pair.realized_pt >= threshold
        survived = in_year & (pair.target_pt >= threshold)
        n_in, n_sur = int(in_year.sum()), int(survived.sum())
        rows.append(
            {
                "year": pair.year,
                "n_matched": len(pair.base),
                "n_in_year": n_in,
                "n_survived": n_sur,
                "survival_rate": (n_sur / n_in) if n_in else float("nan"),
            }
        )
    return pd.DataFrame(rows)


class Fitted(Protocol):
    params: dict[str, float]

    def predict(self, base: pd.Series, residual: pd.Series, weight: pd.Series) -> pd.Series: ...


class Estimator(Protocol):
    name: str

    def fit(
        self,
        pairs: list[YearPair],
        column: str,
        n0: float,
        *,
        shrunk: bool = True,
        weighted: bool = True,
    ) -> Fitted:
        """Fit one coefficient for `column`.

        `shrunk` and `weighted` mirror the evaluation switches in `leave_one_out`
        so a fitted estimator can optimize the SAME loss it is scored on (finding
        A.1). The fixed endpoints ignore them.
        """
        ...


class _FixedK:
    """Endpoint estimator: predict = base + k * weight * residual, k not fitted."""

    def __init__(self, k: float) -> None:
        self.params = {"k": k}

    def predict(self, base: pd.Series, residual: pd.Series, weight: pd.Series) -> pd.Series:
        result: pd.Series = base + self.params["k"] * weight * residual.fillna(0.0)
        return result


class ZeroTransfer:
    """k = 0: ignore the season entirely. This is today's stale-baseline behaviour."""

    name = "k=0"

    def fit(
        self,
        pairs: list[YearPair],
        column: str,
        n0: float,
        *,
        shrunk: bool = True,
        weighted: bool = True,
    ) -> Fitted:
        return _FixedK(0.0)


class FullTransfer:
    """k = 1: move the full (shrunk) surprise."""

    name = "k=1"

    def fit(
        self,
        pairs: list[YearPair],
        column: str,
        n0: float,
        *,
        shrunk: bool = True,
        weighted: bool = True,
    ) -> Fitted:
        return _FixedK(1.0)


def weighted_mse(pred: pd.Series, actual: pd.Series, weight: pd.Series) -> float:
    """Playing-time-weighted MSE, so a 20-PA player's rate cannot dominate."""
    mask = actual.notna() & pred.notna() & (weight > 0)
    if not mask.any():
        return float("nan")
    err = (pred[mask] - actual[mask]) ** 2
    return float((err * weight[mask]).sum() / weight[mask].sum())


def _eval_weight(pair: YearPair, weighted: bool) -> pd.Series:
    """The metric's weight column, per the pre-registered decision (finding A.1).

    Rate coefficients weight by realized year-Y+1 playing time so a 20-PA rate
    cannot dominate a 600-PA one. The playing-time coefficient is UNWEIGHTED:
    weighting the PT target by target_pt is circular, and it assigns weight 0 to
    every non-survivor -- deleting exactly the players whose lost playing time the
    coefficient exists to learn from.
    """
    if weighted:
        return pair.target_pt
    return pd.Series(1.0, index=pair.target_pt.index)


def leave_one_out(
    estimator: Estimator,
    pairs: list[YearPair],
    column: str,
    n0: float,
    *,
    gate: float,
    shrunk: bool = True,
    weighted: bool = True,
) -> pd.DataFrame:
    """Fit on all pairs but one, evaluate on the held-out pair. Spec 6.3.

    `gate` is the realized-playing-time floor from spec 5.4: rows below it are
    excluded from both fitting and evaluation.

    `shrunk=False` is REQUIRED for the playing-time coefficient. The shrink damps
    noisy RATE observations; applying it to the PT residual would damp an injury
    signal in proportion to the playing time the injury suppressed, and would make
    the PT coefficient structurally unable to learn from lost time (spec 5.3).

    `weighted=False` is likewise required for the playing-time coefficient -- see
    `_eval_weight`. Fit and evaluation use the SAME weighting; fitting one loss and
    scoring another would make the held-out comparison meaningless.
    """
    rows = []
    for held in pairs:
        train = [_gated(p, gate) for p in pairs if p.year != held.year]
        fitted = estimator.fit(train, column, n0, shrunk=shrunk, weighted=weighted)
        kept = _gated(held, gate)
        weight = _shrink_weight(kept, n0, shrunk)
        pred = fitted.predict(kept.base[column], kept.residual[column], weight)
        rows.append(
            {
                "estimator": estimator.name,
                "column": column,
                "held_out_year": held.year,
                "n": len(kept.base),
                "error": weighted_mse(pred, kept.target[column], _eval_weight(kept, weighted)),
                **{f"param_{k}": v for k, v in fitted.params.items()},
            }
        )
    return pd.DataFrame(rows)


def _shrink_weight(pair: YearPair, n0: float, shrunk: bool) -> pd.Series:
    """The fold's shrink weight for one pair, or all-ones when unshrunk."""
    if shrunk:
        return shrink(pair.realized_pt, n0)
    return pd.Series(1.0, index=pair.realized_pt.index)


def _gated(pair: YearPair, gate: float) -> YearPair:
    """Restrict a pair to rows clearing the realized-playing-time gate (spec 5.4)."""
    ids = pair.realized_pt.index[pair.realized_pt >= gate]
    return YearPair(
        year=pair.year,
        base=pair.base.loc[ids],
        residual=pair.residual.loc[ids],
        target=pair.target.loc[ids],
        realized_pt=pair.realized_pt.loc[ids],
        target_pt=pair.target_pt.loc[ids],
    )
