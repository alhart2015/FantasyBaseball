"""Restate every season in one run environment before comparing across eras.

Pooling 2000 with 2025 is only legitimate if a stat means the same thing in both. It
does not. Measured on the top-120 hitters / top-90 pitchers by playing time:

    HR/600  16.0 (2014) -> 26.3 (2019)
    SB/600  14.3 (2024) ->  8.4 (2021)      the 2023 pickoff / base-size rules
    K/9      7.54 (2010) ->  9.57 (2021)

A 2004 stolen base and a 2024 stolen base are not the same asset, so a comp matched
on raw SGP would be matching on the era as much as on the player.

Each season's category rates are scaled by `reference_rate / season_rate`, where the
reference is the volume-weighted LEAGUE rate over `REFERENCE_SEASONS`. That window is
not arbitrary: it is what `config/league.yaml`'s SGP denominators were calibrated on
(adopted 2026-07-05 from 2023-2025 finals), so a normalized season is expressed in
the currency the denominators price.

**Rates are league-wide, not taken from a top-N rosterable pool.** A run environment
is a property of the league, and restricting to a pool conflates it with how talent
happens to be concentrated. It also breaks outright on saves: a top-90-by-innings
pool is entirely starters and holds 2-4 saves against ~1,200 league-wide, so the
season factor it implies is noise -- measured between 0.53x and 4.90x, which would
have multiplied every 2005 pitcher's saves by nearly five. League-wide denominators
put that factor in 0.95-1.04 and move K and ERA by under 0.01.

**Volume is never era-normalized.** Playing time is playing time; a 600-PA season is
600 PA in any year. Short schedules are a separate correction and belong to
`panel._scale_short_schedules`.

**`ab_pa` is not normalized either.** It is a structural PA-to-AB ratio (it moves with
the walk rate), not a scoring category. Scaling it would silently reweight the AVG
denominator without changing any category rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fantasy_baseball.sgp.denominators import SgpOverrides

from .panel import score

#: Seasons defining the reference run environment. See the module docstring.
REFERENCE_SEASONS: tuple[int, ...] = (2023, 2024, 2025)

#: ``{rate_column: weighting_denominator}`` -- every rate that feeds a 5x5 category.
RATE_DENOMINATORS: dict[str, dict[str, str]] = {
    "hitter": {"h_ab": "ab", "hr_pa": "pa", "r_pa": "pa", "rbi_pa": "pa", "sb_pa": "pa"},
    "pitcher": {
        "k_ip": "ip",
        "w_ip": "ip",
        "sv_ip": "ip",
        "er_ip": "ip",
        "bb_ip": "ip",
        "h_ip": "ip",
    },
}


def league_rates(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Volume-weighted league category rates per season.

    Rows are seasons, columns are the rate columns of `RATE_DENOMINATORS[kind]`. Each
    is `sum(rate * denominator) / sum(denominator)`, i.e. the league total over the
    league volume -- so a one-inning pitcher carries one inning of weight and no
    minimum-volume filter is needed.
    """
    spec = RATE_DENOMINATORS[kind]
    missing = {denom for denom in spec.values() if denom not in df.columns}
    if missing:
        raise KeyError(
            f"panel must be scored (run panel.score) before normalizing: missing {sorted(missing)}"
        )

    out = {}
    for season, group in df.groupby("season"):
        out[int(season)] = {
            rate: float((group[rate] * group[denom]).sum() / group[denom].sum())
            if group[denom].sum() > 0
            else float("nan")
            for rate, denom in spec.items()
        }
    return pd.DataFrame.from_dict(out, orient="index").sort_index()


def era_factors(
    df: pd.DataFrame,
    kind: str,
    *,
    reference_seasons: tuple[int, ...] = REFERENCE_SEASONS,
) -> pd.DataFrame:
    """``season -> {rate_column: multiplicative factor}`` into the reference environment.

    Split out of `era_normalize` so a frame that is ABOUT one season and carries no
    `season` column -- a ZiPS vintage, an actuals export -- can be restated onto the
    same reference the panel uses. Two independent answers to "what is a 2022 home run
    worth in 2023-2025 terms" is the disagreement this subsystem cannot afford, so
    there is one and both callers read it.
    """
    rates = league_rates(df, kind)
    missing = [s for s in reference_seasons if s not in rates.index]
    if missing:
        # ALL of them, not merely one. Accepting a partial window silently restates
        # every season onto a reference that is not the one league.yaml's denominators
        # were calibrated against -- normalizing onto 2023 alone rather than the
        # 2023-2025 mean shifts every era factor, and therefore every historical SGP,
        # into units the output never mentions. If a narrower window is genuinely
        # wanted, say so by passing `reference_seasons` explicitly.
        raise ValueError(
            f"reference seasons {missing} are not in the panel "
            f"({int(rates.index.min())}-{int(rates.index.max())}), so the run "
            f"environment would be defined by only {sorted(set(reference_seasons) - set(missing))}"
            " -- not the window league.yaml's SGP denominators were calibrated on. "
            "Rebuild the panel to cover them, or pass reference_seasons explicitly."
        )
    present = list(reference_seasons)

    reference = rates.loc[present].mean()
    # A season whose pool rate is 0 gives inf, and NaN propagates from an empty pool.
    # Both mean "no usable adjustment for this season"; neutralize to 1.0 rather than
    # blanking the column or, worse, multiplying a real rate by inf.
    # np.nan, not pd.NA: pd.NA makes the column object-dtype and the later astype(float)
    # raises on it.
    return (reference / rates).replace([np.inf, -np.inf], np.nan)


def normalize_frame(
    frame: pd.DataFrame, season: int, kind: str, factors: pd.DataFrame
) -> pd.DataFrame:
    """Restate one season's rate frame into the reference run environment.

    For frames that are ABOUT a single season and carry no `season` column -- a ZiPS
    vintage, an actuals export -- where `era_normalize` needs a panel to derive the
    factors from. Volume (`pa`/`ip`) and the structural `ab_pa` ratio are left alone
    for the same reasons `era_normalize` leaves them alone; see the module docstring.

    A rate column the frame does not carry is SKIPPED, not an error: the 2027/2028
    ZiPS exports ship with `SV` entirely empty, and refusing a vintage over a missing
    category would reject a file that is otherwise fine.
    """
    out = frame.copy()
    for rate in RATE_DENOMINATORS[kind]:
        if rate not in out.columns:
            continue
        factor = factors[rate].get(season, 1.0)
        out[rate] = out[rate] * (1.0 if pd.isna(factor) else float(factor))
    return out


def era_normalize(
    df: pd.DataFrame,
    kind: str,
    *,
    reference_seasons: tuple[int, ...] = REFERENCE_SEASONS,
    sgp_overrides: SgpOverrides | None = None,
) -> pd.DataFrame:
    """Rescale every season's category rates into the reference run environment.

    Returns a copy with the rate columns adjusted and `sgp` re-scored. `era_factor_*`
    columns are kept so a surprising comp can be traced back to its adjustment rather
    than taken on faith.
    """
    factors = era_factors(df, kind, reference_seasons=reference_seasons)

    out = df.copy()
    for rate in RATE_DENOMINATORS[kind]:
        factor = out["season"].map(factors[rate]).astype(float).fillna(1.0)
        out[f"era_factor_{rate}"] = factor
        out[rate] = out[rate] * factor

    return score(out, kind, sgp_overrides)
