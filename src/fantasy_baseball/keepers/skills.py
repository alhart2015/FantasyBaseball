"""Derive season-to-date true-talent rates from the raw keeper pulls.

Pure and I/O-free like `actuals` and `vintages`: each function takes raw frames
and returns one frame indexed by `mlbam_id`. Fetching belongs to `savant` and
`bref`; wiring a park-factor source belongs to the caller.

Hitters get barrel rate, xwOBA, xBA and wRC+; pitchers get ERA-, FIP, K%,
SwStr%, whiff rate and CSW%.

These are the skill signals that carry to next season, as opposed to the
projections under `data/projections/2027/` and `2028/` -- those are ZiPS
snapshots generated 2026-03-25 and therefore contain no 2026 information at all.

Rates are emitted the way they read: `*_pct` columns are percentages on 0-100,
while `xwoba`/`xba`/`fip` keep their native scale. Missing observations are NaN,
never 0.0 -- see `actuals.safe_ratio`.

Every row is emitted and nothing is regressed to the mean, so a pitcher with a
third of a scoreless inning scores an `era_minus` of 0.0 and sorts above every
real ace. Dropping him here would be the caller's decision to lose, but it means
`pa` and `ip` are output, not decoration: filter or shrink on them before
ranking anything.

Park adjustment is optional. `park_factor` is a Series of venue multipliers
indexed by `mlbam_id` (1.00 neutral, >1.00 inflates the stat), defaulting to
neutral because neither source frame carries a usable team code.
`scripts/fetch_keeper_skills.py` builds it and documents the caveats. Only
`wrc_plus` and `era_minus` are adjusted, so they sit on a different park basis
than `fip` and `k_pct` -- reading a pitcher's ERA- against his FIP picks up part
of that difference as a park artifact.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from fantasy_baseball.data.park_factors import park_neutral_series
from fantasy_baseball.keepers.actuals import (
    HITTER_PT,
    PITCHER_PT,
    index_by_mlbam,
    innings_to_float,
    safe_ratio,
)

# wOBA is published on the OBP scale by dividing the raw linear weights by this
# factor. FanGraphs recomputes it per season in Guts!, which is unreachable (see
# `bref`), so this is the stable 2022-24 value. Park-neutral it is a pure scale
# term and cannot reorder two hitters; with the park adjustment on it interacts
# with the multiplier and can, but only at the margin.
WOBA_SCALE = 1.25

HITTER_SKILLS = ("barrel_pct", "barrel_pa_pct", "xwoba", "xba", "wrc_plus")
PITCHER_SKILLS = ("era_minus", "fip", "k_pct", "swstr_pct", "whiff_pct", "csw_pct")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Column as float, unparseable -> NaN. Raises if the column is absent, so a
    renamed upstream field fails loudly instead of producing an all-NaN skill."""
    if column not in frame.columns:
        raise KeyError(f"expected column {column!r}; got {sorted(frame.columns)}")
    return pd.to_numeric(frame[column], errors="coerce")


def _park_neutral(values: pd.Series, park_factor: pd.Series | None) -> pd.Series:
    """Park-neutralize `values`, or return them unchanged when no factor is given.

    The reindex is required: without it pandas aligns on the union and NaNs out
    every player missing from the bridge. `park_neutral_series` owns the rest.
    """
    if park_factor is None:
        return values
    return park_neutral_series(values, park_factor.reindex(values.index))


def _fip_core(counts: Mapping[str, pd.Series]) -> pd.Series:
    """The FIP numerator, `13*HR + 3*(BB+HBP) - 2*K`.

    Shared by the league constant and the per-player rate so the weights are
    stated once -- the constant is only correct while the two agree, and a
    single-pitcher test cannot catch them drifting apart.
    """
    return 13.0 * counts["HR"] + 3.0 * (counts["BB"] + counts["HBP"]) - 2.0 * counts["SO"]


def _league_ratio(
    numer: pd.Series, denom: pd.Series, what: str, *, usable: pd.Series | None = None
) -> float:
    """Sum `numer/denom` over rows where BOTH are present.

    `Series.sum()` skips NaN, so summing the two independently would drop a
    row's numerator while keeping its denominator -- one blank `ER` in a BBRef
    frame would then halve league ERA and double every *other* pitcher's ERA-.

    Pass `usable` to force a row set shared with another ratio. League ERA and
    the FIP constant need it: masking each on its own numerator would put a
    pitcher's innings into one denominator and not the other, and the constant
    would absorb the difference -- shifting every *other* pitcher's FIP, which
    is exactly the "league FIP equals league ERA" invariant it exists to hold.
    """
    both = numer.notna() & denom.notna()
    if usable is not None:
        both &= usable
    total = float(denom.where(both).sum())
    if total <= 0:
        raise ValueError(f"league {what} is zero; cannot derive the index")
    return float(numer.where(both).sum()) / total


def normalize_hitter_skills(
    expected: pd.DataFrame,
    barrels: pd.DataFrame,
    batting: pd.DataFrame,
    *,
    park_factor: pd.Series | None = None,
) -> pd.DataFrame:
    """Hitter skill rates from the two Savant leaderboards plus a BBRef frame.

    `expected` is `savant.fetch_batter_expected` (xBA/xwOBA/wOBA), `barrels` is
    `savant.fetch_batter_barrels`, and `batting` is `bref.fetch_bref_batting`,
    used only for the league R/PA that wRC+ indexes against.

    Barrel columns are reindexed rather than joined: a hitter with plate
    appearances but no batted-ball event is absent from `barrels`, and NaN is the
    honest barrel rate for him.

    wRC+ is derived from Savant's actual `woba` -- which already applies the
    correct per-season linear weights -- rather than recomputed from components,
    so the only approximated input is `WOBA_SCALE`.
    """
    exp = index_by_mlbam(expected, "player_id")
    brl = index_by_mlbam(barrels, "player_id")

    pa = _numeric(exp, "pa")
    woba = _numeric(exp, "woba")
    lg_woba = _league_ratio(woba * pa, pa, "Savant PA")
    lg_r_pa = _league_ratio(_numeric(batting, "R"), _numeric(batting, "PA"), "BBRef PA")

    # wRAA/PA converted to the runs scale, re-centred on league R/PA, then
    # park-neutralized before indexing to 100.
    wrc_per_pa = (woba - lg_woba) / WOBA_SCALE + lg_r_pa

    return pd.DataFrame(
        {
            HITTER_PT: pa,
            "barrel_pct": _numeric(brl, "brl_percent").reindex(exp.index),
            "barrel_pa_pct": _numeric(brl, "brl_pa").reindex(exp.index),
            "xwoba": _numeric(exp, "est_woba"),
            "xba": _numeric(exp, "est_ba"),
            "wrc_plus": 100.0 * _park_neutral(wrc_per_pa, park_factor) / lg_r_pa,
        },
        index=exp.index,
    )


def normalize_pitcher_skills(
    pitching: pd.DataFrame,
    pitch_mix: pd.DataFrame,
    *,
    park_factor: pd.Series | None = None,
) -> pd.DataFrame:
    """Pitcher skill rates from a BBRef season frame plus Statcast pitch counts.

    `pitching` is `bref.fetch_bref_pitching` and supplies the run- and
    batter-denominated stats (ERA-, FIP, K%); `pitch_mix` is
    `savant.fetch_pitcher_pitch_mix` and supplies the pitch-denominated ones.
    Each stat has exactly one source -- see `bref` for why its own `StL`/`StS`
    are unused.

    Note whiff rate and SwStr% are different stats on different denominators --
    whiffs per SWING (~25%) and per PITCH (~11%) respectively -- and both are
    emitted rather than conflated.

    ERA is recomputed from ER and IP rather than read from the `ERA` column so
    that it shares a denominator with the league ERA it is indexed against. The
    FIP constant is solved from this same frame, which makes league FIP equal
    league ERA by construction -- the definition of the constant, and the reason
    it must not be hardcoded across seasons.
    """
    frame = index_by_mlbam(pitching, "mlbID")
    mix = index_by_mlbam(pitch_mix, "player_id").reindex(frame.index)
    thrown = _numeric(mix, "pitches")
    whiffs = _numeric(mix, "whiffs")

    # BBRef reports IP in baseball notation as a float: 20.1 is 20 1/3, not 20.1.
    ip = frame["IP"].map(innings_to_float)
    # No fillna(0.0): a blank HBP would silently understate FIP by 3*HBP/IP with
    # nothing to signal it. NaN propagates to that pitcher's own rates, and
    # `_league_ratio` drops his innings from the constants along with his counts.
    counts = {name: _numeric(frame, name) for name in ("ER", "HR", "BB", "HBP", "SO")}
    bf = _numeric(frame, "BF")

    core = _fip_core(counts)
    # Both constants over the same pitchers -- see `_league_ratio`.
    complete = counts["ER"].notna() & core.notna()
    lg_era = 9.0 * _league_ratio(counts["ER"], ip, "IP", usable=complete)
    fip_constant = lg_era - _league_ratio(core, ip, "IP", usable=complete)

    era = safe_ratio(9.0 * counts["ER"], ip)
    fip_rate = safe_ratio(core, ip)
    return pd.DataFrame(
        {
            PITCHER_PT: ip,
            "era_minus": 100.0 * _park_neutral(era, park_factor) / lg_era,
            "fip": fip_rate + fip_constant,
            "k_pct": 100.0 * safe_ratio(counts["SO"], bf),
            "swstr_pct": 100.0 * safe_ratio(whiffs, thrown),
            "whiff_pct": 100.0 * safe_ratio(whiffs, _numeric(mix, "swings")),
            "csw_pct": 100.0 * safe_ratio(_numeric(mix, "called_strikes") + whiffs, thrown),
        },
        index=frame.index,
    )
