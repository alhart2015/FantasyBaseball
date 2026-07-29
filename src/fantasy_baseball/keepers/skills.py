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

Park adjustment is optional. `park_factor` is a Series of venue multipliers
indexed by `mlbam_id` (1.00 neutral, >1.00 inflates the stat); it defaults to
neutral because neither source frame carries a usable team code -- BBRef's `Tm`
is an ambiguous city name ("Chicago", "Los Angeles") and is comma-joined for
traded players. `scripts/fetch_keeper_skills.py` builds the Series from the
FanGraphs-coded `Team` column in the ROS projection CSVs.
"""

from __future__ import annotations

import pandas as pd

from fantasy_baseball.keepers.actuals import index_by_mlbam, innings_to_float, safe_ratio

# wOBA is published on the OBP scale by dividing the raw linear weights by this
# factor. FanGraphs recomputes it per season in Guts!, which returns 403 (see
# `bref`), so this is the stable recent-seasons value. It is a league-wide
# constant, so it scales how far wRC+ spreads around 100 but cannot reorder two
# hitters -- a ranking built on `wrc_plus` is unaffected by its drift.
WOBA_SCALE = 1.25

HITTER_SKILLS = ("barrel_pct", "barrel_pa_pct", "xwoba", "xba", "wrc_plus")
PITCHER_SKILLS = ("era_minus", "fip", "k_pct", "swstr_pct", "whiff_pct", "csw_pct")

HITTER_PT = "pa"
PITCHER_PT = "ip"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Column as float, unparseable -> NaN. Raises if the column is absent, so a
    renamed upstream field fails loudly instead of producing an all-NaN skill."""
    if column not in frame.columns:
        raise KeyError(f"expected column {column!r}; got {sorted(frame.columns)}")
    return pd.to_numeric(frame[column], errors="coerce")


def _park_series(park_factor: pd.Series | None, index: pd.Index) -> pd.Series:
    """Align `park_factor` to `index`, defaulting anything missing or nonpositive
    to neutral. A dropped player must not silently divide a rate by NaN."""
    if park_factor is None:
        return pd.Series(1.0, index=index)
    aligned = pd.to_numeric(park_factor.reindex(index), errors="coerce")
    return aligned.where(aligned > 0, 1.0)


def league_r_per_pa(batting: pd.DataFrame) -> float:
    """League runs per plate appearance, the denominator wRC+ indexes against."""
    runs = _numeric(batting, "R").sum()
    pa = _numeric(batting, "PA").sum()
    if pa <= 0:
        raise ValueError("league PA is zero; cannot derive wRC+")
    return float(runs / pa)


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
    total_pa = pa.sum()
    if total_pa <= 0:
        raise ValueError("total PA is zero; cannot derive league wOBA")
    lg_woba = float((woba * pa).sum() / total_pa)
    lg_r_pa = league_r_per_pa(batting)

    # wRAA/PA converted to the runs scale, re-centred on league R/PA, then
    # deflated by the hitter's park before indexing to 100.
    wrc_per_pa = (woba - lg_woba) / WOBA_SCALE + lg_r_pa

    return pd.DataFrame(
        {
            HITTER_PT: pa,
            "barrel_pct": _numeric(brl, "brl_percent").reindex(exp.index),
            "barrel_pa_pct": _numeric(brl, "brl_pa").reindex(exp.index),
            "xwoba": _numeric(exp, "est_woba"),
            "xba": _numeric(exp, "est_ba"),
            "wrc_plus": 100.0 * (wrc_per_pa / _park_series(park_factor, exp.index)) / lg_r_pa,
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
    Each stat has exactly one source: BBRef also publishes `StL`/`StS`, but
    rounded to two decimals, which is too coarse to rank on, so they are unused.

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
    counts = {name: _numeric(frame, name).fillna(0.0) for name in ("ER", "HR", "BB", "HBP", "SO")}
    bf = _numeric(frame, "BF")

    lg_ip = float(ip.sum())
    if lg_ip <= 0:
        raise ValueError("league IP is zero; cannot derive ERA- or FIP")
    lg_era = 9.0 * float(counts["ER"].sum()) / lg_ip
    lg_fip_core = (
        13.0 * float(counts["HR"].sum())
        + 3.0 * (float(counts["BB"].sum()) + float(counts["HBP"].sum()))
        - 2.0 * float(counts["SO"].sum())
    ) / lg_ip
    fip_constant = lg_era - lg_fip_core

    era = safe_ratio(9.0 * counts["ER"], ip)
    fip_core = safe_ratio(
        13.0 * counts["HR"] + 3.0 * (counts["BB"] + counts["HBP"]) - 2.0 * counts["SO"], ip
    )
    return pd.DataFrame(
        {
            PITCHER_PT: ip,
            "era_minus": 100.0 * (era / _park_series(park_factor, frame.index)) / lg_era,
            "fip": fip_core + fip_constant,
            "k_pct": 100.0 * safe_ratio(counts["SO"], bf),
            "swstr_pct": 100.0 * safe_ratio(whiffs, thrown),
            "whiff_pct": 100.0 * safe_ratio(whiffs, _numeric(mix, "swings")),
            "csw_pct": 100.0 * safe_ratio(_numeric(mix, "called_strikes") + whiffs, thrown),
        },
        index=frame.index,
    )
