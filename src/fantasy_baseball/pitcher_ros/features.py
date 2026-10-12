"""Model inputs and answers from the pitcher training table (``pitcher_ros/table.py``, #465).

Answers (``target_frame``): the five roto rates over the rest of the season, as the
scorer reads them (``stats``): K, W and SV per batter faced, ERA and WHIP. Each has a
loss weight (``weight_frame``): the batters he went on to face for K/W/SV, the innings
for ERA/WHIP -- the same weights the scorer's MSE uses, so the net is fit on what it is
graded on.

Inputs (``input_frame``): rates per window (season to date, last 30 days, last season,
last three, career), box-score and pitch-tracking, plus context (age, how much of the
season is left, how much history the store covers, team run support and save chances,
league rates). NaN = unknown (no batters faced, or not measured that season);
``hitter_ros.features.Standardizer`` turns it into 0 plus a missing flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fantasy_baseball.pitcher_ros.stats import PER_BF, TARGETS, rates_from_counts
from fantasy_baseball.pitcher_ros.table import RECENT_WINDOWS

WINDOWS = ("std", "p1", "p3", "car")
# The recent window used as input (l14 is in the table too).
INPUT_RECENT = ("l30",)
assert set(INPUT_RECENT) <= set(RECENT_WINDOWS)
PITCH_GROUPS = ("fb", "fc", "brk", "off")
# The answer counts ``rates_from_counts`` needs, read from the table's ros_ columns.
_ANSWER_COUNTS = ("bf", "outs", "k", "w", "sv", "er", "h", "bb")


def _div(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN where den is 0."""
    den = den.astype(float)
    return num.astype(float) / den.where(den != 0)


def target_frame(t: pd.DataFrame) -> pd.DataFrame:
    """Rest-of-season ``TARGETS`` rates per table row; NaN where there is nothing to
    divide by (no batters faced, or no outs)."""
    counts = pd.DataFrame({c: t[f"ros_{c}"].astype(float) for c in _ANSWER_COUNTS})
    return rates_from_counts(counts)[list(TARGETS)]


def weight_frame(t: pd.DataFrame) -> pd.DataFrame:
    """Each answer's loss weight: rest-of-season batters faced for K/W/SV, innings for
    ERA/WHIP (as the scorer weights them); 0 where the answer is NaN."""
    bf = t["ros_bf"].astype(float)
    ip = t["ros_outs"].astype(float) / 3
    w = pd.DataFrame({s: bf if s in PER_BF else ip for s in TARGETS}, index=t.index)
    return w.where(target_frame(t).notna(), 0.0)


def _box_rates(t: pd.DataFrame, w: str) -> dict[str, pd.Series]:
    def c(name: str) -> pd.Series:
        return t[f"{w}_{name}"].astype(float)

    bf, outs, g = c("bf"), c("outs"), c("g")
    balls_in_play = c("ab") - c("k") - c("hr") + c("sf")
    out = {
        "log_bf": np.log1p(bf),
        "log_g": np.log1p(g),
        "gs_share": _div(c("gs"), g),
        "relief_out_share": _div(c("rel_outs"), outs),
        "outs_per_g": _div(outs, g),
        "outs_per_bf": _div(outs, bf),
        **{f"{s}_bf": _div(c(s), bf) for s in ("k", "bb", "hbp", "h", "hr", "w", "l", "sv", "wp")},
        "era": 27 * _div(c("er"), outs),
        "ra9": 27 * _div(c("r"), outs),
        "whip": 3 * _div(c("h") + c("bb"), outs),
        "babip": _div(c("h") - c("hr"), balls_in_play),
        "gb_share": _div(c("go"), c("go") + c("ao")),
        "pitches_per_bf": _div(c("pitches"), bf),
        "strike_share": _div(c("strikes"), c("pitches")),
        "w_per_gs": _div(c("w"), c("gs")),
        **{f"{s}_per_g": _div(c(s), g) for s in ("svo", "hld", "bs", "gf")},
        "sv_per_svo": _div(c("sv"), c("svo")),
        "irs_share": _div(c("irs"), c("ir")),
    }
    return {f"{w}_{k}": v for k, v in out.items()}


def _mean_sd(t: pd.DataFrame, w: str, name: str) -> tuple[pd.Series, pd.Series]:
    n = t[f"{w}_sc_{name}_n"].astype(float)
    mean = _div(t[f"{w}_sc_{name}_sum"], n)
    sq = _div(t[f"{w}_sc_{name}_sq_sum"], n)
    return mean, np.sqrt((sq - mean**2).clip(lower=0))


def _tracking_rates(t: pd.DataFrame, w: str) -> dict[str, pd.Series]:
    def c(name: str) -> pd.Series:
        return t[f"{w}_sc_{name}"].astype(float)

    pitches, swings, bip = c("pitches"), c("swings"), c("bip")
    velo, velo_sd = _mean_sd(t, w, "velo")
    ev, ev_sd = _mean_sd(t, w, "ev")
    out: dict[str, pd.Series] = {
        "log_pitches": np.log1p(pitches),
        "velo": velo,
        "velo_sd": velo_sd,
        "ext": _mean_sd(t, w, "ext")[0],
        "arm_angle": _mean_sd(t, w, "arm_angle")[0],
        "swing_rate": _div(swings, pitches),
        "whiff_rate": _div(c("whiffs"), swings),
        "csw_rate": _div(c("called_strikes") + c("whiffs"), pitches),
        "zone_rate": _div(c("zone_pitches"), pitches),
        "zone_swing": _div(c("zone_swings"), c("zone_pitches")),
        "zone_contact": _div(c("zone_contacts"), c("zone_swings")),
        "chase_rate": _div(c("chase_pitches"), pitches),
        "chase_swing": _div(c("chase_swings"), c("chase_pitches")),
        "first_strike": _div(c("first_strikes"), c("first_pitches")),
        "lhb_share": _div(c("pitches_vs_lhb"), pitches),
        "ev": ev,
        "ev_sd": ev_sd,
        "la": _mean_sd(t, w, "la")[0],
        "hard_rate": _div(c("hard_hit"), c("ev_n")),
        "barrel_rate": _div(c("barrels"), bip),
        **{f"{s}_rate": _div(c(s), bip) for s in ("gb", "ld", "fb", "pu")},
        "woba": _div(c("woba_sum"), c("woba_n")),
        "xwoba": _div(c("xwoba_sum"), c("xwoba_n")),
        "rv_per_100": 100 * _div(c("rv_sum"), c("rv_n")),
    }
    for g in PITCH_GROUPS:
        out[f"{g}_share"] = _div(c(f"{g}_pitches"), pitches)
        out[f"{g}_whiff"] = _div(c(f"{g}_whiffs"), c(f"{g}_swings"))
        for m in ("velo", "spin", "ivb", "hbreak"):
            out[f"{g}_{m}"] = _mean_sd(t, w, f"{g}_{m}")[0]
    return {f"{w}_{k}": v for k, v in out.items()}


def _league_rates(t: pd.DataFrame, w: str) -> dict[str, pd.Series]:
    def c(name: str) -> pd.Series:
        return t[f"lg_{w}_{name}"].astype(float)

    bf, outs = c("bf"), c("outs")
    out = {
        **{f"{s}_bf": _div(c(s), bf) for s in ("k", "bb", "hr", "w", "sv")},
        "era": 27 * _div(c("er"), outs),
        "whip": 3 * _div(c("h") + c("bb"), outs),
    }
    return {f"lg_{w}_{k}": v for k, v in out.items()}


def _team_rates(t: pd.DataFrame, w: str) -> dict[str, pd.Series]:
    def c(name: str) -> pd.Series:
        return t[f"{w}_team_{name}"].astype(float)

    games = c("games")
    return {
        f"{w}_team_r_per_g": _div(c("r"), games),
        f"{w}_team_svo_per_g": _div(c("svo"), games),
        f"{w}_team_sv_per_g": _div(c("sv"), games),
        f"{w}_team_era": 27 * _div(c("er"), c("outs")),
        f"{w}_team_log_games": np.log1p(games),
    }


COVERAGE = tuple(
    f"{kind}{c}"
    for kind in ("", "pitchtrack_", "battedtrack_")
    for c in ("p1_in_store", "p3_seasons_in_store", "car_seasons_in_store")
)


def input_frame(t: pd.DataFrame) -> pd.DataFrame:
    """Model inputs for every table row. NaN = unknown."""
    cols: dict[str, pd.Series] = {}
    for w in (*WINDOWS, *INPUT_RECENT):
        cols.update(_box_rates(t, w))
        cols.update(_tracking_rates(t, w))
    for w in ("std", "p1", "p3"):
        cols.update(_league_rates(t, w))
    for w in ("std", "p1"):
        cols.update(_team_rates(t, w))
    cols.update(
        {
            "age": t["age"].astype(float),
            "week": t["week"].astype(float),
            "frac_season_left": t["frac_season_left"].astype(float),
            **{c: t[c].astype(float) for c in COVERAGE},
        }
    )
    frame = pd.DataFrame(cols, index=t.index)
    # A ratio of two huge sums can overflow to inf in float32; treat it as unknown.
    return frame.replace([np.inf, -np.inf], np.nan)
