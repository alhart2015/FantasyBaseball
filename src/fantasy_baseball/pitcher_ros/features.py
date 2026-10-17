"""Model inputs and answers from the pitcher training table (``pitcher_ros/table.py``, #465).

Answers (``answers``): the five roto rates over the rest of the season, as the scorer
reads them (``stats``): K, W and SV per batter faced, ERA and WHIP, each with a loss
weight (see :func:`answers` for how they compare with the scorer's).

Inputs (``input_frame``): rates per window (season to date, last 30 days, last season,
last three, career), box-score and pitch-tracking, minor-league pitching (last season,
last three; raw and against his leagues), plus context (age, how much of the season is
left, how much history the store covers, team run support and save chances, league
rates). NaN = unknown (no batters faced, or not measured that season);
``hitter_ros.features.Standardizer`` turns it into 0 plus a missing flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fantasy_baseball.pitcher_ros.stats import PER_BF, TARGETS, rates_from_counts
from fantasy_baseball.pitcher_ros.table import HORIZON_DAYS, MILB_WINDOWS, RECENT_WINDOWS

WINDOWS = ("std", "p1", "p3", "car")
# The recent window used as input (l14 is in the table too).
INPUT_RECENT = ("l30",)
assert set(INPUT_RECENT) <= set(RECENT_WINDOWS)
PITCH_GROUPS = ("fb", "fc", "brk", "off")
# The answer counts ``rates_from_counts`` needs, read from the table's ros_ columns.
_ANSWER_COUNTS = ("bf", "outs", "k", "w", "sv", "er", "h", "bb")

# ERA's fielding-independent pieces (#486): walks, hit batters and home runs per batter
# faced (K per batter faced is already a target). With ``era_pieces`` the net predicts
# them as extra outputs beside the five targets; ERA is still its own output. They are
# what FIP is built from, so their errors show which piece an ERA miss comes from.
ERA_PIECES = ("bb_bf", "hbp_bf", "hr_bf")
_PIECE_COUNTS = ("bb", "hbp", "hr")
# Every count the targets and the pieces are built from, each once.
_RATE_COUNTS = tuple(dict.fromkeys((*_ANSWER_COUNTS, *_PIECE_COUNTS)))


def horizon_columns(horizons: tuple[int, ...] = ()) -> list[str]:
    """The short-horizon output columns: ``d{N}_{stat}`` for each of the ``TARGETS``, per
    horizon of N days (``table.HORIZON_DAYS``)."""
    unknown = sorted(set(horizons) - set(HORIZON_DAYS))
    if unknown:
        raise ValueError(f"no answers for horizons {unknown} days; the table has {HORIZON_DAYS}")
    return [f"d{n}_{s}" for n in horizons for s in TARGETS]


def column_horizon(column: str) -> str:
    """The horizon of an output column: ``d7_k_bf`` -> ``d7``; ``k_bf`` -> ``ros``."""
    head = column.split("_", 1)[0]
    return head if head[:1] == "d" and head[1:].isdigit() else "ros"


def target_stat(column: str) -> str:
    """The stat of an output column: ``d7_k_bf`` -> ``k_bf``; ``k_bf`` -> ``k_bf``."""
    return column if column_horizon(column) == "ros" else column.split("_", 1)[1]


def piece_rates(counts: pd.DataFrame) -> pd.DataFrame:
    """``ERA_PIECES`` from summed counts (``bf``, ``bb``, ``hbp``, ``hr``); NaN where no
    batters were faced."""
    return pd.DataFrame(
        {f"{c}_bf": div(counts[c], counts["bf"]) for c in _PIECE_COUNTS}, index=counts.index
    )


def _target_and_piece_rates(counts: pd.DataFrame) -> pd.DataFrame:
    """``TARGETS`` then ``ERA_PIECES`` from summed ``_RATE_COUNTS``."""
    return pd.concat([rates_from_counts(counts)[list(TARGETS)], piece_rates(counts)], axis=1)


def div(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN where den is 0."""
    den = den.astype(float)
    return num.astype(float) / den.where(den != 0)


# K/W/SV loss weight (see answers). "bf" by default: over 3 seeds x 2022-2026, "bf2"
# (the scorer's own weighting) was worse on K -- preseason +3.1 / +1.1 / +4.3% MSE (<1%,
# 6%, <1% sure better), mid-season +1.2 to +1.8% -- and better on preseason W among the
# fantasy-relevant pitchers (-5.4 / -3.5 / -5.0%, >99% sure better on all 3), the rest
# mixed.
PER_BF_WEIGHTS = ("bf", "bf2")


def answers(
    t: pd.DataFrame,
    per_bf_weight: str = "bf",
    *,
    era_pieces: bool = False,
    horizons: tuple[int, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rest-of-season ``TARGETS`` rates per table row (NaN where there is nothing to
    divide by: no batters faced, or no outs) and each one's loss weight (0 where the
    rate is NaN).

    ERA/WHIP: the innings he went on to pitch -- exactly how the scorer weights them
    (``ip x err^2``). K/W/SV: the scorer charges a rate's error on his actual batters
    faced, ``(bf x err)^2``, so per row its weight is ``bf^2``: ``per_bf_weight`` "bf2"
    matches it; "bf" weights by batters faced, which leans further toward low-volume
    pitchers than the score does.

    ``era_pieces``: also ``ERA_PIECES`` (after the targets), weighted as K is.
    ``horizons``: also the ``TARGETS`` over his next N days (:func:`horizon_columns`,
    after those), weighted the same way over that window's batters and innings."""
    if per_bf_weight not in PER_BF_WEIGHTS:
        raise ValueError(f"unknown per_bf_weight {per_bf_weight!r}; one of {PER_BF_WEIGHTS}")
    horizon_columns(horizons)  # validates
    names = _RATE_COUNTS if era_pieces else _ANSWER_COUNTS
    windows = [("ros_", "", names), *((f"ros_d{n}_", f"d{n}_", _ANSWER_COUNTS) for n in horizons)]
    ys, ws = [], []
    for prefix, tag, cols in windows:
        counts = pd.DataFrame({c: t[f"{prefix}{c}"].astype(float) for c in cols})
        y = (
            _target_and_piece_rates(counts)
            if cols == _RATE_COUNTS
            else rates_from_counts(counts)[list(TARGETS)]
        )
        bf = counts["bf"] ** (2 if per_bf_weight == "bf2" else 1)
        ip = counts["outs"] / 3
        w = pd.DataFrame(
            {s: bf if s in PER_BF or s in ERA_PIECES else ip for s in y.columns}, index=t.index
        )
        ys.append(y.add_prefix(tag))
        ws.append(w.where(y.notna(), 0.0).add_prefix(tag))
    return pd.concat(ys, axis=1), pd.concat(ws, axis=1)


def league_reference(t: pd.DataFrame) -> pd.DataFrame:
    """Each row's league ``TARGETS`` rates known on its as-of date: last season plus this
    season before the date, pooled. The net predicts a pitcher relative to his league
    and is scaled back by this (``NetConfig.relative_target``). Across 2011-2026 last
    season missed the next season's league K rate by 0.43 K per 100 batters faced on
    average, the last three seasons pooled by 0.76 (the rate rose every season to 2020,
    which a longer pool lags); mid-season, with this season so far, 0.59 vs 0.83. NaN
    for a row with no last season in the store. Also the ``ERA_PIECES``' rates."""
    counts = pd.DataFrame(
        {c: t[f"lg_p1_{c}"].astype(float) + t[f"lg_std_{c}"].astype(float) for c in _RATE_COUNTS},
        index=t.index,
    )
    return _target_and_piece_rates(counts).where(t["lg_p1_bf"].astype(float) > 0)


def league_answer_rates(t: pd.DataFrame, horizons: tuple[int, ...] = ()) -> pd.DataFrame:
    """Each row's league ``TARGETS`` (and ``ERA_PIECES``) rates over its answer window: the ``ros_*`` counts of
    every table row of the same season and as-of week pooled. The table has a row for
    every pitcher-season pitching on or after the date, so this is the league's rest of
    the season (the ``lg_*`` columns' population). Uses the answers: a **training target
    denominator only**, never an input or a way back to rates. ``horizons``: also the
    league's ``TARGETS`` over the next N days (:func:`horizon_columns`), the same way."""
    keys = [t["season"], t["week"]]

    def pooled(prefix: str, cols: tuple[str, ...]) -> pd.DataFrame:
        return pd.DataFrame(
            {c: t[f"{prefix}{c}"].astype(float).groupby(keys).transform("sum") for c in cols},
            index=t.index,
        )

    horizon_columns(horizons)  # validates
    return pd.concat(
        [
            _target_and_piece_rates(pooled("ros_", _RATE_COUNTS)),
            *(
                rates_from_counts(pooled(f"ros_d{n}_", _ANSWER_COUNTS))[list(TARGETS)].add_prefix(
                    f"d{n}_"
                )
                for n in horizons
            ),
        ],
        axis=1,
    )


def _box_rates(t: pd.DataFrame, w: str) -> dict[str, pd.Series]:
    def c(name: str) -> pd.Series:
        return t[f"{w}_{name}"].astype(float)

    bf, outs, g = c("bf"), c("outs"), c("g")
    balls_in_play = c("ab") - c("k") - c("hr") + c("sf")
    out = {
        "log_bf": np.log1p(bf),
        "log_g": np.log1p(g),
        "gs_share": div(c("gs"), g),
        "relief_out_share": div(c("rel_outs"), outs),
        "outs_per_g": div(outs, g),
        "outs_per_bf": div(outs, bf),
        **{f"{s}_bf": div(c(s), bf) for s in ("k", "bb", "hbp", "h", "hr", "w", "l", "sv", "wp")},
        "era": 27 * div(c("er"), outs),
        "ra9": 27 * div(c("r"), outs),
        "whip": 3 * div(c("h") + c("bb"), outs),
        "babip": div(c("h") - c("hr"), balls_in_play),
        "gb_share": div(c("go"), c("go") + c("ao")),
        "pitches_per_bf": div(c("pitches"), bf),
        "strike_share": div(c("strikes"), c("pitches")),
        "w_per_gs": div(c("w"), c("gs")),
        **{f"{s}_per_g": div(c(s), g) for s in ("svo", "hld", "bs", "gf")},
        "sv_per_svo": div(c("sv"), c("svo")),
        "irs_share": div(c("irs"), c("ir")),
    }
    return {f"{w}_{k}": v for k, v in out.items()}


def _mean_sd(t: pd.DataFrame, w: str, name: str) -> tuple[pd.Series, pd.Series]:
    n = t[f"{w}_sc_{name}_n"].astype(float)
    mean = div(t[f"{w}_sc_{name}_sum"], n)
    sq = div(t[f"{w}_sc_{name}_sq_sum"], n)
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
        "swing_rate": div(swings, pitches),
        "whiff_rate": div(c("whiffs"), swings),
        "csw_rate": div(c("called_strikes") + c("whiffs"), pitches),
        "zone_rate": div(c("zone_pitches"), pitches),
        "zone_swing": div(c("zone_swings"), c("zone_pitches")),
        "zone_contact": div(c("zone_contacts"), c("zone_swings")),
        "chase_rate": div(c("chase_pitches"), pitches),
        "chase_swing": div(c("chase_swings"), c("chase_pitches")),
        "first_strike": div(c("first_strikes"), c("first_pitches")),
        "lhb_share": div(c("pitches_vs_lhb"), pitches),
        "ev": ev,
        "ev_sd": ev_sd,
        "la": _mean_sd(t, w, "la")[0],
        "hard_rate": div(c("hard_hit"), c("ev_n")),
        "barrel_rate": div(c("barrels"), bip),
        **{f"{s}_rate": div(c(s), bip) for s in ("gb", "ld", "fb", "pu")},
        "woba": div(c("woba_sum"), c("woba_n")),
        "xwoba": div(c("xwoba_sum"), c("xwoba_n")),
        "rv_per_100": 100 * div(c("rv_sum"), c("rv_n")),
    }
    for g in PITCH_GROUPS:
        out[f"{g}_share"] = div(c(f"{g}_pitches"), pitches)
        out[f"{g}_whiff"] = div(c(f"{g}_whiffs"), c(f"{g}_swings"))
        for m in ("velo", "spin", "ivb", "hbreak"):
            out[f"{g}_{m}"] = _mean_sd(t, w, f"{g}_{m}")[0]
    return {f"{w}_{k}": v for k, v in out.items()}


def _league_rates(t: pd.DataFrame, w: str) -> dict[str, pd.Series]:
    def c(name: str) -> pd.Series:
        return t[f"lg_{w}_{name}"].astype(float)

    bf, outs = c("bf"), c("outs")
    out = {
        **{f"{s}_bf": div(c(s), bf) for s in ("k", "bb", "hr", "w", "sv")},
        "era": 27 * div(c("er"), outs),
        "whip": 3 * div(c("h") + c("bb"), outs),
    }
    return {f"lg_{w}_{k}": v for k, v in out.items()}


def _team_rates(t: pd.DataFrame, w: str) -> dict[str, pd.Series]:
    def c(name: str) -> pd.Series:
        return t[f"{w}_team_{name}"].astype(float)

    games = c("games")
    return {
        f"{w}_team_r_per_g": div(c("r"), games),
        f"{w}_team_svo_per_g": div(c("svo"), games),
        f"{w}_team_sv_per_g": div(c("sv"), games),
        f"{w}_team_era": 27 * div(c("er"), c("outs")),
        f"{w}_team_log_games": np.log1p(games),
    }


# Minor-league inputs, 3 seeds x 2022-2026, averaged (runs 006-milb vs 001-plain):
# preseason K MSE 184.7 vs 193.5 (>99% sure better; fantasy-relevant 325.6 vs 336.2,
# 96%). Per seed 186.7 / 187.8 / 185.6 vs 191.3 / 198.9 / 196.2. Mid-season K and the
# other stats level (the lowest: mid-season W on the fantasy-relevant, 6% sure better).
# Batters faced that pull a minor-league K-rate gap halfway to 0
# (``milb_*_k_rel_shrunk``):
MILB_SHRINK_BF = 100


def _milb_rates(t: pd.DataFrame, w: str) -> dict[str, pd.Series]:
    """Minor-league pitching rates for window ``w`` (``table.MILB_WINDOWS``), each line
    also read against its league (``*_rel``: what he allowed minus what his leagues
    allowed, per batter faced, or per 27 outs for ERA). All NaN, log batters faced
    included, where the store has none of the window's seasons: "not stored", not
    "pitched nowhere"."""

    def c(name: str) -> pd.Series:
        return t[f"milb_{w}_{name}"].astype(float)

    bf, outs = c("bf"), c("outs")
    out = {
        "log_bf": np.log1p(bf),
        **{f"{s}_bf": div(c(s), bf) for s in ("k", "bb", "hr", "h")},
        **{f"{s}_rel": div(c(s) - c(f"x{s}"), bf) for s in ("k", "bb", "hr", "h")},
        "k_rel_shrunk": (c("k") - c("xk")) / (bf + MILB_SHRINK_BF),
        "era": 27 * div(c("er"), outs),
        "era_rel": 27 * div(c("er") - c("xer"), outs),
        "whip": 3 * div(c("h") + c("bb"), outs),
        "gs_share": div(c("gs"), c("g")),
        "outs_per_g": div(outs, c("g")),
        "level": div(c("level_bf"), bf),
        "age_vs_level": div(c("age_gap_bf"), c("age_bf")),
    }
    stored = (
        t["milb_p1_in_store"].astype(bool)
        if w == "p1"
        else t["milb_p3_seasons_in_store"].astype(float) > 0
    )
    return {f"milb_{w}_{k}": v.where(stored) for k, v in out.items()}


COVERAGE = (
    *(
        f"{kind}{c}"
        for kind in ("", "pitchtrack_", "battedtrack_")
        for c in ("p1_in_store", "p3_seasons_in_store", "car_seasons_in_store")
    ),
    "milb_p1_in_store",
    "milb_p3_seasons_in_store",
)


# His role per window (saves and save chances, holds, blown saves, games finished,
# start and relief shares): with ``role_windows=False`` the net reads his role from
# the appearance sequence only, where a role change shows as soon as it happens.
# Tried (run 034-norole vs 031, 5 seeds): mid-season SV MSE 8.57 vs 8.54 (49% sure
# better) on the fantasy-relevant, the rest a shade worse; the net still reads the role
# from his other per-window rates. The saves net (model.DEFAULT_SAVES_NET) does it.
ROLE_RATES = (
    "gs_share",
    "relief_out_share",
    "sv_bf",
    "svo_per_g",
    "hld_per_g",
    "bs_per_g",
    "gf_per_g",
    "sv_per_svo",
)


def input_frame(
    t: pd.DataFrame, own_windows: bool = True, role_windows: bool = True
) -> pd.DataFrame:
    """Model inputs for every table row. NaN = unknown. Refuses a table built before the
    minor-league inputs (#465), which would otherwise fail on a bare missing column.

    ``own_windows=False`` leaves out his own big-league rates per window (box score and
    pitch tracking: season to date, last 30 days, last season, last three, career), for
    a net that reads his history only as the appearance sequence. Minor-league, team,
    league and context inputs stay: the sequence holds none of them. The stuff inputs
    (``stuff_features``) are his own windows too: scripts/train_pitcher_ros.py
    --no-own-windows requires --stuff none.

    ``role_windows=False`` leaves out only his role rates per window (``ROLE_RATES``)."""
    if "milb_p1_in_store" not in t.columns:
        raise ValueError(
            "the pitcher table has no minor-league columns (milb_*); rebuild it with "
            "scripts/build_pitcher_ros_table.py"
        )
    cols: dict[str, pd.Series] = {}
    for w in (*WINDOWS, *INPUT_RECENT) if own_windows else ():
        box = _box_rates(t, w)
        if not role_windows:
            box = {k: v for k, v in box.items() if k.removeprefix(f"{w}_") not in ROLE_RATES}
        cols.update(box)
        cols.update(_tracking_rates(t, w))
    for w in ("std", "p1", "p3"):
        cols.update(_league_rates(t, w))
    for w in ("std", "p1"):
        cols.update(_team_rates(t, w))
    for w in MILB_WINDOWS:
        cols.update(_milb_rates(t, w))
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
