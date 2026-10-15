"""Stuff inputs for the pitcher table (#489, stage 2), from the per-pitch models.

For a table row in season T, the season-T stuff model (``pitcher_ros/stuff.py``, trained
on pitches before T) reads every pitch the pitcher threw: before T (his history) and
in T before the row's as-of date. Its predictions are summed per window -- this season
so far (std), last season (p1), the last three (p3), career (car) -- into rates:

* per swing: whiffs (``whiff_swing``); per pitch: called strikes plus whiffs (``csw``)
  and balls (``ball``);
* per ball in play: barrels, hard contact (solid or barrel), ground balls, fly balls and
  line drives (``air``), and xwOBA on contact (``xwoba_con``);
* per pitch: expected damage (xwOBA on contact x chance of contact, ``damage``);
* by pitch family (``FAMILIES``): whiffs per swing and xwOBA on contact.

Every rate is read **relative to the league as the same model sees it**: minus the rate
over every pitch of season T - 1 under model T. Each season's model has its own scale
(PR #420), and season T - 1 is known before T. A row from a season with no model, or a
window with no pitches, is NaN. So are the contact-quality rates (``CONTACT_QUALITY``)
for seasons whose model was trained before Statcast graded contact (2015).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.features import ROW_KEYS, load_feature_file
from fantasy_baseball.pitcher_ros.stuff import (
    BB_TYPES,
    GRADED_CONTACT_FROM,
    OUTCOMES,
    StuffFit,
    predict_stuff,
)

logger = logging.getLogger(__name__)

SWING = ("whiff", "foul", "bip_weak", "bip_flare", "bip_solid", "bip_barrel", "bip_other")
BIP = ("bip_weak", "bip_flare", "bip_solid", "bip_barrel", "bip_other")
assert set(SWING) <= set(OUTCOMES) and set(BIP) <= set(SWING)
assert {"ground_ball", "line_drive", "fly_ball"} <= set(BB_TYPES)
FAMILIES = {"fb": ("ff", "si", "fc"), "brk": ("sl", "sweep", "cu"), "off": ("ch", "fs")}
WINDOWS = ("std", "p1", "p3", "car")

# name -> (numerator sum, denominator sum)
RATES = {
    "whiff_swing": ("whiff", "swing"),
    "csw": ("csw", "n"),
    "ball": ("ball", "n"),
    "barrel_bip": ("barrel", "bip"),
    "hard_bip": ("hard", "bip"),
    "gb_bip": ("gb", "bip"),
    "air_bip": ("air", "bip"),
    "xwoba_con": ("damage", "bip"),
    "damage": ("damage", "n"),
    **{f"{f}_whiff_swing": (f"{f}_whiff", f"{f}_swing") for f in FAMILIES},
    **{f"{f}_xwoba_con": (f"{f}_damage", f"{f}_bip") for f in FAMILIES},
}
STUFF_INPUTS = [f"{w}_stuff_{r}" for w in WINDOWS for r in RATES]
# Named subsets of RATES the pitcher net can read (train_pitcher_ros.py --stuff-rates):
# every rate, the rates over all his pitches (no per-family ones), or a core few.
STUFF_RATE_SETS = {
    "all": tuple(RATES),
    "overall": tuple(r for r in RATES if not r.startswith(tuple(f"{f}_" for f in FAMILIES))),
    "core": ("whiff_swing", "csw", "ball", "gb_bip", "damage"),
}
assert all(set(rates) <= set(RATES) for rates in STUFF_RATE_SETS.values())


def stuff_inputs(rate_set: str) -> list[str]:
    """The ``STUFF_INPUTS`` of the rates in ``STUFF_RATE_SETS[rate_set]``, every window."""
    rates = STUFF_RATE_SETS[rate_set]
    return [c for c in STUFF_INPUTS if c.split("_stuff_", 1)[1] in rates]


# Rates that need graded contact (barrels, hard contact) or Statcast's xwOBA, both from
# 2015 on. A model trained before any such pitch never learned them: left blank.
CONTACT_QUALITY = [
    r for r in RATES if r in ("barrel_bip", "hard_bip", "damage") or r.endswith("xwoba_con")
]


def stuff_features_path(root: Path, name: str) -> Path:
    """Where ``scripts/build_pitcher_stuff.py --features-file <name>`` writes the inputs."""
    return root / f"stuff_{name}.parquet"


def pitch_sums(pitches: pd.DataFrame, fit: StuffFit) -> pd.DataFrame:
    """Per pitch, the summable quantities ``RATES`` divides (``pitches``: a
    ``stuff.stuff_frame``), from ``fit``'s predictions."""
    pr = predict_stuff(fit, pitches)
    p = {o: pr[f"p_{o}"].to_numpy() for o in OUTCOMES}
    bip = sum(p[o] for o in BIP)
    damage = bip * pr["xwoba_bip"].to_numpy()
    out = {
        "n": np.ones(len(pr), np.float32),
        "swing": sum(p[o] for o in SWING),
        "whiff": p["whiff"],
        "csw": p["whiff"] + p["called_strike"],
        "ball": p["ball"],
        "bip": bip,
        "barrel": p["bip_barrel"],
        "hard": p["bip_solid"] + p["bip_barrel"],
        "gb": bip * pr["p_ground_ball"].to_numpy(),
        "air": bip * (pr["p_fly_ball"].to_numpy() + pr["p_line_drive"].to_numpy()),
        "damage": damage,
    }
    for fam, groups in FAMILIES.items():
        on = pitches[[f"pt_{g}" for g in groups]].sum(axis=1).to_numpy() > 0
        out[f"{fam}_swing"] = np.where(on, out["swing"], 0)
        out[f"{fam}_whiff"] = np.where(on, out["whiff"], 0)
        out[f"{fam}_bip"] = np.where(on, bip, 0)
        out[f"{fam}_damage"] = np.where(on, damage, 0)
    frame = pd.DataFrame(out, index=pitches.index).astype(np.float64)
    return frame.assign(
        pitcher=pitches["pitcher"].to_numpy(),
        season=pitches["season"].to_numpy(),
        game_date=pitches["game_date"].to_numpy(),
    )


SUM_COLUMNS = sorted({c for pair in RATES.values() for c in pair})


def _rates(sums: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame(
            {r: sums[a] / sums[b].where(sums[b] > 0) for r, (a, b) in RATES.items()},
            index=sums.index,
        )


def season_features(
    rows: pd.DataFrame, pitches: pd.DataFrame, fit: StuffFit, season: int
) -> pd.DataFrame:
    """``STUFF_INPUTS`` for the table ``rows`` of ``season`` (``ROW_KEYS`` + ``as_of``),
    read by ``fit`` (the season's model, trained before it) from ``pitches`` (every
    season up to and including ``season``)."""
    use = pitches[pitches["season"] <= season]
    sums = pitch_sums(use, fit)
    ref_rows = sums[sums["season"] == season - 1]
    reference = _rates(ref_rows[SUM_COLUMNS].sum().to_frame().T).iloc[0]
    per_season = sums[sums["season"] < season].groupby(["pitcher", "season"])[SUM_COLUMNS].sum()

    def window(keep: Callable[[pd.Index], np.ndarray], name: str) -> pd.DataFrame:
        part = per_season[keep(per_season.index.get_level_values("season"))]
        rates = _rates(part.groupby(level="pitcher").sum()) - reference
        return rates.add_prefix(f"{name}_stuff_")

    hist = pd.concat(
        [
            window(lambda s: s == season - 1, "p1"),
            window(lambda s: s >= season - 3, "p3"),
            window(lambda s: s < season, "car"),
        ],
        axis=1,
    )
    out = rows[[*ROW_KEYS, "as_of"]].reset_index(drop=True)
    out = out.merge(hist, left_on="player_id", right_index=True, how="left")
    out = out.merge(_std(out, sums[sums["season"] == season], reference), on=ROW_KEYS, how="left")
    if season <= GRADED_CONTACT_FROM:
        out[[f"{w}_stuff_{r}" for w in WINDOWS for r in CONTACT_QUALITY]] = np.nan
    return out[[*ROW_KEYS, "as_of", *STUFF_INPUTS]]


def _std(rows: pd.DataFrame, sums: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """std_stuff_* per row: his pitches this season before the as-of date."""
    daily = (
        sums.groupby(["pitcher", "game_date"])[SUM_COLUMNS]
        .sum()
        .groupby(level="pitcher")
        .cumsum()
        .reset_index()
        .rename(columns={"pitcher": "player_id", "game_date": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"]).astype("datetime64[ns]")
    daily["player_id"] = daily["player_id"].astype("int64")
    left = rows[[*ROW_KEYS, "as_of"]].assign(
        as_of=pd.to_datetime(rows["as_of"]).astype("datetime64[ns]"),
        player_id=rows["player_id"].astype("int64"),
    )
    at = pd.merge_asof(
        left.sort_values("as_of"),
        daily.sort_values("date"),
        left_on="as_of",
        right_on="date",
        by="player_id",
        allow_exact_matches=False,  # a game on the as-of date hasn't happened yet
    )
    rates = (_rates(at[SUM_COLUMNS]) - reference).add_prefix("std_stuff_")
    return pd.concat([at[ROW_KEYS].reset_index(drop=True), rates.reset_index(drop=True)], axis=1)


def build_stuff_features(
    table: pd.DataFrame, pitches: pd.DataFrame, models: dict[int, StuffFit]
) -> pd.DataFrame:
    """``STUFF_INPUTS`` for every ``table`` row (``ROW_KEYS`` + ``as_of``), keyed by
    ``ROW_KEYS`` plus ``as_of``. ``models``: season -> that season's stuff model; rows of a
    season with no model get NaN."""
    rows = table[[*ROW_KEYS, "as_of"]]
    parts = []
    for season in sorted(int(s) for s in rows["season"].unique()):
        here = rows[rows["season"] == season]
        if season not in models:
            logger.info("season %d: no stuff model, inputs left blank", season)
            parts.append(here.reset_index(drop=True).assign(**dict.fromkeys(STUFF_INPUTS, np.nan)))
            continue
        logger.info("season %d: reading %d rows", season, len(here))
        parts.append(season_features(here, pitches, models[season], season))
    return pd.concat(parts, ignore_index=True)[[*ROW_KEYS, "as_of", *STUFF_INPUTS]]


def load_stuff_inputs(
    table: pd.DataFrame, root: Path, name: str, rate_set: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    """The ``stuff_inputs(rate_set)`` from ``stuff_<name>.parquet`` under ``root``,
    aligned to ``table``'s rows, and the file's build options."""
    return load_feature_file(
        table,
        stuff_features_path(root, name),
        stuff_inputs(rate_set),
        f"run scripts/build_pitcher_stuff.py --features-file {name} --run <stuff run>",
    )
