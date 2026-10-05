"""Turn training-table counts into model inputs (rates) and answers (rest-of-season rates).

The table (:mod:`hitter_ros.table`) stores counts; this module owns every division. A
rate with a zero denominator is NaN here and is filled after standardizing (so it lands
on the training mean); the window's log volume next to it tells the model how much to
trust it.

:func:`input_frame` reads only named, non-``ros_*`` columns; a test checks that
changing every ``ros_*`` value leaves the inputs unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

WINDOWS = ("std", "p1", "p3", "car")

# Answers: rest-of-season rates, and the counts that weight each one in the loss.
TARGETS = ("r", "hr", "rbi", "sb", "avg")
TARGET_WEIGHT = {"r": "ros_pa", "hr": "ros_pa", "rbi": "ros_pa", "sb": "ros_pa", "avg": "ros_ab"}


def _div(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN where den is 0."""
    return num / den.where(den > 0)


def _window_rates(t: pd.DataFrame, w: str) -> dict[str, pd.Series]:
    def c(name: str) -> pd.Series:
        return t[f"{w}_{name}"].astype(float)

    pa, ab, bip, pitches = c("pa"), c("ab"), c("bip"), c("pitches")
    singles = c("h") - c("hr") - c("b2") - c("b3")
    ev_mean = _div(c("ev_sum"), c("ev_n"))
    la_mean = _div(c("la_sum"), c("la_n"))
    out = {
        "log_pa": np.log1p(pa),
        "log_bip": np.log1p(bip),
        "log_pitches": np.log1p(pitches),
        # box-score rates
        **{f"{s}_pa": _div(c(s), pa) for s in ("hr", "r", "rbi", "sb", "cs", "bb", "k", "hbp")},
        "avg": _div(c("h"), ab),
        "iso": _div(c("b2") + 2 * c("b3") + 3 * c("hr"), ab),
        "steal_attempt_rate": _div(c("sb") + c("cs"), singles + c("bb") + c("hbp")),
        "start_rate": _div(c("starts"), c("games")),
        "lineup_spot": _div(c("spot_sum"), c("starts")),
        # plate discipline
        "swing_rate": _div(c("swings"), pitches),
        "whiff_rate": _div(c("whiffs"), c("swings")),
        "zone_swing": _div(c("zone_swings"), c("zone_pitches")),
        "chase_swing": _div(c("chase_swings"), c("chase_pitches")),
        "zone_contact": _div(c("zone_contacts"), c("zone_swings")),
        "chase_contact": _div(c("chase_contacts"), c("chase_swings")),
        "first_pitch_swing": _div(c("first_pitch_swings"), c("first_pitches")),
        "pitches_per_pa": _div(pitches, pa),
        "share_vs_lhp": _div(c("pitches_vs_lhp"), pitches),
        "share_as_lhb": _div(c("pitches_as_lhb"), pitches),
        # batted balls
        "ev_mean": ev_mean,
        "ev_sd": np.sqrt((_div(c("ev_sq_sum"), c("ev_n")) - ev_mean**2).clip(lower=0)),
        "la_mean": la_mean,
        "la_sd": np.sqrt((_div(c("la_sq_sum"), c("la_n")) - la_mean**2).clip(lower=0)),
        **{f"{s}_rate": _div(c(s), c("ev_n")) for s in ("ev95", "ev100", "ev105")},
        **{f"{s}_rate": _div(c(s), bip) for s in ("barrels", "solid", "gb", "ld", "fb", "pu")},
        "sweet_spot_rate": _div(c("sweet_spot"), c("la_n")),
        "pull_rate": _div(c("pulled"), c("spray_n")),
        "oppo_rate": _div(c("oppo"), c("spray_n")),
        "pulled_air_rate": _div(c("pulled_air"), bip),
        "xwoba_con": _div(c("xwoba_sum"), c("xwoba_n")),
        "xba_con": _div(c("xba_sum"), c("xba_n")),
        "bat_speed": _div(c("bat_speed_sum"), c("bat_speed_n")),
    }
    return {f"{w}_{k}": v for k, v in out.items()}


# League-side rates for the era inputs (#421). Each takes a table-column getter for one
# window and returns that window's rate. They must match the same-named player rates in
# _window_rates (a test checks they do), so the player-vs-league ratio compares like
# with like.
Column = Callable[[str], pd.Series]


def _per_pa(stat: str) -> Callable[[Column], pd.Series]:
    def rate(c: Column) -> pd.Series:
        return _div(c(stat), c("pa"))

    return rate


_ERA_RATES: dict[str, Callable[[Column], pd.Series]] = {
    **{f"{s}_pa": _per_pa(s) for s in ("r", "hr", "rbi", "sb", "bb", "k")},
    "avg": lambda c: _div(c("h"), c("ab")),
    "iso": lambda c: _div(c("b2") + 2 * c("b3") + 3 * c("hr"), c("ab")),
    "steal_attempt_rate": lambda c: _div(
        c("sb") + c("cs"), c("h") - c("hr") - c("b2") - c("b3") + c("bb") + c("hbp")
    ),
}


ERA_MODES = ("none", "relative", "full")
# Table columns the era options read; a table built before #421 lacks them.
ERA_TABLE_COLUMNS = ("lg_std_pa", "lg_p3_pa")


def _era_inputs(t: pd.DataFrame, mode: str, player: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Era inputs (#421). ``relative``: the player's rate divided by the league's in the
    same window. ``full`` adds the league rates themselves and rule flags known before
    the season -- but those are identical for every row of a season, so with ~18
    seasons the net uses them as a season label, memorizes each season's quirks, and
    extrapolates badly to a new one (seen in #421: single test seasons blew up)."""
    out: dict[str, pd.Series] = {}
    if mode == "none":
        return out
    for w in WINDOWS:

        def lg(name: str, w: str = w) -> pd.Series:
            return t[f"lg_{w}_{name}"].astype(float)

        for name, rate in _ERA_RATES.items():
            league = rate(lg)
            out[f"{w}_{name}_vs_lg"] = _div(player[f"{w}_{name}"], league)
            if mode == "full":
                out[f"lg_{w}_{name}"] = league
    if mode == "full":
        season = t["season"]
        out["rules_universal_dh"] = ((season == 2020) | (season >= 2022)).astype(float)
        out["rules_2023"] = (season >= 2023).astype(float)  # pitch clock, bigger bases
    return out


def league_reference(t: pd.DataFrame) -> pd.DataFrame:
    """Each row's league rates for the five answers, known on its as-of date: the last
    three seasons plus this season before the date, pooled. Used to predict a player
    relative to his league and scale back (``relative_target``). Across 2011-2026 the
    3-season average missed next season's league R and HR rates by less than last
    season alone did (#421).

    NaN for a row with no earlier season in the store (its first season): a reference
    built from a few days of this season's games would be mostly noise."""
    counts = pd.DataFrame(
        {c: t[f"lg_p3_{c}"].astype(float) + t[f"lg_std_{c}"].astype(float) for c in COUNTS},
        index=t.index,
    )
    ref = rates_from_counts(counts)
    return ref.where(t["lg_p3_pa"].astype(float) > 0)


def league_answer_rates(t: pd.DataFrame) -> pd.DataFrame:
    """Each row's league rates over its answer window: every table row of the same
    season and as-of week, ``ros_*`` counts pooled. The table has a row for every
    hitter-season who plays on or after the date, so this is the league's rest of the
    season (same population as the ``lg_*`` columns).

    Uses the answers, so it is a **training target denominator only** (#424): dividing
    by it asks "how much better than the league will he be", which needs no forecast of
    the league's level. Never an input, and never used to turn a prediction into rates.
    """
    keys = [t["season"], t["week"]]
    counts = pd.DataFrame(
        {c: t[f"ros_{c}"].astype(float).groupby(keys).transform("sum") for c in COUNTS},
        index=t.index,
    )
    return rates_from_counts(counts)


# Table columns the steal inputs read (#413); a table built before #413 lacks them.
STEAL_TABLE_COLUMNS = ("std_steal_opp2", "std_starts_c", "std_team_sb", "p1_hp_to_1b")


def _steal_inputs(t: pd.DataFrame) -> dict[str, pd.Series]:
    """Steal inputs (#413), per window: how often he is on base with the next base open,
    how often he runs when he is, how often he's safe, and his share of starts at C /
    SS / CF / DH. Plus his team's steal attempts per PA (the manager's green light),
    this season so far and last season, and his home-to-first time and bolt rate."""
    # Local import: table.py pulls in the pitch-data store (duckdb, requests).
    from fantasy_baseball.hitter_ros.table import START_POSITIONS

    out: dict[str, pd.Series] = {}
    for w in WINDOWS:

        def c(name: str, w: str = w) -> pd.Series:
            return t[f"{w}_{name}"].astype(float)

        opp = c("steal_opp2") + c("steal_opp3")
        attempts = c("sb") + c("cs")
        out[f"{w}_steal_opp_pa"] = _div(opp, c("pa"))
        out[f"{w}_attempts_per_opp"] = _div(attempts, opp)
        out[f"{w}_sb_success"] = _div(c("sb"), attempts)
        for pos in (p.lower() for p in START_POSITIONS):
            out[f"{w}_start_share_{pos}"] = _div(c(f"starts_{pos}"), c("starts"))
    for w in ("std", "p1"):
        team = t[f"{w}_team_sb"].astype(float) + t[f"{w}_team_cs"].astype(float)
        out[f"{w}_team_steal_pa"] = _div(team, t[f"{w}_team_pa"].astype(float))
    # Savant's sprint leaderboard, the two previous seasons: home-to-first time, and
    # "bolts" (runs at 30+ ft/s) per competitive run. The leaderboard leaves bolts
    # blank for a runner with none, so blank counts as 0; _div leaves the rate NaN where
    # his runs are unknown.
    for w in ("p1", "p2"):
        runs = t[f"{w}_sprint_runs"].astype(float)
        bolts = t[f"{w}_bolts"].astype(float).fillna(0.0)
        out[f"{w}_hp_to_1b"] = t[f"{w}_hp_to_1b"].astype(float)
        out[f"{w}_bolt_rate"] = _div(bolts, runs)
    return out


def input_frame(t: pd.DataFrame, era: str = "none", steal: bool = False) -> pd.DataFrame:
    """Model inputs for every table row: rates per window plus context. NaN = unknown.
    ``era``: see :func:`_era_inputs`. ``steal``: add :func:`_steal_inputs`."""
    if era not in ERA_MODES:
        raise ValueError(f"unknown era mode {era!r}")
    cols: dict[str, pd.Series] = {}
    for w in WINDOWS:
        cols.update(_window_rates(t, w))
    cols.update(
        {
            "age": t["age"].astype(float),
            "week": t["week"].astype(float),
            "frac_season_left": t["frac_season_left"].astype(float),
            "p1_in_store": t["p1_in_store"].astype(float),
            "p3_seasons_in_store": t["p3_seasons_in_store"].astype(float),
            "car_seasons_in_store": t["car_seasons_in_store"].astype(float),
            "p1_sprint_speed": t["p1_sprint_speed"].astype(float),
            "p2_sprint_speed": t["p2_sprint_speed"].astype(float),
            "p1_sprint_runs": np.log1p(t["p1_sprint_runs"].astype(float)),
            "p2_sprint_runs": np.log1p(t["p2_sprint_runs"].astype(float)),
            "std_team_r_pa": _div(t["std_team_r"].astype(float), t["std_team_pa"].astype(float)),
            "p1_team_r_pa": _div(t["p1_team_r"].astype(float), t["p1_team_pa"].astype(float)),
            "std_team_log_pa": np.log1p(t["std_team_pa"].astype(float)),
        }
    )
    cols.update(_era_inputs(t, era, cols))
    if steal:
        cols.update(_steal_inputs(t))
    return pd.DataFrame(cols, index=t.index)


# The counts the five answer rates are built from.
COUNTS = ("pa", "ab", "h", "r", "hr", "rbi", "sb")


def rates_from_counts(df: pd.DataFrame) -> pd.DataFrame:
    """The five answer rates from counts ``pa, ab, h, r, hr, rbi, sb``: R/HR/RBI/SB per PA
    and AVG = H/AB, NaN with no PA/AB. The one definition used to train and to score."""
    pa, ab = df["pa"].astype(float), df["ab"].astype(float)
    out = pd.DataFrame({s: _div(df[s].astype(float), pa) for s in ("r", "hr", "rbi", "sb")})
    out["avg"] = _div(df["h"].astype(float), ab)
    return out[list(TARGETS)]


def target_frame(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rest-of-season rates, loss weights). A rate with no PA/AB is NaN with weight 0."""
    ros = t[[f"ros_{c}" for c in COUNTS]]
    rates = rates_from_counts(ros.rename(columns=lambda c: c.removeprefix("ros_")))
    weights = pd.DataFrame({k: t[v].astype(float) for k, v in TARGET_WEIGHT.items()}, index=t.index)
    return rates, weights


class Standardizer:
    """Mean/std scaling fit on training rows; unknown (NaN) inputs become 0 = the mean.

    An input never seen in training (e.g. bat speed, which starts in 2023, when training
    on earlier seasons) is dropped: the net has no weight to give it. An input that is
    sometimes known and sometimes not gets a 0/1 ``<name>_missing`` column; one that is
    always known gets none, so no flag can take a value training never showed."""

    def __init__(self) -> None:
        self.mean: pd.Series | None = None
        self.std: pd.Series | None = None
        self.columns: list[str] = []
        self.missing_cols: list[str] = []

    def fit(self, x: pd.DataFrame) -> Standardizer:
        known = x.notna()
        self.columns = [c for c in x.columns if known[c].any()]
        self.missing_cols = [c for c in self.columns if not known[c].all()]
        self.mean = x[self.columns].mean()
        self.std = x[self.columns].std().replace(0, 1).fillna(1)
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        assert self.mean is not None and self.std is not None, "fit first"
        z = ((x[self.columns] - self.mean) / self.std).fillna(0.0)
        flags = x[self.missing_cols].isna().astype(float).add_suffix("_missing")
        return np.asarray(pd.concat([z, flags], axis=1), dtype=np.float32)

    @property
    def n_features(self) -> int:
        assert self.mean is not None, "fit first"
        return len(self.mean) + len(self.missing_cols)


SEASON_PARTS = 5


def balance_by_season_time(weights: pd.DataFrame, frac_season_left: pd.Series) -> pd.DataFrame:
    """Rescale loss weights so each fifth of the season carries the same total weight.

    PA-weighting alone gives late-season rows (few PA left) ~4% of the weight though
    they are ~16% of the rows, so the net barely learns late-season projections. Within
    each fifth, rows still count in proportion to their PA. Each target column is
    balanced on its own, and the overall total is unchanged.
    """
    part = np.minimum((1 - frac_season_left.to_numpy()) * SEASON_PARTS, SEASON_PARTS - 1)
    return balance_by_group(weights, part.astype(int))


def balance_by_group(weights: pd.DataFrame, group: np.ndarray) -> pd.DataFrame:
    """Rescale loss weights so every group (labels 0..k-1) carries the same total weight.
    Within a group rows keep their relative weights; each target column is balanced on
    its own; the overall total is unchanged."""
    n_groups = int(group.max()) + 1 if len(group) else 0
    out = weights.copy()
    for col in weights.columns:
        w = weights[col].to_numpy(dtype=float)
        totals = np.bincount(group, weights=w, minlength=n_groups)
        present = totals > 0
        target = w.sum() / present.sum()
        scale = np.where(present, target / np.where(present, totals, 1), 0.0)
        out[col] = w * scale[group]
    return out
