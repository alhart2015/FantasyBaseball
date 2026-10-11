"""Turn training-table counts into model inputs (rates) and answers (rest-of-season rates).

The table (:mod:`hitter_ros.table`) stores counts; this module owns every division. A
rate with a zero denominator is NaN here and is filled after standardizing (so it lands
on the training mean); the window's log volume next to it tells the model how much to
trust it.

:func:`input_frame` reads only named, non-``ros_*`` columns; a test checks that
changing every ``ros_*`` value leaves the inputs unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WINDOWS = ("std", "p1", "p3", "car")
# Recent form (#419): the last 7 / 14 days before the date (table.RECENT_WINDOWS).
RECENT_WINDOWS = ("l7", "l14")

# Answers: rest-of-season rates, and the counts that weight each one in the loss.
TARGETS = ("r", "hr", "rbi", "sb", "avg")
TARGET_WEIGHT = {"r": "ros_pa", "hr": "ros_pa", "rbi": "ros_pa", "sb": "ros_pa", "avg": "ros_ab"}
# AVG as its pieces (#433): AVG = HR/AB + BABIP x (1 - K/AB - HR/AB), with balls in play
# BIP = AB - K - HR (so the identity is exact; sacrifice flies aren't in AB). Each piece
# is a success rate out of trials: K and HR out of AB, hits on balls in play out of BIP.
PIECES = ("k_ab", "hr_ab", "babip")
PIECE_COUNTS = ("ab", "h", "hr", "k")
# SB as its pieces (#413): SB/PA = opportunities/PA x attempts/opportunity x SB/attempt,
# exact whenever all three are defined. An opportunity is being on 1B with 2B open, or on
# 2B with 3B open, at the first pitch of a PA (table.STEAL_COUNTS); attempts = SB + CS.
# Opportunities and attempts are counts (Poisson: opportunities with PA as exposure,
# attempts with opportunities as exposure -- a runner can try twice on one opportunity,
# 2B then 3B, so attempts per opportunity can pass 1); SB out of attempts is a success
# rate (binomial).
SB_POISSON_PIECES = ("opp_pa", "att_opp")
SB_BINOMIAL_PIECES = ("sb_att",)
SB_PIECES = (*SB_POISSON_PIECES, *SB_BINOMIAL_PIECES)
SB_PIECE_COUNTS = ("pa", "sb", "cs", "steal_opp2", "steal_opp3")


def _div(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN where den is 0."""
    return num / den.where(den > 0)


# Strike zone for the zone / chase inputs and the pitch tokens' height (#433):
# "statcast" = Savant's zone (sz_top / sz_bot), "fixed" = FIXED_ZONE for every season
# (the table's fzone_ / fchase_ columns).
ZONES = ("statcast", "fixed")
# A fixed strike zone, the same box for every season and hitter. Savant's zone comes
# from sz_top / sz_bot, which the 2026 ABS system records differently (mean top 3.43 ->
# 3.22 ft, spread halved), so its in-zone share fell from 50.5% to 47.5% while pitch
# locations didn't move. The box tests the ball's center: sideways, within the plate's
# half-width plus a ball (0.83 ft); up and down, 1.5-3.5 ft, with no ball added (chosen
# so it holds 49-50% of pitches in 2023-2026 and agrees with Savant's zone on 96% of
# pitches before 2026). (half-width, bottom, top) in feet. Files counted with it record
# it (:func:`zone_options`), so changing it asks for a rebuild.
FIXED_ZONE = (0.83, 1.5, 3.5)


def zone_options(zone: str) -> dict[str, Any]:
    """The strike-zone build options a file made with ``zone`` records next to it."""
    if zone not in ZONES:
        raise ValueError(f"unknown zone {zone!r}")
    return {"zone": zone, **({"fixed_zone": list(FIXED_ZONE)} if zone == "fixed" else {})}


def read_build_options(path: Path) -> dict[str, Any]:
    """The build options saved next to a data file (``<file>.json``); ``{}`` without one."""
    options_path = path.with_suffix(".json")
    out: dict[str, Any] = json.loads(options_path.read_text()) if options_path.exists() else {}
    return out


def _window_rates(t: pd.DataFrame, w: str, zone: str = "statcast") -> dict[str, pd.Series]:
    def c(name: str) -> pd.Series:
        return t[f"{w}_{name}"].astype(float)

    z = "f" if zone == "fixed" else ""  # the same input names either way

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
        "zone_swing": _div(c(f"{z}zone_swings"), c(f"{z}zone_pitches")),
        "chase_swing": _div(c(f"{z}chase_swings"), c(f"{z}chase_pitches")),
        "zone_contact": _div(c(f"{z}zone_contacts"), c(f"{z}zone_swings")),
        "chase_contact": _div(c(f"{z}chase_contacts"), c(f"{z}chase_swings")),
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
# Table columns the era options and league_reference read; a table built before #421
# lacks them.
ERA_TABLE_COLUMNS = ("lg_std_pa", "lg_p3_pa", "lg_p1_pa")


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


# SB's league level moves faster than the other answers' (the 2023 rules, the 2026
# cool-off), so its reference is last season plus this season before the date, this
# season's games counted SB_STD_WEIGHT times. Across 2016-2026 it missed the league's
# rest-of-season SB rate by 0.69 SB/600 preseason and 0.61 mid-season (weeks 1-22),
# vs. 0.99 and 0.96 for the 3-season pool (#413).
SB_STD_WEIGHT = 3


def league_reference(
    t: pd.DataFrame, *, pieces: bool = False, sb_pieces: bool = False
) -> pd.DataFrame:
    """Each row's league rates for the five answers (and, with ``pieces``, AVG's
    pieces), known on its as-of date: the last
    three seasons plus this season before the date, pooled. Used to predict a player
    relative to his league and scale back (``relative_target``). Across 2011-2026 the
    3-season average missed next season's league R and HR rates by less than last
    season alone did (#421). SB is the exception: last season plus this season,
    this season weighted ``SB_STD_WEIGHT`` (#413).

    NaN for a row with no earlier season in the store (its first season): a reference
    built from a few days of this season's games would be mostly noise. SB, which
    reads only last season, is NaN without last season too.

    ``pieces``: also the ``PIECES`` rates (#433); ``sb_pieces``: the ``SB_PIECES``, from
    SB's window (#413)."""
    names = answer_counts(pieces=pieces)
    counts = pd.DataFrame(
        {c: t[f"lg_p3_{c}"].astype(float) + t[f"lg_std_{c}"].astype(float) for c in names},
        index=t.index,
    )
    ref = _with_pieces(counts, pieces)
    # SB, and its pieces with ``sb_pieces``: the fast window, so the pieces multiply
    # back to the SB reference.
    fast_names = SB_PIECE_COUNTS if sb_pieces else ("pa", "sb")
    fast = pd.DataFrame(
        {
            c: t[f"lg_p1_{c}"].astype(float) + SB_STD_WEIGHT * t[f"lg_std_{c}"].astype(float)
            for c in fast_names
        },
        index=t.index,
    )
    last_season = t["lg_p1_pa"].astype(float) > 0
    ref["sb"] = _div(fast["sb"], fast["pa"]).where(last_season)
    if sb_pieces:
        ref = pd.concat([ref, sb_piece_rates(fast).where(last_season)], axis=1)
    return ref.where(t["lg_p3_pa"].astype(float) > 0)


def league_answer_rates(
    t: pd.DataFrame, *, pieces: bool = False, sb_pieces: bool = False
) -> pd.DataFrame:
    """Each row's league rates over its answer window: every table row of the same
    season and as-of week, ``ros_*`` counts pooled. The table has a row for every
    hitter-season who plays on or after the date, so this is the league's rest of the
    season (same population as the ``lg_*`` columns).

    Uses the answers, so it is a **training target denominator only** (#424): dividing
    by it asks "how much better than the league will he be", which needs no forecast of
    the league's level. Never an input, and never used to turn a prediction into rates.

    ``pieces``: also the ``PIECES`` rates (#433); ``sb_pieces``: the ``SB_PIECES`` (#413).
    """
    keys = [t["season"], t["week"]]
    names = answer_counts(pieces=pieces, sb_pieces=sb_pieces)
    counts = pd.DataFrame(
        {c: t[f"ros_{c}"].astype(float).groupby(keys).transform("sum") for c in names},
        index=t.index,
    )
    return _with_pieces(counts, pieces, sb_pieces)


def _with_pieces(counts: pd.DataFrame, pieces: bool, sb_pieces: bool = False) -> pd.DataFrame:
    parts = [rates_from_counts(counts)]
    if pieces:
        parts.append(piece_rates(counts))
    if sb_pieces:
        parts.append(sb_piece_rates(counts))
    return pd.concat(parts, axis=1)


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


def input_frame(
    t: pd.DataFrame,
    era: str = "none",
    steal: bool = False,
    recent: bool = False,
    zone: str = "statcast",
    xba_ab: bool = False,
    blend: bool = False,
) -> pd.DataFrame:
    """Model inputs for every table row: rates per window plus context. NaN = unknown.
    ``era``: see :func:`_era_inputs`. ``steal``: add :func:`_steal_inputs`. ``recent``:
    add the same per-window rates over the last 7 and 14 days (#419). ``zone``: which
    strike zone the zone / chase inputs use (``ZONES``, #433). ``xba_ab``: add
    :func:`_xba_per_ab` per window (#433). ``blend``: add :func:`_blend_inputs` (#451)."""
    if era not in ERA_MODES:
        raise ValueError(f"unknown era mode {era!r}")
    if zone not in ZONES:
        raise ValueError(f"unknown zone {zone!r}")
    cols: dict[str, pd.Series] = {}
    for w in WINDOWS:
        cols.update(_window_rates(t, w, zone))
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
    if recent:
        for w in RECENT_WINDOWS:
            cols.update(_window_rates(t, w, zone))
    if steal:
        cols.update(_steal_inputs(t))
    if xba_ab:
        for w in (*WINDOWS, *(RECENT_WINDOWS if recent else ())):
            cols[f"{w}_xba_ab"] = _xba_per_ab(t, w, cols[f"{w}_xba_con"])
    if blend:
        cols.update(_blend_inputs(t))
    return pd.DataFrame(cols, index=t.index)


def _xba_per_ab(t: pd.DataFrame, w: str, xba_con: pd.Series) -> pd.Series:
    """Expected AVG per at-bat in window ``w``, strikeouts as outs (#433): xBA on contact
    times the share of at-bats that weren't strikeouts. The net has both pieces, but
    hitters with 300-700 MLB PA showed it under-used them: nudging its AVG toward this
    season's xBA per AB ordered them better in every season 2022-2026. ``xba_con``:
    the window's xBA on contact, from :func:`_window_rates`."""
    ab = t[f"{w}_ab"].astype(float)
    return xba_con * _div(ab - t[f"{w}_k"].astype(float), ab)


# Blended inputs (#451): this season to date mixed with a prior, one mix per entry here,
# each counting this many PA of prior. Walk-forward 2022-2026, the plain net left this
# season's signal on the table: late in the season about a fifth of the gap between a
# hitter's season-to-date rate and our call (SB: over half) showed up again in his rest
# of season. The net has the rate and the PA as separate inputs and must learn "trust
# the rate more as PA grow" itself; these do that mix for it, a ladder of strengths it
# can combine.
BLEND_PA = (50, 150, 450, 1350)


# The counts _blend_parts reads from each window.
BLEND_COUNTS = (
    "pa",
    "ab",
    "h",
    "r",
    "hr",
    "rbi",
    "sb",
    "bb",
    "k",
    "cs",
    "steal_opp2",
    "steal_opp3",
)


def _blend_parts(t: pd.DataFrame, w: str) -> dict[str, tuple[pd.Series, pd.Series]]:
    """(successes, trials) for each blended rate in window ``w``: the answers, BB/PA,
    AVG's pieces (:func:`piece_parts`) and SB's pieces (:func:`sb_piece_parts`)."""
    counts = t[[f"{w}_{n}" for n in BLEND_COUNTS]].astype(float)
    counts.columns = list(BLEND_COUNTS)
    pa = counts["pa"]
    return {
        **{f"{s}_pa": (counts[s], pa) for s in ("r", "hr", "rbi", "sb", "bb")},
        "avg": (counts["h"], counts["ab"]),
        **piece_parts(counts),
        **sb_piece_parts(counts),
    }


def _blend_inputs(t: pd.DataFrame) -> dict[str, pd.Series]:
    """Season-to-date rates blended with a prior, ``bl{K}_{rate}`` for each K in
    ``BLEND_PA`` (#451), Marcel-style:

    * prior = (last season x 2 + the two before) shrunk toward the league's rate over
      the same seasons by K PA: (successes + K' x league rate) / (trials + K').
    * blend = this season to date shrunk toward that prior by K PA the same way.

    K' is K PA in the rate's own trials (league trials per PA x K), so attempts per
    opportunity or SB per attempt get as much prior as K PA would bring. NaN where the
    league rate is unknown (a row in the store's first season)."""

    std, p1, p3 = (_blend_parts(t, w) for w in ("std", "p1", "p3"))
    lg_p1, lg_p3 = (_blend_parts(t, w) for w in ("lg_p1", "lg_p3"))
    lg_pa = t["lg_p1_pa"].astype(float) + t["lg_p3_pa"].astype(float)
    out = {}
    for name, (made, tried) in std.items():
        lg_made, lg_tried = lg_p1[name][0] + lg_p3[name][0], lg_p1[name][1] + lg_p3[name][1]
        lg_rate, tried_per_pa = _div(lg_made, lg_tried), _div(lg_tried, lg_pa)
        prior_made, prior_tried = p1[name][0] + p3[name][0], p1[name][1] + p3[name][1]
        for k in BLEND_PA:
            shrink = k * tried_per_pa
            prior = (prior_made + shrink * lg_rate) / (prior_tried + shrink)
            out[f"bl{k}_{name}"] = (made + shrink * prior) / (tried + shrink)
    return out


# The box-score rates of box_score_inputs: (count, per, input name).
BOX_RATES = (
    ("r", "pa", "r"),
    ("hr", "pa", "hr"),
    ("rbi", "pa", "rbi"),
    ("sb", "pa", "sb"),
    ("h", "ab", "avg"),
    ("k", "ab", "k"),
    ("bb", "pa", "bb"),
)


def _relative_blend_inputs(t: pd.DataFrame) -> dict[str, pd.Series]:
    """:func:`_blend_inputs` on the league's scale (#413): each window's successes over
    the successes a league-average hitter would have had in the same trials (observed
    over expected, 1 = the league), so a jump in the league's rate (the 2023 rules)
    moves none of them.

    * prior = last season x 2 + the two before, observed over expected, shrunk toward 1
      by K PA: (successes + K' x pool rate) / (expected + K' x pool rate), the pool being
      the league over those same seasons.
    * blend = this season to date, observed over expected at this season's league rate,
      shrunk toward that prior by K PA the same way.

    K' is K PA in the rate's own trials, as in :func:`_blend_inputs`. Before any game this
    season (preseason) the blend is the prior. NaN where the league rate is unknown."""
    std, p1, p3 = (_blend_parts(t, w) for w in ("std", "p1", "p3"))
    lg = {w: _blend_parts(t, f"lg_{w}") for w in ("std", "p1", "p3")}
    lg_pa = t["lg_p1_pa"].astype(float) + t["lg_p3_pa"].astype(float)
    out = {}
    for name, (made, tried) in std.items():
        pool_made, pool_tried = (
            lg["p1"][name][0] + lg["p3"][name][0],
            lg["p1"][name][1] + lg["p3"][name][1],
        )
        pool_rate, tried_per_pa = _div(pool_made, pool_tried), _div(pool_tried, lg_pa)
        prior_made = p1[name][0] + p3[name][0]
        prior_expected = p1[name][1] * _div(*lg["p1"][name]) + p3[name][1] * _div(*lg["p3"][name])
        # This season's league rate; the pool's before any game, where it is unknown.
        std_rate = _div(*lg["std"][name]).fillna(pool_rate)
        for k in BLEND_PA:
            shrink = k * tried_per_pa
            prior = (prior_made + shrink * pool_rate) / (prior_expected + shrink * pool_rate)
            out[f"bl{k}_{name}"] = (made + shrink * std_rate * prior) / (
                tried * std_rate + shrink * std_rate
            )
    return out


def box_score_inputs(t: pd.DataFrame, relative: bool = False) -> pd.DataFrame:
    """The SB net's inputs (``NetConfig.sb_inputs`` "box", #451): :func:`_blend_inputs`,
    each window's box-score rates (``BOX_RATES``) and log PA / AB, the week, the share of
    the season left and age. Steal opportunities come in only through the blends of SB's
    pieces (opportunities per PA, attempts per opportunity); none of :func:`input_frame`'s
    batted-ball, plate-discipline, speed, team, position or green-light inputs. NaN =
    unknown.

    ``relative`` ("box_relative", #413): every rate on the league's scale instead -- each
    window's rate over the league's in the same window, and :func:`_relative_blend_inputs`
    -- so the inputs, like the league-relative answers the net learns, carry no league
    level. With raw rates, a league-wide jump (SB after the 2023 rules) reads as every
    hitter getting better than his league, and is then counted again when the prediction
    is multiplied back by the new league rate."""
    cols = dict(_relative_blend_inputs(t) if relative else _blend_inputs(t))
    for count, per, name in BOX_RATES:
        for w in WINDOWS:
            rate = _div(t[f"{w}_{count}"].astype(float), t[f"{w}_{per}"].astype(float))
            if relative:
                league = _div(t[f"lg_{w}_{count}"].astype(float), t[f"lg_{w}_{per}"].astype(float))
                rate = _div(rate, league)
            cols[f"{w}_{name}"] = rate
    for w in WINDOWS:
        cols[f"{w}_log_pa"] = np.log1p(t[f"{w}_pa"].astype(float))
        cols[f"{w}_log_ab"] = np.log1p(t[f"{w}_ab"].astype(float))
    for c in ("week", "frac_season_left", "age"):
        cols[c] = t[c].astype(float)
    return pd.DataFrame(cols, index=t.index)


# A feature file built for the table (probes #417, minor-league inputs #435) has one row
# per table row, keyed by these, plus the row's as-of date to check it against.
ROW_KEYS = ["player_id", "season", "week"]


def check_aligned(table: pd.DataFrame, frame: pd.DataFrame) -> str | None:
    """Why a feature file ``frame`` can't be used with ``table`` (None if it can):
    duplicate keys, a table row it lacks, or an as-of date that differs (a file built for
    another table would have cut its history at the wrong date)."""
    if frame.duplicated(ROW_KEYS).any():
        return "it has duplicate (player, season, week) rows"
    merged = table[[*ROW_KEYS, "as_of"]].merge(
        frame[[*ROW_KEYS, "as_of"]], on=ROW_KEYS, how="left", suffixes=("", "_file")
    )
    missing = int(merged["as_of_file"].isna().sum())
    if missing:
        return f"it lacks {missing} table rows"
    moved = int((pd.to_datetime(merged["as_of"]) != pd.to_datetime(merged["as_of_file"])).sum())
    if moved:
        return f"{moved} rows have a different as-of date than the table"
    return None


def load_feature_file(
    table: pd.DataFrame,
    path: Path,
    columns: list[str],
    rebuild: str,
    expected_options: dict[str, Any] | None = None,
    expected_label: str = "the expected",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``columns`` of the feature file at ``path``, aligned to ``table``'s rows, and the
    build options saved next to it (``<file>.json``; ``{}`` without one). ValueError
    ending in ``rebuild`` (how to make the file) when it is missing, was built with
    options other than ``expected_options`` (when given), lacks a column, or was built
    for another table."""
    if not path.exists():
        raise ValueError(f"{path} is missing; {rebuild}")
    options = read_build_options(path)
    if expected_options is not None and options != expected_options:
        raise ValueError(
            f"{path} was built with {options or 'unknown options'}, not {expected_label} "
            f"{expected_options}; {rebuild}"
        )
    frame = pd.read_parquet(path)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} lacks {len(missing)} features (e.g. {missing[0]}); {rebuild}")
    problem = check_aligned(table, frame)
    if problem:
        raise ValueError(f"{path}: {problem}; {rebuild}")
    return aligned_inputs(table, frame, columns), options


def aligned_inputs(table: pd.DataFrame, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """``columns`` of a feature file aligned to ``table``'s rows by ``ROW_KEYS``; NaN for
    a row the file doesn't cover."""
    merged = table[ROW_KEYS].merge(
        frame[[*ROW_KEYS, *columns]], on=ROW_KEYS, how="left", validate="one_to_one"
    )
    merged.index = table.index
    return merged[columns]


# The counts the five answer rates are built from.
COUNTS = ("pa", "ab", "h", "r", "hr", "rbi", "sb")
# COUNTS plus what AVG's pieces need (K), #433.
COUNTS_WITH_PIECES = (*COUNTS, *(c for c in PIECE_COUNTS if c not in COUNTS))
# Every answer count: COUNTS_WITH_PIECES plus what SB's pieces need (CS and steal
# opportunities), #413. The table's short-horizon answers count all of them.
ALL_ANSWER_COUNTS = (
    *COUNTS_WITH_PIECES,
    *(c for c in SB_PIECE_COUNTS if c not in COUNTS_WITH_PIECES),
)


def answer_counts(*, pieces: bool = False, sb_pieces: bool = False) -> tuple[str, ...]:
    """The counts the answers need: ``COUNTS``, plus AVG's piece counts with ``pieces``
    and SB's with ``sb_pieces``, in ``ALL_ANSWER_COUNTS`` order."""
    wanted = {*COUNTS, *(PIECE_COUNTS if pieces else ()), *(SB_PIECE_COUNTS if sb_pieces else ())}
    return tuple(c for c in ALL_ANSWER_COUNTS if c in wanted)


def rates_from_counts(df: pd.DataFrame) -> pd.DataFrame:
    """The five answer rates from counts ``pa, ab, h, r, hr, rbi, sb``: R/HR/RBI/SB per PA
    and AVG = H/AB, NaN with no PA/AB. The one definition used to train and to score."""
    pa, ab = df["pa"].astype(float), df["ab"].astype(float)
    out = pd.DataFrame({s: _div(df[s].astype(float), pa) for s in ("r", "hr", "rbi", "sb")})
    out["avg"] = _div(df["h"].astype(float), ab)
    return out[list(TARGETS)]


def piece_parts(df: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Each ``PIECES`` rate's (successes, trials) from counts ``ab, h, hr, k`` (#433): K
    and HR out of AB, hits on balls in play (H - HR) out of BIP (AB - K - HR). The one
    place they are defined."""
    ab, h, hr, k = (df[c].astype(float) for c in PIECE_COUNTS)
    return {"k_ab": (k, ab), "hr_ab": (hr, ab), "babip": (h - hr, ab - k - hr)}


def piece_rates(df: pd.DataFrame) -> pd.DataFrame:
    """The ``PIECES`` rates from counts ``ab, h, hr, k`` (#433): K/AB, HR/AB and BABIP;
    NaN with no AB (no BIP for BABIP)."""
    return pd.DataFrame(
        {name: _div(made, tried) for name, (made, tried) in piece_parts(df).items()},
        index=df.index,
    )


def avg_from_pieces(k_ab: Any, hr_ab: Any, babip: Any) -> Any:
    """AVG from its ``PIECES`` (#433): HR/AB + BABIP x (1 - K/AB - HR/AB)."""
    return hr_ab + babip * (1 - k_ab - hr_ab)


def sb_piece_rates(df: pd.DataFrame) -> pd.DataFrame:
    """The ``SB_PIECES`` rates from counts ``pa, sb, cs, steal_opp2, steal_opp3`` (#413):
    opportunities per PA, attempts per opportunity, SB per attempt; NaN where the
    denominator is 0."""
    return pd.DataFrame(
        {name: _div(made, tried) for name, (made, tried) in sb_piece_parts(df).items()},
        index=df.index,
    )


def sb_piece_parts(df: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Each ``SB_PIECES`` rate's (successes, trials) from counts ``pa, sb, cs,
    steal_opp2, steal_opp3`` (#413): opportunities (``table.STEAL_COUNTS``) out of PA,
    attempts (SB + CS) out of opportunities, SB out of attempts. The one place they are
    defined."""
    sb = df["sb"].astype(float)
    opp = df["steal_opp2"].astype(float) + df["steal_opp3"].astype(float)
    attempts = sb + df["cs"].astype(float)
    return {
        "opp_pa": (opp, df["pa"].astype(float)),
        "att_opp": (attempts, opp),
        "sb_att": (sb, attempts),
    }


def sb_piece_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Each ``SB_PIECES`` rate's trials (its denominator and loss weight), from
    :func:`sb_piece_parts`."""
    return pd.DataFrame(
        {name: tried for name, (_, tried) in sb_piece_parts(df).items()}, index=df.index
    )


def sb_from_pieces(opp_pa: Any, att_opp: Any, sb_att: Any) -> Any:
    """SB per PA from its ``SB_PIECES`` (#413)."""
    return opp_pa * att_opp * sb_att


def horizon_columns(horizons: tuple[int, ...] = (), stats: tuple[str, ...] = TARGETS) -> list[str]:
    """Output column names: the rest-of-season ``stats`` (default ``TARGETS``), then
    ``n{N}_{stat}`` for each short horizon (#419)."""
    return [*stats, *(f"n{n}_{s}" for n in horizons for s in stats)]


def target_stat(column: str) -> str:
    """The stat of an output column: ``n25_hr`` -> ``hr``; ``hr`` -> ``hr``;
    ``n25_k_ab`` -> ``k_ab``."""
    return column.split("_", 1)[1] if column_horizon(column) != "ros" else column


def column_horizon(column: str) -> str:
    """The horizon of an output column: ``n25_hr`` -> ``n25``; ``hr`` -> ``ros``."""
    head = column.split("_", 1)[0]
    return head if head[:1] == "n" and head[1:].isdigit() else "ros"


def target_frame(
    t: pd.DataFrame,
    horizons: tuple[int, ...] = (),
    *,
    pieces: bool = False,
    sb_pieces: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rates, loss weights), one column per ``horizon_columns(horizons)``: rest of season,
    then the next N PA (#419). A rate with no PA/AB -- or a horizon he didn't reach -- is
    NaN with weight 0. Weights are the PA (AB for AVG) in that answer's window.

    ``pieces``: also the ``PIECES`` per window (#433), after all of those columns,
    weighted by their trials (AB for K/AB and HR/AB, BIP for BABIP). ``sb_pieces``: the
    ``SB_PIECES`` per window (#413), after those, weighted by ``sb_piece_trials``."""
    rate_parts, weight_parts, piece_rate_parts, piece_weight_parts = [], [], [], []
    sb_rate_parts, sb_weight_parts = [], []
    names = answer_counts(pieces=pieces, sb_pieces=sb_pieces)
    for prefix, tag in (("ros_", ""), *((f"ros_n{n}_", f"n{n}_") for n in horizons)):
        counts = t[[f"{prefix}{c}" for c in names]].rename(
            columns=lambda c, p=prefix: c.removeprefix(p)
        )
        rate_parts.append(rates_from_counts(counts).add_prefix(tag))
        weights = pd.DataFrame(
            {
                f"{tag}{k}": t[v.replace("ros_", prefix, 1)].astype(float)
                for k, v in TARGET_WEIGHT.items()
            },
            index=t.index,
        )
        weight_parts.append(weights.fillna(0.0))
        if pieces:
            piece_rate_parts.append(piece_rates(counts).add_prefix(tag))
            piece_weights = pd.DataFrame(
                {f"{tag}{name}": tried for name, (_, tried) in piece_parts(counts).items()},
                index=t.index,
            )
            piece_weight_parts.append(piece_weights.clip(lower=0).fillna(0.0))
        if sb_pieces:
            sb_rate_parts.append(sb_piece_rates(counts).add_prefix(tag))
            sb_weights = sb_piece_trials(counts).add_prefix(tag)
            sb_weight_parts.append(sb_weights.clip(lower=0).fillna(0.0))
    rates = pd.concat([*rate_parts, *piece_rate_parts, *sb_rate_parts], axis=1)
    weights = pd.concat([*weight_parts, *piece_weight_parts, *sb_weight_parts], axis=1)
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
