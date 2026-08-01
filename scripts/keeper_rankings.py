"""Rank keeper candidates on skill, luck, batted-ball, future and age, then price
them in VAR.

Reads what `fetch_keeper_skills.py` cached, computes each player's actual roto
value (SGP) for the season, blends the five families in percentile space using
the weights fitted in `keepers/composite.py`, then converts that ordinal
composite into projected value with an error bar via `keepers/projection.py`.

Writes `data/cache/keeper_skills/keeper_rankings_{kind}_{year}.csv`:

    value_pct        percentile of actual SGP this season (= skill + luck)
    skill_pct        SKILL   -- percentile of the peripherals
    luck_pct         LUCK    -- value_pct - skill_pct, the unsupported part of the line
    batted_ball_pct  BATTED_BALL -- percentile of avg-xba / fip-era; carries a
                     NEGATIVE weight, clawing the batted-ball luck back out of LUCK
    future_pct       FUTURE  -- percentile of blended out-year ZiPS projected SGP
    age_pct          AGE     -- percentile of age, younger better
    composite        the fitted five-family blend, ordinal within pool
    proj_sgp    projected 2027 SGP implied by that composite
    sd          predictive SD of proj_sgp for ONE player, not a group mean
    proj_var    proj_sgp plus a mean-centred positional scarcity adjustment
    pos         the position that adjustment came from

Rows are ranked by `proj_var`, not by composite. The composite is a within-pool
percentile and cannot be compared across pools; `proj_var` is in standings-gain
points, so a catcher and a closer and an outfielder can share one list. The
positional term is a scarce-position bonus rather than a subtracted floor; see
`keepers.scarcity`, and regenerate it with `--scarcity`.

MID-SEASON CAVEAT. `proj_sgp`, `sd` and `proj_var` are fitted on COMPLETE seasons,
so running this partway through one scores a truncated pool against full-season
constants. Truncation removes the players who have not yet cleared MIN_PT, and
those are mostly the low-value ones, so a SURVIVOR ranks lower inside the smaller
pool than he would in the full one and his printed absolutes come out LOW.

The distortion is strongly uneven: it is several times larger mid-board than at
the top, which means GAPS between tiers are unreliable mid-season as well as
levels. Only within-pool ORDER survives, and only NEARLY -- not because the
truncation is a monotone remap, which it is not: `skill_pct` averages several
per-stat percentiles that each remap differently under a change of pool, so a
small number of pairs genuinely cross.

`--study` prints all of it -- per-quintile shift and rank correlation, at this
run's actual pool size -- for the same reason the numbers left this file
elsewhere: an earlier version of this paragraph had the direction backwards.

Read the ranking in TIERS, not by row. Adjacent players are separated by far less
than `sd`, so consecutive ranks are close to coin flips; `sd` is there to stop a
single-rank gap being read as real.

`--roster` answers the decision directly: P(each of my players finishes among my
N best). That is joint and set-dependent -- it needs the exact rivals -- so it
lives there rather than in this pool-wide CSV. `--league` does the same for all ten
teams, computing each team's P(keep) over ITS OWN roster, never league-wide.

CROSS-POOL CAVEAT for `--league`. `proj_var` is in SGP so a hitter and a pitcher
CAN share one list, but the two fits regress to their own pool's mean at very
different rates, so no pitcher reaches the top of a mixed board. That is a real
difference in PREDICTABILITY rather than a scale artifact -- top-decile hitters go
on to earn substantially more than top-decile pitchers, and pitchers are several
times more likely to collapse to nothing. `--study` prints both, per pool, for the
same reason the other numbers left this file: nothing here would regenerate them.
Read a mixed board as expected value only, and read the pitcher list on its own --
its top is compressed to a fraction of one sd, so that ORDER means nothing.

`luck` carries a POSITIVE weight and `batted_ball` a NEGATIVE one: `luck` rewards
outperforming the peripherals (mostly a playing-time/role signal) while
`batted_ball` claws back the part of that gap which is pure batted-ball luck and
does not repeat. `future` is discounted for staleness. All three are
counterintuitive and argued in `keepers/composite.py`; `--study` reproduces the
evidence and `--backtest` the weights.

Usage:
    python scripts/keeper_rankings.py
    python scripts/keeper_rankings.py --roster          # P(top-3) on my roster
    python scripts/keeper_rankings.py --league --top 50 # every team's board
    python scripts/keeper_rankings.py --backtest        # refit the family weights
    python scripts/keeper_rankings.py --fit             # refit projection.py constants
    python scripts/keeper_rankings.py --study           # the supporting diagnostics
    python scripts/keeper_rankings.py --scarcity        # re-measure positional credits
    python scripts/keeper_rankings.py --year 2025
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Player names carry accents (Luis Garcia Jr., Julio Rodriguez) and this box's
# stdout is cp1252, which renders them as "Garc?a". Names come from data, not from
# source, so this is the documented exception to the ASCII-only rule.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fantasy_baseball.config import load_config
from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.data.fangraphs import load_projection_set
from fantasy_baseball.keepers.actuals import index_by_mlbam, innings_to_float
from fantasy_baseball.keepers.appearances import season_eligibility
from fantasy_baseball.keepers.composite import (
    FAMILIES,
    FITTED_WEIGHTS,
    FUTURE_BLEND,
    HITTER_SKILLS,
    LOWER_IS_BETTER,
    PITCHER_SKILLS,
    batted_ball,
    check_known_families,
    composite,
    durability,
    future_percentile,
    luck,
    percentile,
    skill_percentile,
    speed,
)
from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season
from fantasy_baseball.keepers.positions import load_positions
from fantasy_baseball.keepers.projection import (
    RESIDUAL_QUANTILE_LEVELS,
    expected_sgp,
    probability_top_n,
    sgp_sd,
)
from fantasy_baseball.keepers.scarcity import (
    NATIVE_CREDITS,
    centred_credits,
    credit_levels,
    marginal_starter_floors,
    slot_capacities,
)
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.positions import HITTER_ELIGIBLE, PITCHER_ELIGIBLE
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import Category
from fantasy_baseball.utils.name_utils import normalize_name

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
SKILLS_DIR = PROJECT_ROOT / "data" / "cache" / "keeper_skills"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"

# Below these a percentile is noise: the skills module regresses nothing, so a
# 3-inning pitcher can post the league's best ERA-. See its module docstring.
MIN_PT = {"hitter": 250, "pitcher": 50}
# Used only when a player is absent from the position map. UTIL is the deepest
# hitter floor, so an unknown hitter is charged the harshest replacement level
# rather than flattered by a scarce one. The pitcher token is inert because SP and
# RP never appear in the credits table at all: it ships a single "P", which
# `var._pitcher_floor_key` falls back to. See `keepers.scarcity`.
FALLBACK_POS = {"hitter": ["UTIL"], "pitcher": ["P"]}
POOLS: tuple[str, ...] = (PlayerType.HITTER, PlayerType.PITCHER)
# Display schema for the per-pool tables; the CSV keeps every column.
SHOWN = [
    "rank",
    "name",
    "age",
    "pt",
    "pos",
    "skill_pct",
    "luck_pct",
    "batted_ball_pct",
    "future_pct",
    "composite",
    "proj_sgp",
    "sd",
    "proj_var",
    "keeper_of",
]
BACKTEST_FIT_YEARS = (2022, 2023)
BACKTEST_HOLDOUT = 2024
# The fit and the diagnostics that justify it must cover the same seasons.
ALL_TRANSITION_YEARS = (*BACKTEST_FIT_YEARS, BACKTEST_HOLDOUT)


def _raw(year: int, table: str) -> pd.DataFrame:
    """A cached BBRef pull: `table` is "batting" or "pitching"."""
    return pd.read_csv(SKILLS_DIR / f"raw_{year}" / f"bref_{table}_{year}.v2.csv")


def _sgp(lines: pd.DataFrame, denoms) -> pd.Series:
    return lines.apply(lambda row: calculate_player_sgp(row, denoms=denoms), axis=1)


def season_value(year: int, kind: str, denoms) -> pd.DataFrame:
    """Actual roto value, age and playing time for `year`, indexed by mlbam_id.

    Taken from the same BBRef pulls the skills come from, so both sides of the
    join share one provenance and one id.
    """
    if kind == "hitter":
        frame = index_by_mlbam(_raw(year, "batting"), "mlbID")
        pt = pd.to_numeric(frame["PA"], errors="coerce")
        lines = pd.DataFrame(
            {
                "r": pd.to_numeric(frame["R"], errors="coerce"),
                "hr": pd.to_numeric(frame["HR"], errors="coerce"),
                "rbi": pd.to_numeric(frame["RBI"], errors="coerce"),
                "sb": pd.to_numeric(frame["SB"], errors="coerce"),
                "ab": pd.to_numeric(frame["AB"], errors="coerce"),
                "avg": pd.to_numeric(frame["BA"], errors="coerce"),
            }
        ).fillna(0.0)
        lines["player_type"] = PlayerType.HITTER
    else:
        frame = index_by_mlbam(_raw(year, "pitching"), "mlbID")
        pt = frame["IP"].map(innings_to_float)
        lines = pd.DataFrame(
            {
                "w": pd.to_numeric(frame["W"], errors="coerce"),
                "k": pd.to_numeric(frame["SO"], errors="coerce"),
                "sv": pd.to_numeric(frame["SV"], errors="coerce"),
                "ip": pt,
                "era": pd.to_numeric(frame["ERA"], errors="coerce"),
                "whip": pd.to_numeric(frame["WHIP"], errors="coerce"),
            }
        ).fillna(0.0)
        lines["player_type"] = PlayerType.PITCHER

    out = pd.DataFrame(
        {
            "age": pd.to_numeric(frame["Age"], errors="coerce"),
            "pt": pt,
            "sgp": _sgp(lines, denoms),
        },
        index=frame.index,
    )
    # The raw rate the batted-ball family differences against its expected skill
    # (avg vs xba, era vs fip). Emitted under its real name; neither collides with a
    # skills column, so `_observed`'s join adds no `_sk` suffix. Re-derived from the
    # raw column rather than the fillna(0.0) `lines`: batted_ball must see NaN for a
    # blank rate so composite mean-fills it to neutral, not to a phantom 0.0 extreme.
    rate, raw_col = ("avg", "BA") if kind == "hitter" else ("era", "ERA")
    out[rate] = pd.to_numeric(frame[raw_col], errors="coerce")
    # The `speed` family's numerator, converted here because this is where `denoms`
    # lives; `composite.speed` divides it by PT. Hitters only -- there is no pitcher
    # analogue, and `FAMILIES["pitcher"]` omits the family.
    if kind == "hitter":
        out["sb_sgp"] = lines["sb"] / denoms[Category.SB]
    return out


# `(year, kind, denoms)` -> SGP of that ZiPS export. A multi-season run asks for the
# same export twice, because `year + 2` for one season is `year + 1` for the next, and
# `load_projection_set` parses BOTH pools' CSVs on every call. Memoizing takes
# `--scarcity` from 16 loads to 5. Not an `lru_cache`: `denoms` is a dict. `denoms` IS
# in the key because the cached SGP is scored through it -- omitting it hands a second
# call with different denominators (a comparison run, a test) the first call's series.
_OUT_YEAR_CACHE: dict[tuple[int, str, tuple[tuple[str, float], ...]], pd.Series] = {}


def _denoms_key(denoms) -> tuple[tuple[str, float], ...]:
    """A hashable, order-stable key component for a `denoms` dict."""
    return tuple(sorted((str(k), float(v)) for k, v in denoms.items()))


def zips_out_year_sgp(year: int, kind: str, denoms) -> pd.Series:
    """SGP of the ZiPS projection for `year`, indexed by mlbam_id.

    `load_projection_set` resolves the filename variants, strips the BOM, validates
    the required columns and renames to the lowercase stat-line keys
    `calculate_player_sgp` wants. It also keeps SV, which `keepers.vintages` drops
    -- the reason this does not route through there.

    A pitcher export with no saves at all is warned about rather than described in
    prose: the 2027 and 2028 files ship SV blank on every row where 2022-2026
    populate it, so a closer's `future_pct` is computed with sv=0 and understated.
    The check is per file, so it goes quiet by itself once a fresher export lands
    instead of leaving a stale caveat behind for someone to re-verify by hand.
    """
    cache_key = (year, str(kind), _denoms_key(denoms))
    cached = _OUT_YEAR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    directory = PROJECTIONS_DIR / str(year)
    if not directory.is_dir():
        return pd.Series(dtype=float)
    hitters, pitchers = load_projection_set(directory, "zips")
    frame = hitters if kind == "hitter" else pitchers
    if frame.empty:
        return pd.Series(dtype=float)
    lines = index_by_mlbam(frame, "mlbam_id")
    lines["player_type"] = PlayerType.HITTER if kind == "hitter" else PlayerType.PITCHER
    if kind == PlayerType.PITCHER and not (pd.to_numeric(lines["sv"], errors="coerce") > 0).any():
        print(
            f"  WARNING: the {year} ZiPS pitcher export carries no saves, so every"
            " closer's projected value is understated."
        )
    out = _sgp(lines, denoms)
    _OUT_YEAR_CACHE[cache_key] = out
    return out


def pricing_table() -> tuple[dict[str, list[str]], dict[str, float]]:
    """Positions and mean-centred credits -- neither varies by pool or by season.

    Hoisted out of `build` so one run makes one position lookup instead of two:
    it is the only network touch in this script.
    """
    return load_positions(), credit_levels()


def _observed(year: int, kind: str, denoms) -> pd.DataFrame:
    """Actual value, age, playing time and skills for one season and pool."""
    value = season_value(year, kind, denoms)
    skills = pd.read_csv(SKILLS_DIR / f"{kind}_skills_{year}.csv").set_index("mlbam_id")
    return value.join(skills, how="inner", rsuffix="_sk")


def _prior_pt_percentile(year: int, kind: str) -> pd.Series:
    """PT percentile in `year - 1`, the memory half of the `durability` family.

    Percentiled over EVERY player in that season, deliberately NOT over the
    `MIN_PT`-qualified pool: a veteran who took 100 PA while hurt has to score
    genuinely low here. Filtering him out would send him to `durability`'s
    missing-prior fallback and hand him a clean slate, which is the exact failure
    the family exists to prevent.

    An uncached prior season returns empty, which degrades the whole pool to
    current-season-only rather than raising -- the same fallback one player with no
    prior season gets. `ALL_TRANSITION_YEARS` needs raw_2021 onward for this to
    bind; without it the family is silently just `pt`.

    Reads playing time straight off the raw pull rather than going through
    `season_value`: only PT is wanted, and `season_value` would run a per-row SGP
    apply over a whole extra season per transition to produce columns nothing here
    reads. That also keeps this free of `denoms`, so it cannot fail on a partial
    denominator dict.
    """
    if not (SKILLS_DIR / f"raw_{year - 1}").exists():
        return pd.Series(dtype=float)
    table = "batting" if kind == "hitter" else "pitching"
    frame = index_by_mlbam(_raw(year - 1, table), "mlbID")
    pt = (
        pd.to_numeric(frame["PA"], errors="coerce")
        if kind == "hitter"
        else frame["IP"].map(innings_to_float)
    )
    return percentile(pt)


def _qualified_families(
    frame: pd.DataFrame, kind: str, prior_pt_pct: pd.Series | None = None
) -> pd.DataFrame:
    """Apply the playing-time floor and build the same-season families.

    Computes ALL candidate families (`skill`, `luck`, `pt`, `batted_ball`, `age`);
    `future` is left to the caller (the ranking uses out-years, the backtest a stale
    same-year projection). The active pool's `FAMILIES[kind]` selects which ones the
    composite actually blends, so the ranking and the backtest cannot drift into
    validating different feature definitions.
    """
    # An empty pool (empty actuals/skills join, or MIN_PT filtering everyone out early in
    # a season) flows through as an empty frame, not a raise: `--study` builds intentional
    # empty sub-pools it skips, and the live board renders empty. `composite` blends an
    # empty pool to an empty result; `_require_mandatory_families` no-ops on it.
    # Computed BEFORE the MIN_PT filter and kept on the same all-players base as
    # `_prior_pt_percentile`, so `durability` blends two commensurable percentiles.
    # `pt_pct` below is deliberately different: it ranks within the qualified pool,
    # which is what compresses an injury-shortened star to mid-pack and is the
    # reason `durability` exists as a separate family rather than a tweak to it.
    all_pt_pct = percentile(frame["pt"])
    qualified = frame[frame["pt"] >= MIN_PT[kind]].copy()
    qualified["value_pct"] = percentile(qualified["sgp"])
    qualified["skill_pct"] = skill_percentile(qualified, kind)
    qualified["luck_pct"] = luck(qualified["value_pct"], qualified["skill_pct"])
    qualified["pt_pct"] = percentile(qualified["pt"])
    # Keyed on the POOL, not on whether the column happens to be there: `speed` is a
    # shipped hitter family, so a hitter frame that cannot supply it has to fail loud
    # here rather than quietly omit `speed_pct` and let `composite` drop the family.
    # Pitchers have no analogue and `FAMILIES["pitcher"]` omits it.
    if kind == "hitter":
        qualified["speed_pct"] = percentile(speed(qualified))
    prior = pd.Series(dtype=float) if prior_pt_pct is None else prior_pt_pct
    qualified["durability_pct"] = durability(all_pt_pct.reindex(qualified.index), prior)
    # `batted_ball` returns avg-xba / fip-era, both signed higher = luckier already.
    qualified["batted_ball_pct"] = percentile(batted_ball(qualified, kind))
    qualified["age_pct"] = percentile(qualified["age"], higher_is_better=False)
    return qualified


def _slots_for(positions: dict[str, list[str]], name: str, kind: str) -> list[str]:
    """Eligible slots for pricing, constrained to the pool being scored.

    Yahoo lists Ohtani as UTIL, so in the PITCHER pool `calculate_var` would take
    the hitter branch and net his pitching projection against the UTIL floor. A
    row is only ever priced against its own pool's floors.
    """
    eligible = PITCHER_ELIGIBLE if kind == PlayerType.PITCHER else HITTER_ELIGIBLE
    slots = positions.get(normalize_name(str(name)), [])
    # An allowlist, not a denylist: a bench or IL token is not a position to price
    # against, and a slot added to the enum lands in the right pool by itself.
    return [slot for slot in slots if slot in eligible] or FALLBACK_POS[kind]


def composite_pct(
    frame: pd.DataFrame,
    kind: str,
    weights: tuple[float, ...] | None = None,
    *,
    family_order: tuple[str, ...] | None = None,
    strict: bool = False,
) -> pd.Series:
    """The composite, re-ranked to 0-1 -- the x-axis everything downstream uses.

    ONE definition on purpose. `projection`'s constants are fitted against this
    exact quantity by `--fit`, and `expected_sgp`/`sgp_sd` are then applied to it
    in `build`. Two independent spellings would let the slope and intercept keep
    being applied to a subtly different variable, moving every proj_sgp, sd,
    proj_var and p_keep with no test failing.

    The re-rank matters: `luck` is a difference centred on zero while the other
    families (including `batted_ball`, which is a percentile) span 0-1, so the raw
    weighted mean is not on a percentile scale. Ranking is order-preserving, so it
    changes only how the number reads.

    `strict` forwards to `composite`: the fit/backtest callers pass True so an
    all-NaN family fails loud instead of being silently dropped from a blend whose
    number is then persisted as constants or used to decide the shipped model.
    """
    # `composite` owns the family_order/weights co-supply guard so every caller is
    # covered; pass the RAW `family_order` (not the resolved `order`) so it sees the
    # live path's None rather than a defaulted set. `_family_columns` still needs the
    # resolved order to build the columns.
    order = family_order if family_order is not None else FAMILIES[kind]
    # Reject a typo'd family_order HERE, before `_family_columns` does `frame["{typo}_pct"]`
    # and raises an opaque pandas KeyError -- composite's own guard runs after that access.
    check_known_families(set(order))
    return percentile(
        composite(
            _family_columns(frame, order),
            kind,
            weights=weights,
            family_order=family_order,
            strict=strict,
        )
    )


def _family_columns(frame: pd.DataFrame, family_order: tuple[str, ...]) -> dict[str, pd.Series]:
    """The `{family: series}` mapping `composite` expects, for the active families."""
    return {family: frame[f"{family}_pct"] for family in family_order}


def _require_mandatory_families(
    qualified: pd.DataFrame, family_order: tuple[str, ...] | None, kind: str, year: int
) -> None:
    """Fail loud if any family the active blend uses arrives entirely NaN.

    The live board runs `composite` with strict=False so a genuinely ABSENT family (a
    scarcity board built without an out-year term) drops cleanly -- but a family that is
    present yet all-NaN is a data outage, not an intended omission, and would be SILENTLY
    DROPPED, shipping a valid-looking board ranked on the wrong weights. Any shipped
    family can reach here all-NaN: `avg`/`era`/`age` coerce to NaN on a malformed raw
    column, skills can come back present-but-empty, and `future_pct` reindexes to NaN
    when the ZiPS out-year files are absent -- so guard them all. `future` and
    `batted_ball` get source-pointing messages (their absence is the likeliest and its
    fix specific); the rest fall back to a generic one. `future` all-NaN ignores the
    out-year projection; `batted_ball` all-NaN reverts to the pre-#277 blend that
    over-sells lucky everyday bats (Rafaela, Otto Lopez).

    Check only families the active blend uses: a backtest candidate board may
    legitimately omit one (baseline/A carry no `batted_ball`).
    """
    # An EMPTY pool is not a family outage -- there are simply no players (an early-season
    # board, a `--study` truncation sub-pool). It flows through as an empty board, so
    # no-op here rather than misfire: `.isna().all()` is vacuously True on empty columns
    # and would otherwise raise about the first family on every empty pool.
    if qualified.empty:
        return
    order = family_order if family_order is not None else FAMILIES[kind]
    for family in order:
        if not qualified[f"{family}_pct"].isna().all():
            continue
        if family == "future":
            raise FileNotFoundError(
                f"no {kind} ZiPS out-year projection reached the pool for "
                f"{year + 1}/{year + 2}; `future` is all-NaN -- check the files exist "
                f"under {PROJECTIONS_DIR} and their mlbam ids match"
            )
        if family == "batted_ball":
            raise ValueError(
                f"no {kind} batted-ball inputs (avg/xba or fip/era) for {year}; the "
                "`batted_ball` claw-back cannot be computed -- check the Savant/skills pull"
            )
        raise ValueError(
            f"{kind} family `{family}` is entirely NaN for {year}; the board would "
            "silently drop it and rank on the wrong weights -- check the source pull"
        )


def projected(
    year: int,
    kind: str,
    denoms,
    *,
    family_order: tuple[str, ...] | None = None,
    weights: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Everything up to and including `proj_sgp`/`sd`, with NO positional term.

    Split out because `keepers.scarcity` measures its floors from `proj_sgp`, and
    `--scarcity` therefore has to score a board WITHOUT consuming the credits it is
    about to produce. Calling `build` there would be inert today -- pricing is
    applied after `proj_sgp` is set -- but a reader would have to prove that by
    hand, and a later edit could make it false. This way the circularity cannot
    exist. It also skips a per-row pricing loop that is ~90% of a warm build.

    `family_order`/`weights` override the shipped blend, for the backtest's candidate
    boards; pass BOTH or NEITHER (`composite` rejects one without the other), and the
    live path leaves them None.
    """
    qualified = _qualified_families(
        _observed(year, kind, denoms), kind, _prior_pt_percentile(year, kind)
    )

    # Rank the projections WITHIN the qualified pool, not within all ~1900 ZiPS
    # rows. Most of that file is minor leaguers, so ranking there would put every
    # established regular above the 90th percentile and say nothing.
    near = zips_out_year_sgp(year + 1, kind, denoms).reindex(qualified.index)
    far = zips_out_year_sgp(year + 2, kind, denoms).reindex(qualified.index)
    qualified["future_pct"] = future_percentile(near, far)
    # The live board's loud all-NaN check, raising a source-pointing message BEFORE
    # composite runs -- which is why composite_pct below stays strict=False here and is
    # not redundant with it: composite's own `strict` path is the generic guard the
    # offline fit/backtest callers use. Each path runs exactly one of the two.
    _require_mandatory_families(qualified, family_order, kind, year)

    qualified["composite"] = composite_pct(
        qualified, kind, weights=weights, family_order=family_order
    )

    # The composite is ordinal; this is what puts it on a value scale and lets
    # hitters and pitchers share one list. See `keepers.projection`.
    qualified["proj_sgp"] = expected_sgp(qualified["composite"], kind)
    qualified["sd"] = sgp_sd(qualified["composite"], kind)
    return qualified


def build(
    year: int,
    kind: str,
    denoms,
    keepers: dict[str, str],
    pricing: tuple[dict[str, list[str]], dict[str, float]] | None = None,
    *,
    family_order: tuple[str, ...] | None = None,
    weights: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """`projected` plus the positional adjustment, ranked and labelled.

    Like `projected`, `family_order`/`weights` are pass-both-or-neither -- `composite`
    rejects one without the other rather than silently blending mismatched defaults.
    """
    # Mean-centred credits: a display offset only, see `keepers.scarcity`. Resolved (and
    # guarded) BEFORE the expensive `projected` so a mis-shaped credits table fails fast.
    positions, floors = pricing_table() if pricing is None else pricing
    # build hands `calculate_var` no "ip"/role_ip: the credits ship a single "P", so
    # `_pitcher_floor_key` falls back to it and never routes by role. If the table ever
    # gains SP/RP keys that fallback stops and `role_from_ip(0.0)` routes EVERY starter --
    # a 200-IP ace included -- to the reliever floor, silently. Fail loud rather than
    # misprice: an SP/RP split requires build to also pass a full-season-equivalent IP
    # (see `keepers.scarcity`'s note and `sgp.var.calculate_var`). A raise beats a comment.
    if {"SP", "RP"} & set(floors):
        raise ValueError(
            "keeper credits contain SP/RP floor keys but `build` passes no role_ip; every "
            "starter would be priced at the RP floor -- pass a full-season-equivalent IP "
            "(see keepers.scarcity)."
        )

    qualified = projected(year, kind, denoms, family_order=family_order, weights=weights)
    priced = [
        calculate_var(
            pd.Series(
                {
                    "total_sgp": proj,
                    "positions": _slots_for(positions, name, kind),
                }
            ),
            floors,
            return_position=True,
        )
        for name, proj in zip(qualified["name"], qualified["proj_sgp"], strict=True)
    ]
    qualified["proj_var"] = [var for var, _ in priced]
    qualified["pos"] = [pos for _, pos in priced]
    # keeper_of is the ONE deliberate bare-name join left: config.keepers is a hand-authored
    # {name, team} list with no player_type, and this drives only the display "*" marker, never
    # scoring -- so a same-name collision over-marks cosmetically, it does not mis-price (#282).
    qualified["keeper_of"] = [keepers.get(normalize_name(str(n)), "") for n in qualified["name"]]

    ranked = qualified.sort_values("proj_var", ascending=False)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


# --- backtest -------------------------------------------------------------


def _transition(year: int, kind: str, denoms) -> pd.DataFrame:
    """Features observed in `year` against the SGP percentile realized in year+1."""

    feat = _qualified_families(
        _observed(year, kind, denoms), kind, _prior_pt_percentile(year, kind)
    )
    nxt = _observed(year + 1, kind, denoms)
    # The out-year analogue: a projection FOR `year` was built before `year`, so it
    # sits the same two seasons forward from its data as ZiPS 2027 does from 2026.
    # Using next year's projection here would flatter `future` badly -- a fresh
    # projection scores 0.67/0.52 alone against this stale one's 0.52/0.35.
    feat["future_pct"] = percentile(zips_out_year_sgp(year, kind, denoms).reindex(feat.index))
    # A player who does not appear next season scores 0 rather than dropping
    # out: vanishing is the outcome a keeper decision most wants to avoid.
    feat["target"] = percentile(nxt["sgp"]).reindex(feat.index).fillna(0.0)
    feat["target_sgp"] = nxt["sgp"].reindex(feat.index).fillna(0.0)
    feat["target_pt"] = percentile(nxt["pt"]).reindex(feat.index).fillna(0.0)
    next_rate = nxt["sgp"] / nxt["pt"].where(nxt["pt"] > 0)
    feat["target_rate"] = percentile(next_rate).reindex(feat.index).fillna(0.0)
    return feat.dropna(subset=["value_pct", "skill_pct", "age_pct"])


# Weight grids, coarse on purpose -- two fit seasons cannot resolve a finer step.
# The `mid` grid (luck / batted_ball) spans below zero so a shrunk-to-zero or
# negative weight is observable; the shipped 0.4 floor would hide it.
_GRID_MID = (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
_GRID_PT = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
_GRID_FUTURE = (0.0, 0.2, 0.4, 0.6, 0.8)
_GRID_AGE = (0.0, 0.15, 0.3, 0.45)
_FAMILY_GRID: dict[str, tuple[float, ...]] = {
    "skill": (1.0,),  # pinned; every other family is measured against it
    "pt": _GRID_PT,
    # Spans below zero like the other mid grids: a speed family that the fit wants
    # to shrink to nothing, or to charge for, has to be observable rather than
    # floored out of sight.
    "speed": _GRID_MID,
    "durability": _GRID_PT,
    "luck": _GRID_MID,
    "batted_ball": _GRID_MID,
    "future": _GRID_FUTURE,
    "age": _GRID_AGE,
}
# The parameterizations the bake-off compares, per pool. C keeps `luck` but adds a
# `batted_ball` term free to go negative, so the grid can claw back the batted-ball
# part `luck` over-includes without dropping luck's PT/role/SB signal.
CANDIDATES: dict[str, tuple[str, ...]] = {
    "baseline": ("skill", "luck", "future", "age"),
    "A: pt+luck": ("skill", "pt", "luck", "future", "age"),
    "B: pt+batted_ball": ("skill", "pt", "batted_ball", "future", "age"),
    "C: luck-batted_ball": ("skill", "luck", "batted_ball", "future", "age"),
    # D is what SHIPS (#288): no residual, every family measuring one named thing.
    # For pitchers `speed` drops out and D is literally B, which lost this holdout
    # in #277 -- kept side by side on purpose so the tie stays visible rather than
    # being re-discovered as a regression. See the composite docstring for why a
    # tie is the correct outcome and why D ships anyway.
    "D: direct": ("skill", "speed", "pt", "batted_ball", "future", "age"),
    # E swaps raw `pt` for `durability` -- same set, but the volume term gains a
    # season of memory. This is the only candidate carrying information the #277
    # bake-off never had, and the only place a predictive GAIN could come from.
    "E: durability (shipped)": (
        "skill",
        "speed",
        "durability",
        "batted_ball",
        "future",
        "age",
    ),
}
# `speed` has no pitcher analogue; the pool's D collapses to B rather than erroring
# on a family the frame cannot supply.
CANDIDATE_FAMILY_SKIP: dict[str, frozenset[str]] = {"pitcher": frozenset({"speed"})}
# Hitters the bake-off must move the right way: the lucky everyday bats should fall,
# the genuinely skilled everyday bat (Alvarez) should not.
WATCHLIST = ("Ceddanne Rafaela", "Otto Lopez", "Yordan Alvarez")


def _weighted_rho(
    frame: pd.DataFrame,
    weights: tuple[float, ...],
    kind: str,
    *,
    family_order: tuple[str, ...],
) -> float:
    # `family_order` is required, not defaulted: `weights` is grid-searched per family
    # set, so the two are inseparable, and `composite`'s co-supply guard raises on
    # weights-without-family_order anyway. `strict`: a fit/holdout rho computed with a
    # family silently dropped would decide the model on the wrong blend.
    blended = composite_pct(frame, kind, weights=weights, family_order=family_order, strict=True)
    return float(blended.corr(frame["target"], method="spearman"))


def _best_weights(
    fit: list[pd.DataFrame], family_order: tuple[str, ...], kind: str
) -> tuple[tuple[float, ...], list[float]]:
    """Grid-search the family weights maximizing mean fit rho; skill pinned at 1.0.

    Returns the best weight tuple and the per-fit-season rho at those weights (the
    latter feeds the noise floor).
    """
    axes = [_FAMILY_GRID[f] for f in family_order]
    best_weights: tuple[float, ...] | None = None
    best_per: list[float] = []
    best_mean = -2.0
    for weights in product(*axes):
        # Keep the winner's per-season rhos as the search goes, rather than re-running
        # every `_weighted_rho` (blend + re-rank + Spearman) for the winner afterward.
        per = [_weighted_rho(f, weights, kind, family_order=family_order) for f in fit]
        mean = sum(per) / len(per)
        if mean > best_mean:
            best_weights, best_mean, best_per = weights, mean, per
    # Not an assert: `python -O` strips those, and a None here (every combo's mean rho
    # NaN, so `mean > best_mean` never fires -- a degenerate/constant backtest panel)
    # would fall through into `_weighted_rho(..., weights=None)` and surface as
    # `composite`'s opaque co-supply error far from the real cause.
    if best_weights is None:
        raise ValueError(
            f"no {kind} weight combination produced a finite fit rho; "
            "the backtest panel is degenerate (all correlations NaN)"
        )
    return best_weights, best_per


def _watchlist_moves(year: int, denoms, best: dict[str, tuple[float, ...]]) -> None:
    """Print each watchlist hitter's board rank under baseline vs each candidate."""
    pricing = pricing_table()
    ranks: dict[str, dict[str, int]] = {}
    for label in CANDIDATES:
        board = build(
            year,
            "hitter",
            denoms,
            {},
            pricing=pricing,
            family_order=CANDIDATES[label],
            weights=best[label],
        )
        # setdefault keeps the FIRST (higher proj_var) of any two board hitters that
        # normalize to the same name -- the board is sorted proj_var-descending, so
        # this tie-breaks by VAR like the old `.iloc[0]` did, not last-write-wins.
        rank_by_name: dict[str, int] = {}
        for n, r in zip(board["name"], board["rank"], strict=True):
            rank_by_name.setdefault(normalize_name(str(n)), int(r))
        ranks[label] = rank_by_name
    print(f"\n  hitter watchlist ranks on the {year} board (lower = better):")
    print("    " + f"{'player':<18}" + "".join(f"{label:>20}" for label in CANDIDATES))
    for name in WATCHLIST:
        target = normalize_name(name)
        # "n/a", not a number: a watchlist hitter absent from the board (below the PT
        # floor, or a name mismatch) must not print as a rank under "lower = better".
        cells = [ranks[label].get(target, "n/a") for label in CANDIDATES]
        print(f"    {name:<18}" + "".join(f"{c!s:>20}" for c in cells))


def run_backtest(denoms) -> None:
    best_by_pool: dict[str, dict[str, tuple[float, ...]]] = {}
    for kind in POOLS:
        fit = [_transition(y, kind, denoms) for y in BACKTEST_FIT_YEARS]
        hold = _transition(BACKTEST_HOLDOUT, kind, denoms)
        print(
            f"\n{'=' * 70}\n{kind.upper()}  fit={list(BACKTEST_FIT_YEARS)} holdout={BACKTEST_HOLDOUT}"
        )
        print(f"  {'candidate':<20}{'holdout':>9}{'fit':>9}{'noise':>9}  best weights")
        best_by_pool[kind] = {}
        skip = CANDIDATE_FAMILY_SKIP.get(kind, frozenset())
        seen: set[tuple[str, ...]] = set()
        for label, requested in CANDIDATES.items():
            family_order = tuple(f for f in requested if f not in skip)
            # Dropping `speed` makes pitcher-D identical to B. Say so in the table
            # instead of grid-searching the same family set twice -- the identity is
            # the point (#277 already held this holdout against it) and a silently
            # duplicated row would read as two independent results.
            if family_order in seen:
                same = next(
                    k
                    for k, v in CANDIDATES.items()
                    if tuple(f for f in v if f not in skip) == family_order and k != label
                )
                print(f"  {label:<20}{'':>27}  == {same} here ({'/'.join(sorted(skip))} n/a)")
                continue
            seen.add(family_order)
            weights, per_season = _best_weights(fit, family_order, kind)
            best_by_pool[kind][label] = weights
            holdout = _weighted_rho(hold, weights, kind, family_order=family_order)
            fit_rho = sum(per_season) / len(per_season)
            noise = max(per_season) - min(per_season)  # in-sample rho spread
            shown = " ".join(f"{f}={w:+.2f}" for f, w in zip(family_order, weights, strict=True))
            print(f"  {label:<20}{holdout:>9.4f}{fit_rho:>9.4f}{noise:>9.4f}  {shown}")
        # The currently-shipped model at its shipped weights, for reference against
        # the candidate rows above. The `baseline` candidate row reproduces the
        # pre-change number (0.7085 hitters / 0.4962 pitchers), which is the
        # generalization's no-behaviour-change check.
        shipped = _weighted_rho(hold, FITTED_WEIGHTS[kind], kind, family_order=FAMILIES[kind])
        print(f"  {'shipped':<20}{shipped:>9.4f}   (current model)")
        # The pure-null floor: skill percentile alone, no volume/luck/future/age term.
        # A candidate that fails to clear this by more than the pool's noise is buying
        # nothing the peripherals do not already say.
        skill_only = _weighted_rho(hold, (1.0,), kind, family_order=("skill",))
        print(f"  {'skill only':<20}{skill_only:>9.4f}   (null floor)")
    # The watchlist is a hitter question; build 2026 boards under each candidate's
    # best hitter weights.
    _watchlist_moves(2026, denoms, best_by_pool["hitter"])
    print(
        "\n  Rho cannot separate these: the candidates span less than the fit-season"
        "\n  noise in both pools, though all of them clear the skill-only null floor."
        "\n  D (no residual; every family measures one named thing) is what SHIPS, and"
        "\n  it TIED rather than won -- for pitchers it is literally B, which lost this"
        "\n  holdout in #277. That is the correct outcome by construction: `pt` is what"
        "\n  the `luck` residual was already proxying, so D re-expresses the same"
        "\n  information under honest names. It ships on interpretability and on having"
        "\n  the lowest fit-season noise, not on rho -- see #288 and the composite"
        "\n  docstring before re-litigating from the holdout column alone. A predictive"
        "\n  gain has to come from a durability estimator that beats raw `pt`."
        "\n  Weights are applied by hand to `composite.FITTED_WEIGHTS`."
    )


def _kv_payload(key: CacheKey):
    """Unwrapped value for one cache key, or None if the KV is unreachable.

    Import-local and failure-tolerant, like `keepers.positions`: everything else in
    this script runs offline and only the roster reports need the network. Keyed
    through `CacheKey` rather than a "cache:..." literal, which is what that enum
    exists to prevent.
    """
    try:
        from fantasy_baseball.data.kv_store import get_kv

        blob = get_kv().get(redis_key(key))
        if not blob:
            return None
        # Inside the try on purpose: a corrupt blob (truncated / invalid JSON) must
        # degrade to None like an unreachable KV, not crash the whole report with a
        # JSONDecodeError -- the caller only knows how to handle "no data".
        payload = json.loads(blob) if isinstance(blob, str) else blob
    except Exception:
        return None
    if isinstance(payload, dict) and "_data" in payload:
        return payload["_data"]
    return payload


def _match_key(name, player_type) -> tuple[str, str]:
    """The `(normalized_name, str(player_type))` join key, built ONE way for both sides of
    the roster<->board match so a roster entry and a board row can never spell it
    differently. Not `make_player_key`: this normalizes (accent-folds, lowercases) to join
    two data sources, whereas that key uses the raw display name. See #282.
    """
    return (normalize_name(str(name)), str(player_type))


def _roster_entries(entries) -> list[tuple[str, str]]:
    """`_match_key` per roster entry, DUPLICATES preserved.

    A bare normalized name is ambiguous: Yahoo splits a two-way player (Ohtani) into a
    hitter and a pitcher entry that can sit on different teams, and 2022 alone had a
    hitter and a pitcher both named `Luis Garcia`. Roster blobs carry `player_type`
    ('hitter'/'pitcher') but no mlbam_id, so `(name, player_type)` is the only key that
    joins to the board's `(name, kind)` -- it resolves every cross-TYPE collision. It is a
    LIST, not a set: two same-name SAME-type spots are distinct roster slots, and
    collapsing them to one made `unscored` go negative. Residual limit (#282): two
    same-name same-type players on DIFFERENT teams still cannot be told apart -- the board
    and roster share no unique id -- but that is rarer than the cross-type case this fixes.
    """
    if not isinstance(entries, list):
        return []
    return [_match_key(e["name"], e.get("player_type", "")) for e in entries if e.get("name")]


def load_roster_keys() -> list[tuple[str, str]]:
    """`(normalized_name, player_type)` entries on my roster, from the live KV blob."""
    return _roster_entries(_kv_payload(CacheKey.ROSTER))


def load_league_rosters(my_team: str) -> dict[str, list[tuple[str, str]]]:
    """`(normalized_name, player_type)` entries per team for the WHOLE league.

    `cache:roster` holds only my team and `cache:opp_rosters` only the other nine, so the
    league is the union. Same import-local, failure-tolerant shape as `load_roster_keys`:
    everything else in this script runs offline.
    """
    opponents = _kv_payload(CacheKey.OPP_ROSTERS)
    rosters: dict[str, list[tuple[str, str]]] = {}
    if isinstance(opponents, dict):
        for team, players in opponents.items():
            keys = _roster_entries(players)
            if keys:
                rosters[str(team)] = keys
    mine = load_roster_keys()
    if mine:
        rosters[my_team] = mine
    return rosters


def _board_row_keys(board: pd.DataFrame) -> list[tuple[str, str]]:
    """`_match_key` per board row -- the key that joins a board row (indexed by mlbam_id,
    carrying a `name` and a `kind`) to a roster entry, since the roster blobs carry
    `player_type` but no mlbam_id. The board side of the same `_match_key` `_roster_entries`
    builds; used for ownership, roster filtering, and the not-scored set. See #282.
    """
    return [_match_key(n, k) for n, k in zip(board["name"], board["kind"], strict=True)]


def _score_roster(frame: pd.DataFrame, slots: int) -> pd.DataFrame:
    """Dedupe a roster's two-way players to their better side, THEN attach P(keep) over that
    roster. The dedupe-before-score ordering is the core of #282: a two-way player must
    collapse to one keeper candidate within a roster before `probability_top_n` spreads the
    `slots` mass over it. Both `--league` (per team) and `--roster` (my whole roster) go
    through here so the two paths cannot drift. `_dedupe_two_way` returns a fresh frame, so
    the caller needs no defensive copy.
    """
    part = _dedupe_two_way(frame)
    part["p_keep"] = probability_top_n(part["proj_var"], part["sd"], part["kind"], slots)
    return part


def _below_floor(roster_keys: list[tuple[str, str]], board_keys: list[tuple[str, str]]) -> Counter:
    """Roster entries that did NOT clear MIN_PT, per (name, type), never negative.

    A roster entry is below the floor iff its (name, type) has no row among `board_keys` (the
    roster's board rows taken BEFORE the two-way dedupe) AND that player is not already on the
    board under his OTHER type. The two clauses together cover the two-way player Yahoo splits
    into a separate hitter and pitcher entry:
      - both sides clear the floor: both keys are on the board, so neither is below it, and the
        later mlbam dedupe collapses them to one keeper candidate;
      - only one side clears it (he bats but does not pitch this year): the other side has no
        board row, but he IS kept via the side that did, so the cross-type clause drops it
        rather than brand a scored keeper as "not competing".
    Only a CROSS-type board row suppresses. Two same-(name, type) players -- two 'Luis Garcia'
    pitchers, one below the floor -- are distinct people sharing a normalized key, so the
    below-floor one is still counted (his name is on the board only under his OWN type).

    `.total()` is the single count both `--roster` (its NOT-SCORED set) and `--league` (its
    per-team count) read, so the two agree about a roster, and it can never go negative.

    The cross-type clause is a NAME PROXY, not proof of identity: with no shared id it assumes a
    name on the board under a different type is the same two-way player. Two DIFFERENT people
    sharing a normalized name across types (a hitter and an unrelated below-floor pitcher both
    named 'Will Smith', both mine) would defeat it, dropping the below-floor one silently. That,
    and two same-(name, type) players on DIFFERENT teams, are the irreducible #282 limit -- both
    astronomically rarer than the two-way case this serves, and neither changes a scored P(keep);
    both reports print collision WARNINGs where an ambiguity DOES move a scored number.
    """
    # Counter's binary `-` IS the positive multiset difference: subtract, then drop every
    # non-positive count -- a key the board carries at least as often as the roster holds it
    # falls out (fully covered), and nothing zero or negative survives.
    raw = Counter(roster_keys) - Counter(board_keys)
    on_board: dict[str, set[str]] = {}
    for name, ptype in board_keys:
        on_board.setdefault(name, set()).add(ptype)
    # Keep a below-floor key only when the board carries that name under NO other type: an empty
    # set (a genuine below-floor player) or just his own type (a same-type namesake) both pass;
    # a DIFFERENT type on the board means he is a two-way player kept via his other side -- drop.
    return Counter(
        {(name, t): n for (name, t), n in raw.items() if on_board.get(name, set()) <= {t}}
    )


def _fmt_keys(keys: list[tuple[str, str]]) -> str:
    """`(name, type)` keys as a human list -- "name (type), name (type)" -- shared by the
    report NOT-SCORED line and both collision WARNINGs, which all print the normalized key the
    rosters carry (not a display name), so the three read identically. Pass them pre-ordered.
    """
    return ", ".join(f"{name} ({ptype})" for name, ptype in keys)


def run_fit(denoms) -> None:
    """Refit and print `keepers.projection`'s constants, paste-ready.

    Same features and transitions as `--backtest`, but regressing realized SGP in
    LEVELS rather than correlating percentiles -- that is what puts the ordinal
    composite on a value scale. Without this the value half of a two-stage model
    would be unreproducible while the ordinal half is a flag away.
    """
    seasons = ALL_TRANSITION_YEARS
    print(f"# Refitted over {', '.join(f'{y}->{y + 1}' for y in seasons)}")
    fit, sd_fit, quantiles = {}, {}, {}
    for kind in POOLS:
        frames = []
        for year in seasons:
            feat = _transition(year, kind, denoms)
            frames.append(
                pd.DataFrame(
                    {
                        # strict: these constants are pasted into projection.py, so a
                        # family silently dropped for a bad transition year would be
                        # baked in permanently rather than noticed.
                        "c": composite_pct(feat, kind, strict=True),
                        "sgp": feat["target_sgp"],
                    }
                )
            )
        panel = pd.concat(frames, ignore_index=True)
        x, y = panel["c"].to_numpy(), panel["sgp"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (intercept + slope * x)
        # SD as a function of the composite, via E|residual| scaled for a normal.
        sd_slope, sd_intercept = np.polyfit(x, np.abs(residual), 1)
        scale = math.sqrt(math.pi / 2)
        fit[kind] = (round(float(intercept), 3), round(float(slope), 3))
        sd_fit[kind] = (round(scale * float(sd_intercept), 3), round(scale * float(sd_slope), 3))
        standardized = residual / (scale * (sd_intercept + sd_slope * x))
        quantiles[kind] = np.percentile(
            standardized, [level * 100 for level in RESIDUAL_QUANTILE_LEVELS]
        )
        r2 = 1.0 - residual.var() / y.var()
        print(f"#   {kind}: n={len(panel)} R2={r2:.3f}")
    for name, table in (("SGP_FIT", fit), ("SGP_SD_FIT", sd_fit)):
        body = ", ".join(f'"{k}": {v}' for k, v in table.items())
        print(f"{name} = {{{body}}}")
    print("STD_RESIDUAL_QUANTILES = {")
    for kind, values in quantiles.items():
        print(f'    "{kind}": (' + ", ".join(f"{v:.3f}" for v in values) + "),")
    print("}")


def _rho(left: pd.Series, right: pd.Series) -> float:
    return float(left.corr(right, method="spearman"))


def _mean_rho(frames: list[pd.DataFrame], left: str, right: str) -> float:
    return sum(_rho(f[left], f[right]) for f in frames) / len(frames)


SKILL_VARIANTS: dict[str, dict[str, tuple[str, ...]]] = {
    PlayerType.HITTER: {
        "all 5 (shipped)": HITTER_SKILLS,
        "drop wrc_plus": tuple(c for c in HITTER_SKILLS if c != "wrc_plus"),
    },
    PlayerType.PITCHER: {
        "all 6 (shipped)": PITCHER_SKILLS,
        "drop era_minus": tuple(c for c in PITCHER_SKILLS if c != "era_minus"),
        "drop era- and fip": tuple(c for c in PITCHER_SKILLS if c not in ("era_minus", "fip")),
    },
}


def _variant_percentile(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """`skill_percentile` over an arbitrary subset, for the ablation below."""
    parts = [percentile(frame[col], higher_is_better=col not in LOWER_IS_BETTER) for col in columns]
    return pd.concat(parts, axis=1).mean(axis=1)


def _truncation_shift(kind: str, denoms, year: int, pool_size: int) -> pd.DataFrame:
    """What running mid-season does to a COMPLETE season's own numbers.

    Simulates the truncation by keeping only the `pool_size` highest-playing-time
    rows and rebuilding every family from scratch -- which is what a partial season
    actually does, since each family is a percentile computed WITHIN the pool that
    cleared MIN_PT. Regenerates the MID-SEASON CAVEAT at the top of this module.
    """
    observed = _observed(year, kind, denoms)
    out_year = zips_out_year_sgp(year, kind, denoms)
    prior = _prior_pt_percentile(year, kind)

    def priced(frame: pd.DataFrame) -> pd.Series:
        feat = _qualified_families(frame, kind, prior)
        feat["future_pct"] = percentile(out_year.reindex(feat.index))
        feat = feat.dropna(subset=["value_pct", "skill_pct", "age_pct"])
        return expected_sgp(composite_pct(feat, kind, strict=True), kind)

    both = pd.DataFrame(
        {"full": priced(observed), "small": priced(observed.nlargest(pool_size, "pt"))}
    ).dropna()
    both["delta"] = both["small"] - both["full"]
    return both


def _print_truncation(kind: str, denoms, pool_size: int) -> None:
    print("")
    print(f"  truncation to the live pool size ({pool_size} rows), on COMPLETE seasons:")
    print("   year   n  rho    mean    Q1     Q2     Q3     Q4     Q5   top10")
    for year in ALL_TRANSITION_YEARS:
        shift = _truncation_shift(kind, denoms, year, pool_size)
        if shift.empty:
            continue
        quintiles = shift.groupby(pd.qcut(shift["full"], 5, labels=False, duplicates="drop"))
        cells = "".join(f"{value:>7.2f}" for value in quintiles["delta"].mean())
        top10 = shift.nlargest(10, "full")["delta"].mean()
        print(
            f"   {year} {len(shift):>3} {_rho(shift['full'], shift['small']):.3f}"
            f"{shift['delta'].mean():>7.2f}{cells}{top10:>7.2f}"
        )
    print(
        "    -> levels come out LOW, and much more so mid-board than at the top, so"
        " GAPS are distorted too; rho shows order is only NEAR-invariant."
    )


# Extends a season past `ALL_TRANSITION_YEARS` on purpose: measuring a floor
# needs only that season's own board, not a following season to score against,
# so 2025 is usable here where the backtest cannot use it.
SCARCITY_YEARS = (2022, 2023, 2024, 2025)


def _season_eligibility(year: int, board_index, kind: str) -> dict[object, set[str]]:
    """Per-season eligibility keyed by the board's MLBAM index.

    Pitchers are `{P}` (the pool has one slot). A hitter with no >= 10-game hitter
    slot is priced at `{DH}` -- UTIL-only via `can_fill_slot`, matching the old
    `FALLBACK_POS` intent but derived per season instead of from the 2026 Yahoo map.
    `HITTER_ELIGIBLE` drops the pitcher slot a two-way player's line carries.
    """
    if kind == PlayerType.PITCHER:
        return {idx: {"P"} for idx in board_index}
    derived = season_eligibility(fetch_mlb_season(SKILLS_DIR, year, "fielding"))
    return {
        idx: {s for s in derived.get(int(idx), set()) if s in HITTER_ELIGIBLE} or {"DH"}
        for idx in board_index
    }


def run_scarcity(denoms) -> None:
    """Re-measure the positional credits and print them paste-ready.

    The floors `keepers.scarcity` ships are measured, not assumed, so this is what
    regenerates them. Each season is scored against its OWN >= 10-game eligibility
    (`keepers.appearances`), not the current Yahoo map. It prints the per-season
    centred credit alongside the average because the single-season spread is large,
    and an average of four seasons is the only defensible summary.
    """
    config = load_config(CONFIG_PATH)
    capacities = slot_capacities(config.roster_slots, config.num_teams)
    print(f"  league starting slots (bench and IL excluded): {capacities}")
    # Which slots belong to which pool is a property of the league, not the season.
    wanted = {
        kind: {
            slot: n
            for slot, n in capacities.items()
            if (slot in PITCHER_ELIGIBLE) == (kind == PlayerType.PITCHER)
        }
        for kind in POOLS
    }
    # The derived eligibility is fielding-based and records pitchers only as "P" -- it
    # cannot tell SP from RP (a role, not a fielding position). If the league ever
    # splits P into SP/RP slots, `_season_eligibility` would seat no pitcher in them
    # and the pitcher credit would silently vanish, so fail loudly instead of shipping
    # a table with a hole. A role split needs an IP-based source; see `keepers.scarcity`.
    split = set(wanted[PlayerType.PITCHER]) - {"P"}
    if split:
        raise ValueError(
            f"pitcher slots {sorted(split)} need a role-based (IP) eligibility source; "
            "the fielding-derived map yields only 'P'."
        )

    per_season = []
    coverage: list[tuple[int, int, int]] = []  # (year, hitter_rows, dh_fallback)
    for year in SCARCITY_YEARS:
        floors: dict[str, float] = {}
        for kind in POOLS:
            # Each season is scored against ITS OWN eligibility (>= 10 games at a
            # position that season), not the current Yahoo map, which has no history
            # and routed older seasons' gone-by-now players to UTIL.
            board = projected(year, kind, denoms)
            eligible = _season_eligibility(year, board.index, kind)
            floors.update(marginal_starter_floors(board["proj_sgp"], eligible, wanted[kind]))
            if kind == PlayerType.HITTER:
                dh = sum(1 for slots in eligible.values() if slots == {"DH"})
                coverage.append((year, len(board), dh))
        per_season.append((year, floors))

    seasons = pd.DataFrame({year: floors for year, floors in per_season}).T
    slots = sorted(seasons.columns)
    print("\n  measured floor per season (on this model's proj_sgp scale):")
    print("    year  " + "".join(f"{slot:>8}" for slot in slots))
    for year, floors in per_season:
        print(
            f"    {year}  " + "".join(f"{floors.get(slot, float('nan')):>8.2f}" for slot in slots)
        )

    print("\n  hitter map coverage per season (rows priced at a real position vs DH):")
    print("    year   rows   real    DH")
    for year, rows, dh in coverage:
        print(f"    {year}  {rows:>5}  {rows - dh:>5}  {dh:>4}")

    # The Yahoo map was complete by construction; a derived fielding pull can be
    # truncated or stale (an interrupted cache write, a mid-season partial), which
    # thins every dedicated slot's leftover pool and silently skews the credits. Real
    # seasons fall back to DH for ~2-3% of the board, so an implausibly high share is a
    # bad cache -- refuse to emit credits from it rather than let it be pasted.
    thin = [(year, dh / rows) for year, rows, dh in coverage if rows and dh / rows > 0.25]
    if thin:
        detail = ", ".join(f"{year}: {share:.0%} DH" for year, share in thin)
        raise ValueError(
            f"fielding coverage is implausibly thin ({detail}); the cached fielding pull "
            "is likely truncated or stale. Refetch before trusting these credits."
        )

    print("\n  centred credit per season (the year-to-year spread, as a flag not prose):")
    print("    year  " + "".join(f"{slot:>8}" for slot in slots))
    for year, floors in per_season:
        credits = centred_credits(floors)
        print(f"    {year}  " + "".join(f"{credits.get(s, float('nan')):>8.2f}" for s in slots))

    mean_floor = seasons.mean().to_dict()
    fresh = centred_credits(mean_floor)
    order = sorted(fresh, key=lambda slot: -fresh[slot])
    print("\n  averaged, centred, against what is shipped:")
    print(f"    {'slot':<7}{'floor':>8}{'credit':>9}{'shipped':>10}{'delta':>8}")
    for slot in order:
        shipped = NATIVE_CREDITS.get(slot, float("nan"))
        print(
            f"    {slot:<7}{mean_floor[slot]:>8.2f}{fresh[slot]:>9.2f}"
            f"{shipped:>10.2f}{fresh[slot] - shipped:>8.2f}"
        )
    _validate_against_yahoo(denoms)

    print("\n  paste into keepers/scarcity.py:")
    print("NATIVE_CREDITS: dict[str, float] = {")
    for slot in order:
        print(f'    "{slot}": {fresh[slot]:.3f},')
    print("}")


def _validate_against_yahoo(denoms) -> None:
    """Sanity-check the 10-game rule against the real Yahoo map, on 2025 (a COMPLETE
    season -- what Yahoo's 2026 map is largely built from; 2026 is half-played, so a
    derived-2026 map would spuriously under-match). Bridges the derived map (MLBAM) to
    Yahoo (name) through the 2025 board's own clean name column."""
    yahoo, _ = pricing_table()  # normalized-name -> Yahoo slots
    derived = season_eligibility(fetch_mlb_season(SKILLS_DIR, 2025, "fielding"))
    board = projected(2025, PlayerType.HITTER, denoms)
    slots = ("C", "1B", "2B", "3B", "SS", "OF")
    hit = {s: [0, 0] for s in slots}  # [derived-agrees, yahoo-has]
    for idx, name in zip(board.index, board["name"], strict=True):
        y = set(yahoo.get(normalize_name(str(name)), []))
        d = derived.get(int(idx), set())
        for s in slots:
            if s in y:
                hit[s][1] += 1
                if s in d:
                    hit[s][0] += 1
    print("\n  derived-2025 vs Yahoo-2026 agreement (share of Yahoo's players recovered):")
    for s in slots:
        agree, total = hit[s]
        pct = 100.0 * agree / total if total else float("nan")
        print(f"    {s:<4} {agree:>4}/{total:<4} = {pct:5.1f}%")


def _print_cross_pool(denoms) -> None:
    """Why no pitcher reaches the top of a mixed board.

    Regenerates the CROSS-POOL CAVEAT at the top of this module. The claim it makes
    is that the hitter/pitcher gap is a difference in PREDICTABILITY rather than a
    scale artifact, and that is only checkable against realized outcomes.
    """
    print("")
    print("  cross-pool: what a TOP-DECILE composite went on to earn, by pool")
    print(f"    {'pool':<9}{'n':>5}{'mean':>8}{'sd':>7}{'P(~0)':>8}{'composite 1.0 ->':>19}")
    for kind in POOLS:
        realized = pd.concat(
            [
                frame.loc[composite_pct(frame, kind, strict=True) >= 0.9, "target_sgp"]
                for frame in (_transition(year, kind, denoms) for year in ALL_TRANSITION_YEARS)
            ]
        )
        ceiling = float(expected_sgp(pd.Series([1.0]), kind).iloc[0])
        print(
            f"    {kind:<9}{len(realized):>5}{realized.mean():>8.2f}{realized.std():>7.2f}"
            f"{(realized <= 0.001).mean() * 100:>7.1f}%{ceiling:>19.2f}"
        )
    print(
        "    -> the pitcher shortfall is in REALIZED value, not just in the fitted"
        " ceiling, so a mixed board is expected value and nothing more."
    )


def run_study(denoms, live_year: int) -> None:
    """Print the diagnostics the module docstrings argue from.

    Every claim in `keepers/composite.py` about WHY the families are shaped the way
    they are is reproduced here, so none of them can drift into being wrong without
    this command disagreeing. Same for this module's own two caveats: the mid-season
    one and the cross-pool one.
    """
    _print_cross_pool(denoms)
    seasons = ALL_TRANSITION_YEARS
    for kind in POOLS:
        frames = [_transition(year, kind, denoms) for year in seasons]
        header = "=" * 66
        print("")
        print(header)
        print(f"{kind.upper()}  ({len(frames)} transitions)")

        # Measured, not assumed: the caveat is about THIS run's pool.
        live = _observed(live_year, kind, denoms)
        _print_truncation(kind, denoms, int((live["pt"] >= MIN_PT[kind]).sum()))

        print("  what predicts next season, split into volume and rate:")
        print(f"    {'predictor':<16}{'-> SGP':>9}{'-> PT':>9}{'-> RATE':>9}")
        for label, column in (
            ("last-yr value", "value_pct"),
            ("skills", "skill_pct"),
            ("speed", "speed_pct"),
            ("luck (dropped)", "luck_pct"),
            ("playing time", "pt_pct"),
            ("batted-ball", "batted_ball_pct"),
            ("age (younger)", "age_pct"),
            ("future (stale)", "future_pct"),
        ):
            if not any(column in f.columns for f in frames):
                continue  # `speed` in the pitcher pool
            cells = [
                _mean_rho(frames, column, target)
                for target in ("target", "target_pt", "target_rate")
            ]
            print(f"    {label:<16}" + "".join(f"{c:>9.3f}" for c in cells))

        # The #288 argument-of-record: `luck` was a residual whose largest component
        # was volume, not luck. Printed rather than asserted so the composite
        # docstring's table cannot drift away from the data.
        print("  what the DROPPED `luck` residual was actually made of:")
        for label, column in (
            ("playing time", "pt_pct"),
            ("speed", "speed_pct"),
            ("batted-ball", "batted_ball_pct"),
            ("age (younger)", "age_pct"),
        ):
            if not any(column in f.columns for f in frames):
                continue
            print(f"    {label:<16}{_mean_rho(frames, column, 'luck_pct'):>9.3f}")

        # Kept after #288 dropped the family: this is WHY the residual read as
        # signal, and the next block shows `pt` doing the same job under its own
        # name. Delete either and the docstring's argument stops being checkable.
        print("  the dropped `luck` wanted a POSITIVE weight (skill + w*luck -> next SGP):")
        for weight in (-1.0, -0.5, 0.0, 0.5, 1.0):
            scores = [_rho(f["skill_pct"] + weight * f["luck_pct"], f["target"]) for f in frames]
            print(f"    w={weight:>5.1f}  rho={sum(scores) / len(scores):+.4f}")

        print("  ...and `pt` earns it under its own name (skill + w*pt -> next SGP):")
        for weight in (-1.0, -0.5, 0.0, 0.5, 1.0):
            scores = [_rho(f["skill_pct"] + weight * f["pt_pct"], f["target"]) for f in frames]
            print(f"    w={weight:>5.1f}  rho={sum(scores) / len(scores):+.4f}")

        near = zips_out_year_sgp(2027, kind, denoms)
        far = zips_out_year_sgp(2028, kind, denoms).reindex(near.index)
        if near.notna().any() and far.notna().any():
            blend = FUTURE_BLEND[0] * near + FUTURE_BLEND[1] * far.fillna(near)
            print(
                f"  out-year 2027 vs 2028 rho = {_rho(near, far):+.3f}; "
                f"blend vs 2027 alone = {_rho(blend, near):+.3f}"
                "  <- why the second year adds little"
            )

        print("  a FRESH next-season projection vs the stale out-year analogue:")
        for label, offset in (("stale (T, shipped)", 0), ("fresh (T+1)", 1)):
            scores = []
            for year, frame in zip(seasons, frames, strict=True):
                projected = zips_out_year_sgp(year + offset, kind, denoms).reindex(frame.index)
                scores.append(_rho(percentile(projected).fillna(0.0), frame["target"]))
            print(f"    {label:<20} rho = {sum(scores) / len(scores):+.4f}")

        print("  skill-family ablation (-> next-year RATE), why the impure inputs stay:")
        for label, columns in SKILL_VARIANTS[kind].items():
            scores = [_rho(_variant_percentile(f, columns), f["target_rate"]) for f in frames]
            print(f"    {label:<20} rho = {sum(scores) / len(scores):+.4f}")


def _dedupe_two_way(board: pd.DataFrame) -> pd.DataFrame:
    """Collapse a two-way player's two pool rows into his better one.

    He qualifies in BOTH pools, so without this he appears twice, draws
    independent outcomes and competes against himself for the keeper slots --
    Ohtani absorbed 0.33 of the 3.00 slot mass for one roster spot.

    Keyed on MLBAM id, which is the frame's index, NOT on name. The index name is
    asserted rather than assumed: adding a `reset_index` anywhere upstream would
    make every label unique, silently restore the double-count, and leave the
    tests green, since a synthetic index passes them just as well. 2022 alone had two
    different Will Smiths and two different Diego Castillos across the pools plus
    two different Luis Garcias inside one, and a name-keyed drop deletes a real
    rival: `probability_top_n` then spreads the same slot mass over fewer people,
    inflating everyone's P KEEP while the sum-to-slots check still passes. Expects
    `board` already sorted best-first, so `keep="first"` keeps the better side.
    """
    if board.index.name != "mlbam_id":
        raise ValueError(
            f"expected an mlbam_id index to dedupe on, got {board.index.name!r}; "
            "a reset_index upstream would silently un-fix the two-way double-count"
        )
    return board[~board.index.duplicated(keep="first")].reset_index(drop=True)


def _fail_if_empty_board(board: pd.DataFrame, year: int, pools: tuple[str, ...] = ()) -> bool:
    """Print why and return True when a LIVE keeper board is empty or missing a pool.

    The shared math (`composite`, `build`, `_qualified_families`) tolerates an empty pool
    so the diagnostics can build intentional empty sub-pools -- but an EMPTY live board is
    either a broken actuals/skills join (a 0-row inner join on an mlbam id mismatch) or a
    season too early for anyone to clear MIN_PT. Neither is a board to act on, so the
    live-board commands fail rather than emit a silent empty CSV/report. (Base `feat/273`
    failed loud here via a vacuous guard; #277's empty-tolerance would otherwise mask it.)

    `pools` guards the COMBINED roster/league board (which carries a `kind` column): if any
    expected pool contributed ZERO rows it also fails, because a one-pool join break (drifted
    mlbam ids on just the pitcher side, say) merges a full board with an empty one, and the
    whole-board `.empty` check above would pass while the report SILENTLY omits an entire
    pool. Unlike the too-early cause, a join break is not season-gated -- it hits mid-season
    exactly when decisions are made. The per-kind CSV path checks each pool and passes none.
    """
    if board.empty:
        print(
            f"no players qualified for {year} (>= the MIN_PT floors); the keeper board is "
            "empty -- either the season is too early, or the actuals/skills join is broken "
            "(mlbam id mismatch)."
        )
        return True
    missing = set(pools) - set(board["kind"]) if pools else set()
    if missing:
        print(
            f"the {'/'.join(sorted(str(p) for p in missing))} pool is empty for {year}; a "
            "partial keeper board would silently omit it -- check that pool's actuals/skills "
            "join (mlbam id overlap) and its MIN_PT floor."
        )
        return True
    return False


def _scored_board(year: int, denoms, keepers: dict[str, str], pricing) -> pd.DataFrame:
    """Both pools on ONE proj_var scale, indexed by mlbam_id, proj_var-descending.

    Shared by `--roster` and `--league` so the same player cannot be scored two different
    ways depending on which report asked. Cross-pool comparison is what `proj_var` exists
    for; the composite alone could not do it.

    Two-way players are NOT deduped here: their hitter and pitcher rows share one mlbam_id
    but can belong to DIFFERENT teams (`--league`), so collapsing before ownership is known
    deletes one team's copy (#282). Each caller applies `_dedupe_two_way` per roster, after
    ownership -- within a team for `--league`, over my roster for `--roster`.
    """
    scored = []
    for kind in POOLS:
        part = build(year, kind, denoms, keepers, pricing=pricing)
        part["kind"] = kind
        scored.append(part)  # index is mlbam_id, which the per-roster dedupe needs
    return pd.concat(scored).sort_values("proj_var", ascending=False)


def league_report(year: int, denoms, keepers: dict[str, str], slots: int, top: int) -> int:
    """The league-wide keeper board, then each team's best `slots` candidates.

    P(keep) is deliberately NOT computed over the league: it is the probability a
    player finishes among the best on HIS OWN roster, which is the decision each
    manager actually faces, and it depends on the exact rivals. So it is computed
    once per team over that team's whole scoreable roster -- not over the top five
    shown, or the numbers would not sum to the slot count.
    """
    config = load_config(CONFIG_PATH)
    rosters = load_league_rosters(config.team_name)
    if not rosters:
        print("No league rosters available (needs the live KV blobs); nothing to score.")
        return 1
    # A "LEAGUE KEEPER BOARD" that silently covers only some teams is worse than none:
    # one present KV blob (e.g. cache:roster without cache:opp_rosters) unions to a
    # handful of teams and the header still reads "league". Refuse rather than mislead.
    if len(rosters) < config.num_teams:
        print(
            f"partial league: only {len(rosters)} of {config.num_teams} team rosters loaded "
            f"from the KV ({config.num_teams - len(rosters)} missing) -- the board would omit "
            "whole teams with no signal. Refresh the roster blobs and retry."
        )
        return 1

    # Owner map keyed on `(normalized_name, player_type)`, NOT a bare name: two teams can
    # roster the same normalized name (Ohtani hitter here, pitcher there; two Luis Garcias),
    # and a bare key would hand both board rows to whichever team iterated last (#282).
    owner_of = {key: team for team, keys in rosters.items() for key in keys}
    board = _scored_board(year, denoms, keepers, pricing_table())
    if _fail_if_empty_board(board, year, POOLS):
        return 1
    board["owner"] = [owner_of.get(key) for key in _board_row_keys(board)]
    board = board.dropna(subset=["owner"])
    if board.empty:
        print("No rostered players matched the board (roster/board name mismatch).")
        return 1

    # The one case (name, type) cannot resolve: the SAME key on two DIFFERENT teams (two
    # same-type players sharing a normalized name). owner_of silently keeps the last team, so
    # say so out loud rather than mis-credit in silence -- the board and roster share no
    # unique id. Warn only for a key that actually reaches a SCORED row, and AFTER the board
    # is built: a collision among sub-floor players (never on the board) corrupts nothing
    # shown, and a warning printed before the empty-board guard would name a phantom.
    #
    # This warning names arbitrary OWNERSHIP; it does NOT repair the downstream counts, which
    # are the same id-less #282 residual: the team that loses owner_of's last-writer tiebreak
    # still shows its qualified colliding player as "below the floor", and a same-(name, type)
    # FREE AGENT (colliding with exactly ONE team, so len(on_teams) == 1) is scored into that
    # team with no warning. A roster mlbam id is the real fix (#284).
    on_teams: dict[tuple[str, str], set[str]] = {}
    for team, keys in rosters.items():
        for key in keys:
            on_teams.setdefault(key, set()).add(team)
    board_keys = set(_board_row_keys(board))
    ambiguous = sorted(key for key in board_keys if len(on_teams.get(key, ())) > 1)
    if ambiguous:
        shown = _fmt_keys(ambiguous)
        print(
            f"  WARNING: {len(ambiguous)} name(s) are rostered by more than one team and cannot be "
            f"told apart without a player id, so ownership of them is arbitrary: {shown}."
        )

    # One grouping, and P(keep) per team over that team's own roster -- never league-wide.
    # Iterating the frame's groups rather than `rosters` means a team with nothing scoreable
    # simply has no group, so there is no zero-fill to initialize and no emptiness guard to
    # forget. The two-way dedupe happens HERE, per team: a player's hitter and pitcher rows
    # can belong to different teams, so collapsing before ownership deletes one team's copy.
    #
    # "Below the floor" is counted from the team's PRE-dedupe group, not `len(rosters) -
    # len(part)`: a roster entry cleared MIN_PT iff its (name, type) has a row in the group,
    # and a two-way owner's two entries collapse to one scored row without either being below
    # the floor. Counter (not a length difference) keeps that right and never negative even
    # when a same-(name,type) collision hands one team both rows (#282).
    by_team: dict[str, pd.DataFrame] = {}
    unscored_by_team: dict[str, int] = {}
    for team, group in board.groupby("owner", sort=False):
        unscored_by_team[str(team)] = _below_floor(rosters[team], _board_row_keys(group)).total()
        by_team[str(team)] = _score_roster(group, slots)

    # The league table needs the PER-TEAM-DEDUPED board (a two-way player owned by one team
    # collapses to his better side), which is the concat of the groups re-sorted -- `board`
    # itself is no longer deduped now that the dedupe moved into the loop above.
    rostered = pd.concat(by_team.values()).sort_values("proj_var", ascending=False)
    print(f"\n{'=' * 72}")
    print(f"LEAGUE KEEPER BOARD -- top {top} of {len(rostered)} scoreable rostered players")
    print(f"{'=' * 72}")
    print(
        f"{'':4}{'PLAYER':<20}{'POS':>4}{'AGE':>4}{'PROJ VAR':>10}"
        f"{'RAW SGP':>9}{'+/-SD':>7}{'OWNER':>30}"
    )
    print("-" * 88)
    for rank, row in enumerate(rostered.head(top).itertuples(), start=1):
        mine = "*" if row.owner == config.team_name else " "
        print(
            f"{rank:>3}{mine}{row.name:<20}{row.pos:>4}{row.age:>4}"
            f"{row.proj_var:>10.2f}{row.proj_sgp:>9.2f}{row.sd:>7.2f}{row.owner:>30}"
        )

    print(f"\n{'=' * 72}")
    print(f"EACH TEAM'S TOP {slots} KEEPER CANDIDATES  (P KEEP is within that roster)")
    print(f"{'=' * 72}")
    # `head(slots)` means "his best" only because each group inherits the board's
    # descending order through the groupby.
    strongest = sorted(by_team, key=lambda t: -by_team[t]["proj_var"].head(slots).sum())
    for team in strongest:
        part = by_team[team]
        unscored = unscored_by_team[team]
        mine = " *" if team == config.team_name else "  "
        print(f"\n{mine}{team}  ({len(part)} scoreable, {unscored} below the floor)")
        for row in part.head(slots).itertuples():
            print(
                f"      {row.name:<22}{row.pos:>4}{row.age:>4}"
                f"{row.proj_var:>9.2f}{row.p_keep * 100:>7.0f}%"
            )
    print(
        "\n  A player below the qualifying floor has no percentile and is not"
        " competing, so he inflates his own team's P KEEP."
    )
    return 0


def roster_report(year: int, denoms, keepers: dict[str, str], slots: int) -> int:
    """Score one roster and give each player P(he is among its `slots` best).

    That is the keeper question directly -- "would I be right to keep him" -- and
    it only means anything against a specific set of rivals, so it is computed
    here over the roster rather than shipped in the pool-wide CSV.
    """
    roster = load_roster_keys()  # my roster as (name, type) keys; a LIST, so an owned
    if not roster:  # below-floor DUPLICATE of a name is not collapsed away
        print("No roster available (needs the live KV blob); nothing to score.")
        return 1
    roster_set = set(roster)

    board = _scored_board(year, denoms, keepers, pricing_table())
    if _fail_if_empty_board(board, year, POOLS):
        return 1
    # Filter to MY roster by (normalized_name, player_type), not a bare name: a CROSS-type
    # namesake (a hitter "Will Smith" when I own the reliever) must not be scored into my
    # competing set and dilute every real keeper's P(keep) (#282). This resolves cross-TYPE
    # collisions; two same-TYPE players sharing a normalized name (two "Luis Garcia" pitchers)
    # share this key and cannot be separated without a shared id -- the irreducible #282 residual
    # (a roster mlbam id is the real fix; #284). Normalize the whole board ONCE here.
    all_keys = _board_row_keys(board)
    mask = [key in roster_set for key in all_keys]
    board = board[mask]
    board_keys = [key for key, keep in zip(all_keys, mask, strict=True) if keep]  # PRE-dedupe
    # "Below the floor" is decided from the PRE-dedupe board (see `_below_floor`): deriving it
    # from the post-dedupe frame would brand a two-way player's deliberately-collapsed side as
    # below the floor -- it qualified -- and mask an owned below-floor duplicate behind its twin.
    below = _below_floor(roster, board_keys)
    # A (name, type) the board carries MORE often than I roster it is an unowned same-type
    # namesake scored into my competition, spreading my slot mass over a player I do not own.
    # This flags that board-SURPLUS case only; it CANNOT catch a 1-for-1 swap (I roster a
    # below-floor "Luis Garcia" while an unowned qualifying one takes the board row) -- the counts
    # cancel, so that stays silent. Both are the same id-less #282 residual, not fully fixable here.
    diluted = Counter(board_keys) - Counter(roster)
    board = _score_roster(board, slots)  # dedupe two-way to his better side, then P(keep)

    print(f"\n=== {len(board)} scoreable players, {slots} keeper slots ===")
    print(f"{'':1}{'PLAYER':<20}{'POS':>4}{'AGE':>4}{'PROJ VAR':>10}{'+/-SD':>7}{'P KEEP':>8}")
    print("-" * 54)
    for row in board.itertuples():
        mine = "*" if row.keeper_of else " "
        print(
            f"{mine}{row.name:<20}{row.pos:>4}{row.age:>4}"
            f"{row.proj_var:>10.2f}{row.sd:>7.2f}{row.p_keep * 100:>7.0f}%"
        )
    awarded = min(slots, len(board))
    print(f"\n  P KEEP sums to {board['p_keep'].sum():.2f}, i.e. exactly the {awarded} slots.")
    if diluted:
        shown = _fmt_keys(sorted(diluted))
        print(
            f"  WARNING: {diluted.total()} board row(s) share a name and type with a roster spot "
            f"but a different player id, so an unowned namesake may be inflating your competition "
            f"and understating P KEEP: {shown}."
        )
    if below:
        # `.total()` counts ENTRIES below the floor, the same view --league reads, so the two
        # reports agree even when one name below the floor is held twice; the list names the
        # distinct (name, type) keys, since same-(name, type) twins cannot be told apart (#282).
        shown = _fmt_keys(sorted(below))
        print(
            f"  NOT SCORED ({below.total()}), below the qualifying floor so they have no "
            f"percentile: {shown}."
        )
        print("  Their absence inflates everyone else's P KEEP -- they are not competing.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, help="defaults to config season_year")
    parser.add_argument("--backtest", action="store_true", help="refit the family weights")
    parser.add_argument("--fit", action="store_true", help="refit projection.py's constants")
    parser.add_argument("--scarcity", action="store_true", help="re-measure the positional credits")
    parser.add_argument("--study", action="store_true", help="print the supporting diagnostics")
    parser.add_argument("--roster", action="store_true", help="score my roster for P(top-N)")
    parser.add_argument(
        "--league", action="store_true", help="league-wide board plus each team's best"
    )
    parser.add_argument(
        "--slots", type=int, default=3, help="keeper slots, for --roster and --league"
    )
    parser.add_argument("--top", type=int, default=20, help="rows to print per pool")
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    year = config.season_year if args.year is None else args.year
    denoms = get_sgp_denominators(getattr(config, "sgp_overrides", None))
    keepers = {normalize_name(k["name"]): k["team"] for k in (config.keepers or [])}

    if args.backtest:
        run_backtest(denoms)
        return 0
    if args.fit:
        run_fit(denoms)
        return 0
    if args.scarcity:
        run_scarcity(denoms)
        return 0
    if args.study:
        run_study(denoms, year)
        return 0
    if args.league:
        return league_report(year, denoms, keepers, args.slots, args.top)
    if args.roster:
        return roster_report(year, denoms, keepers, args.slots)

    pricing = pricing_table()
    for kind in POOLS:
        table = build(year, kind, denoms, keepers, pricing=pricing)
        if _fail_if_empty_board(table, year):
            return 1
        out_path = SKILLS_DIR / f"keeper_rankings_{kind}_{year}.csv"
        table.to_csv(out_path)
        shown = SHOWN
        print(
            f"\n=== {kind.upper()} ({len(table)} qualified, >= {MIN_PT[kind]}) -> {out_path.name}"
        )
        print(table[shown].head(args.top).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
