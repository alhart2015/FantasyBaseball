"""Rank keeper candidates on skill, luck, future and age, then price them in VAR.

Reads what `fetch_keeper_skills.py` cached, computes each player's actual roto
value (SGP) for the season, blends the four families in percentile space using
the weights fitted in `keepers/composite.py`, then converts that ordinal
composite into projected value with an error bar via `keepers/projection.py`.

Writes `data/cache/keeper_skills/keeper_rankings_{kind}_{year}.csv`:

    value_pct   percentile of actual SGP this season (= skill + luck)
    skill_pct   SKILL   -- percentile of the peripherals
    luck_pct    LUCK    -- value_pct - skill_pct, the unsupported part of the line
    future_pct  FUTURE  -- percentile of blended out-year ZiPS projected SGP
    age_pct     AGE     -- percentile of age, younger better
    composite   the fitted four-family blend, ordinal within pool
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
different rates -- composite 1.0 maps to 13.37 for hitters and 8.92 for pitchers --
so no pitcher reaches the top of a mixed board. That is a real predictability gap,
not a scale artifact: top-decile hitters went on to earn 14.42 next season against
10.10 for pitchers, and pitchers were 4.6% likely to fall to zero against 1.0%.
Read a mixed board as expected value only, and read the pitcher list on its own --
its top twelve span 0.46 SGP against an sd near 5.9, so their ORDER means nothing.

`luck` carries a POSITIVE weight and `future` is discounted for staleness. Both
are counterintuitive and both are argued in `keepers/composite.py`; `--study`
reproduces the evidence.

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
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.data.fangraphs import load_projection_set
from fantasy_baseball.keepers.actuals import index_by_mlbam, innings_to_float
from fantasy_baseball.keepers.composite import (
    FAMILIES,
    FITTED_WEIGHTS,
    FUTURE_BLEND,
    HITTER_SKILLS,
    LOWER_IS_BETTER,
    PITCHER_SKILLS,
    composite,
    future_percentile,
    luck,
    percentile,
    skill_percentile,
)
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
# RP resolve to the same floor -- see `keepers.scarcity`.
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
    return out


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
    return _sgp(lines, denoms)


def pricing_table(denoms) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Positions and mean-centred floors -- neither varies by pool.

    Hoisted out of `build` so one run makes one position lookup instead of two:
    it is the only network touch in this script.
    """
    return load_positions(), credit_levels()


def _observed(year: int, kind: str, denoms) -> pd.DataFrame:
    """Actual value, age, playing time and skills for one season and pool."""
    value = season_value(year, kind, denoms)
    skills = pd.read_csv(SKILLS_DIR / f"{kind}_skills_{year}.csv").set_index("mlbam_id")
    return value.join(skills, how="inner", rsuffix="_sk")


def _qualified_families(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Apply the playing-time floor and build the three same-season families.

    Shared by the shipped ranking and the backtest that justifies its weights, so
    the two cannot drift into validating different feature definitions. `future`
    is left to the caller: the ranking uses out-years, the backtest a stale
    same-year projection.
    """
    qualified = frame[frame["pt"] >= MIN_PT[kind]].copy()
    qualified["value_pct"] = percentile(qualified["sgp"])
    qualified["skill_pct"] = skill_percentile(qualified, kind)
    qualified["luck_pct"] = luck(qualified["value_pct"], qualified["skill_pct"])
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
    frame: pd.DataFrame, kind: str, weights: tuple[float, float, float, float] | None = None
) -> pd.Series:
    """The composite, re-ranked to 0-1 -- the x-axis everything downstream uses.

    ONE definition on purpose. `projection`'s constants are fitted against this
    exact quantity by `--fit`, and `expected_sgp`/`sgp_sd` are then applied to it
    in `build`. Two independent spellings would let the slope and intercept keep
    being applied to a subtly different variable, moving every proj_sgp, sd,
    proj_var and p_keep with no test failing.

    The re-rank matters: `luck` is a difference centred on zero while the other
    three families span 0-1, so the raw weighted mean is not on a percentile scale.
    Ranking is order-preserving, so it changes only how the number reads.
    """
    return percentile(composite(_family_columns(frame), kind, weights=weights))


def _family_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """The `{family: series}` mapping `composite` expects, keyed off FAMILIES so a
    fifth family cannot be wired into one call site and not the other."""
    return {family: frame[f"{family}_pct"] for family in FAMILIES}


def build(
    year: int,
    kind: str,
    denoms,
    keepers: dict[str, str],
    pricing: tuple[dict[str, list[str]], dict[str, float]] | None = None,
) -> pd.DataFrame:
    qualified = _qualified_families(_observed(year, kind, denoms), kind)

    # Rank the projections WITHIN the qualified pool, not within all ~1900 ZiPS
    # rows. Most of that file is minor leaguers, so ranking there would put every
    # established regular above the 90th percentile and say nothing.
    near = zips_out_year_sgp(year + 1, kind, denoms).reindex(qualified.index)
    far = zips_out_year_sgp(year + 2, kind, denoms).reindex(qualified.index)
    qualified["future_pct"] = future_percentile(near, far)
    if qualified["future_pct"].isna().all():
        # `composite` mean-fills a missing family and the mean of an all-NaN
        # column is NaN, so this would silently write a CSV of NaN, rank
        # alphabetically, and hand --roster 100% keep odds to the first three
        # names. Fail instead.
        raise FileNotFoundError(
            f"no {kind} ZiPS out-year projection under {PROJECTIONS_DIR} for "
            f"{year + 1}/{year + 2}; `future` cannot be computed"
        )

    qualified["composite"] = composite_pct(qualified, kind)

    # The composite is ordinal; this is what puts it on a value scale and lets
    # hitters and pitchers share one list. See `keepers.projection`.
    qualified["proj_sgp"] = expected_sgp(qualified["composite"], kind)
    qualified["sd"] = sgp_sd(qualified["composite"], kind)

    # Mean-centred credits: a display offset only, see `keepers.scarcity`.
    positions, floors = pricing_table(denoms) if pricing is None else pricing
    priced = [
        calculate_var(
            pd.Series(
                {
                    "total_sgp": proj,
                    "positions": _slots_for(positions, name, kind),
                    "ip": pt,
                }
            ),
            floors,
            return_position=True,
        )
        for name, proj, pt in zip(
            qualified["name"], qualified["proj_sgp"], qualified["pt"], strict=True
        )
    ]
    qualified["proj_var"] = [var for var, _ in priced]
    qualified["pos"] = [pos for _, pos in priced]
    qualified["keeper_of"] = [keepers.get(normalize_name(str(n)), "") for n in qualified["name"]]

    ranked = qualified.sort_values("proj_var", ascending=False)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


# --- backtest -------------------------------------------------------------


def _transition(year: int, kind: str, denoms) -> pd.DataFrame:
    """Features observed in `year` against the SGP percentile realized in year+1."""

    feat = _qualified_families(_observed(year, kind, denoms), kind)
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


def _weighted_rho(
    frame: pd.DataFrame, weights: tuple[float, float, float, float], kind: str
) -> float:
    blended = composite_pct(frame, kind, weights=weights)
    return float(blended.corr(frame["target"], method="spearman"))


def run_backtest(denoms) -> None:
    for kind in POOLS:
        fit = [_transition(y, kind, denoms) for y in BACKTEST_FIT_YEARS]
        hold = _transition(BACKTEST_HOLDOUT, kind, denoms)
        print(
            f"\n{'=' * 70}\n{kind.upper()}  fit={list(BACKTEST_FIT_YEARS)} holdout={BACKTEST_HOLDOUT}"
        )

        # skill is pinned at 1.0; every other family is measured against it.
        rows = []
        for w_luck, w_future, w_age in product(
            (0.4, 0.6, 0.8, 1.0, 1.2), (0.0, 0.2, 0.4, 0.6, 0.8), (0.0, 0.15, 0.3, 0.45)
        ):
            weights = (1.0, w_luck, w_future, w_age)
            rows.append(
                {
                    "luck": w_luck,
                    "future": w_future,
                    "age": w_age,
                    "fit_rho": sum(_weighted_rho(f, weights, kind) for f in fit) / len(fit),
                    "holdout_rho": _weighted_rho(hold, weights, kind),
                }
            )
        table = pd.DataFrame(rows).sort_values("fit_rho", ascending=False)
        print("  skill pinned at 1.0; top blends by fit:")
        print(table.head(5).round(4).to_string(index=False))
        print("\n  baselines (holdout):")
        shipped = FITTED_WEIGHTS[kind]
        for label, weights in [
            ("shipped", shipped),
            ("no future", (1.0, shipped[1], 0.0, shipped[3])),
            ("skill+luck only", (1.0, shipped[1], 0.0, 0.0)),
            ("skill only", (1.0, 0.0, 0.0, 0.0)),
        ]:
            print(f"    {label:17s} rho = {_weighted_rho(hold, weights, kind):.4f}")


def _as_payload(blob):
    """Unwrap a KV value: JSON string or dict, with the `_data` envelope removed."""
    if not blob:
        return None
    payload = json.loads(blob) if isinstance(blob, str) else blob
    if isinstance(payload, dict) and "_data" in payload:
        return payload["_data"]
    return payload


def _normalized_names(entries) -> set[str]:
    if not isinstance(entries, list):
        return set()
    return {normalize_name(str(e.get("name", ""))) for e in entries if e.get("name")}


def load_roster_names() -> set[str]:
    """Normalized names on my roster, from the live KV blob.

    Import-local and failure-tolerant, like `keepers.positions`: the rest of this
    script runs offline, and only this one report needs the network.
    """
    try:
        from fantasy_baseball.data.kv_store import get_kv

        return _normalized_names(_as_payload(get_kv().get("cache:roster")))
    except Exception:
        return set()


def load_league_rosters(my_team: str) -> dict[str, set[str]]:
    """Normalized names per team for the WHOLE league, from the live KV blobs.

    `cache:roster` holds only my team and `cache:opp_rosters` only the other
    nine, so the league is the union. Same import-local, failure-tolerant shape
    as `load_roster_names`: everything else in this script runs offline.
    """
    try:
        from fantasy_baseball.data.kv_store import get_kv

        kv = get_kv()
        opponents = _as_payload(kv.get("cache:opp_rosters"))
    except Exception:
        return {}
    rosters: dict[str, set[str]] = {}
    if isinstance(opponents, dict):
        for team, players in opponents.items():
            names = _normalized_names(players)
            if names:
                rosters[str(team)] = names
    mine = load_roster_names()
    if mine:
        rosters[my_team] = mine
    return rosters


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
                        "c": composite_pct(feat, kind),
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

    def priced(frame: pd.DataFrame) -> pd.Series:
        feat = _qualified_families(frame, kind)
        feat["future_pct"] = percentile(out_year.reindex(feat.index))
        feat = feat.dropna(subset=["value_pct", "skill_pct", "age_pct"])
        return expected_sgp(composite_pct(feat, kind), kind)

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


SCARCITY_YEARS = (2022, 2023, 2024, 2025)


def run_scarcity(denoms) -> None:
    """Re-measure the positional credits and print them paste-ready.

    The floors `keepers.scarcity` ships are measured, not assumed, so this is what
    regenerates them. It prints the per-season table as well as the average because
    the single-season spread is large -- the catcher credit has ranged from 0.50 to
    2.18 -- and an average of four seasons is the only defensible summary.
    """
    config = load_config(CONFIG_PATH)
    positions = load_positions()
    capacities = slot_capacities(config.roster_slots, config.num_teams)
    print(f"  league starting slots (bench and IL excluded): {capacities}")

    per_season = []
    for year in SCARCITY_YEARS:
        floors: dict[str, float] = {}
        for kind in POOLS:
            board = build(year, kind, denoms, {}, pricing=(positions, credit_levels()))
            eligible = {
                idx: set(_slots_for(positions, name, kind))
                for idx, name in zip(board.index, board["name"], strict=True)
            }
            wanted = {
                slot: n for slot, n in capacities.items() if (slot == "P") == (kind == "pitcher")
            }
            floors.update(marginal_starter_floors(board["proj_sgp"], eligible, wanted))
        per_season.append((year, floors))

    slots = sorted({slot for _, f in per_season for slot in f})
    print("\n  measured floor per season (on this model's proj_sgp scale):")
    print("    year  " + "".join(f"{slot:>8}" for slot in slots))
    for year, floors in per_season:
        print(
            f"    {year}  " + "".join(f"{floors.get(slot, float('nan')):>8.2f}" for slot in slots)
        )

    mean_floor = {
        slot: sum(f[slot] for _, f in per_season if slot in f)
        / sum(1 for _, f in per_season if slot in f)
        for slot in slots
    }
    fresh = centred_credits(mean_floor)
    print("\n  averaged, centred, against what is shipped:")
    print(f"    {'slot':<7}{'floor':>8}{'credit':>9}{'shipped':>10}{'delta':>8}")
    for slot in sorted(fresh, key=lambda s: -fresh[s]):
        shipped = NATIVE_CREDITS.get(slot, float("nan"))
        print(
            f"    {slot:<7}{mean_floor[slot]:>8.2f}{fresh[slot]:>9.2f}"
            f"{shipped:>10.2f}{fresh[slot] - shipped:>8.2f}"
        )
    print("\n  paste into keepers/scarcity.py:")
    print("NATIVE_CREDITS: dict[str, float] = {")
    for slot in sorted(fresh, key=lambda s: -fresh[s]):
        print(f'    "{slot}": {fresh[slot]:.3f},')
    print("}")


def run_study(denoms, live_year: int) -> None:
    """Print the diagnostics the module docstrings argue from.

    Every claim in `keepers/composite.py` about WHY the families are shaped the way
    they are is reproduced here, so none of them can drift into being wrong without
    this command disagreeing.
    """
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
            ("luck", "luck_pct"),
            ("age (younger)", "age_pct"),
            ("future (stale)", "future_pct"),
        ):
            cells = [
                _mean_rho(frames, column, target)
                for target in ("target", "target_pt", "target_rate")
            ]
            print(f"    {label:<16}" + "".join(f"{c:>9.3f}" for c in cells))

        print("  luck needs a POSITIVE weight (composite = skill + w*luck -> next SGP):")
        for weight in (-1.0, -0.5, 0.0, 0.5, 1.0):
            scores = [_rho(f["skill_pct"] + weight * f["luck_pct"], f["target"]) for f in frames]
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

        # Does the POSITIONAL credit match what players at that position actually
        # realize? This is what decides whether the spread is supported.


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


def _scored_board(year: int, denoms, keepers: dict[str, str], pricing) -> pd.DataFrame:
    """Both pools on ONE proj_var scale, two-way players collapsed.

    Shared by `--roster` and `--league` so the same player cannot be scored two
    different ways depending on which report asked. Cross-pool comparison is what
    `proj_var` exists for; the composite alone could not do it.
    """
    scored = []
    for kind in POOLS:
        part = build(year, kind, denoms, keepers, pricing=pricing).copy()
        part["kind"] = kind
        scored.append(part)  # index is mlbam_id, and the dedupe below needs it
    return _dedupe_two_way(pd.concat(scored).sort_values("proj_var", ascending=False))


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

    owner_of = {name: team for team, names in rosters.items() for name in names}
    board = _scored_board(year, denoms, keepers, pricing_table(denoms))
    board["owner"] = board["name"].map(lambda n: owner_of.get(normalize_name(str(n)), ""))
    rostered = board[board["owner"] != ""].copy()

    # P(keep) per team, over that team's full scoreable roster.
    rostered["p_keep"] = 0.0
    for team in rosters:
        rows = rostered["owner"] == team
        if not rows.any():
            continue
        part = rostered[rows]
        rostered.loc[rows, "p_keep"] = probability_top_n(
            part["proj_var"], part["sd"], part["kind"], slots
        )

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
    for team in sorted(
        rosters, key=lambda t: -rostered.loc[rostered["owner"] == t, "proj_var"].head(slots).sum()
    ):
        part = rostered[rostered["owner"] == team]
        unscored = len(rosters[team]) - len(part)
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
    roster = load_roster_names()
    if not roster:
        print("No roster available (needs the live KV blob); nothing to score.")
        return 1

    board = _scored_board(year, denoms, keepers, pricing_table(denoms))
    board = board[board["name"].map(lambda n: normalize_name(str(n)) in roster)].copy()

    board["p_keep"] = probability_top_n(board["proj_var"], board["sd"], board["kind"], slots)
    missing = sorted(roster - {normalize_name(str(n)) for n in board["name"]})

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
    if missing:
        print(
            f"  NOT SCORED ({len(missing)}), below the qualifying floor so they have no "
            f"percentile: {', '.join(missing)}."
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

    pricing = pricing_table(denoms)
    for kind in POOLS:
        table = build(year, kind, denoms, keepers, pricing=pricing)
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
