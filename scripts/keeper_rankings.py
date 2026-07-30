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
positional term is a scarce-position bonus rather than a subtracted floor, for
scale reasons `projection.scarcity_floors` sets out -- and the offset is cosmetic,
so it changes no ordering.

Read the ranking in TIERS, not by row. Adjacent players are separated by far less
than `sd`, so consecutive ranks are close to coin flips; `sd` is there to stop a
single-rank gap being read as real.

`--roster` answers the decision directly: P(each of my players finishes among my
N best). That is joint and set-dependent -- it needs the exact rivals -- so it
lives there rather than in this pool-wide CSV.

`luck` carries a POSITIVE weight, which the name does not suggest: the gap
encodes role and durability as well as noise, and forcing it negative collapses
the fit. For trades read it the other way round. `future` is deliberately
discounted for staleness -- the out-year files have never seen this season.
Both are argued in `keepers/composite.py`.

Usage:
    python scripts/keeper_rankings.py
    python scripts/keeper_rankings.py --roster          # P(top-3) on my roster
    python scripts/keeper_rankings.py --backtest        # refit the family weights
    python scripts/keeper_rankings.py --fit             # refit projection.py constants
    python scripts/keeper_rankings.py --study           # the supporting diagnostics
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

from fantasy_baseball.config import load_config
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.data.fangraphs import load_projection_set
from fantasy_baseball.keepers.actuals import index_by_mlbam, innings_to_float
from fantasy_baseball.keepers.composite import (
    FAMILIES,
    FITTED_WEIGHTS,
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
    scarcity_floors,
    sgp_sd,
)
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.name_utils import normalize_name

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
SKILLS_DIR = PROJECT_ROOT / "data" / "cache" / "keeper_skills"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"

# Below these a percentile is noise: the skills module regresses nothing, so a
# 3-inning pitcher can post the league's best ERA-. See its module docstring.
MIN_PT = {"hitter": 250, "pitcher": 50}
# Used only when a player is absent from the position map. UTIL and SP are the
# DEEPEST floors, so an unknown position is charged the harshest replacement
# level rather than flattered by a scarce one.
FALLBACK_POS = {"hitter": ["UTIL"], "pitcher": ["P"]}
POOLS: tuple[str, ...] = (PlayerType.HITTER, PlayerType.PITCHER)
# Display schema for the per-pool tables; the CSV keeps every column.
SHOWN = (
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
)
BACKTEST_FIT_YEARS = (2022, 2023)
BACKTEST_HOLDOUT = 2024


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
    if kind == "pitcher":
        games = pd.to_numeric(frame["G"], errors="coerce")
        starts = pd.to_numeric(frame["GS"], errors="coerce")
        out["start_share"] = (starts / games.where(games > 0)).fillna(0.0)
    return out


def zips_out_year_sgp(year: int, kind: str, denoms) -> pd.Series:
    """SGP of the ZiPS projection for `year`, indexed by mlbam_id.

    `load_projection_set` resolves the filename variants, strips the BOM, validates
    the required columns and renames to the lowercase stat-line keys
    `calculate_player_sgp` wants. It also keeps SV, which `keepers.vintages` drops
    -- the reason this does not route through there.

    CAVEAT on the out-years specifically: SV is present as a column but BLANK on
    every row of the 2027 and 2028 exports, where 2022-2026 all carry it. So a
    closer's `future_pct` is computed with sv=0 and is understated -- measurably,
    zeroing SV on the 2026 file costs a 20-save reliever ~0.13 of percentile. The
    backtest reads the saves-bearing same-year files, so the `future` weight was
    validated on a slightly richer feature than production gets. Nothing to fix in
    code; re-check when fresher out-years land.
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
    return _sgp(lines, denoms)


def pricing_table(denoms) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Positions and mean-centred floors -- neither varies by pool.

    Hoisted out of `build` so one run makes one position lookup instead of two:
    it is the only network touch in this script.
    """
    return load_positions(), scarcity_floors(position_aware_replacement_levels(denoms=denoms))


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


# Full-season innings that stand in for a ROLE, either side of the 100-IP bar
# `sgp.replacement.role_from_ip` splits on. Not projections -- they exist only to
# name the role, which is why they are round numbers.
STARTER_ROLE_IP = 180.0
RELIEVER_ROLE_IP = 60.0


def _slots_for(positions: dict[str, list[str]], name: str, kind: str) -> list[str]:
    """Eligible slots for pricing, constrained to the pool being scored.

    Yahoo lists Ohtani as UTIL, so in the PITCHER pool `calculate_var` would take
    the hitter branch and net his pitching projection against the UTIL floor. A
    row is only ever priced against its own pool's floors.
    """
    slots = positions.get(normalize_name(str(name)), [])
    if kind == "pitcher":
        pitching = [slot for slot in slots if slot in ("P", "SP", "RP")]
        return pitching or FALLBACK_POS[kind]
    batting = [slot for slot in slots if slot not in ("P", "SP", "RP")]
    return batting or FALLBACK_POS[kind]


def _role_equivalent_ip(frame: pd.DataFrame, kind: str) -> list[float | None]:
    """Which replacement floor each pitcher should net against.

    `calculate_var` routes SP vs RP off `role_ip`, and its docstring asks for a
    FULL-SEASON-equivalent figure precisely so a partial line does not flip the
    role mid-season. Passing the to-date innings -- which is what the frame
    already holds -- is a no-op, and in late July it put 120 of 198 qualified
    pitchers on the RP floor, the shallowest of the nine: every starter under 100
    innings so far was priced as a reliever, worth a spurious ~1.9 SGP against a
    genuine one.

    Role comes from START SHARE instead, which a partial season measures fine.
    """
    if kind != "pitcher":
        return [None] * len(frame)
    share = frame.get("start_share")
    if share is None:
        return [None] * len(frame)
    return [STARTER_ROLE_IP if value >= 0.5 else RELIEVER_ROLE_IP for value in share]


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

    blended = composite(_family_columns(qualified), kind)
    # Re-rank the blend to 0-1. `luck` is a difference centred on zero while the
    # other three span 0-1, so the raw weighted mean is not on a percentile scale
    # and would read as if everyone were mid-pack. Ranking is order-preserving,
    # so this changes only how the number reads.
    qualified["composite"] = percentile(blended)

    # The composite is ordinal; this is what puts it on a value scale and lets
    # hitters and pitchers share one list. See `keepers.projection`.
    qualified["proj_sgp"] = expected_sgp(qualified["composite"], kind)
    qualified["sd"] = sgp_sd(qualified["composite"], kind)

    # Mean-centred floors: a display offset only, see `projection.scarcity_floors`.
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
            role_ip=role,
        )
        for name, proj, pt, role in zip(
            qualified["name"],
            qualified["proj_sgp"],
            qualified["pt"],
            _role_equivalent_ip(qualified, kind),
            strict=True,
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
    blended = composite(_family_columns(frame), kind, weights=weights)
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


def load_roster_names() -> set[str]:
    """Normalized names on my roster, from the live KV blob.

    Import-local and failure-tolerant, like `keepers.positions`: the rest of this
    script runs offline, and only this one report needs the network.
    """
    try:
        from fantasy_baseball.data.kv_store import get_kv

        blob = get_kv().get("cache:roster")
    except Exception:
        return set()
    if not blob:
        return set()
    payload = json.loads(blob) if isinstance(blob, str) else blob
    entries = payload.get("_data", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return set()
    return {normalize_name(str(e.get("name", ""))) for e in entries if e.get("name")}


def run_fit(denoms) -> None:
    """Refit and print `keepers.projection`'s constants, paste-ready.

    Same features and transitions as `--backtest`, but regressing realized SGP in
    LEVELS rather than correlating percentiles -- that is what puts the ordinal
    composite on a value scale. Without this the value half of a two-stage model
    would be unreproducible while the ordinal half is a flag away.
    """
    seasons = [*BACKTEST_FIT_YEARS, BACKTEST_HOLDOUT]
    print(f"# Refitted over {', '.join(f'{y}->{y + 1}' for y in seasons)}")
    fit, sd_fit, quantiles = {}, {}, {}
    for kind in POOLS:
        frames = []
        for year in seasons:
            feat = _transition(year, kind, denoms)
            frames.append(
                pd.DataFrame(
                    {
                        "c": percentile(composite(_family_columns(feat), kind)),
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


def run_study(denoms) -> None:
    """Print the diagnostics the module docstrings argue from.

    Every claim in `keepers/composite.py` about WHY the families are shaped the way
    they are is reproduced here, so none of them can drift into being wrong without
    this command disagreeing.
    """
    seasons = (*BACKTEST_FIT_YEARS, BACKTEST_HOLDOUT)
    for kind in POOLS:
        frames = [_transition(year, kind, denoms) for year in seasons]
        header = "=" * 66
        print("")
        print(header)
        print(f"{kind.upper()}  ({len(frames)} transitions)")

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

    scored = []
    pricing = pricing_table(denoms)
    for kind in POOLS:
        table = build(year, kind, denoms, keepers, pricing=pricing)
        on_roster = table["name"].map(lambda n: normalize_name(str(n)) in roster)
        part = table[on_roster].copy()
        part["kind"] = kind
        scored.append(part)
    board = pd.concat(scored).sort_values("proj_var", ascending=False).reset_index(drop=True)

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
    parser.add_argument("--study", action="store_true", help="print the supporting diagnostics")
    parser.add_argument("--roster", action="store_true", help="score my roster for P(top-N)")
    parser.add_argument("--slots", type=int, default=3, help="keeper slots, for --roster")
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
    if args.study:
        run_study(denoms)
        return 0
    if args.roster:
        return roster_report(year, denoms, keepers, args.slots)

    pricing = pricing_table(denoms)
    for kind in POOLS:
        table = build(year, kind, denoms, keepers, pricing=pricing)
        out_path = SKILLS_DIR / f"keeper_rankings_{kind}_{year}.csv"
        table.to_csv(out_path)
        shown = list(SHOWN)
        print(
            f"\n=== {kind.upper()} ({len(table)} qualified, >= {MIN_PT[kind]}) -> {out_path.name}"
        )
        print(table[shown].head(args.top).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
