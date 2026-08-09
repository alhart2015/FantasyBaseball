"""Out-of-sample bake-off between the trajectory matchers (#312, #313).

Answers the question the default rests on: does `shape` actually predict better than
level matching, and on which players? Every claim in `trajectory/__init__.py` and in
`shape.py`'s docstring comes from this script -- it exists so those numbers can be
re-measured rather than trusted, and so a regression in the estimator shows up as a
changed table instead of a stale docstring.

**The query player is removed from the panel entirely** before either estimator is
built, so neither can match him to himself. That is the whole point: an in-sample
comparison would flatter `shape`, which fits a model, over `comps`, which averages.

Slice, always. A random sample of the panel is dominated by fringe players -- 173 of
249 in the first run -- and the pooled number said 3% RMSE where the decision-relevant
slice said 18-20%. Pooled accuracy is not the thing being bought.

Usage:
    python scripts/backtest_trajectory.py                      # hitters, elite slices
    python scripts/backtest_trajectory.py --pool pitcher       # the #313 question
    python scripts/backtest_trajectory.py --sample 400         # random rather than elite
    python scripts/backtest_trajectory.py --horizon 2 --elite-floor 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from keeper_persistence import TRANSITIONS as KEEPER_TRANSITIONS

from fantasy_baseball.config import load_config
from fantasy_baseball.sgp.denominators import SgpOverrides
from fantasy_baseball.trajectory.board import season_slots
from fantasy_baseball.trajectory.comps import comp_trajectory
from fantasy_baseball.trajectory.era import era_normalize
from fantasy_baseball.trajectory.panel import DEFAULT_PANEL_DIR, load_scored_panel
from fantasy_baseball.trajectory.panel import score as panel_score
from fantasy_baseball.trajectory.shape import build_history, shape_trajectory
from fantasy_baseball.trajectory.value import STARTER_SHARE, best_floor, resolve_slots
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD

#: Columns the role bucket needs, and the rule for a season split across two rows.
#: `collapse_split_seasons` keeps only `sgp` and `age`, so a traded pitcher's counting
#: columns have to be re-summed here or a mid-season trade reads as two half-roles --
#: the same reason `trajectory.board._SPLIT_RULES` re-sums `starts`/`games`.
_ROLE_SUMS = ("starts", "games", "sv")

#: Every year-pair the persistence fit could use, bounded by actuals coverage
#: (`data/stats/{pool}-{Y}.csv` exists for 2022-2025). Imported rather than restated so
#: this cannot drift from the fit `keeper_persistence` actually validated.
ALL_TRANSITIONS = KEEPER_TRANSITIONS


def transitions_for(base_year: int, mode: str) -> tuple[tuple[int, int], ...]:
    """Which (year, year+1) transitions the persistence fit may use for `base_year`.

    `loto` drops ONLY the transition being predicted. It does not make the fit causal,
    and the difference matters: for base 2022 both survivors are LATER than the
    transition predicted, so the fit trains on the future. That is disclosed in the
    writeup as a third advantage keeper-value keeps, rather than silently corrected,
    because a strictly causal rule leaves base 2022 with nothing to fit on at all and
    base 2023 with one transition -- which would delete the +2 horizon and with it the
    multi-year claim this evaluation exists to make.

    `causal` is the sensitivity variant, and it is only informative for base 2023:

        base 2022  loto = 2 transitions, both future   causal = 0   not computable
        base 2023  loto = 2, one future                causal = 1   THIS is the check
        base 2024  loto = 2, none future               causal = 2   identical, measures nothing
    """
    if mode not in {"loto", "causal"}:
        raise ValueError(f"mode must be 'loto' or 'causal', got {mode!r}")
    if mode == "loto":
        predicted = (base_year, base_year + 1)
        return tuple(t for t in ALL_TRANSITIONS if t != predicted)
    return tuple(t for t in ALL_TRANSITIONS if t[1] <= base_year)


#: The last season that can serve as an OUTCOME. 2026 is in progress, and the only
#: tool for comparing it against full seasons -- `panel.prorate_partial` -- is
#: straight-line and explicitly assumes the player stays healthy. Pacing an outcome
#: season would scale an injured player's line up as if he had not been hurt, which is
#: exactly the confound the injury-excluded view exists to remove.
LAST_OUTCOME_SEASON = 2025


def horizons_for(base_year: int) -> tuple[int, ...]:
    """Which forward years are scoreable from `base_year`.

    The single source of truth for this, so the shape side, the keeper side and the
    slice counts cannot disagree about which base years support a multi-year target.
    """
    return tuple(h for h in (1, 2) if base_year + h <= LAST_OUTCOME_SEASON)


def historical_panel(
    raw_panel: pd.DataFrame,
    kind: str,
    base_year: int,
    sgp_overrides: SgpOverrides | None,
) -> pd.DataFrame:
    """Era-normalize on the FULL panel, THEN truncate to `base_year`.

    Not the other order, and this is not a stylistic preference. `era_normalize` raises
    when any of `REFERENCE_SEASONS = (2023, 2024, 2025)` is missing -- deliberately, so
    a partial window cannot silently restate every season into units the output never
    mentions. A panel truncated to `season <= 2022` contains none of them, so computing
    factors after truncation aborts base years 2022 and 2023 outright.

    The factor table is therefore informed by seasons after `base_year`. That is a
    limitation, not an advantage to either estimator: a run environment is a league-wide
    fact and both sides are restated by the same one. It is also what the shipped
    harness already does -- it normalizes the full panel and filters queries afterwards.

    Called once per base year. `era_normalize` re-scores every one of ~18,000 seasons
    row-wise, so calling it per query would be hours of identical work; `without_player`
    is the cheap per-query half.
    """
    normalized = era_normalize(raw_panel, kind, sgp_overrides=sgp_overrides)
    return normalized[normalized["season"] <= base_year].copy()


def without_player(panel: pd.DataFrame, query_id: int) -> pd.DataFrame:
    """The panel both estimators see for one query: no self-matching.

    Cheap by design and called in the inner loop. An in-sample comparison would flatter
    `shape`, which fits a model, over an estimator that averages.
    """
    return panel[panel["mlbam_id"] != query_id]


def keeper_value_sgp(
    frame: pd.DataFrame, kind: str, sgp_overrides: SgpOverrides | None
) -> pd.Series:
    """SGP for a keeper-value forecast frame, scored by the PANEL's own scorer.

    This is what puts the two estimators on one scale rather than merely in one unit.
    `forecast_pool` emits the canonical rate/PT schema -- `keepers.actuals.HITTER_PT`
    is literally `"pa"` and `HITTER_RATES` are character-for-character the columns
    `panel.score` reconstructs from -- so the forecast can be handed to the scorer the
    realized seasons went through, with no translation step to disagree about.

    Deliberately NOT via `keeper_forecast.to_counting`, which renames to `PA`/`IP` and
    finishes `AVG`/`ERA`/`WHIP`: that output is for display and would need a second
    scoring path.
    """
    scored = panel_score(frame.reset_index(), kind, sgp_overrides)
    return scored.set_index("mlbam_id")["sgp"]


def var_for(
    sgp_by_id: pd.Series,
    kind: str,
    base_year: int,
    cache_dir: Path,
    levels: dict[str, float],
) -> pd.Series:
    """SGP above the position-aware floor, using year-`base_year` eligibility.

    Year Y, not the outcome year: Y is the information set the keeper decision actually
    has, and outcome-year eligibility would be hindsight. The catcher-to-outfield floor
    spread is 2.3 SGP a year, larger than the margins this backtest is trying to
    resolve, so the choice is not cosmetic.

    Reuses `trajectory.board.season_slots` rather than re-reading the fielding cache: a
    second eligibility path would be free to price the backtest's catchers differently
    from the live board's. Its existing fallback carries through -- a missing or corrupt
    cache degrades to UTIL, the HIGHEST hitter floor, so an unknown player is
    understated rather than credited with scarcity he may not have.
    """
    eligibility = season_slots(cache_dir, base_year)
    floors = {}
    for pid in sgp_by_id.index:
        slots = resolve_slots(set(eligibility.get(int(pid), frozenset())), kind)
        floors[pid] = best_floor(slots, levels)[1]
    return sgp_by_id - pd.Series(floors, dtype=float).reindex(sgp_by_id.index)


def roles(panel: pd.DataFrame) -> pd.Series:
    """``(mlbam_id, season) -> "SP" / "closer" / "RP"``.

    #313 asks for the pitcher result split by role, because a closer's SGP is
    saves-dominated and saves are a job rather than a skill: a pooled pitcher number can
    average two opposite effects into a null.

    The cuts are BORROWED, not invented. `STARTER_SHARE` is the same `starts / games`
    split `trajectory.value` routes a pitcher's replacement floor on, and
    `CLOSER_SV_THRESHOLD` is the same save count the draft board buckets closers at. A
    third rule defined here would be one more thing to disagree with them.

    **Pass the RAW panel, not the era-normalized one.** `era_normalize` rescales
    `sv_ip` and `panel.score` then rebuilds `sv` from it, so a 20-save threshold on a
    normalized frame is a threshold on restated saves -- which is meaningless, because
    a closer is a JOB and 20 saves is a count of real ones. Measured on the live panel,
    that mistake moves 8 of 17,947 seasons across the bucket line. Refused below rather
    than documented, since the two frames are otherwise interchangeable to look at.
    """
    normalized = [c for c in panel.columns if c.startswith("era_factor_")]
    if normalized:
        raise ValueError(
            "roles() needs the RAW panel: this frame is era-normalized "
            f"(carries {normalized[:3]}...), so its `sv` has been restated into the "
            "reference run environment and a 20-save cut no longer means 20 saves. "
            "Pass the frame from load_scored_panel, before era_normalize."
        )
    missing = [c for c in _ROLE_SUMS if c not in panel.columns]
    if missing:
        raise KeyError(f"pitcher panel is missing role columns {missing}")
    agg = panel.groupby(["mlbam_id", "season"])[list(_ROLE_SUMS)].sum()
    games = agg["games"].to_numpy(dtype=float)
    starts = agg["starts"].to_numpy(dtype=float)
    saves = agg["sv"].to_numpy(dtype=float)
    # games == 0 cannot be a starter; guard the divide rather than letting it warn.
    share = np.divide(starts, games, out=np.zeros_like(starts), where=games > 0)
    bucket = np.where(
        share >= STARTER_SHARE, "SP", np.where(saves >= CLOSER_SV_THRESHOLD, "closer", "RP")
    )
    return pd.Series(bucket, index=agg.index, name="role")


def score(
    panel: pd.DataFrame,
    queries: pd.DataFrame,
    kind: str,
    horizon: int,
    role_by_season: pd.Series | None = None,
) -> pd.DataFrame:
    """Predict `horizon` years ahead for each query, with that player held out."""
    index = panel.set_index(["mlbam_id", "season"])["sgp"]
    rows = []
    for i, q in enumerate(queries.itertuples(index=False), start=1):
        if i % 100 == 0:
            print(f"  {i}/{len(queries)}...", flush=True)
        actual = float(index.get((q.mlbam_id, q.season + horizon), 0.0))
        # No self-matching: the player is gone from the panel both estimators see.
        clean = panel[panel["mlbam_id"] != q.mlbam_id]
        age, current, prior = int(q.age), float(q.current), float(q.prior)
        level = comp_trajectory(clean, kind=kind, age=age, sgp=current, horizons=(horizon,))
        curve, _ = shape_trajectory(
            clean, kind=kind, age=age, sgp=current, prior_sgp=prior, horizons=(horizon,)
        )
        if level.path[0].n == 0 or np.isnan(curve.path[0].mean):
            continue
        # `track` is `current` plus a HARD band on the prior season (#305) -- the same
        # two anchors shape uses, bounded instead of kernel-weighted. Passing prior_sgp
        # is what selects it; `comp_trajectory` defaults to level matching without it.
        #
        # Fitted AFTER the guard so a row that is about to be discarded does not pay for
        # a third full-panel scan. Its own emptiness is deliberately NOT part of that
        # guard: the two-mode comparison was already published from this harness, and
        # dropping rows track cannot score would silently change the current-vs-shape
        # population. Track records NaN there and is reported on its own defined subset.
        tracked = comp_trajectory(
            clean, kind=kind, age=age, sgp=current, prior_sgp=prior, horizons=(horizon,)
        )
        rows.append(
            {
                "mlbam_id": q.mlbam_id,
                "season": q.season,
                "age": age,
                "prior": prior,
                "now": current,
                "actual": actual,
                "current": level.path[0].mean,
                "shape": curve.path[0].mean,
                "track": (float("nan") if tracked.path[0].n == 0 else tracked.path[0].mean),
                # The role of the QUERY season -- the one both anchors describe.
                "role": (
                    role_by_season.get((q.mlbam_id, q.season), "")
                    if role_by_season is not None
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str) -> dict | None:
    if len(df) < 10:
        # Say so rather than printing nothing. A slice that silently vanishes reads as
        # "not applicable" when it means "too thin to measure" -- which for the role
        # splits in #313 is itself the finding.
        print(f"  {label:30s} n={len(df):4d}   (under 10, not reported)")
        return None
    out = {}
    for mode in ("current", "shape"):
        err = df[mode] - df["actual"]
        out[mode] = (float(np.sqrt((err**2).mean())), float(err.abs().mean()), float(err.mean()))
    wins = float(((df["shape"] - df["actual"]).abs() < (df["current"] - df["actual"]).abs()).mean())
    print(
        f"  {label:30s} n={len(df):4d}   "
        f"RMSE {out['current'][0]:5.2f} -> {out['shape'][0]:5.2f}   "
        f"MAE {out['current'][1]:5.2f} -> {out['shape'][1]:5.2f}   "
        f"bias {out['current'][2]:+5.2f} -> {out['shape'][2]:+5.2f}   "
        f"shape wins {wins:.0%}"
    )
    return {"slice": label, "n": len(df), "wins": wins}


def report_track(df: pd.DataFrame, label: str) -> None:
    """Three-way on the subset where `track` found any comps.

    Separate from `report` on purpose. `track`'s hard prior band leaves some queries
    with an empty cohort, and folding those drops into the shared row filter would move
    the current-vs-shape population that was already measured and published. So the
    three-way runs on track's own defined subset, and the coverage is printed rather
    than left for the reader to infer from a shrinking n.
    """
    defined = df.dropna(subset=["track"])
    coverage = f"{len(defined)}/{len(df)}"
    if len(defined) < 10:
        print(f"  {label:30s} track scored {coverage:>9}   (under 10, not reported)")
        return
    stats = {}
    for mode in ("current", "track", "shape"):
        err = defined[mode] - defined["actual"]
        stats[mode] = (float(np.sqrt((err**2).mean())), float(err.mean()))
    beats_track = float(
        (
            (defined["shape"] - defined["actual"]).abs()
            < (defined["track"] - defined["actual"]).abs()
        ).mean()
    )
    print(
        f"  {label:30s} track scored {coverage:>9}   "
        f"RMSE cur {stats['current'][0]:5.2f} / track {stats['track'][0]:5.2f} / "
        f"shape {stats['shape'][0]:5.2f}   "
        f"bias track {stats['track'][1]:+5.2f} shape {stats['shape'][1]:+5.2f}   "
        f"shape beats track {beats_track:.0%}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=("hitter", "pitcher"), default="hitter")
    parser.add_argument("--horizon", type=int, default=1, help="years ahead to predict")
    parser.add_argument(
        "--elite-floor",
        type=float,
        default=14.0,
        help="prior-season SGP at or above which a query counts as elite",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="score a RANDOM sample of this size instead of every elite season",
    )
    parser.add_argument("--min-age", type=int, default=24)
    parser.add_argument("--max-age", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--out", type=Path, help="write the scored queries to this CSV")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")

    if not args.panel_dir.is_absolute():
        args.panel_dir = PROJECT_ROOT / args.panel_dir

    overrides = load_config(PROJECT_ROOT / "config" / "league.yaml").sgp_overrides
    # Kept separately: the estimators want the era-normalized frame, but `roles` needs
    # the raw one -- see its docstring. Everything below reads `panel` except that one
    # call.
    raw_panel = load_scored_panel(args.pool, panel_dir=args.panel_dir, sgp_overrides=overrides)
    panel = era_normalize(raw_panel, args.pool, sgp_overrides=overrides)
    last = int(panel["season"].max())

    # `build_history` supplies both anchors and censors seasons whose prior predates
    # the panel -- the same rows a real query would have.
    pool = build_history(panel)
    pool = pool[
        pool["age"].between(args.min_age, args.max_age) & (pool["season"] + args.horizon <= last)
    ]
    if args.sample:
        queries = pool.sample(min(args.sample, len(pool)), random_state=args.seed)
        header = f"random sample of {len(queries)}"
    else:
        queries = pool[pool["prior"] >= args.elite_floor]
        header = f"every season with a prior >= {args.elite_floor:g} SGP"

    print(
        f"{args.pool.upper()}S, +{args.horizon}: {header}, ages "
        f"{args.min_age}-{args.max_age}, {len(queries)} queries\n"
    )
    role_by_season = roles(raw_panel) if args.pool == "pitcher" else None
    df = score(panel, queries, args.pool, args.horizon, role_by_season)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    print(f"\nscored {len(df)} (query player held out of the panel each time)")
    print(f"{'':32s}       current -> shape")
    report(df, "ALL")
    elite = df[df["prior"] >= args.elite_floor]
    report(elite, f"elite (prior >= {args.elite_floor:g})")
    report(elite[elite["now"] < elite["prior"] * 0.8], "elite down year (<80% of prior)")
    report(elite[elite["now"] < elite["prior"] * 0.7], "elite big drop (<70% of prior)")
    report(elite[elite["now"] >= elite["prior"] * 0.8], "elite holding steady")
    report(df[df["now"] > df["prior"] * 1.25], "breakout (up >25%)")

    # The two-mode table above races shape against LEVEL matching only. `track` uses the
    # same two anchors shape does, so it is the closer competitor -- and retiring it
    # (#325) without ever racing it would be retiring an unmeasured alternative.
    print("\n  -- three-way, including track (hard prior band) --")
    report_track(df, "ALL")
    report_track(elite, f"elite (prior >= {args.elite_floor:g})")
    report_track(elite[elite["now"] < elite["prior"] * 0.7], "elite big drop (<70% of prior)")
    report_track(elite[elite["now"] >= elite["prior"] * 0.8], "elite holding steady")

    if args.pool == "pitcher":
        # #313: a pooled pitcher number can average a starter effect and a closer effect
        # into a null, so the roles are reported separately rather than trusted to agree.
        print("\n  -- by role of the query season --")
        for role in ("SP", "RP", "closer"):
            report(df[df["role"] == role], f"{role}")
            report(df[(df["role"] == role) & (df["prior"] >= args.elite_floor)], f"{role} elite")
        # 15% of pitcher-seasons score below replacement against 7.7% for hitters, and
        # the linear form was never checked against a negative anchor.
        print("\n  -- negative anchors --")
        report(df[(df["now"] < 0) | (df["prior"] < 0)], "either anchor negative")
        report(df[df["now"] < 0], "current season negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
