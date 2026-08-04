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

from fantasy_baseball.config import load_config
from fantasy_baseball.trajectory.comps import comp_trajectory
from fantasy_baseball.trajectory.era import era_normalize
from fantasy_baseball.trajectory.panel import DEFAULT_PANEL_DIR, load_scored_panel
from fantasy_baseball.trajectory.shape import build_history, shape_trajectory
from fantasy_baseball.trajectory.value import STARTER_SHARE
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD

#: Columns the role bucket needs, and the rule for a season split across two rows.
#: `collapse_split_seasons` keeps only `sgp` and `age`, so a traded pitcher's counting
#: columns have to be re-summed here or a mid-season trade reads as two half-roles --
#: the same reason `trajectory.board._SPLIT_RULES` re-sums `starts`/`games`.
_ROLE_SUMS = ("starts", "games", "sv")


def roles(panel: pd.DataFrame) -> pd.Series:
    """``(mlbam_id, season) -> "SP" / "closer" / "RP"``.

    #313 asks for the pitcher result split by role, because a closer's SGP is
    saves-dominated and saves are a job rather than a skill: a pooled pitcher number can
    average two opposite effects into a null.

    The cuts are BORROWED, not invented. `STARTER_SHARE` is the same `starts / games`
    split `trajectory.value` routes a pitcher's replacement floor on, and
    `CLOSER_SV_THRESHOLD` is the same save count the draft board buckets closers at. A
    third rule defined here would be one more thing to disagree with them.
    """
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
    panel = era_normalize(
        load_scored_panel(args.pool, panel_dir=args.panel_dir, sgp_overrides=overrides),
        args.pool,
        sgp_overrides=overrides,
    )
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
    role_by_season = roles(panel) if args.pool == "pitcher" else None
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
