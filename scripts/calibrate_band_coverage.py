"""Is the trajectory band honest on the rows the board flags with `(!)`?

`trajectory_board.py` tells the reader to trust the BAND over the point estimate on an
extrapolated row. That instruction rests on a coverage claim -- "measured on breakouts
the interval is calibrated for hitters" -- which appears in exactly one place, a comment
string, and whose tooling (`scripts/backtest_trajectory.py`) was deleted in #325. The
coverage table that DOES carry a sample size (`model.PathPoint.p10`, n=729 hitters at
h=3) is general-population: 445 of the live board's 602 hitters sit above 30% local
support, so a number measured over everyone says almost nothing about the six below 5%.

This script measures tail coverage SLICED BY `local_support`, which is the slice the
`(!)` flag is drawn on and the one nobody has measured. Each tail should hold 10%. A
low-support bucket at 10%/10% means the flagged bands are honest and an extrapolated row
can be ranked on its band; a bucket at 25% means they are too narrow and every `(!)` row
needs a haircut before it competes with a supported one.

WHAT IS MEASURED

    per horizon   the band on one forward season, as `PathPoint.p10`/`p90` report it
    3-year sum    p10 and p90 SUMMED across h=1,2,3, which is what `sweep.totals` does
                  and therefore what the board's headline 2027-29 column prints

The second is the decision-relevant one and it is not the same question. `totals` sums
the yearly bands, which assumes the three years move together; the docstring says so and
calls it "the conservative direction". Whether that is conservative by 2 points or by 20
is measurable, and it has not been measured either.

METHOD, matching `tune_shape_windows.py` so the two cannot disagree

  * The query player is held out of the panel for his WHOLE career -- every season of
    his is a query and they all want the same panel, so `prepare` is hoisted above him.
  * `last_complete_season` stays the FULL panel's max, so dropping a player never moves
    the censoring cutoff for everybody else.
  * A forward season the player did not play is a real 0, not a missing value. See
    `model.played`: SGP is genuinely negative for a below-replacement season, so testing
    `> 0` would file every one of those as a career ending.
  * `bootstrap_draws` defaults to `sweep.SWEEP_DRAWS` (250), the board's own setting, so
    what is calibrated here is the band the board actually prints.

RAW SGP, NOT VAR, and that is not a shortcut. `shape_trajectory` documents `y_var =
y_sgp - replacement` as affine with the intercept column exactly the vector subtracted,
so `mean`, `median`, `p10` and `p90` are all "the raw fit's minus the floor" (#331). The
actual shifts by the same floor. A constant subtracted from the prediction, both band
edges and the outcome cannot change which side of an edge the outcome landed on, so tail
coverage is IDENTICAL on the two scales -- and running on SGP avoids having to
reconstruct per-season slot eligibility back to 2000, which the panel does not carry.

Coverage estimates come with a Wilson interval, because the whole point is the thin
bucket: `<5%` support may hold a few dozen rows, and a bare "24%" off n=30 is exactly the
kind of number that would let this script repeat the mistake it was written to correct.

    python scripts/calibrate_band_coverage.py                      # hitters, 1500 queries
    python scripts/calibrate_band_coverage.py --pool pitcher
    python scripts/calibrate_band_coverage.py --pool both --sample 0 --out cov.csv
    python scripts/calibrate_band_coverage.py --from-csv cov.csv   # re-analyse, no re-run

Build the panel first (one time, ~1 minute):
    python scripts/build_pt_panel.py --start 2000 --end 2026 --out-dir data/trajectory
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from fantasy_baseball.config import load_config
from fantasy_baseball.trajectory.model import MIN_LOCAL_SUPPORT
from fantasy_baseball.trajectory.panel import DEFAULT_PANEL_DIR, load_scored_panel
from fantasy_baseball.trajectory.shape import (
    build_history,
    collapsed_index,
    prepare,
    shape_trajectory,
)
from fantasy_baseball.trajectory.sweep import SWEEP_DRAWS

#: Support buckets. The edge at `MIN_LOCAL_SUPPORT` is the one that matters -- it is the
#: `(!)` threshold -- and it is split in two because 3% and 9% are both flagged and are
#: not obviously the same population. Above it, two buckets are enough to show a trend.
BUCKETS = (0.0, 0.05, MIN_LOCAL_SUPPORT, 0.30, 1.01)
BUCKET_LABELS = ("<5% (!)", "5-10% (!)", "10-30%", ">30%")

#: Nominal tail mass on each side. `shape` reads the band off the 0.10/0.90 weighted
#: residual quantiles, so each tail should hold this share of outcomes.
NOMINAL = 0.10


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share.

    Not `p +/- z*sqrt(p(1-p)/n)`: the normal interval is badly wrong at the small n and
    near-0.1 p this script exists to report, and can hand back a negative lower bound,
    which would print as a coverage rate below zero.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    d = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def score_pool(
    panel: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    kind: str,
    horizons: tuple[int, ...],
    draws: int,
) -> pd.DataFrame:
    """One row per (query, horizon): band edges, the outcome, and the query's support."""
    _, index = collapsed_index(panel)
    last = int(panel["season"].max())

    records: list[tuple] = []
    players = list(queries.groupby("mlbam_id", sort=False))
    scored = 0
    started = time.time()
    for done, (mlbam_id, seasons) in enumerate(players, start=1):
        if done % 100 == 0:
            rate = (time.time() - started) / max(scored, 1)
            print(
                f"  {done}/{len(players)} players, {scored}/{len(queries)} queries, "
                f"{len(records)} rows, eta {rate * (len(queries) - scored):.0f}s",
                flush=True,
            )
        # Held out once for the whole career, `last` from the FULL panel -- see METHOD.
        clean = panel[panel["mlbam_id"] != mlbam_id]
        prepared = prepare(clean, kind=kind, horizons=horizons, last_complete_season=last)
        for q in seasons.itertuples(index=False):
            observable = tuple(h for h in horizons if q.season + h <= last)
            if not observable:
                continue
            scored += 1
            curve, _ = shape_trajectory(
                prepared,
                kind=kind,
                age=int(q.age),
                sgp=float(q.current),
                prior_sgp=float(q.prior),
                horizons=observable,
                last_complete_season=last,
                bootstrap_draws=draws,
            )
            for point in curve.path:
                if point.n == 0:
                    # No comps at this horizon: `_empty_point`, carrying no estimate.
                    continue
                actual = float(index.get((q.mlbam_id, q.season + point.horizon), 0.0))
                records.append(
                    (
                        kind,
                        q.mlbam_id,
                        q.season,
                        int(q.age),
                        float(q.prior),
                        float(q.current),
                        point.horizon,
                        curve.local_support,
                        point.mean,
                        point.p10,
                        point.p90,
                        actual,
                        point.band_fell_back,
                    )
                )
    return pd.DataFrame(
        records,
        columns=[
            "pool",
            "mlbam_id",
            "season",
            "age",
            "prior",
            "now",
            "horizon",
            "support",
            "predicted",
            "p10",
            "p90",
            "actual",
            "band_fell_back",
        ],
    )


def tails(df: pd.DataFrame) -> dict:
    """Both tail shares for one slice, with Wilson intervals and the bucket's n."""
    n = len(df)
    below = int((df["actual"] < df["p10"]).sum())
    above = int((df["actual"] > df["p90"]).sum())
    lo_b, hi_b = wilson(below, n)
    lo_a, hi_a = wilson(above, n)
    return {
        "n": n,
        "below_p10": below / n if n else float("nan"),
        "below_ci": (lo_b, hi_b),
        "above_p90": above / n if n else float("nan"),
        "above_ci": (lo_a, hi_a),
        "med_width": float((df["p90"] - df["p10"]).median()) if n else float("nan"),
    }


def bucketed(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage by support bucket, long format, one row per bucket."""
    out = []
    cut = pd.cut(df["support"], BUCKETS, labels=BUCKET_LABELS, include_lowest=True)
    for label in BUCKET_LABELS:
        sub = df[cut == label]
        out.append({"bucket": label, **tails(sub)})
    return pd.DataFrame(out)


def report(df: pd.DataFrame, title: str) -> None:
    """Print one coverage table. Each tail should read 10%."""
    print(f"\n{title}")
    print("-" * len(title))
    if df.empty or df["n"].sum() == 0:
        print("  (no rows)")
        return
    print(
        f"  {'support':<11}{'n':>6}  {'below p10':>10} {'95% CI':>14}   "
        f"{'above p90':>10} {'95% CI':>14}  {'med width':>9}"
    )
    for r in df.itertuples(index=False):
        if not r.n:
            print(f"  {r.bucket:<11}{0:>6}  {'--':>10}")
            continue
        bci = f"{r.below_ci[0]:.0%}-{r.below_ci[1]:.0%}"
        aci = f"{r.above_ci[0]:.0%}-{r.above_ci[1]:.0%}"
        # Flag a tail whose interval EXCLUDES the nominal 10%, which is the only sense in
        # which a bucket is measurably miscalibrated rather than noisily off.
        bad_b = "*" if r.below_ci[0] > NOMINAL or r.below_ci[1] < NOMINAL else " "
        bad_a = "*" if r.above_ci[0] > NOMINAL or r.above_ci[1] < NOMINAL else " "
        print(
            f"  {r.bucket:<11}{r.n:>6}  {r.below_p10:>9.0%}{bad_b} {bci:>14}   "
            f"{r.above_p90:>9.0%}{bad_a} {aci:>14}  {r.med_width:>9.1f}"
        )
    print("    * = 95% interval excludes the nominal 10%, i.e. measurably miscalibrated")


def three_year(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Per-query totals over ALL of `horizons`, bands summed the way `sweep.totals` sums.

    Queries missing any horizon are dropped rather than summed short: a two-year sum and
    a three-year sum are not the same quantity and averaging their coverage together
    would report neither.
    """
    wanted = set(horizons)
    keep = df[df["horizon"].isin(wanted)]
    grouped = keep.groupby(["mlbam_id", "season"], sort=False)
    full = grouped["horizon"].transform("nunique") == len(wanted)
    keep = keep[full]
    if keep.empty:
        return keep
    return keep.groupby(["mlbam_id", "season"], sort=False).agg(
        pool=("pool", "first"),
        support=("support", "first"),
        predicted=("predicted", "sum"),
        p10=("p10", "sum"),
        p90=("p90", "sum"),
        actual=("actual", "sum"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=("hitter", "pitcher", "both"), default="hitter")
    parser.add_argument(
        "--sample",
        type=int,
        default=1500,
        help="score this many queries per pool, drawn at random; 0 means every query",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--draws",
        type=int,
        default=SWEEP_DRAWS,
        help="bootstrap refits per query; the default is the board's own setting",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--out", type=Path, help="write the scored rows here")
    parser.add_argument("--from-csv", type=Path, help="re-analyse a previous --out, no re-run")
    args = parser.parse_args()
    if not args.panel_dir.is_absolute():
        args.panel_dir = PROJECT_ROOT / args.panel_dir
    horizons = tuple(sorted(args.horizons))

    if args.from_csv:
        scored = pd.read_csv(args.from_csv)
        print(f"re-analysing {len(scored)} rows from {args.from_csv}")
    else:
        config = load_config(PROJECT_ROOT / "config" / "league.yaml")
        pools = ["hitter", "pitcher"] if args.pool == "both" else [args.pool]
        frames = []
        for kind in pools:
            panel = load_scored_panel(
                kind, panel_dir=args.panel_dir, sgp_overrides=config.sgp_overrides
            )
            history = build_history(panel)
            queries = history
            if args.sample and args.sample < len(history):
                queries = history.sample(args.sample, random_state=args.seed)
            # Grouped by player so the held-out `prepare` is paid once per career.
            queries = queries.sort_values("mlbam_id")
            print(
                f"\n{kind}: {len(panel)} panel rows, {len(queries)} queries "
                f"({queries['mlbam_id'].nunique()} players), horizons {horizons}, "
                f"{args.draws} draws",
                flush=True,
            )
            started = time.perf_counter()
            frames.append(
                score_pool(panel, queries, kind=kind, horizons=horizons, draws=args.draws)
            )
            print(f"  scored in {time.perf_counter() - started:.0f}s", flush=True)
        scored = pd.concat(frames, ignore_index=True)
        if args.out:
            scored.to_csv(args.out, index=False)
            print(f"\n  wrote {len(scored)} rows to {args.out}")

    for kind, pool_rows in scored.groupby("pool", sort=False):
        span = f"{min(horizons)}-{max(horizons)}"
        for h in sorted(pool_rows["horizon"].unique()):
            report(bucketed(pool_rows[pool_rows["horizon"] == h]), f"{kind}, horizon +{h}")
        totals = three_year(pool_rows, horizons)
        if not totals.empty:
            report(bucketed(totals), f"{kind}, SUMMED h{span} (what the board prints)")

    print(
        "\n  Each tail should hold 10%. A LOW below-p10 share means the band's downside is"
        "\n  too generous -- outcomes land inside it more often than advertised, so the"
        "\n  interval is WIDER than it needs to be. A HIGH share means it is too narrow"
        "\n  and the row is riskier than the board shows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
