"""Select shape's kernel half-widths by held-out prediction error (#310).

`shape.AGE_WINDOW` (2 years) and `shape.PRIOR_WINDOW` (8 SGP) were CHOSEN. They are the
only two knobs that decide which seasons inform a query, so "chosen" means the estimator's
neighbourhood was never measured -- and #310 says so explicitly. This script sweeps a grid
of both and reports out-of-sample error, so the defaults are a measurement rather than a
guess and a later change to either has a number to beat.

**The query player is held out of the panel**, exactly as `backtest_trajectory.py` does it,
so the fit can never match him to himself. Held out ONCE PER PLAYER rather than per query:
every one of his seasons is a query and every one of them wants the same panel, so the
`prepare` that dominates a naive loop is hoisted above his whole career (#311's state,
reused here for a different reason).

**Complete cases only.** A narrow kernel refuses more queries -- `MIN_EFFECTIVE_ROWS`
gates on the effective sample size, and a tight window starves it -- and the ones it
refuses are the thin, hard ones. Scoring each grid point on whatever it happened to answer
would hand the narrowest setting the easiest rows and call it accurate. So error is
computed on the intersection: the (player, season, horizon) rows EVERY grid point answered.
Coverage is reported separately, because refusing to answer is a real cost and belongs in
the decision rather than hidden inside the RMSE.

**Selection is cross-validated by player.** Picking the grid's argmin and then quoting its
error is the same error twice -- the winner is partly fit to the sample it won on. Folds
are cut by player (never by row: a player's seasons are correlated, and splitting them
across folds leaks him into his own training set), the argmin is taken on the other folds,
and the held-out fold is scored at that choice. What that comparison buys is the honest
question: does tuning beat the shipped default on data neither of them saw.

**Every cell carries a confidence, not just a rank.** The surface is flat enough that an
argmin means very little on its own -- 0.1% below the default and 3% below it print the
same way -- so each grid point is compared to the shipped setting by a paired bootstrap
resampled by player, and the report says which cells are actually distinguishable. On the
hitter panel that turns "8 is the winner" into the more useful "anything from 4 to 16 is
the same answer, and turning the kernel off costs 1.7% on the elite slice".

Usage:
    python scripts/tune_shape_windows.py                       # hitters, default grid
    python scripts/tune_shape_windows.py --pool pitcher         # the #313 pool
    python scripts/tune_shape_windows.py --horizons 1 2 3 4 5   # longer horizons
    python scripts/tune_shape_windows.py --sample 0 --out preds.csv
    python scripts/tune_shape_windows.py --from-csv preds.csv  # re-analyse, no re-sweep
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
from fantasy_baseball.trajectory.era import era_normalize
from fantasy_baseball.trajectory.panel import DEFAULT_PANEL_DIR, load_scored_panel
from fantasy_baseball.trajectory.shape import (
    AGE_WINDOW,
    PRIOR_WINDOW,
    build_history,
    collapsed_index,
    prepare,
    shape_trajectory,
)

#: Half-widths swept by default. Wide enough on both ends that an argmin landing on an
#: edge is visible as an edge rather than being mistaken for an interior optimum -- the
#: report says so when it happens.
#:
#: The top of each range is deliberately past anything anyone would ship: age 10 spans
#: the whole panel once `age_window + 1` is applied, and 100 SGP is wider than the entire
#: observed spread. Those cells are the kernel switched OFF, which is the control the
#: question needs -- if error barely moves between a tight window and no window at all,
#: the answer is that the two linear anchors are doing the work and the neighbourhood is
#: nearly irrelevant, which is a different finding from "8 is the right width".
DEFAULT_AGE_WINDOWS = (1, 2, 3, 4, 6, 10)
DEFAULT_PRIOR_WINDOWS = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 40.0, 100.0)

#: The bootstrap drives `PathPoint.se`, `spread` and the p10/p90 band. None of the three
#: touch `mean`, which is the only column scored here, and the bootstrap is ~90% of a
#: query's cost -- so it is turned down to the minimum the API accepts rather than paid
#: for once per grid point per query.
TUNING_DRAWS = 2

#: Memory the plateau bootstrap's working arrays may occupy, which sets its batch size at
#: `PLATEAU_BYTES // (24 * players)`. Same budget and same reasoning as
#: `shape.BOOTSTRAP_BYTES`: a batch is pure vectorization width, so trading it away costs
#: a little speed and nothing else -- the draws, and therefore the answer, are unchanged.
PLATEAU_BYTES = 32 * 1024 * 1024


def _folds(ids: np.ndarray, folds: int) -> np.ndarray:
    """Assign each row a fold from its PLAYER id, so a career never spans two folds.

    `id % folds` rather than a shuffle: deterministic without carrying a second seed, and
    mlbam ids have no structure that aligns with talent.
    """
    return ids % folds


def _n_queries(df: pd.DataFrame) -> int:
    """Query-horizons behind a slice. `len(df)` counts each one once per GRID POINT."""
    return int(df.groupby(["mlbam_id", "season", "horizon"], sort=False).ngroups)


def score_grid(
    panel: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    kind: str,
    horizons: tuple[int, ...],
    age_windows: tuple[int, ...],
    prior_windows: tuple[float, ...],
) -> pd.DataFrame:
    """Predict every query at every grid point, with the query's player held out.

    Long format -- one row per (query, horizon, age_window, prior_window) -- because the
    slicing afterwards is by all four and a wide frame would have to be melted anyway.
    """
    _, index = collapsed_index(panel)
    last = int(panel["season"].max())
    grid = [(aw, pw) for aw in age_windows for pw in prior_windows]

    records: list[tuple] = []
    players = list(queries.groupby("mlbam_id", sort=False))
    scored = 0
    started = time.time()
    for done, (mlbam_id, seasons) in enumerate(players, start=1):
        if done % 250 == 0:
            # Extrapolate on QUERIES, not players. The elite slice is concatenated first
            # and elite players have long careers, so the leading groups carry ~5 seasons
            # against an overall average near 2 -- a per-player rate read off them put the
            # ETA at 24 minutes on a 12-minute run.
            rate = (time.time() - started) / max(scored, 1)
            print(
                f"  {done}/{len(players)} players, {scored}/{len(queries)} queries, "
                f"{len(records)} rows, eta {rate * (len(queries) - scored):.0f}s",
                flush=True,
            )
        # Held out once for the whole career: every season of his is a query and they all
        # want the same panel. `last` stays the FULL panel's, so dropping a player cannot
        # move the censoring cutoff for everyone else.
        clean = panel[panel["mlbam_id"] != mlbam_id]
        prepared = prepare(clean, kind=kind, horizons=horizons, last_complete_season=last)
        for q in seasons.itertuples(index=False):
            observable = tuple(h for h in horizons if q.season + h <= last)
            if not observable:
                continue
            scored += 1
            actual = {h: float(index.get((q.mlbam_id, q.season + h), 0.0)) for h in observable}
            for age_window, prior_window in grid:
                curve, _ = shape_trajectory(
                    prepared,
                    kind=kind,
                    age=int(q.age),
                    sgp=float(q.current),
                    prior_sgp=float(q.prior),
                    horizons=observable,
                    age_window=age_window,
                    prior_window=prior_window,
                    last_complete_season=last,
                    bootstrap_draws=TUNING_DRAWS,
                )
                for point in curve.path:
                    records.append(
                        (
                            kind,
                            q.mlbam_id,
                            q.season,
                            int(q.age),
                            float(q.prior),
                            float(q.current),
                            point.horizon,
                            age_window,
                            prior_window,
                            point.mean,
                            actual[point.horizon],
                        )
                    )
    return pd.DataFrame(
        records,
        columns=[
            # Carried on every row so `--from-csv` can REFUSE a pool it was not swept on.
            # Nothing downstream reads it; it exists because the analysis is pool-blind and
            # would otherwise print pitcher predictions under a HITTERS header. Same failure
            # `shape.Prepared.kind` was added to stop, one layer out.
            "pool",
            "mlbam_id",
            "season",
            "age",
            "prior",
            "now",
            "horizon",
            "age_window",
            "prior_window",
            "predicted",
            "actual",
        ],
    )


def complete_cases(df: pd.DataFrame, n_grid: int) -> tuple[pd.DataFrame, float]:
    """Rows every grid point answered, and the share of rows that survived.

    A query the `MIN_EFFECTIVE_ROWS` gate refused comes back as NaN. Dropping those per
    grid point would score each setting on a different population -- and systematically
    kinder to the narrow ones, whose refusals are exactly the thin queries.
    """
    answered = df.dropna(subset=["predicted"])
    counts = answered.groupby(["mlbam_id", "season", "horizon"]).size()
    keep = counts[counts == n_grid].index
    if not len(keep):
        return answered.iloc[:0], 0.0
    key = pd.MultiIndex.from_arrays([answered["mlbam_id"], answered["season"], answered["horizon"]])
    complete = answered[key.isin(keep)]
    return complete, len(keep) / max(len(counts), 1)


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of query-horizons each grid point actually answered."""
    return (
        df.assign(answered=df["predicted"].notna().astype(float))
        .pivot_table(index="prior_window", columns="age_window", values="answered", aggfunc="mean")
        .sort_index()
    )


def error_table(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE at each grid point, prior_window down the rows and age_window across."""
    return (
        df.assign(sq=(df["predicted"] - df["actual"]) ** 2)
        .pivot_table(index="prior_window", columns="age_window", values="sq", aggfunc="mean")
        .pipe(np.sqrt)
        .sort_index()
    )


def plateau(
    df: pd.DataFrame, *, draws: int = 2000, seed: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each grid point's RMSE gap to the shipped default, and how sure that gap is.

    The grid alone cannot answer the question that matters. An argmin 0.1% below the
    default and an argmin 3% below it print the same way, and reading a winner off a
    surface this flat is reading noise -- so every cell is compared to the default by a
    cluster bootstrap and the answer comes back as a confidence, not a ranking.

    Resampled by PLAYER, matching `cross_validate`'s fold cut and for the same reason: his
    seasons share whatever the model gets wrong about him, so resampling rows would treat
    ~11k correlated observations as independent and shrink every interval.

    Paired -- both cells are scored on the resampled players, so the comparison differences
    out the draw's difficulty and measures only the setting. The bootstrap runs over all 60
    grid points at once as a single matrix product, off per-player squared-error sums, which
    is exact: complete cases guarantee every player contributes the same query-horizons at
    every grid point, so the sums are comparable across columns by construction.

    Returns (delta, confidence): RMSE minus the default's, so POSITIVE is worse than the
    default; and the share of draws in which the default came out ahead. ~0.5 means the two
    are indistinguishable, which on this surface is the common case and the whole point.
    """
    squared = df.assign(sq=(df["predicted"] - df["actual"]) ** 2)
    sums = squared.pivot_table(
        index="mlbam_id", columns=["prior_window", "age_window"], values="sq", aggfunc="sum"
    )
    at_default = squared[
        (squared["age_window"] == AGE_WINDOW) & (squared["prior_window"] == PRIOR_WINDOW)
    ]
    counts = at_default.groupby("mlbam_id").size().reindex(sums.index).to_numpy(dtype=float)

    n = len(sums)
    per_player = sums.to_numpy()
    rng = np.random.default_rng(seed)
    # BATCHED over draws, under a byte budget, for the reason `shape.BOOTSTRAP_BYTES`
    # exists: the offset-bincount idiom borrowed below holds three (batch, n) arrays alive
    # at once, so a fixed batch grows without bound in the number of players. At the
    # shipped defaults that is ~29 MB and unbatched would be fine; at `--sample 0
    # --draws 10000` it is half a gigabyte. The batch is pure vectorization width -- draws
    # are consumed in the same order at any width, so the answer does not depend on it.
    chunk = max(1, min(draws, PLATEAU_BYTES // (24 * n)))
    rmse = np.empty((draws, per_player.shape[1]))
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        # How many times each player was drawn, as one offset bincount over the whole
        # batch rather than `size` separate ones -- `shape._bootstrap_predictions`'s idiom.
        picks = rng.integers(0, n, (size, n))
        picks += (np.arange(size) * n)[:, None]
        weights = np.bincount(picks.ravel(), minlength=size * n).reshape(size, n).astype(float)
        rmse[start : start + size] = np.sqrt((weights @ per_player) / (weights @ counts)[:, None])

    # Unstacked off the pivot's own MultiIndex rather than reshaped into an assumed
    # (prior, age) order -- a transposed grid would still print as a plausible table.
    at = list(sums.columns).index((PRIOR_WINDOW, AGE_WINDOW))
    observed = np.sqrt(per_player.sum(axis=0) / counts.sum())
    delta = pd.Series(observed - observed[at], index=sums.columns)
    # Ties count as half a win each. A cell that never differs from the default is the
    # perfectly indistinguishable case, and a bare `>` scores it 0.00 -- the same reading
    # as "this setting beat the default in every draw", which is the opposite claim.
    baseline = rmse[:, at][:, None]
    confidence = pd.Series(
        ((rmse > baseline) + 0.5 * (rmse == baseline)).mean(axis=0), index=sums.columns
    )
    # The default's own cell is a comparison with itself, which `>` scores as 0.00 -- a
    # number that reads exactly like "this setting always beat the default". Blank it.
    confidence.iloc[at] = float("nan")
    return (
        delta.unstack("age_window").sort_index(),
        confidence.unstack("age_window").sort_index(),
    )


def print_table(table: pd.DataFrame, title: str, *, fmt: str = "{:6.3f}") -> None:
    print(f"\n{title}")
    print("  prior\\age  " + "".join(f"{c:>8d}" for c in table.columns))
    for prior_window, row in table.iterrows():
        cells = "".join(("--" if np.isnan(v) else fmt.format(v)).rjust(8) for v in row)
        print(f"  {prior_window:8.1f}  {cells}")


def report_best(table: pd.DataFrame, label: str, n: int) -> tuple[int, float]:
    """The argmin, how much it beats the shipped default, and whether it sits on an edge."""
    flat = table.stack()
    prior_window, age_window = flat.idxmin()
    best = float(flat.min())
    default = float(table.loc[PRIOR_WINDOW, AGE_WINDOW])
    edge = age_window in (table.columns.min(), table.columns.max()) or prior_window in (
        table.index.min(),
        table.index.max(),
    )
    print(
        f"  {label:34s} n={n:5d}  best age={age_window} prior={prior_window:g} "
        f"RMSE {best:.3f} vs default(age={AGE_WINDOW}, prior={PRIOR_WINDOW:g}) "
        f"{default:.3f}  ({(default - best) / default:+.1%})"
        + ("   [ON A GRID EDGE]" if edge else "")
    )
    return int(age_window), float(prior_window)


def cross_validate(df: pd.DataFrame, folds: int) -> None:
    """Select on the other folds, score on this one, pooled over folds.

    Reports the tuned choice against the shipped default on rows neither saw. The two are
    scored on the SAME rows every fold, so the comparison is paired.
    """
    fold_of = _folds(df["mlbam_id"].to_numpy(), folds)
    tuned_sq: list[np.ndarray] = []
    default_sq: list[np.ndarray] = []
    picks: list[tuple[int, float]] = []
    for f in range(folds):
        train, test = df[fold_of != f], df[fold_of == f]
        if train.empty or test.empty:
            continue
        prior_window, age_window = error_table(train).stack().idxmin()
        picks.append((int(age_window), float(prior_window)))
        chosen = test[(test["age_window"] == age_window) & (test["prior_window"] == prior_window)]
        shipped = test[(test["age_window"] == AGE_WINDOW) & (test["prior_window"] == PRIOR_WINDOW)]
        tuned_sq.append(((chosen["predicted"] - chosen["actual"]) ** 2).to_numpy())
        default_sq.append(((shipped["predicted"] - shipped["actual"]) ** 2).to_numpy())

    if not tuned_sq:
        print("  (too few players to cross-validate)")
        return
    tuned = float(np.sqrt(np.concatenate(tuned_sq).mean()))
    default = float(np.sqrt(np.concatenate(default_sq).mean()))
    print(
        f"  cross-validated RMSE: tuned {tuned:.3f} vs default "
        f"{default:.3f}  ({(default - tuned) / default:+.1%})"
    )
    unique = sorted(set(picks))
    print(f"  fold picks (age, prior): {picks}")
    if len(unique) > 1:
        print("  -- the folds DISAGREE, so any single winner is partly sample noise")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=("hitter", "pitcher"), default="hitter")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--age-windows", type=int, nargs="+", default=list(DEFAULT_AGE_WINDOWS))
    parser.add_argument(
        "--prior-windows", type=float, nargs="+", default=list(DEFAULT_PRIOR_WINDOWS)
    )
    parser.add_argument(
        "--elite-floor",
        type=float,
        default=14.0,
        help="prior-season SGP at or above which a query counts as elite",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=3000,
        help=(
            "score this many NON-elite queries, drawn at random; every elite query is "
            "always scored. 0 scores the whole pool."
        ),
    )
    parser.add_argument("--min-age", type=int, default=22)
    parser.add_argument("--max-age", type=int, default=38)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--draws", type=int, default=2000, help="player-bootstrap draws behind the plateau test"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--out", type=Path, help="write the raw predictions to this CSV")
    parser.add_argument(
        "--from-csv",
        type=Path,
        help="re-analyse an earlier --out instead of sweeping again (seconds, not minutes)",
    )
    args = parser.parse_args()
    if min(args.horizons) < 1:
        parser.error("--horizons must all be at least 1")
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    if not args.panel_dir.is_absolute():
        args.panel_dir = PROJECT_ROOT / args.panel_dir

    if args.from_csv:
        # The sweep is minutes and the analysis is seconds. Changing a slice or a bootstrap
        # setting should not cost another sweep, and re-running one to do it invites
        # comparing two tables that came from different query samples.
        #
        # EVERYTHING here is read off the file -- pool, grid, horizons, counts. The panel is
        # not loaded at all. It used to be: the query set was rebuilt from the CLI defaults,
        # printed as the header, and then thrown away, so `--from-csv pitchers.csv` announced
        # HITTERS and `--sample 25` announced 25 fringe queries above an analysis of 3000.
        # The numbers below it were right the whole time, which is what made it dangerous.
        # Refused, not ignored -- `player_trajectory.py` does the same for flags that cannot
        # affect the chosen mode. These all describe how to SWEEP, and the sweep already
        # happened; accepting `--sample 25` here and answering from 3000 saved queries is
        # how the old header came to disagree with its own table.
        sweeps_only = (
            "horizons",
            "age_windows",
            "prior_windows",
            "min_age",
            "max_age",
            "sample",
            "panel_dir",
            "out",
        )
        overridden = [
            f"--{name.replace('_', '-')}"
            for name in sweeps_only
            if getattr(args, name) != parser.get_default(name)
        ]
        if overridden:
            verb = "describes" if len(overridden) == 1 else "describe"
            parser.error(
                f"{', '.join(overridden)} {verb} the sweep and cannot apply to "
                f"--from-csv; the saved file already fixes that"
            )
        df = pd.read_csv(args.from_csv)
        if "pool" not in df.columns:
            parser.error(
                f"{args.from_csv} predates the `pool` column and cannot be checked against "
                f"--pool {args.pool}; re-run the sweep to regenerate it"
            )
        pools = sorted(set(df["pool"]))
        if pools != [args.pool]:
            parser.error(
                f"{args.from_csv} holds {'/'.join(pools)} predictions but --pool is "
                f"{args.pool}; the analysis is pool-blind and would label them wrongly"
            )
        horizons = tuple(sorted(set(df["horizon"])))
        age_windows = tuple(sorted(set(df["age_window"])))
        prior_windows = tuple(sorted(set(df["prior_window"])))
        ages = (int(df["age"].min()), int(df["age"].max()))
        # QUERIES, matching the sweep branch's `len(queries)` -- one per (player, season),
        # not per query-horizon, or the same header line would mean two different things
        # depending on which branch printed it.
        n_queries = int(df.groupby(["mlbam_id", "season"], sort=False).ngroups)
        elite_rows = df[df["prior"] >= args.elite_floor]
        n_elite = int(elite_rows.groupby(["mlbam_id", "season"], sort=False).ngroups)
    else:
        overrides = load_config(PROJECT_ROOT / "config" / "league.yaml").sgp_overrides
        panel = era_normalize(
            load_scored_panel(args.pool, panel_dir=args.panel_dir, sgp_overrides=overrides),
            args.pool,
            sgp_overrides=overrides,
        )
        last = int(panel["season"].max())
        horizons = tuple(sorted(set(args.horizons)))
        age_windows = tuple(sorted(set(args.age_windows)))
        prior_windows = tuple(sorted(set(args.prior_windows)))
        ages = (args.min_age, args.max_age)

        history = build_history(panel)
        history = history[
            history["age"].between(args.min_age, args.max_age)
            & (history["season"] + min(horizons) <= last)
        ]
        elite = history[history["prior"] >= args.elite_floor]
        fringe = history[history["prior"] < args.elite_floor]
        if args.sample:
            fringe = fringe.sample(min(args.sample, len(fringe)), random_state=args.seed)
        queries = pd.concat([elite, fringe])
        n_queries, n_elite = len(queries), len(elite)

    if AGE_WINDOW not in age_windows or PRIOR_WINDOW not in prior_windows:
        parser.error(
            f"the grid must contain the shipped default (age={AGE_WINDOW}, "
            f"prior={PRIOR_WINDOW:g}); everything here is reported against it"
        )
    n_grid = len(age_windows) * len(prior_windows)
    print(
        f"{args.pool.upper()}S, horizons {list(horizons)}, ages {ages[0]}-{ages[1]}\n"
        f"{n_queries} queries ({n_elite} elite at prior >= {args.elite_floor:g}, "
        f"{n_queries - n_elite} fringe) x {n_grid} grid points\n"
        f"age windows {list(age_windows)}, prior windows "
        f"{[f'{p:g}' for p in prior_windows]}\n"
    )
    if args.from_csv:
        print(f"re-analysing {len(df)} saved predictions from {args.from_csv}\n")
    else:
        df = score_grid(
            panel,
            queries,
            kind=args.pool,
            horizons=horizons,
            age_windows=age_windows,
            prior_windows=prior_windows,
        )
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"\nwrote {args.out}")

    print_table(coverage_table(df), "COVERAGE (share of query-horizons answered)", fmt="{:6.1%}")
    complete, kept = complete_cases(df, n_grid)
    print(
        f"\ncomplete cases: {kept:.1%} of query-horizons answered by every grid point "
        f"({len(complete) // n_grid} of {len(df) // n_grid})"
    )
    if complete.empty:
        print("no query-horizon was answered at every grid point; widen the grid")
        return 1

    slices = {
        "ALL": complete,
        f"elite (prior >= {args.elite_floor:g})": complete[complete["prior"] >= args.elite_floor],
        "elite down year (<80% of prior)": complete[
            (complete["prior"] >= args.elite_floor) & (complete["now"] < complete["prior"] * 0.8)
        ],
        "fringe (prior < 8)": complete[complete["prior"] < 8],
    }
    for label, subset in slices.items():
        if subset.empty:
            continue
        table = error_table(subset)
        print_table(table, f"RMSE -- {label}, horizons {list(horizons)} pooled")
        print()
        report_best(table, label, _n_queries(subset))
        delta, confidence = plateau(subset, draws=args.draws, seed=args.seed)
        print_table(
            delta,
            f"  ... minus the default's RMSE (positive = worse than age={AGE_WINDOW}, "
            f"prior={PRIOR_WINDOW:g})",
            fmt="{:+6.3f}",
        )
        print_table(
            confidence,
            "  ... share of player-bootstrap draws in which the DEFAULT won "
            "(~0.5 = indistinguishable)",
            fmt="{:6.2f}",
        )

    for h in horizons:
        at_h = complete[complete["horizon"] == h]
        if not at_h.empty:
            print(f"\n+{h} only:")
            report_best(error_table(at_h), "ALL", _n_queries(at_h))
            elite_h = at_h[at_h["prior"] >= args.elite_floor]
            if not elite_h.empty:
                report_best(
                    error_table(elite_h),
                    f"elite (prior >= {args.elite_floor:g})",
                    _n_queries(elite_h),
                )

    print(f"\nCROSS-VALIDATED SELECTION ({args.folds} folds, cut by player)")
    for label, subset in slices.items():
        if subset.empty:
            continue
        print(f"\n  {label}, n={_n_queries(subset)}")
        cross_validate(subset, args.folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
