"""Fit the conformal band calibration and write it to `data/trajectory/`.

`calibrate_band_coverage.py` MEASURES the band. This one CORRECTS it: same held-out
sweep, but the output is a multiplier table `shape` applies so each tail holds its
nominal 10% instead of the 1.3%-to-26% it held before (see
`docs/trajectory-band-calibration-2026-09-04.md`).

    python scripts/build_band_calibration.py                      # sweep, fit, write
    python scripts/build_band_calibration.py --from-csv held.csv  # reuse a saved sweep
    python scripts/build_band_calibration.py --validate           # rolling-origin check
    python scripts/build_band_calibration.py --dry-run            # print, write nothing

The sweep is ~35 minutes for both pools at horizons 1-5, so `--out-heldout` saves the
predictions and `--from-csv` refits from them in seconds. Refitting from a saved sweep is
also what keeps a validation run and the shipped table describing the same data.

WHEN TO RE-RUN: whenever `data/trajectory/` is rebuilt. The artifact carries the panel
filenames it was fitted on and `BandCalibration.load` REFUSES a mismatch rather than
warning -- a calibration fitted on one panel and applied to another prints bands at a
scale nothing measured, and it looks completely normal on screen.

VALIDATION lives here rather than in a notebook because it is what makes the guarantee
checkable. `--validate` runs rolling origin: for each season Y, fit on every query whose
outcome was already observable before Y and measure coverage on the queries resolving in
Y. That is exactly the production refresh, so its numbers are the ones to believe -- a
single early-vs-late split extrapolates nine years and reads ~2 points hotter than the
tool will ever actually be.
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
from fantasy_baseball.trajectory.calibration import (
    BUCKET_LABELS,
    CALIBRATION_PATH,
    MAX_HORIZON,
    TARGETS,
    BandCalibration,
    bucket_of,
    build_table,
    fit_rows,
    newest_outcome,
    panel_vintage_of,
    span_frame,
)
from fantasy_baseball.trajectory.panel import DEFAULT_PANEL_DIR, load_scored_panel
from fantasy_baseball.trajectory.shape import build_history
from fantasy_baseball.trajectory.sweep import SWEEP_DRAWS

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from calibrate_band_coverage import score_pool

#: Seasons the rolling-origin validation reports. Earlier origins leave too little fitted
#: history to be informative about a production refresh.
VALIDATE_FROM = 2017


def sweep(panel_dir: Path, draws: int, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Held-out predictions for every query in both pools."""
    config = load_config(PROJECT_ROOT / "config" / "league.yaml")
    frames = []
    for kind in ("hitter", "pitcher"):
        panel = load_scored_panel(kind, panel_dir=panel_dir, sgp_overrides=config.sgp_overrides)
        queries = build_history(panel).sort_values("mlbam_id")
        print(
            f"\n{kind}: {len(panel)} panel rows, {len(queries)} queries "
            f"({queries['mlbam_id'].nunique()} players), horizons {horizons}",
            flush=True,
        )
        started = time.perf_counter()
        frames.append(score_pool(panel, queries, kind=kind, horizons=horizons, draws=draws))
        print(f"  scored in {time.perf_counter() - started:.0f}s", flush=True)
    return pd.concat(frames, ignore_index=True)


def _coverage(frame: pd.DataFrame, table: BandCalibration, target: str) -> tuple[int, float, float]:
    """`(n, below_p10, above_p90)` for one target under `table`."""
    below = above = n = 0
    for pool, rows in frame.groupby("pool", sort=False):
        cells = span_frame(rows, target)
        if cells.empty:
            continue
        corrected = [
            table.apply(p, lo, hi, pool=str(pool), target=target, support=s)
            for p, lo, hi, s in zip(
                cells["predicted"], cells["p10"], cells["p90"], cells["support"], strict=True
            )
        ]
        actual = cells["actual"].to_numpy()
        below += int((actual < np.array([c[0] for c in corrected])).sum())
        above += int((actual > np.array([c[1] for c in corrected])).sum())
        n += len(cells)
    return n, (below / n if n else float("nan")), (above / n if n else float("nan"))


def report_fit(frame: pd.DataFrame, table: BandCalibration) -> None:
    """In-sample coverage per bucket. A plumbing check: this MUST land on ~10%/10%.

    It is not evidence the calibration generalises -- the quantile was fitted on these
    rows, so hitting the nominal here is arithmetic. It catches the failure that matters
    at build time: a table applied differently from how it was fitted, which reads as a
    plausible band and is wrong by several points.
    """
    print(f"\n{'=' * 74}\nIN-SAMPLE (plumbing -- must read ~10%/10%)\n{'=' * 74}")
    newest = newest_outcome(frame)
    for pool in sorted(table.multipliers):
        rows = frame[frame["pool"] == pool]
        for target in TARGETS:
            # `fit_rows`, NOT `span_frame`: the table is fitted on a recent-outcomes
            # window, so evaluating it against the whole frame measures generalisation
            # rather than plumbing -- and reads 1-3 points off nominal for a table that
            # is exactly right. Same call the fit makes, so the two cannot drift.
            cells = fit_rows(rows, target, window_years=table.window_years, newest=newest)
            if cells.empty:
                continue
            for label in BUCKET_LABELS:
                sub = cells[cells["support"].map(bucket_of) == label]
                if sub.empty:
                    continue
                corrected = [
                    table.apply(p, lo, hi, pool=pool, target=target, support=s)
                    for p, lo, hi, s in zip(
                        sub["predicted"], sub["p10"], sub["p90"], sub["support"], strict=True
                    )
                ]
                actual = sub["actual"].to_numpy()
                below = float((actual < np.array([c[0] for c in corrected])).mean())
                above = float((actual > np.array([c[1] for c in corrected])).mean())
                lo, hi = table.scale(pool=pool, target=target, support=sub["support"].iloc[0])
                bad = "  <-- BUG" if not (0.085 <= below <= 0.11 and 0.085 <= above <= 0.11) else ""
                print(
                    f"  {pool:<8}{target:<4}{label:<7} n={len(sub):>6}  "
                    f"below {below:>5.1%}  above {above:>5.1%}   "
                    f"x{lo:.2f}/x{hi:.2f}{bad}"
                )


def report_validation(frame: pd.DataFrame) -> None:
    """Rolling origin: fit on outcomes before Y, measure on outcomes resolving in Y.

    A span is keyed on its LAST outcome year (`season + k`), not on the query season --
    a 1..3 total is not observable until all three years are in, so keying it on the
    query would train on outcomes that had not happened yet.
    """
    print(
        f"\n{'=' * 74}\nROLLING ORIGIN (the production refresh -- these are the real numbers)\n{'=' * 74}"
    )
    for target in ("y1", f"y{MAX_HORIZON}", "s3", f"s{MAX_HORIZON}"):
        k = int(target[1:])
        print(f"\n  --- {target} ---")
        collected: list[tuple[int, float, float, bool]] = []
        for year in range(VALIDATE_FROM, int(frame["season"].max()) + MAX_HORIZON + 1):
            # `season + k` for BOTH kinds: a year-k row and a 1..k span both resolve on
            # the same season, so there is nothing to branch on here.
            last = frame["season"] + k
            fit, test = frame[last < year], frame[last == year]
            if len(test) < 300 or fit.empty:
                continue
            table = build_table(fit, panel_vintage="validation")
            n, below, above = _coverage(test, table, target)
            if not n:
                continue
            # A span touching 2020 is not an exchangeable draw: a 60-game season scaled
            # to 162 (`panel.py`) carries ~2.7x the sampling variance of a full one, so
            # its outcomes are genuinely more dispersed than anything the band was fitted
            # on. Reported, and excluded from the headline mean -- pricing a pandemic
            # into every future band permanently is the worse error.
            covid = (year - k) <= 2020 <= year
            collected.append((n, below, above, covid))
            print(
                f"    {year}  n={n:>5}   below {below:>5.1%}  above {above:>5.1%}"
                + ("   [span contains 2020]" if covid else "")
            )
        clean = [(b, a) for _, b, a, c in collected if not c]
        if clean:
            print(
                f"    mean excluding 2020 spans:  below {np.mean([b for b, _ in clean]):.1%}"
                f"  above {np.mean([a for _, a in clean]):.1%}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-csv", type=Path, help="refit from a saved sweep instead of running one"
    )
    parser.add_argument("--out-heldout", type=Path, help="save the sweep's predictions here")
    parser.add_argument(
        "--out", type=Path, default=None, help=f"artifact path (default {CALIBRATION_PATH})"
    )
    parser.add_argument("--draws", type=int, default=SWEEP_DRAWS)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--validate", action="store_true", help="run the rolling-origin check")
    parser.add_argument("--dry-run", action="store_true", help="print everything, write nothing")
    args = parser.parse_args()
    if not args.panel_dir.is_absolute():
        args.panel_dir = PROJECT_ROOT / args.panel_dir
    out = args.out or (PROJECT_ROOT / CALIBRATION_PATH)

    horizons = tuple(range(1, MAX_HORIZON + 1))
    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
        print(f"refitting from {len(frame)} saved rows in {args.from_csv}")
        missing = set(horizons) - set(frame["horizon"].unique())
        if missing:
            # Every span up to MAX_HORIZON must be fittable, or a board rendered at that
            # horizon silently falls through to an uncorrected band.
            parser.error(f"{args.from_csv} is missing horizons {sorted(missing)}; re-sweep")
    else:
        frame = sweep(args.panel_dir, args.draws, horizons)
        if args.out_heldout:
            frame.to_csv(args.out_heldout, index=False)
            print(f"\n  wrote {len(frame)} held-out rows to {args.out_heldout}")

    vintage = panel_vintage_of(args.panel_dir)
    table = build_table(frame, panel_vintage=vintage)
    report_fit(frame, table)
    if table.fallbacks:
        print(f"\n  THIN CELLS pooled across buckets: {table.fallbacks}")
    if args.validate:
        report_validation(frame)

    print(f"\n{'=' * 74}\nMULTIPLIERS  (x1.00 = band unchanged)\n{'=' * 74}")
    for pool in sorted(table.multipliers):
        for target in TARGETS:
            cells = table.multipliers[pool].get(target)
            if not cells:
                continue
            shown = "   ".join(f"{lab} x{v[0]:.2f}/x{v[1]:.2f}" for lab, v in cells.items())
            print(f"  {pool:<8}{target:<4}  {shown}")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0
    table.save(out)
    print(f"\n  wrote {out}  (panel vintage {vintage})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
