"""Score hitter outcome ranges (pt_model.ranges, #505) on held-out seasons.

Leave one season out: fit on the other seasons, give every hitter in the held-out one
his P10 and P90 of PA, R, HR, RBI, SB and AVG, and compare with what he did. Twice:
with the player-specific PT range (QuantilePT) and the one-size baseline (BandPT).

Projected playing time: the PT system's preseason file per season,
data/projections/<y>/<system>-hitters*.csv (one file per folder), or with
--pt-system last-season, his PA the season before. Seasons, history and actuals: the
MLB Stats API hitting leaderboards (pt_model.range_inputs), cached in
data/cache/keeper_skills/. Our rates: the --rates-run seeds' preseason rows
(data/hitter_ros/runs/<run>-s<N>/).

Writes data/hitter_ranges/backtest-<name>/scored.parquet and summary.md, and prints
the summary: share of seasons below P10 and above P90 (target 10% each), overall and
for expected regulars (rank PA 450+) and by season; quantile loss vs the baseline,
with how sure the difference is (0-100, bootstrap over players).

    python scripts/backtest_hitter_ranges.py --name base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from fantasy_baseball.pt_model import range_inputs as ri
from fantasy_baseball.pt_model.range_report import summarize
from fantasy_baseball.pt_model.range_specs import HITTER
from fantasy_baseball.pt_model.ranges import BandPT, QuantilePT, backtest

PROJECTIONS = PROJECT_ROOT / "data" / "projections"
CACHE = PROJECT_ROOT / "data" / "cache" / "keeper_skills"
PEOPLE = CACHE / "mlb_people_all_2000_2026.csv"
RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"
OUT = PROJECT_ROOT / "data" / "hitter_ranges"
LEVELS = (0.1, 0.5, 0.9)
# The PA middle check: predicted-median PA bins.
MEDIAN_BINS = (0, 300, 450, 550, 620, 660, 700, 800)
REGULAR_PA = 450
# AVG is scored where he had this many AB.
MIN_SCORED_AB = 100


# --pt-system that ranks hitters by last season's actual PA instead of a projection.
LAST_SEASON = "last-season"


def pt_file(system: str, season: int, kind: str = "hitters") -> Path:
    """The PT system's preseason ``<system>-<kind>*.csv`` for ``season`` (exactly one)."""
    found = sorted((PROJECTIONS / str(season)).glob(f"{system}-{kind}*.csv"))
    if len(found) != 1:
        raise SystemExit(
            f"want one {system}-{kind}*.csv in {PROJECTIONS / str(season)}, found {len(found)}"
        )
    return found[0]


def pt_projection(system: str, season: int, seasons: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """The PT source's ``pa``/``ab`` for ``season``; last-season reads ``seasons``."""
    if system == LAST_SEASON:
        return seasons[season - 1][["pa", "ab"]]
    return ri.load_pt(pt_file(system, season))


def load_inputs(
    seasons: list[int], system: str, *, with_actuals: bool = True
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """(every season the rows for ``seasons`` read, by year -- ``seasons`` themselves
    only ``with_actuals``; the people info)."""
    years = set(ri.history_seasons(seasons)) | (set(seasons) if with_actuals else set())
    if system == LAST_SEASON:
        years |= {s - 1 for s in seasons}
    people = pd.read_csv(PEOPLE, usecols=["id", "birthDate", "mlbDebutDate"])
    return ri.load_seasons(CACHE, years), ri.player_info(people)


def training_rows(
    seasons: list[int], system: str, run: str, seeds: list[int]
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Every season's rows, with actuals, and every hitter's PA by season (shared with
    scripts/hitter_ranges.py)."""
    loaded, info = load_inputs(seasons, system)
    rates = ri.load_rates(RUNS / f"{run}-s{s}" for s in seeds)
    parts = []
    for season in seasons:
        rows, dropped = ri.build_rows(
            season, pt_projection(system, season, loaded), rates, loaded, info, with_actuals=True
        )
        print(f"{season}: {len(rows)} hitters ({dropped} ranked but without our rates, left out)")
        parts.append(rows)
    return pd.concat(parts), ri.actual_pa(loaded, seasons)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2022, 2027)))
    parser.add_argument("--pt-system", default="steamer")
    parser.add_argument("--rates-run", default="414-seq")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if len(args.seasons) < 3:
        parser.error("--seasons: need 3+ (each held-out fit is itself cross-fitted by season)")
    out = OUT / f"backtest-{args.name}"
    out.mkdir(parents=True, exist_ok=True)

    rows, pa = training_rows(sorted(args.seasons), args.pt_system, args.rates_run, args.seeds)
    scored = []
    for label, make in (("player-specific", QuantilePT), ("one-size", BandPT)):
        rng = np.random.default_rng(args.seed)
        scored.append(backtest(rows, pa, make, rng, HITTER, LEVELS, args.draws).assign(model=label))
        print(f"{label} done")
    all_scored = pd.concat(scored, ignore_index=True)
    all_scored.to_parquet(out / "scored.parquet", index=False)
    md = [
        f"### Hitter ranges backtest `{args.name}`",
        "",
        f"Seasons {args.seasons}, PT from {args.pt_system}, rates from {args.rates_run} seeds {args.seeds}, "
        f"{args.draws} draws, seed {args.seed}.",
        "",
        "```",
        *summarize(
            all_scored,
            HITTER,
            np.random.default_rng(args.seed),
            regular=REGULAR_PA,
            median_bins=MEDIAN_BINS,
            min_scored={"avg": MIN_SCORED_AB},
            by_season=("r", "hr", "rbi"),
        ),
        "```",
    ]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
