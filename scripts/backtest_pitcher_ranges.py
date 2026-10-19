"""Score pitcher outcome ranges (pt_model.ranges, #505) on held-out seasons.

As scripts/backtest_hitter_ranges.py, per role: starters (range_specs.STARTER) and
relievers (RELIEVER), each ranked, fit and scored within the role. Every pitcher in a
held-out season gets his P10 and P90 of IP, K, W, SV, ERA and WHIP, from the
player-specific PT range (QuantilePT) and the one-size baseline (BandPT).

Projected playing time and role: the PT system's preseason file per season,
data/projections/<y>/<system>-pitchers*.csv, or with --pt-system last-season, his
line the season before. Seasons, history and actuals: the MLB Stats API pitching
leaderboards (pt_model.range_inputs), cached in data/cache/keeper_skills/. Our rates:
--rates-run's preseason rows (data/pitcher_ros/runs/<run>/predictions.parquet, its
seeds averaged).

Writes data/pitcher_ranges/backtest-<name>/scored.parquet and summary.md (a section per
role) and prints the summary.

    python scripts/backtest_pitcher_ranges.py --name base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd
from backtest_hitter_ranges import CACHE, LAST_SEASON, LEVELS, PEOPLE, pt_file

from fantasy_baseball.pt_model import range_inputs as ri
from fantasy_baseball.pt_model.range_report import summarize
from fantasy_baseball.pt_model.range_specs import RELIEVER, STARTER
from fantasy_baseball.pt_model.ranges import BandPT, QuantilePT, Spec, backtest

RUNS = PROJECT_ROOT / "data" / "pitcher_ros" / "runs"
OUT = PROJECT_ROOT / "data" / "pitcher_ranges"
ROLES = (STARTER, RELIEVER)
# Per role: an expected regular's ranked IP, and the IP middle check's bins.
REGULAR_IP = {STARTER.name: 140.0, RELIEVER.name: 55.0}
MEDIAN_BINS = {
    STARTER.name: (0, 60, 100, 130, 155, 175, 190, 260),
    RELIEVER.name: (0, 20, 35, 45, 55, 62, 70, 200),
}
# ERA and WHIP are scored where he threw this many innings.
MIN_SCORED_IP = 20.0


def pt_projection(system: str, season: int, seasons: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """The PT source's ``ip``/``bf``/``gs_share`` for ``season``."""
    if system == LAST_SEASON:
        return ri.last_season_pitcher_pt(seasons, season)
    return ri.load_pitcher_pt(pt_file(system, season, "pitchers"))


def load_inputs(
    seasons: list[int], system: str, *, with_actuals: bool = True
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """(every pitching season the rows for ``seasons`` read, by year -- ``seasons``
    themselves only ``with_actuals``; the people info)."""
    years = set(ri.history_seasons(seasons)) | (set(seasons) if with_actuals else set())
    if system == LAST_SEASON:
        years |= {s - 1 for s in seasons}
    people = pd.read_csv(PEOPLE, usecols=["id", "birthDate", "mlbDebutDate"])
    return ri.load_pitching_seasons(CACHE, years), ri.player_info(people)


def training_rows(
    seasons: list[int], system: str, run: str
) -> dict[str, tuple[pd.DataFrame, dict[int, np.ndarray]]]:
    """Per role: every season's rows, with actuals, and the role's IP by season
    (shared with scripts/pitcher_ranges.py)."""
    loaded, info = load_inputs(seasons, system)
    rates = ri.load_pitcher_rates(RUNS / run)
    out = {}
    for spec in ROLES:
        parts = []
        for season in seasons:
            rows, dropped = ri.build_pitcher_rows(
                season,
                spec,
                pt_projection(system, season, loaded),
                rates,
                loaded,
                info,
                with_actuals=True,
            )
            print(
                f"{spec.name} {season}: {len(rows)} ({dropped} ranked but without our rates, left out)"
            )
            parts.append(rows)
        out[spec.name] = (pd.concat(parts), ri.actual_ip(loaded, seasons, spec))
    return out


def report(scored: pd.DataFrame, spec: Spec, seed: int) -> list[str]:
    return summarize(
        scored,
        spec,
        np.random.default_rng(seed),
        regular=REGULAR_IP[spec.name],
        median_bins=MEDIAN_BINS[spec.name],
        min_scored={"era": MIN_SCORED_IP, "whip": MIN_SCORED_IP},
        by_season=("k", "era", "whip") if spec.role == "starter" else ("k", "sv", "era"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2022, 2027)))
    parser.add_argument("--pt-system", default="steamer")
    parser.add_argument("--rates-run", default="035-saves-net")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if len(args.seasons) < 3:
        parser.error("--seasons: need 3+ (each held-out fit is itself cross-fitted by season)")
    out = OUT / f"backtest-{args.name}"
    out.mkdir(parents=True, exist_ok=True)

    by_role = training_rows(sorted(args.seasons), args.pt_system, args.rates_run)
    md = [
        f"### Pitcher ranges backtest `{args.name}`",
        "",
        f"Seasons {args.seasons}, PT from {args.pt_system}, rates from {args.rates_run}, "
        f"{args.draws} draws, seed {args.seed}.",
    ]
    scored = []
    for spec in ROLES:
        rows, ip = by_role[spec.name]
        parts = []
        for label, make in (("player-specific", QuantilePT), ("one-size", BandPT)):
            rng = np.random.default_rng(args.seed)
            parts.append(
                backtest(rows, ip, make, rng, spec, LEVELS, args.draws).assign(model=label)
            )
            print(f"{spec.name} {label} done")
        role = pd.concat(parts, ignore_index=True).assign(role=spec.name)
        scored.append(role)
        md += [
            "",
            f"#### {spec.name.capitalize()}s",
            "",
            "```",
            *report(role, spec, args.seed),
            "```",
        ]
    pd.concat(scored, ignore_index=True).to_parquet(out / "scored.parquet", index=False)
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
