"""Score hitter outcome ranges (pt_model.ranges, #505) on held-out seasons.

Leave one season out: fit on the other seasons, give every hitter in the held-out one
his P10 and P90 of PA, R, HR, RBI, SB and AVG, and compare with what he did. Twice:
with the player-specific PT range (QuantilePT) and the one-size baseline (BandPT).

Projected playing time: the PT system's preseason file per season,
data/projections/<y>/<system>-hitters*.csv (one file per folder), or with
--pt-system last-season, his PA the season before (data/stats; from 2023, the first
season with a stats file before it). Our rates: the
--rates-run seeds' preseason rows (data/hitter_ros/runs/<run>-s<N>/).

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
from fantasy_baseball.pt_model.ranges import (
    STATS,
    BandPT,
    QuantilePT,
    backtest,
    pinball,
)

PROJECTIONS = PROJECT_ROOT / "data" / "projections"
STATS_DIR = PROJECT_ROOT / "data" / "stats"
PANEL = PROJECT_ROOT / "data" / "playing_time" / "hitter_pt_panel_2010_2026.csv"
PEOPLE = PROJECT_ROOT / "data" / "cache" / "keeper_skills" / "mlb_people_all_2000_2026.csv"
RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"
OUT = PROJECT_ROOT / "data" / "hitter_ranges"
LEVELS = (0.1, 0.5, 0.9)
# The PA middle check: predicted-median PA bins.
MEDIAN_BINS = (0, 300, 450, 550, 620, 660, 700, 800)
REGULAR_PA = 450
# AVG is scored where he had this many AB.
MIN_SCORED_AB = 100
BOOTSTRAPS = 4000


# --pt-system that ranks hitters by last season's actual PA instead of a projection.
LAST_SEASON = "last-season"


def pt_file(system: str, season: int) -> Path:
    found = sorted((PROJECTIONS / str(season)).glob(f"{system}-hitters*.csv"))
    if len(found) != 1:
        raise SystemExit(
            f"want one {system}-hitters*.csv in {PROJECTIONS / str(season)}, found {len(found)}"
        )
    return found[0]


def pt_projection(system: str, season: int) -> pd.DataFrame:
    """The PT source's ``pa``/``ab`` for ``season``."""
    if system == LAST_SEASON:
        return ri.load_season_stats(STATS_DIR, season - 1)[["pa", "ab"]]
    return ri.load_pt(pt_file(system, season))


def training_rows(seasons: list[int], system: str, run: str, seeds: list[int]) -> pd.DataFrame:
    """Every season's rows, with actuals (shared with scripts/hitter_ranges.py)."""
    panel = pd.read_csv(PANEL)
    people = pd.read_csv(PEOPLE, usecols=["id", "birthDate"])
    info = ri.player_info(panel, people)
    lines = ri.season_lines(panel, STATS_DIR, ri.history_seasons(seasons))
    rates = ri.load_rates(RUNS / f"{run}-s{s}" for s in seeds)
    parts = []
    for season in seasons:
        rows, dropped = ri.build_rows(
            season,
            pt_projection(system, season),
            rates,
            lines,
            info,
            actual=ri.load_season_stats(STATS_DIR, season),
        )
        print(f"{season}: {len(rows)} hitters ({dropped} ranked but without our rates, left out)")
        parts.append(rows)
    return pd.concat(parts)


def sure_better(
    loss_a: pd.Series, loss_b: pd.Series, players: pd.Series, rng: np.random.Generator
) -> float:
    """How sure (0-100) a's loss is lower than b's, resampling players."""
    gain = (loss_b - loss_a).groupby(players.to_numpy()).sum().to_numpy()
    draws = rng.integers(0, len(gain), (BOOTSTRAPS, len(gain)))
    return float((gain[draws].sum(axis=1) > 0).mean() * 100)


def middle(scored: pd.DataFrame) -> str:
    p = scored[(scored["stat"] == "pa") & (scored["model"] == "player-specific")]
    bins = pd.cut(p["q50"], MEDIAN_BINS)
    t = p.groupby(bins, observed=True).apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "predicted": g["q50"].median(),
                "real": g["actual"].median(),
                "below %": (g["actual"] < g["q50"]).mean() * 100,
            }
        ),
        include_groups=False,
    )
    return t.round(1).to_string()


def summarize(scored: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    s = scored[~((scored["stat"] == "avg") & (scored["ab"] < MIN_SCORED_AB))].copy()
    s["below"], s["above"] = s["actual"] < s["q10"], s["actual"] > s["q90"]
    s["loss"] = pinball(s["actual"].to_numpy(), s["q10"].to_numpy(), 0.1) + pinball(
        s["actual"].to_numpy(), s["q90"].to_numpy(), 0.9
    )
    order = ["pa", *STATS]

    def cover(df: pd.DataFrame) -> str:
        t = df.groupby(["stat", "model"])[["below", "above"]].mean().unstack("model").reindex(order)
        return (t * 100).round(1).to_string()

    md = [
        "Share of seasons below P10 / above P90, % (target 10 / 10):",
        cover(s),
        "",
        f"Expected regulars (rank PA {REGULAR_PA}+):",
        cover(s[s["rank_pa"] >= REGULAR_PA]),
        "",
        "R, HR and RBI by season:",
        (
            s[s["stat"].isin(["r", "hr", "rbi"])]
            .groupby(["season", "model"])[["below", "above"]]
            .mean()
            .unstack("model")
            * 100
        )
        .round(1)
        .to_string(),
        "",
        "PA middle, player-specific: share of real PA below the predicted median, % (50 is",
        "right), by predicted median:",
        middle(scored),
        "",
        "Quantile loss (P10 + P90), player-specific vs one-size:",
    ]
    wide = s.pivot_table(
        index=["player_id", "season", "stat"], columns="model", values="loss"
    ).dropna()
    for stat in order:
        w = wide.xs(stat, level="stat")
        sure = sure_better(
            w["player-specific"],
            w["one-size"],
            w.index.get_level_values("player_id").to_series(),
            rng,
        )
        change = w["player-specific"].mean() / w["one-size"].mean() - 1
        md.append(f"  {stat:4s} {change:+.1%}  (how sure better: {sure:.0f})")
    return md


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

    rows = training_rows(sorted(args.seasons), args.pt_system, args.rates_run, args.seeds)
    pa = ri.actual_pa(STATS_DIR, args.seasons)
    scored = []
    for label, make in (("player-specific", QuantilePT), ("one-size", BandPT)):
        rng = np.random.default_rng(args.seed)
        scored.append(backtest(rows, pa, make, rng, LEVELS, args.draws).assign(model=label))
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
        *summarize(all_scored, np.random.default_rng(args.seed)),
        "```",
    ]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
