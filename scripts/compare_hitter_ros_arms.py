"""Compare training arms on what actually happened, with the luck of the draw shown.

An arm is a set of runs that differ only by seed: data/hitter_ros/runs/<arm>-s<N>/ (or
one run named <arm>). Each arm's predicted rates are averaged over its seeds and scored
against the first arm listed (the base), on the same hitters:

* Mid-season, by stretch of the season: rest-of-season rates from
  scored_horizons.parquet (the net's own output, every hitter with
  ``horizons.ROS_MIN_PA``+ PA left, every week of every test season).
* Preseason: scored_preseason.parquet (hitters with ``backtest.PRESEASON_MIN_PA``+
  actual PA).

Each cell: arm minus base in gap-weighted pairwise points (the main score; positive =
the arm orders hitters better), the 95% interval and how sure the arm is better or
worse (``evaluate.sure``: "96% sure better" means what it says; 95%+ is marked real),
both from resampling hitters (``evaluate.pairwise_bootstrap``: each hitter redrawn once
across every season-week, so overlapping weeks aren't counted as independent), then how
many seeds were better on their own (seeds paired by number). How sure counts only
which hitters were scored; the seed count shows the training noise.

Judge a change here first, on actuals over every week and season, before checking it
against FanGraphs: the FanGraphs mid-season snapshots are one season (#453).

Usage:
    python scripts/compare_hitter_ros_arms.py 451-base 451-dd50
    python scripts/compare_hitter_ros_arms.py 451-base 451-dd25 451-dd50 --seasons 2022 2023
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.backtest import sure_text, to_markdown
from fantasy_baseball.hitter_ros.evaluate import order_scores, pairwise_bootstrap
from fantasy_baseball.hitter_ros.features import TARGETS

RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"
FINISHED_MARKER = "summary.md"
# Stretches of the season: (first week, last week, label); week 0 is the preseason.
WEEK_BANDS = ((1, 6, "wk1-6"), (7, 13, "wk7-13"), (14, 20, "wk14-20"), (21, 99, "wk21+"))
PRESEASON = "preseason"
BANDS = (*(label for *_, label in WEEK_BANDS), PRESEASON)
KEYS = ["band", "season", "snapshot", "player_id", "stat"]


def arm_runs(arm: str) -> dict[str, Path]:
    """The arm's finished runs by seed label: ``<arm>-s<N>`` folders, or the one run
    ``<arm>`` (label "")."""
    seeded = {
        m.group(1): p
        for p in RUNS.glob(f"{arm}-s*")
        if (m := re.fullmatch(rf"{re.escape(arm)}-s(\d+)", p.name))
        and (p / FINISHED_MARKER).exists()
    }
    if seeded:
        return dict(sorted(seeded.items(), key=lambda kv: int(kv[0])))
    if (RUNS / arm / FINISHED_MARKER).exists():
        return {"": RUNS / arm}
    raise ValueError(f"no finished run {arm} or {arm}-s<N> in {RUNS}")


def band_of(snapshot: str) -> str:
    """The band of a scored unit: "pre" is the preseason, "<season>-w<week>" a week."""
    if snapshot == "pre":
        return PRESEASON
    week = int(snapshot.split("-w")[1])
    for lo, hi, label in WEEK_BANDS:
        if lo <= week <= hi:
            return label
    raise ValueError(f"week {week} is in no band")


def run_scores(run: Path) -> pd.DataFrame:
    """One run's scored rates (``KEYS`` + projected, actual): rest of season mid-season
    and the preseason. ``snapshot`` is the season-week, "pre" for the preseason."""
    parts = []
    hz = run / "scored_horizons.parquet"
    if hz.exists():
        h = pd.read_parquet(hz)
        h = h[(h["horizon"] == "ros") & (h["comparison"] == "all") & (h["system"] == "head")]
        parts.append(h.assign(band=h["snapshot"].map(band_of)))
    pre = run / "scored_preseason.parquet"
    if pre.exists():
        p = pd.read_parquet(pre)
        parts.append(p[p["system"] == "ours"].assign(band=PRESEASON, snapshot="pre"))
    if not parts:
        raise ValueError(f"{run} has no scored_horizons or scored_preseason file")
    return pd.concat(parts)[[*KEYS, "projected", "actual"]]


def arm_scores(arm: str, seasons: list[int] | None) -> dict[str, pd.DataFrame]:
    """Each seed's scored rates, by seed label."""
    out = {}
    for label, run in arm_runs(arm).items():
        s = run_scores(run)
        out[label] = s[s["season"].isin(seasons)] if seasons else s
    return out


def seed_mean(scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Projected rates averaged over seeds. Every seed must have scored the same rows
    (which hitters are scored depends only on what they did, not on the projection)."""
    frames = list(scores.values())
    first = frames[0].set_index(KEYS).sort_index()
    for f in frames[1:]:
        if not f.set_index(KEYS).sort_index().index.equals(first.index):
            raise ValueError("an arm's seeds scored different rows; were they run alike?")
    mean = pd.concat(frames).groupby(KEYS)["projected"].mean()
    return first[["actual"]].join(mean).reset_index()


def band_means(scored: pd.DataFrame) -> pd.DataFrame:
    """Gap-weighted pairwise (%) per band x stat, each season-week counting once."""
    per = order_scores(scored.assign(system="x"))
    per["band"] = per["snapshot"].map(band_of)
    return per.groupby(["band", "stat"])["pairwise_w"].mean().unstack("stat") * 100


def compare_arm(
    base: dict[str, pd.DataFrame], arm: dict[str, pd.DataFrame], n_boot: int
) -> pd.DataFrame:
    """Bands x stats: "diff [lo, hi] how sure, wins/seeds" for ``arm`` minus ``base``."""
    b, a = seed_mean(base), seed_mean(arm)
    both = b.merge(a, on=KEYS, suffixes=("_b", "_a"))
    if len(both) != len(b) or len(both) != len(a):
        raise ValueError("the two arms scored different rows; compare like with like")
    seeds = [s for s in base if s in arm]
    wins = sum(((band_means(arm[s]) - band_means(base[s])) > 0).astype(int) for s in seeds)
    cells: dict[str, dict[str, str]] = {}
    for band in BANDS:
        g = both[both["band"] == band]
        if g.empty:
            continue
        long = pd.concat(
            [
                g[[*KEYS, "actual_b"]].assign(system=name, projected=g[col])
                for name, col in (("base", "projected_b"), ("arm", "projected_a"))
            ]
        ).rename(columns={"actual_b": "actual"})
        boot = pairwise_bootstrap(long, "arm", "base", n_boot=n_boot)
        cells[band] = {
            s: f"{r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] {sure_text(r['sure'])}"
            + (f", {int(wins.loc[band, s])}/{len(seeds)} seeds" if seeds else "")
            for s, r in boot.iterrows()
            if s in TARGETS
        }
    return pd.DataFrame(cells).T[list(TARGETS)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", help="the arm the others are compared with")
    parser.add_argument("arms", nargs="+")
    parser.add_argument("--seasons", type=int, nargs="+", help="only these test seasons")
    parser.add_argument("--n-boot", type=int, default=300, help="hitter resamples")
    args = parser.parse_args()
    try:
        base = arm_scores(args.base, args.seasons)
        arms = {name: arm_scores(name, args.seasons) for name in args.arms}
    except ValueError as err:
        parser.error(str(err))
    print(
        "Gap-weighted pairwise points, arm minus base (positive = arm better), mean of "
        "seeds' predictions: diff [95% interval] how sure the arm is better or worse, "
        "seeds better/seeds"
    )
    for name, arm in arms.items():
        print(f"\n**{name} minus {args.base}** ({len(arm)} vs {len(base)} seeds)\n")
        print(to_markdown(compare_arm(base, arm, args.n_boot)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
