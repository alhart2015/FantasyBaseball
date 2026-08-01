"""Fit and validate the per-stat persistence share S from historical seasons.

S answers: of the gap between a player's projection and what he actually did, how
much repeats next year? See `keepers/persistence.py` for the model and
`docs/keeper-value-teardown-2026-08-01.md` for why the rate/volume split matters.

Data used (local only, no network):

    projection for year Y   data/projections/{Y}/zips-{hitters,pitchers}.csv
    actual for year Y       data/stats/{hitters,pitchers}-{Y}.csv

Transitions available: 2022->23, 2023->24, 2024->25.

Usage:
    python scripts/keeper_persistence.py                 # fit + leave-one-out validation
    python scripts/keeper_persistence.py --counting      # add the blended counting-stat fit
    python scripts/keeper_persistence.py --terciles      # is a CONSTANT S good enough?
    python scripts/keeper_persistence.py --min-pa 400 --min-ip 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.keepers.actuals import (
    HITTER_PT,
    HITTER_RATES,
    PITCHER_PT,
    PITCHER_RATES,
    index_by_mlbam,
)
from fantasy_baseball.keepers.persistence import (
    HITTER_COUNTING,
    PITCHER_COUNTING,
    Share,
    apply_share,
    evaluate_shares,
    fit_counting_share,
    fit_share,
    gap,
    rmse,
)
from fantasy_baseball.keepers.vintages import decompose_hitters, decompose_pitchers

PROJECTIONS = PROJECT_ROOT / "data" / "projections"
STATS = PROJECT_ROOT / "data" / "stats"

TRANSITIONS = ((2022, 2023), (2023, 2024), (2024, 2025))

POOLS = {
    "hitter": {
        "pt": HITTER_PT,
        "rates": HITTER_RATES,
        "counting": HITTER_COUNTING,
        "file": "hitters",
        "decompose": decompose_hitters,
    },
    "pitcher": {
        "pt": PITCHER_PT,
        "rates": PITCHER_RATES,
        "counting": PITCHER_COUNTING,
        "file": "pitchers",
        "decompose": decompose_pitchers,
    },
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _projection_path(year: int, stem: str) -> Path:
    exact = PROJECTIONS / str(year) / f"zips-{stem}.csv"
    if exact.exists():
        return exact
    matches = sorted((PROJECTIONS / str(year)).glob(f"zips-{stem}-*.csv"))
    if not matches:
        raise FileNotFoundError(f"no ZiPS {stem} export for {year}")
    return matches[-1]


def load_rates(year: int, kind: str, *, source: str) -> pd.DataFrame:
    """Decomposed rate/PT frame for one (year, pool, source), indexed by mlbam_id.

    `decompose_*` works on BOTH sources because the ZiPS export and our actuals CSVs
    carry the same counting-stat column names (PA/AB/H/HR/R/RBI/SB, IP/SO/W/SV/ER/BB/H)
    keyed by MLBAMID. That is why there is no separate actuals decomposer.
    """
    pool = POOLS[kind]
    path = (
        _projection_path(year, str(pool["file"]))
        if source == "projection"
        else STATS / f"{pool['file']}-{year}.csv"
    )
    frame = pool["decompose"](_read(path))  # type: ignore[operator]
    # A ZiPS export can list a player twice (multi-team rows). Keep the highest-volume
    # entry rather than the first, so the survivor is the real line and not a 12-PA stub.
    pt = str(pool["pt"])
    ordered = frame.sort_values(pt, ascending=False)
    return ordered.loc[~ordered.index.duplicated(keep="first")]


def load_counts(year: int, kind: str, *, source: str) -> pd.DataFrame:
    """Raw counting stats for the 5x5 categories, indexed by mlbam_id."""
    pool = POOLS[kind]
    path = (
        _projection_path(year, str(pool["file"]))
        if source == "projection"
        else STATS / f"{pool['file']}-{year}.csv"
    )
    raw = index_by_mlbam(_read(path), "MLBAMID")
    cols = [*pool["counting"], "PA" if kind == "hitter" else "IP"]
    out = raw[cols].astype(float)
    volume = "PA" if kind == "hitter" else "IP"
    ordered = out.sort_values(volume, ascending=False)
    return ordered.loc[~ordered.index.duplicated(keep="first")]


def build_transition(
    year: int, kind: str, *, min_pt: float, min_next_pt: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aligned (projection_Y, actual_Y, actual_Y+1) panel plus a sample-attrition record.

    Columns are suffixed `_proj`, `_obs`, `_next`. The filter is deliberately asymmetric:
    `min_pt` on year Y selects the population we would actually consider keeping, while
    `min_next_pt` on Y+1 is only high enough to make a rate meaningful. Requiring a Y+1
    appearance at all is a SURVIVORSHIP filter -- players who washed out are dropped, and
    their gaps were the least persistent -- so the counts are reported, not hidden.
    """
    pool = POOLS[kind]
    pt = str(pool["pt"])
    proj = load_rates(year, kind, source="projection")
    obs = load_rates(year, kind, source="actual")
    nxt = load_rates(year + 1, kind, source="actual")

    qualified = obs.index[obs[pt] >= min_pt]
    survived = qualified.intersection(nxt.index[nxt[pt] >= min_next_pt])
    idx = survived.intersection(proj.index)
    attrition = {
        "qualified_year_Y": len(qualified),
        "survived_to_Y1": len(survived),
        "also_projected": len(idx),
    }

    parts = {}
    for name, frame in (("proj", proj), ("obs", obs), ("next", nxt)):
        for col in (pt, *pool["rates"]):
            parts[f"{col}_{name}"] = frame.loc[idx, col]
    return pd.DataFrame(parts, index=idx), attrition


def build_counting_transition(
    year: int, kind: str, *, min_pt: float, min_next_pt: float
) -> pd.DataFrame:
    pool = POOLS[kind]
    volume = "PA" if kind == "hitter" else "IP"
    proj = load_counts(year, kind, source="projection")
    obs = load_counts(year, kind, source="actual")
    nxt = load_counts(year + 1, kind, source="actual")
    idx = (
        obs.index[obs[volume] >= min_pt]
        .intersection(nxt.index[nxt[volume] >= min_next_pt])
        .intersection(proj.index)
    )
    parts = {}
    for name, frame in (("proj", proj), ("obs", obs), ("next", nxt)):
        for col in (*pool["counting"], volume):
            parts[f"{col}_{name}"] = frame.loc[idx, col]
    return pd.DataFrame(parts, index=idx)


def _pooled(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack transitions. The index repeats across transitions by design -- the same
    player contributes one observation per transition -- so it is reset to keep the
    row count honest."""
    return pd.concat(frames, ignore_index=True)


def _fit_column(panel: pd.DataFrame, col: str, pt: str) -> Share:
    return fit_share(
        gap(panel[f"{col}_obs"], panel[f"{col}_proj"]),
        gap(panel[f"{col}_next"], panel[f"{col}_proj"]),
        column=col,
        weights=panel[f"{pt}_obs"],
    )


def run_fit(kind: str, panels: dict[tuple[int, int], pd.DataFrame]) -> dict[str, Share]:
    pool = POOLS[kind]
    pt = str(pool["pt"])
    pooled = _pooled(list(panels.values()))
    fits = {col: _fit_column(pooled, col, pt) for col in (pt, *pool["rates"])}

    print(
        f"\n{'=' * 78}\n{kind.upper()}S -- persistence share S, pooled over {len(panels)} transitions"
    )
    print(f"{'=' * 78}")
    print(f"{'column':<10} {'S':>7} {'+/-':>6} {'intercept':>11} {'r2':>7} {'n':>6}  signal")
    print("-" * 78)
    for col, fit in fits.items():
        flag = "yes" if fit.separable_from_zero else "NO -- gap is noise"
        label = f"{col} (volume)" if col == pt else col
        print(
            f"{label:<10} {fit.share:>7.3f} {fit.stderr:>6.3f} {fit.intercept:>11.5f} "
            f"{fit.r2:>7.3f} {fit.n:>6}  {flag}"
        )
    return fits


def run_validation(kind: str, panels: dict[tuple[int, int], pd.DataFrame]) -> None:
    """Leave-one-transition-out: fit on the others, score on the held-out one.

    With three transitions this is three folds. Reported per fold rather than averaged,
    because the spread ACROSS folds is the noise floor -- a difference between models
    smaller than the spread between folds is not a difference.
    """
    pool = POOLS[kind]
    pt = str(pool["pt"])
    columns = (pt, *pool["rates"])
    print(f"\n{kind.upper()}S -- leave-one-transition-out RMSE (weighted by year-Y volume)")
    print(f"{'-' * 78}")
    print(
        f"{'column':<10} {'holdout':<12} {'S(fit)':>7} {'s=0':>10} {'s=1':>10} {'fitted':>10}  verdict"
    )
    print("-" * 78)
    for col in columns:
        for held, panel in panels.items():
            train = _pooled([p for k, p in panels.items() if k != held])
            fit = _fit_column(train, col, pt)
            scores = evaluate_shares(
                panel[f"{col}_proj"],
                gap(panel[f"{col}_obs"], panel[f"{col}_proj"]),
                panel[f"{col}_next"],
                fit,
                weights=panel[f"{pt}_obs"],
            )
            best = min(scores, key=lambda k: scores[k])
            verdict = "fitted wins" if best == "fitted" else f"{best} wins"
            print(
                f"{col:<10} {held[0]}->{str(held[1])[2:]:<7} {fit.share:>7.3f} "
                f"{scores['s0']:>10.4f} {scores['s1']:>10.4f} {scores['fitted']:>10.4f}  {verdict}"
            )
        print()


def run_counting(kind: str, args: argparse.Namespace) -> None:
    """The blended counting-stat fit, for comparison against the rate fit."""
    pool = POOLS[kind]
    volume = "PA" if kind == "hitter" else "IP"
    min_pt = args.min_pa if kind == "hitter" else args.min_ip
    min_next = args.min_next_pa if kind == "hitter" else args.min_next_ip
    frames = [
        build_counting_transition(y, kind, min_pt=min_pt, min_next_pt=min_next)
        for y, _ in TRANSITIONS
    ]
    pooled = _pooled(frames)
    print(f"\n{kind.upper()}S -- COUNTING-stat S (volume+rate blended) vs the matching RATE S")
    print(f"{'-' * 78}")
    print(f"{'stat':<8} {'S(counting)':>12} {'S(rate)':>9} {'S(volume)':>10}   read")
    print("-" * 78)

    rate_panels = {
        (y, y + 1): build_transition(y, kind, min_pt=min_pt, min_next_pt=min_next)[0]
        for y, _ in TRANSITIONS
    }
    rate_pooled = _pooled(list(rate_panels.values()))
    pt = str(pool["pt"])
    vol_fit = _fit_column(rate_pooled, pt, pt)

    for stat, rate_col in pool["counting"].items():  # type: ignore[union-attr]
        counting = fit_counting_share(
            pooled[f"{stat}_proj"],
            pooled[f"{stat}_obs"],
            pooled[f"{stat}_next"],
            column=stat,
            weights=pooled[f"{volume}_obs"],
        )
        rate_fit = _fit_column(rate_pooled, rate_col, pt)
        drift = counting.share - rate_fit.share
        read = "counting inflated by volume" if drift > 0.05 else "close -- little PT leakage"
        print(
            f"{stat:<8} {counting.share:>12.3f} {rate_fit.share:>9.3f} "
            f"{vol_fit.share:>10.3f}   {read}"
        )


def run_vs_fresh(kind: str, args: argparse.Namespace) -> None:
    """Score the S-adjusted STALE projection against a real FRESH one.

    This is the calibration that matters for the live use case. For 2027 we hold only
    a ZiPS out-year generated before the 2026 season -- there is no fresh 2027 line to
    fall back on. Historically we DO have one (ZiPS Y+1, built with year-Y information),
    so this measures what a fresh projection would have been worth and therefore how
    much the S-adjustment recovers of it.

    `fresh` is the ceiling, `s0` (the untouched stale projection) is the floor, and the
    fraction of that span the fitted model closes is the headline number.
    """
    pool = POOLS[kind]
    pt = str(pool["pt"])
    min_pt = args.min_pa if kind == "hitter" else args.min_ip
    min_next = args.min_next_pa if kind == "hitter" else args.min_next_ip

    print(f"\n{kind.upper()}S -- S-adjusted STALE projection vs a real FRESH projection")
    print(f"{'-' * 78}")
    print(f"{'column':<10} {'stale (s0)':>11} {'fitted':>10} {'fresh Y+1':>11} {'gap closed':>11}")
    print("-" * 78)

    panels = {
        (y, y + 1): build_transition(y, kind, min_pt=min_pt, min_next_pt=min_next)[0]
        for y, _ in TRANSITIONS
    }
    for col in (pt, *pool["rates"]):
        stale_e, fit_e, fresh_e = [], [], []
        for (y, y1), panel in panels.items():
            train = _pooled([p for k, p in panels.items() if k != (y, y1)])
            fit = _fit_column(train, col, pt)
            g = gap(panel[f"{col}_obs"], panel[f"{col}_proj"])
            w = panel[f"{pt}_obs"]
            fresh = load_rates(y1, kind, source="projection")[col].reindex(panel.index)
            stale_e.append(
                rmse(
                    apply_share(
                        panel[f"{col}_proj"], g, Share(col, 0.0, fit.intercept, 0, 0.0, 0.0)
                    ),
                    panel[f"{col}_next"],
                    w,
                )
            )
            fit_e.append(rmse(apply_share(panel[f"{col}_proj"], g, fit), panel[f"{col}_next"], w))
            fresh_e.append(rmse(fresh, panel[f"{col}_next"], w))
        stale, fitted, fresh_m = (sum(v) / len(v) for v in (stale_e, fit_e, fresh_e))
        span = stale - fresh_m
        closed = (
            f"{100.0 * (stale - fitted) / span:.0f}%" if span > 1e-12 else "n/a (fresh no better)"
        )
        print(f"{col:<10} {stale:>11.4f} {fitted:>10.4f} {fresh_m:>11.4f} {closed:>11}")


def run_terciles(kind: str, panels: dict[tuple[int, int], pd.DataFrame]) -> None:
    """Refit S within playing-time terciles.

    A constant S assumes a 250-PA gap is as believable as a 650-PA gap. It is not:
    the small-sample gap carries more noise, so its true share should be LOWER. If S
    rises monotonically across terciles, a constant is the wrong shape and the fix is
    a reliability form, S(n) = n / (n + k), fit per stat.
    """
    pool = POOLS[kind]
    pt = str(pool["pt"])
    pooled = _pooled(list(panels.values()))
    edges = pooled[f"{pt}_obs"].quantile([1 / 3, 2 / 3]).to_list()
    bands = {
        f"low (<{edges[0]:.0f})": pooled[pooled[f"{pt}_obs"] < edges[0]],
        "mid": pooled[(pooled[f"{pt}_obs"] >= edges[0]) & (pooled[f"{pt}_obs"] < edges[1])],
        f"high (>{edges[1]:.0f})": pooled[pooled[f"{pt}_obs"] >= edges[1]],
    }
    print(f"\n{kind.upper()}S -- S by year-Y volume tercile (is a CONSTANT S the right shape?)")
    print(f"{'-' * 78}")
    header = "".join(f"{name:>20}" for name in bands)
    print(f"{'column':<10}{header}   shape")
    print("-" * 78)
    for col in (pt, *pool["rates"]):
        shares = []
        for band in bands.values():
            try:
                shares.append(_fit_column(band, col, pt).share)
            except ValueError:
                shares.append(float("nan"))
        cells = "".join(f"{s:>20.3f}" for s in shares)
        rising = shares[0] < shares[1] < shares[2]
        shape = "RISING -- constant S too crude" if rising else "flat/mixed"
        print(f"{col:<10}{cells}   {shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-pa", type=float, default=300, help="year-Y hitter PA floor")
    parser.add_argument("--min-ip", type=float, default=50, help="year-Y pitcher IP floor")
    parser.add_argument("--min-next-pa", type=float, default=50, help="year-Y+1 hitter PA floor")
    parser.add_argument("--min-next-ip", type=float, default=20, help="year-Y+1 pitcher IP floor")
    parser.add_argument("--counting", action="store_true", help="add the blended counting fit")
    parser.add_argument("--terciles", action="store_true", help="refit S by volume tercile")
    parser.add_argument(
        "--vs-fresh", action="store_true", help="score against a real fresh Y+1 projection"
    )
    parser.add_argument("--pool", choices=("hitter", "pitcher"), help="restrict to one pool")
    args = parser.parse_args()

    kinds = [args.pool] if args.pool else ["hitter", "pitcher"]
    for kind in kinds:
        min_pt = args.min_pa if kind == "hitter" else args.min_ip
        min_next = args.min_next_pa if kind == "hitter" else args.min_next_ip
        panels: dict[tuple[int, int], pd.DataFrame] = {}
        print(f"\n### {kind}s: sample attrition (floor {min_pt:g} in year Y)")
        for y, y1 in TRANSITIONS:
            panel, attrition = build_transition(y, kind, min_pt=min_pt, min_next_pt=min_next)
            panels[(y, y1)] = panel
            lost = attrition["qualified_year_Y"] - attrition["survived_to_Y1"]
            pct = 100.0 * lost / max(attrition["qualified_year_Y"], 1)
            print(
                f"  {y}->{y1}: {attrition['qualified_year_Y']} qualified, "
                f"{attrition['survived_to_Y1']} survived ({lost} lost, {pct:.0f}% "
                f"-- SURVIVORSHIP, inflates S), {attrition['also_projected']} also projected"
            )
        run_fit(kind, panels)
        run_validation(kind, panels)
        if args.counting:
            run_counting(kind, args)
        if args.vs_fresh:
            run_vs_fresh(kind, args)
        if args.terciles:
            run_terciles(kind, panels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
